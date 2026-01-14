"""
Multi-stage verification node for perfect-scoring solutions.

Implements:
1. Chain of Verification (CoVE) - self-questioning
2. LLM-as-judge verification
3. Adversarial edge case generation and testing
4. Augmentation testing
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional

from ..schema import AgentState
from ..actions.code_execution import execute_transformation_code, calculate_grid_results
from ..augmentation import augment_task_data


def _all_examples_success(results):
    """Helper to check if all examples succeeded."""
    if not results:
        return False
    return all(bool(r.get('code_success', False)) for r in results)


def get_perfect_solutions(state: AgentState) -> List[Dict]:
    """Extract solutions with 100% training accuracy."""
    solutions = state.get('solutions_list', [])
    perfect = []
    
    for sol in solutions:
        results = sol.get('training_results', [])
        if _all_examples_success(results):
            perfect.append(sol)
    
    return perfect


def verify_solutions_node(state: AgentState, llm, transformation_llm, code_llm) -> AgentState:
    """Multi-stage verification of perfect-scoring solutions.
    
    Pipeline:
    1. Basic training verification (always enabled)
    2. Chain of Verification (CoVE) if enabled
    3. LLM-as-judge if enabled
    4. Adversarial testing if enabled
    5. Augmentation testing (always if augmentation enabled)
    6. Final confidence scoring and decision
    """
    task_id = state.get('task_id', 'unknown')
    print(f"\n{'='*60}")
    print(f"Task {task_id} [VERIFICATION STAGE]")
    print(f"{'='*60}\n")
    
    # Get configuration
    cove_enabled = state.get('cove_verification', False)
    llm_judge_enabled = state.get('llm_as_judge_verification', False)
    adversarial_enabled = state.get('adversarial_verification', False)
    confidence_threshold = state.get('verification_confidence_threshold', 0.75)
    num_augmentations = state.get('verification_num_augmentations', 10)
    
    # Get all perfect solutions
    perfect_solutions = get_perfect_solutions(state)
    
    if not perfect_solutions:
        print("[verify] No perfect solutions found, continuing evolution...")
        state['verification_passed'] = False
        state['verification_confidence'] = 0.0
        return state
    
    print(f"[verify] Found {len(perfect_solutions)} perfect solution(s)")
    print(f"[verify] Verification enabled: CoVE={cove_enabled}, LLM-Judge={llm_judge_enabled}, Adversarial={adversarial_enabled}")
    
    # Run verification on each perfect solution
    verification_results = []
    for i, solution in enumerate(perfect_solutions):
        print(f"\n--- Verifying Solution {i+1}/{len(perfect_solutions)} ---")
        
        result = {
            'solution': solution,
            'training_score': 1.0,  # Already perfect on training
            'cove_score': None,
            'llm_judge_score': None,
            'adversarial_pass_rate': None,
            'augmentation_pass_rate': None,
            'overall_confidence': 0.0,
            'concerns': [],
            'recommendation': ''
        }
        
        # Stage 1: Chain of Verification
        if cove_enabled:
            print("  [1/4] Running Chain of Verification...")
            cove_result = run_chain_of_verification(state, solution, llm)
            result['cove_score'] = cove_result['score']
            result['concerns'].extend(cove_result.get('concerns', []))
            print(f"  → CoVE Score: {result['cove_score']:.2%}")
        
        # Stage 2: LLM-as-Judge
        if llm_judge_enabled:
            print("  [2/4] Running LLM-as-Judge verification...")
            judge_result = run_llm_as_judge(state, solution, llm)
            result['llm_judge_score'] = judge_result['score']
            result['concerns'].extend(judge_result.get('concerns', []))
            print(f"  → LLM-Judge Score: {result['llm_judge_score']:.2%}")
        
        # Stage 3: Adversarial Testing
        if adversarial_enabled:
            print("  [3/4] Running adversarial edge case testing...")
            adv_result = run_adversarial_testing(state, solution, llm)
            result['adversarial_pass_rate'] = adv_result['pass_rate']
            result['concerns'].extend(adv_result.get('concerns', []))
            print(f"  → Adversarial Pass Rate: {result['adversarial_pass_rate']:.2%}")
        
        # Stage 4: Augmentation Testing (if num_augmentations > 0)
        if num_augmentations > 0:
            print("  [4/4] Testing on augmented examples...")
            aug_result = run_augmentation_testing(state, solution, num_augmentations)
            result['augmentation_pass_rate'] = aug_result['pass_rate']
            if aug_result['pass_rate'] < 0.8:
                result['concerns'].append(f"Low augmentation pass rate: {aug_result['pass_rate']:.1%}")
            print(f"  → Augmentation Pass Rate: {result['augmentation_pass_rate']:.2%}")
        
        # Calculate overall confidence
        result['overall_confidence'] = calculate_overall_confidence(
            result, cove_enabled, llm_judge_enabled, adversarial_enabled, num_augmentations > 0
        )
        result['recommendation'] = make_recommendation(result['overall_confidence'], confidence_threshold)
        
        print(f"  → Overall Confidence: {result['overall_confidence']:.2%}")
        print(f"  → Recommendation: {result['recommendation']}")
        
        verification_results.append(result)
    
    # Select best verified solution
    best = max(verification_results, key=lambda x: x['overall_confidence'])
    
    # Store results in state
    state['verification_results'] = verification_results
    state['best_verified_solution'] = best['solution']
    state['verification_confidence'] = best['overall_confidence']
    state['verification_concerns'] = best['concerns']
    
    # Decision: pass if confidence meets threshold
    if best['overall_confidence'] >= confidence_threshold:
        print(f"\n✅ VERIFICATION PASSED (confidence: {best['overall_confidence']:.2%})")
        state['verification_passed'] = True
    else:
        print(f"\n⚠️  VERIFICATION UNCERTAIN (confidence: {best['overall_confidence']:.2%})")
        print(f"   Concerns: {', '.join(best['concerns'][:3])}")
        state['verification_passed'] = False
        
        # Add feedback for next evolution round
        state['verification_feedback'] = {
            'concerns': best['concerns'],
            'recommendation': 'continue_search'
        }
    
    return state


def run_chain_of_verification(state: AgentState, solution: Dict, llm) -> Dict:
    """Implement Chain of Verification (CoVE).
    
    Process:
    1. Generate verification questions about the solution
    2. Answer each question independently
    3. Cross-check answers for consistency
    """
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    
    # Step 1: Generate verification questions
    question_prompt = f"""You have a solution that scores 100% on training examples for an ARC task.
Generate 5 critical verification questions to test if this solution truly generalizes.

Training Examples: {json.dumps(task_data.get('train', [])[:2])}  # Only first 2 for brevity

Solution Code:
```python
{code}
```

Generate questions that probe:
1. What is the core transformation rule?
2. What assumptions does the code make?
3. What edge cases might break it?
4. Is it specific to training or general?
5. How would it handle variations?

Return ONLY a JSON object:
{{"questions": ["Question 1?", "Question 2?", ...]}}
"""
    
    try:
        questions_response = llm.invoke(question_prompt)
        questions = parse_json_response(questions_response, 'questions', [])
        
        if not questions:
            return {'score': 0.5, 'concerns': ['Could not generate verification questions']}
        
        # Step 2: Answer each question
        answers = []
        for q in questions[:5]:  # Limit to 5 questions
            answer_prompt = f"""Answer this verification question about the ARC solution:

Question: {q}

Solution Code:
```python
{code}
```

Provide a brief, analytical answer."""
            
            answer_response = llm.invoke(answer_prompt)
            answer_text = answer_response if isinstance(answer_response, str) else getattr(answer_response, 'content', str(answer_response))
            answers.append({'question': q, 'answer': answer_text})
        
        # Step 3: Cross-check for confidence
        consistency_prompt = f"""Review these Q&A pairs about an ARC solution.
Rate your confidence that this solution will generalize well.

Q&A:
{json.dumps(answers, indent=2)}

Return JSON:
{{
  "confidence_score": 0.0-1.0,
  "concerns": ["concern 1", ...]
}}
"""
        
        consistency_response = llm.invoke(consistency_prompt)
        consistency = parse_json_response(consistency_response)
        
        return {
            'score': consistency.get('confidence_score', 0.5),
            'concerns': consistency.get('concerns', []),
            'questions': answers
        }
    
    except Exception as e:
        print(f"  Warning: CoVE failed: {e}")
        return {'score': 0.5, 'concerns': [f'CoVE error: {str(e)}']}


def run_llm_as_judge(state: AgentState, solution: Dict, llm) -> Dict:
    """Use LLM to judge solution quality holistically."""
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    training_results = solution.get('training_results', [])
    
    prompt = f"""You are an expert judge evaluating an ARC solution.

Task: {json.dumps(task_data.get('train', []))}

Solution Code:
```python
{code}
```

Training Results: All {len(training_results)} examples passed ✓

Evaluate this solution on:
1. Generalization potential (0-1)
2. Code quality and clarity (0-1)
3. Robustness to variations (0-1)
4. Likelihood it's overfitted (0-1, where 1=definitely overfitted)

Return JSON:
{{
  "generalization": 0.0-1.0,
  "quality": 0.0-1.0,
  "robustness": 0.0-1.0,
  "overfit_risk": 0.0-1.0,
  "concerns": ["..."]
}}
"""
    
    try:
        response = llm.invoke(prompt)
        result = parse_json_response(response)
        
        # Calculate composite score (lower overfit_risk is better)
        score = (
            result.get('generalization', 0.5) * 0.4 +
            result.get('quality', 0.5) * 0.2 +
            result.get('robustness', 0.5) * 0.3 +
            (1.0 - result.get('overfit_risk', 0.5)) * 0.1
        )
        
        return {
            'score': score,
            'concerns': result.get('concerns', [])
        }
    
    except Exception as e:
        print(f"  Warning: LLM-as-Judge failed: {e}")
        return {'score': 0.5, 'concerns': [f'Judge error: {str(e)}']}


def run_adversarial_testing(state: AgentState, solution: Dict, llm) -> Dict:
    """Generate and test adversarial edge cases."""
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    
    # Generate edge cases
    prompt = f"""Generate 3-5 adversarial test cases for this ARC solution.

Training Examples: {json.dumps(task_data.get('train', [])[:2])}

Solution Code:
```python
{code}
```

Generate edge cases with different:
- Grid sizes
- Color patterns
- Sparse/dense layouts

Return JSON:
{{"edge_cases": [{{"input": [[...]], "rationale": "..."}}]}}
"""
    
    try:
        response = llm.invoke(prompt)
        edge_cases = parse_json_response(response, 'edge_cases', [])
        
        if not edge_cases:
            return {'pass_rate': 0.5, 'concerns': ['Could not generate edge cases']}
        
        # Test each edge case
        passed = 0
        failed_cases = []
        
        for case in edge_cases[:5]:  # Limit to 5
            input_grid = case.get('input')
            if not input_grid:
                continue
            
            try:
                output, error = execute_transformation_code(code, input_grid, timeout_seconds=5)
                if error is None and output is not None:
                    passed += 1
                else:
                    failed_cases.append(case.get('rationale', 'Unknown'))
            except:
                failed_cases.append(case.get('rationale', 'Execution error'))
        
        total = len(edge_cases[:5])
        pass_rate = passed / total if total > 0 else 0.0
        
        concerns = []
        if pass_rate < 0.6:
            concerns.append(f"Failed {len(failed_cases)} adversarial cases: {', '.join(failed_cases[:2])}")
        
        return {'pass_rate': pass_rate, 'concerns': concerns}
    
    except Exception as e:
        print(f"  Warning: Adversarial testing failed: {e}")
        return {'pass_rate': 0.5, 'concerns': [f'Adversarial error: {str(e)}']}


def run_augmentation_testing(state: AgentState, solution: Dict, num_augmentations: int) -> Dict:
    """Test solution on augmented examples."""
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    
    try:
        # Generate augmented examples
        augmented = augment_task_data(task_data, num_augmentations)
        aug_examples = augmented.get('train', [])
        
        if not aug_examples:
            return {'pass_rate': 0.5, 'concerns': ['No augmented examples generated']}
        
        # Test on augmented examples
        passed = 0
        for example in aug_examples:
            input_grid = example.get('input')
            expected_output = example.get('output')
            
            try:
                predicted_output, error = execute_transformation_code(code, input_grid, timeout_seconds=5)
                
                if predicted_output and expected_output:
                    match = (predicted_output == expected_output)
                    if match:
                        passed += 1
            except:
                pass
        
        pass_rate = passed / len(aug_examples)
        
        return {'pass_rate': pass_rate, 'concerns': []}
    
    except Exception as e:
        print(f"  Warning: Augmentation testing failed: {e}")
        return {'pass_rate': 0.5, 'concerns': [f'Augmentation error: {str(e)}']}


def calculate_overall_confidence(result: Dict, cove_enabled: bool, llm_judge_enabled: bool, 
                                 adversarial_enabled: bool, augmentation_enabled: bool) -> float:
    """Calculate weighted confidence score from all enabled verifications."""
    scores = []
    weights = []
    
    # Training score (always counted)
    scores.append(result['training_score'])
    weights.append(0.3)
    
    # CoVE score
    if cove_enabled and result['cove_score'] is not None:
        scores.append(result['cove_score'])
        weights.append(0.25)
    
    # LLM-as-judge score
    if llm_judge_enabled and result['llm_judge_score'] is not None:
        scores.append(result['llm_judge_score'])
        weights.append(0.25)
    
    # Adversarial score
    if adversarial_enabled and result['adversarial_pass_rate'] is not None:
        scores.append(result['adversarial_pass_rate'])
        weights.append(0.15)
    
    # Augmentation score
    if augmentation_enabled and result['augmentation_pass_rate'] is not None:
        scores.append(result['augmentation_pass_rate'])
        weights.append(0.15)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return result['training_score']
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Calculate weighted average
    confidence = sum(s * w for s, w in zip(scores, normalized_weights))
    return confidence


def make_recommendation(confidence: float, threshold: float) -> str:
    """Make recommendation based on confidence."""
    if confidence >= threshold:
        return "accept"
    elif confidence >= threshold - 0.15:
        return "refine"
    else:
        return "continue_search"


def parse_json_response(response, key=None, default=None):
    """Parse JSON from LLM response."""
    content = response if isinstance(response, str) else getattr(response, 'content', str(response))
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    
    try:
        data = json.loads(content)
        if key:
            return data.get(key, default)
        return data
    except:
        if key:
            return default
        return {}
