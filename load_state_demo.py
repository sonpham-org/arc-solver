"""Small demo utility to load a saved `latest_state.json` and present a
brief summary using the `AgentState` schema as the base type.

This script is intentionally lightweight: it reads the JSON file, casts it
to `AgentState` (from `agent.schema`) for typing purposes, performs a few
runtime sanity checks, and prints a short human-readable summary.
"""
from typing import Optional, cast, Any, Dict, List
import json
import os
import sys

from agent.schema import AgentState


def load_latest_state(path: Optional[str] = None) -> AgentState:
	"""Load `latest_state.json` and return it as an `AgentState`.

	Args:
		path: Optional path to the JSON file. If omitted, defaults to
			  `data/sample_states/latest_state.json` relative to the repo root.

	Returns:
		An object typed as `AgentState` (no deep validation beyond a few
		sanity checks).
	"""
	if path is None:
		repo_root = os.path.dirname(os.path.abspath(__file__))
		path = os.path.join(repo_root, "data", "sample_states", "latest_state.json")

	if not os.path.isfile(path):
		raise FileNotFoundError(f"latest_state.json not found at: {path}")

	with open(path, 'r') as fh:
		loaded = json.load(fh)

	# Basic runtime sanity checks (keys that AgentState usually contains)
	if not isinstance(loaded, dict):
		raise TypeError("Loaded state is not a JSON object/dict")

	# Cast to AgentState for callers & type checkers. TypedDict is not
	# enforced at runtime, so we do a couple of simple checks below.
	state = cast(AgentState, loaded)

	# Sanity checks (non-exhaustive)
	required_keys = ["task_id", "task_data", "task_folder", "solutions_list"]
	for k in required_keys:
		if k not in state:
			print(f"Warning: expected key '{k}' missing from loaded state")

	# Ensure lists exist
	for list_key in ("solutions_list", "seed_solutions_list", "fused_solutions_list", "mutated_solutions_list"):
		if state.get(list_key) is None:
			state[list_key] = []  # type: ignore[index]

	return state


def summarize_state(state: AgentState) -> None:
	"""Print a short summary of the loaded `AgentState`."""
	tid = state.get('task_id') or '<unknown>'
	folder = state.get('task_folder') or '<none>'
	cur_gen = int(state.get('current_generation') or 0)
	max_gen = int(state.get('max_generations') or 0)
	cur_loop = int(state.get('current_loop') or 0)

	sols = state.get('solutions_list') or []
	num_sols = len(sols)

	print(f"Loaded AgentState for task: {tid}")
	print(f"  task_folder: {folder}")
	print(f"  current_generation: {cur_gen} / {max_gen}")
	print(f"  current_loop: {cur_loop}")
	print(f"  total solutions in `solutions_list`: {num_sols}")

	# Compute simple aggregate metrics across solutions if present
	if num_sols:
		train_rates: List[float] = []
		overlap_avgs: List[float] = []
		for s in sols:
			try:
				tr = float(s.get('training_success_rate', 0.0))
			except Exception:
				tr = 0.0
			try:
				ov = float(s.get('training_overlap_average', 0.0))
			except Exception:
				ov = 0.0
			train_rates.append(tr)
			overlap_avgs.append(ov)

		avg_train = sum(train_rates) / len(train_rates) if train_rates else 0.0
		avg_overlap = sum(overlap_avgs) / len(overlap_avgs) if overlap_avgs else 0.0
		max_train = max(train_rates) if train_rates else 0.0
		max_overlap = max(overlap_avgs) if overlap_avgs else 0.0

		print(f"  avg training success: {avg_train:.2%}")
		print(f"  max training success: {max_train:.2%}")
		print(f"  avg training overlap: {avg_overlap:.2f}%")
		print(f"  max training overlap: {max_overlap:.2f}%")

		# Print top 3 solutions by training success
		ranked = sorted(enumerate(sols), key=lambda iv: float(iv[1].get('training_success_rate', 0.0)), reverse=True)
		print("  Top solutions (index: training_success_rate, training_overlap):")
		for idx, sol in ranked[:3]:
			ts = float(sol.get('training_success_rate', 0.0))
			to = float(sol.get('training_overlap_average', 0.0))
			print(f"    {idx}: {ts:.2%}, overlap {to:.2f}%")
	else:
		print("  No solutions found in `solutions_list`.")


def main(argv: Optional[List[str]] = None) -> int:
	argv = argv if argv is not None else sys.argv[1:]
	path = None
	if argv:
		path = argv[0]

	try:
		state = load_latest_state(path)
		summarize_state(state)
	except Exception as e:
		print(f"Failed to load/parse latest_state.json: {e}")
		return 2

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

