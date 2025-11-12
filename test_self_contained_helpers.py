#!/usr/bin/env python3
"""
Test script to verify that helper databases are created in run output directories,
making each run self-contained.
"""

import os
import tempfile
import shutil
from pathlib import Path

# Import the helper database class
from arc_prompt_with_helpers import HelperDatabase, run_with_helper_learning

def test_helper_db_in_output_dir():
    """Test that helper database is created in the output directory."""
    
    print("🧪 Testing self-contained helper database placement...")
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   Created test output directory: {output_dir}")
        
        # Mock minimal task data for testing
        mock_tasks = [
            ("test_task_1", {"train": [], "test": []}),
        ]
        
        mock_solutions = {"test_task_1": []}
        
        # Test with helper database enabled
        print("\n   Testing helper database creation...")
        
        try:
            # This should create helpers.db inside the output directory
            results = run_with_helper_learning(
                selected_tasks=mock_tasks,
                solutions=mock_solutions,
                model="gemini-2.5-flash-lite-preview-06-17",
                provider="google",
                output_dir=str(output_dir),
                use_sanitized_task=False,
                use_sanitized_id=True,
                print_input=False,
                print_output=False,
                print_json=False,
                use_smart_router=False,
                max_reflections=1,
                task_retries=0,
                parallel=False,
                use_helpers=True,
                redo_tasks=0,
                helper_db_path="helpers.db"  # This should be ignored in favor of output_dir
            )
            
            # Check if database was created in output directory
            expected_db_path = output_dir / "helpers.db"
            
            if expected_db_path.exists():
                print(f"   ✅ SUCCESS: Helper database created at {expected_db_path}")
                
                # Test that we can connect to the database and it has the right structure
                helper_db = HelperDatabase(str(expected_db_path))
                stats = helper_db.get_database_stats()
                print(f"   ✅ Database is functional with {stats['total_helpers']} helpers")
                
                return True
            else:
                print(f"   ❌ FAILURE: Helper database not found at {expected_db_path}")
                
                # List what files are actually in the output directory
                print(f"   Files in output directory:")
                for item in output_dir.iterdir():
                    print(f"     - {item.name}")
                
                return False
                
        except Exception as e:
            print(f"   ❌ FAILURE: Exception occurred: {e}")
            return False

def test_multiple_runs_isolated():
    """Test that multiple runs have isolated helper databases."""
    
    print("\n🧪 Testing isolation between multiple runs...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create two separate run directories
        run1_dir = Path(temp_dir) / "run1"
        run2_dir = Path(temp_dir) / "run2"
        
        run1_dir.mkdir(parents=True, exist_ok=True)
        run2_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   Created run directories: {run1_dir} and {run2_dir}")
        
        # Create helper databases in each run directory
        db1_path = run1_dir / "helpers.db"
        db2_path = run2_dir / "helpers.db"
        
        db1 = HelperDatabase(str(db1_path))
        db2 = HelperDatabase(str(db2_path))
        
        # Add a helper to the first database
        test_helper_code = """
def test_helper():
    '''A test helper function'''
    return "test_result"
"""
        
        db1.store_helper_function(test_helper_code, "task1", True)
        
        # Check that first database has the helper but second doesn't
        stats1 = db1.get_database_stats()
        stats2 = db2.get_database_stats()
        
        if stats1['total_helpers'] == 1 and stats2['total_helpers'] == 0:
            print("   ✅ SUCCESS: Databases are properly isolated")
            
            # Verify the databases are at different paths
            if db1_path.exists() and db2_path.exists() and db1_path != db2_path:
                print("   ✅ SUCCESS: Database files are at different locations")
                return True
            else:
                print("   ❌ FAILURE: Database files not properly separated")
                return False
        else:
            print(f"   ❌ FAILURE: Database isolation failed. DB1: {stats1['total_helpers']}, DB2: {stats2['total_helpers']}")
            return False

if __name__ == "__main__":
    print("Testing self-contained helper database functionality...\n")
    
    test1_passed = test_helper_db_in_output_dir()
    test2_passed = test_multiple_runs_isolated()
    
    print(f"\n{'='*60}")
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Helper databases are properly self-contained.")
    else:
        print("❌ Some tests failed. Check output above for details.")
    print(f"{'='*60}")