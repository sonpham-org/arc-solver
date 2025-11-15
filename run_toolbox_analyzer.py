#!/usr/bin/env python3
"""
Command-line interface for analyzing run-specific ARC toolboxes.
Since each run creates its own isolated toolbox, this tool helps analyze
toolboxes from specific runs or compare across multiple runs.
"""
import argparse
import sys
import os
import glob
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persistent_toolbox import PersistentToolbox


def find_run_toolboxes():
    """Find all available run toolboxes in the output directory."""
    toolbox_paths = []
    for run_dir in glob.glob("output/*/"):
        toolbox_db = os.path.join(run_dir, "toolbox.db")
        if os.path.exists(toolbox_db):
            toolbox_paths.append(run_dir)
    return sorted(toolbox_paths)


def cmd_list_runs(args):
    """List all available run toolboxes."""
    runs = find_run_toolboxes()
    
    if not runs:
        print("No run toolboxes found in output directory.")
        return
    
    print("🔧 Available ARC Run Toolboxes")
    print("=" * 50)
    for run_dir in runs:
        try:
            toolbox = PersistentToolbox(run_dir)
            stats = toolbox.get_statistics()
            print(f"📁 {os.path.basename(run_dir)}")
            print(f"   Functions: {stats['total_functions']}")
            print(f"   Usage records: {stats['total_usage_records']}")
            if os.path.exists(os.path.join(run_dir, "params.json")):
                with open(os.path.join(run_dir, "params.json")) as f:
                    params = json.load(f)
                    print(f"   Model: {params.get('model', 'unknown')}")
                    print(f"   Mode: {params.get('mode', 'unknown')}")
        except Exception as e:
            print(f"📁 {os.path.basename(run_dir)} (error: {e})")
        print()


def cmd_stats(args):
    """Display toolbox statistics for a specific run."""
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory '{args.run_dir}' not found.")
        return
        
    if not os.path.exists(os.path.join(args.run_dir, "toolbox.db")):
        print(f"Error: No toolbox found in '{args.run_dir}'.")
        return
    
    toolbox = PersistentToolbox(args.run_dir)
    stats = toolbox.get_statistics()
    
    print(f"🔧 ARC Run Toolbox Statistics: {os.path.basename(args.run_dir)}")
    print("=" * 60)
    print(f"Total functions: {stats['total_functions']}")
    print(f"Total usage records: {stats['total_usage_records']}")
    
    if stats['categories']:
        print("\nFunctions by category:")
        for category, info in stats['categories'].items():
            print(f"  {category}: {info['count']} functions (avg success: {info['avg_success']:.1%})")
    
    # Show top performing functions
    functions = toolbox.get_all_functions(max_functions=5)
    if functions:
        print("\nTop performing functions:")
        for func in functions[:5]:
            print(f"  {func['name']}: {func['usage_count']} uses, {func['success_rate']:.1%} success")


def cmd_list(args):
    """List helper functions from a specific run."""
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory '{args.run_dir}' not found.")
        return
        
    if not os.path.exists(os.path.join(args.run_dir, "toolbox.db")):
        print(f"Error: No toolbox found in '{args.run_dir}'.")
        return
    
    toolbox = PersistentToolbox(args.run_dir)
    
    functions = toolbox.get_all_functions(
        category=args.category,
        min_success_rate=args.min_success_rate / 100.0,
        max_functions=args.limit
    )
    
    if not functions:
        print("No functions found matching criteria.")
        return
    
    print(f"Found {len(functions)} helper function(s):")
    print("-" * 80)
    
    for func in functions:
        print(f"📋 {func['name']} ({func['category']})")
        print(f"   📖 {func['description']}")
        print(f"   📊 Usage: {func['usage_count']}, Success: {func['success_rate']:.1%}")
        if args.show_code:
            print(f"   💻 Code:")
            for line in func['code'].split('\n'):
                print(f"      {line}")
        print()


def cmd_export(args):
    """Export a run's toolbox to JSON."""
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory '{args.run_dir}' not found.")
        return
        
    if not os.path.exists(os.path.join(args.run_dir, "toolbox.db")):
        print(f"Error: No toolbox found in '{args.run_dir}'.")
        return
    
    toolbox = PersistentToolbox(args.run_dir)
    export_path = toolbox.export_toolbox(args.output)
    print(f"Toolbox exported to: {export_path}")


def cmd_compare(args):
    """Compare toolboxes from multiple runs."""
    if len(args.run_dirs) < 2:
        print("Error: Need at least 2 run directories to compare.")
        return
    
    print("🔧 ARC Toolbox Comparison")
    print("=" * 50)
    
    for run_dir in args.run_dirs:
        if not os.path.exists(run_dir):
            print(f"Warning: Run directory '{run_dir}' not found, skipping.")
            continue
            
        if not os.path.exists(os.path.join(run_dir, "toolbox.db")):
            print(f"Warning: No toolbox found in '{run_dir}', skipping.")
            continue
        
        try:
            toolbox = PersistentToolbox(run_dir)
            stats = toolbox.get_statistics()
            print(f"\n📁 {os.path.basename(run_dir)}:")
            print(f"   Functions: {stats['total_functions']}")
            print(f"   Usage records: {stats['total_usage_records']}")
            
            # Show unique functions by category
            if stats['categories']:
                for category, info in stats['categories'].items():
                    print(f"   {category}: {info['count']} functions")
                    
        except Exception as e:
            print(f"Error analyzing {run_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ARC run-specific toolboxes")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List runs command
    list_runs_parser = subparsers.add_parser("list-runs", help="List all available run toolboxes")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show toolbox statistics for a run")
    stats_parser.add_argument("run_dir", help="Run directory (e.g., output/2024-11-13T10-30-45-123456)")
    
    # List functions command
    list_parser = subparsers.add_parser("list", help="List functions in a run's toolbox")
    list_parser.add_argument("run_dir", help="Run directory")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.add_argument("--min-success-rate", type=float, default=0.0, 
                           help="Minimum success rate percentage (0-100)")
    list_parser.add_argument("--limit", type=int, help="Maximum functions to show")
    list_parser.add_argument("--show-code", action="store_true", help="Show function code")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export run toolbox to JSON")
    export_parser.add_argument("run_dir", help="Run directory")
    export_parser.add_argument("output", nargs="?", help="Output JSON file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple run toolboxes")
    compare_parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")
    
    args = parser.parse_args()
    
    if args.command == "list-runs":
        cmd_list_runs(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())