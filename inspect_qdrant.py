#!/usr/bin/env python3
"""
Interactive Qdrant Database Inspector

This script allows you to:
1. Browse past runs using arrow keys
2. Inspect the contents of a Qdrant database
3. View the first few rows of stored reasoning traces
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Try to import required packages
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    print("Error: qdrant-client not installed. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False
    sys.exit(1)

try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    print("Error: curses not available (required for arrow key navigation)")
    CURSES_AVAILABLE = False
    sys.exit(1)


def find_runs(base_dir: str = "output/output_agent") -> List[Dict[str, Any]]:
    """Find all runs with qdrant_db directories.
    
    Returns:
        List of dicts with 'path', 'name', 'timestamp', and 'has_qdrant' keys
    """
    runs = []
    
    if not os.path.exists(base_dir):
        return runs
    
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        
        qdrant_path = os.path.join(entry_path, "qdrant_db")
        has_qdrant = os.path.exists(qdrant_path)
        
        # Get modification time
        try:
            mtime = os.path.getmtime(entry_path)
            timestamp = datetime.fromtimestamp(mtime)
        except Exception:
            timestamp = None
        
        runs.append({
            'path': entry_path,
            'name': entry,
            'timestamp': timestamp,
            'has_qdrant': has_qdrant,
            'qdrant_path': qdrant_path if has_qdrant else None
        })
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x['timestamp'] or datetime.min, reverse=True)
    
    return runs


def get_collection_info(qdrant_path: str) -> Optional[Dict[str, Any]]:
    """Load collection info from qdrant_db directory.
    
    Returns:
        Dict with collection metadata or None if not found
    """
    info_path = os.path.join(qdrant_path, "collection_info.json")
    if not os.path.exists(info_path):
        return None
    
    try:
        with open(info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading collection info: {e}")
        return None


def inspect_qdrant_collection(qdrant_path: str, num_rows: int = 5) -> None:
    """Inspect the contents of a Qdrant collection.
    
    Args:
        qdrant_path: Path to the qdrant_db directory
        num_rows: Number of rows to display (default: 5)
    """
    # Load collection info
    collection_info = get_collection_info(qdrant_path)
    if not collection_info:
        print(f"\n❌ No collection_info.json found in {qdrant_path}")
        return
    
    collection_name = collection_info.get('collection_name')
    if not collection_name:
        print(f"\n❌ No collection_name in collection_info.json")
        return
    
    print(f"\n{'='*80}")
    print(f"Qdrant Collection: {collection_name}")
    print(f"Database Path: {qdrant_path}")
    print(f"{'='*80}\n")
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(path=qdrant_path)
        
        # Check if collection exists
        if not client.collection_exists(collection_name):
            print(f"❌ Collection '{collection_name}' does not exist in the database")
            return
        
        # Get collection info
        collection = client.get_collection(collection_name)
        print(f"Collection Status: {collection.status}")
        print(f"Points Count: {collection.points_count}")
        
        # Get collection stats for vector info
        try:
            # Try to get vectors count from config
            if hasattr(collection, 'config') and hasattr(collection.config, 'params'):
                vector_size = collection.config.params.vectors.size
                print(f"Vector Size: {vector_size}")
        except Exception:
            pass
        
        print(f"\n{'='*80}")
        print(f"First {num_rows} Records (ReasoningTraceRecord):")
        print(f"{'='*80}\n")
        
        # Scroll through records
        records = client.scroll(
            collection_name=collection_name,
            limit=num_rows,
            with_payload=True,
            with_vectors=False  # Don't fetch vectors (they're large)
        )
        
        points, _ = records
        
        if not points:
            print("No records found in collection")
            return
        
        # Display each record
        for i, point in enumerate(points, 1):
            print(f"\n--- Record {i} (ID: {point.id}) ---")
            
            payload = point.payload or {}
            
            # Display ReasoningTraceRecord fields
            # These are the fields stored by store_record() in actions.py
            print(f"  Record Type: ReasoningTraceRecord")
            
            # Display reasoning summary (concise strategy)
            if 'reasoning_summary' in payload:
                summary = payload['reasoning_summary']
                print(f"  Reasoning Summary: {summary}")
            
            # Display concepts identified
            if 'concepts' in payload:
                concepts = payload['concepts']
                if concepts:
                    print(f"  Concepts ({len(concepts)}): {', '.join(concepts)}")
                else:
                    print(f"  Concepts: None")
            
            # Display helper functions count
            if 'helpers' in payload:
                helpers = payload['helpers']
                if helpers and isinstance(helpers, list):
                    print(f"  Helper Functions: {len(helpers)} defined")
                    # Show first few helper names if available
                    helper_names = []
                    for h in helpers[:3]:
                        if isinstance(h, dict) and 'name' in h:
                            helper_names.append(h['name'])
                    if helper_names:
                        preview = ', '.join(helper_names)
                        if len(helpers) > 3:
                            preview += f" ... (+{len(helpers)-3} more)"
                        print(f"    Names: {preview}")
                else:
                    print(f"  Helper Functions: None")
            
            # Display reasoning text preview (full reasoning analysis)
            if 'reasoning_text' in payload:
                trace = payload['reasoning_text']
                if trace:
                    # Show first 300 chars
                    preview = trace[:300].replace('\n', ' ').strip()
                    if len(trace) > 300:
                        preview += "..."
                    print(f"  Reasoning Text (preview): {preview}")
                else:
                    print(f"  Reasoning Text: Empty")
            
            # Count of any other unexpected fields
            expected_fields = {'reasoning_text', 'reasoning_summary', 'concepts', 'helpers'}
            other_fields = [k for k in payload.keys() if k not in expected_fields]
            if other_fields:
                print(f"  Other fields ({len(other_fields)}): {', '.join(other_fields)}")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error inspecting collection: {e}")
    finally:
        try:
            client.close()
        except Exception:
            pass


def draw_menu(stdscr, runs: List[Dict[str, Any]], selected_idx: int) -> None:
    """Draw the selection menu using curses.
    
    Args:
        stdscr: Curses window object
        runs: List of run dictionaries
        selected_idx: Currently selected index
    """
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    
    # Title
    title = "Qdrant Database Inspector - Select a Run"
    stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
    stdscr.addstr(1, 0, "=" * width)
    
    # Instructions
    instructions = "Use ↑/↓ arrows to navigate, Enter to select, 'q' to quit"
    stdscr.addstr(2, (width - len(instructions)) // 2, instructions)
    stdscr.addstr(3, 0, "=" * width)
    
    # Calculate visible window
    visible_start = max(0, selected_idx - (height - 8) // 2)
    visible_end = min(len(runs), visible_start + height - 8)
    
    # Display runs
    for i in range(visible_start, visible_end):
        run = runs[i]
        y_pos = 4 + (i - visible_start)
        
        # Format run info
        qdrant_marker = "✓" if run['has_qdrant'] else "✗"
        timestamp_str = run['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if run['timestamp'] else "Unknown"
        line = f" {qdrant_marker} [{timestamp_str}] {run['name']}"
        
        # Truncate if needed
        if len(line) > width - 1:
            line = line[:width-4] + "..."
        
        # Highlight selected
        if i == selected_idx:
            stdscr.addstr(y_pos, 0, line, curses.A_REVERSE)
        else:
            stdscr.addstr(y_pos, 0, line)
    
    # Status bar
    status = f"Showing {visible_start + 1}-{visible_end} of {len(runs)} runs"
    stdscr.addstr(height - 1, 0, status, curses.A_DIM)
    
    stdscr.refresh()


def select_run_interactive(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Interactive run selection using arrow keys.
    
    Args:
        runs: List of run dictionaries
        
    Returns:
        Selected run dict or None if cancelled
    """
    if not runs:
        print("No runs found in output/output_agent")
        return None
    
    def _select(stdscr):
        curses.curs_set(0)  # Hide cursor
        selected_idx = 0
        
        while True:
            draw_menu(stdscr, runs, selected_idx)
            
            # Get key
            key = stdscr.getch()
            
            # Handle navigation
            if key == curses.KEY_UP and selected_idx > 0:
                selected_idx -= 1
            elif key == curses.KEY_DOWN and selected_idx < len(runs) - 1:
                selected_idx += 1
            elif key in [curses.KEY_ENTER, 10, 13]:  # Enter key
                return runs[selected_idx]
            elif key in [ord('q'), ord('Q'), 27]:  # q or Escape
                return None
    
    return curses.wrapper(_select)


def main():
    """Main entry point for the inspector."""
    if not QDRANT_AVAILABLE or not CURSES_AVAILABLE:
        return 1
    
    print("Loading runs...")
    runs = find_runs()
    
    if not runs:
        print("No runs found in output/output_agent")
        return 1
    
    # Filter to only runs with Qdrant databases
    qdrant_runs = [r for r in runs if r['has_qdrant']]
    
    if not qdrant_runs:
        print(f"Found {len(runs)} runs, but none have Qdrant databases (qdrant_db folder)")
        return 1
    
    print(f"Found {len(qdrant_runs)} runs with Qdrant databases")
    
    # Interactive selection
    selected_run = select_run_interactive(qdrant_runs)
    
    if not selected_run:
        print("\nSelection cancelled")
        return 0
    
    # Inspect the selected run
    num_rows = 5
    
    # Check if user wants to see more rows
    print(f"\nInspecting: {selected_run['name']}")
    try:
        response = input(f"How many records to display? (default: {num_rows}): ").strip()
        if response:
            num_rows = int(response)
    except (ValueError, KeyboardInterrupt):
        pass
    
    inspect_qdrant_collection(selected_run['qdrant_path'], num_rows=num_rows)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
