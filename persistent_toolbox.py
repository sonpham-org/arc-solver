#!/usr/bin/env python3
"""
Persistent Helper Function Storage for ARC LangGraph Agent.

This module manages a growing toolbox of helper functions that persists
across different task runs, allowing the agent to accumulate knowledge
and become more capable over time.
"""
import json
import os
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlite3


class PersistentToolbox:
    """
    Manages a run-specific storage of helper functions for a single ARC task run.
    Each run creates its own isolated toolbox that starts fresh but can grow during the run.
    Uses SQLite for efficient storage and retrieval.
    """
    
    def __init__(self, run_output_dir: str):
        """
        Initialize the run-specific persistent toolbox.
        
        Args:
            run_output_dir: Path to the run's output directory where toolbox will be stored
        """
        os.makedirs(run_output_dir, exist_ok=True)
        self.storage_path = os.path.join(run_output_dir, "toolbox.db")
        self.run_output_dir = run_output_dir
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            # Helper functions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS helper_functions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    code TEXT NOT NULL,
                    code_hash TEXT UNIQUE NOT NULL,
                    category TEXT DEFAULT 'general',
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_task_id TEXT,
                    complexity_score INTEGER DEFAULT 1,
                    dependencies TEXT DEFAULT '[]'
                )
            """)
            
            # Usage tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS function_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    function_name TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    execution_time REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (function_name) REFERENCES helper_functions (name)
                )
            """)
            
            # Task success tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    functions_used TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL,
                    UNIQUE(task_id, attempt_number)
                )
            """)
            
            conn.commit()
    
    def _get_code_hash(self, code: str) -> str:
        """Generate a hash for the code to detect duplicates."""
        return hashlib.md5(code.encode('utf-8')).hexdigest()
    
    def add_helper_function(self, name: str, description: str, code: str, 
                          category: str = "general", source_task_id: str = None,
                          complexity_score: int = 1, dependencies: List[str] = None) -> bool:
        """
        Add a new helper function to the toolbox.
        
        Args:
            name: Function name
            description: What the function does
            code: Function implementation
            category: Function category (grid_ops, pattern_detection, etc.)
            source_task_id: Task ID where this function was first extracted
            complexity_score: Complexity rating (1-5)
            dependencies: List of other function names this depends on
            
        Returns:
            True if added successfully, False if duplicate
        """
        code_hash = self._get_code_hash(code)
        timestamp = datetime.now().isoformat()
        dependencies_json = json.dumps(dependencies or [])
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO helper_functions 
                    (name, description, code, code_hash, category, created_at, updated_at, 
                     source_task_id, complexity_score, dependencies)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, description, code, code_hash, category, timestamp, timestamp,
                      source_task_id, complexity_score, dependencies_json))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Function already exists
                return False
    
    def get_all_functions(self, category: str = None, min_success_rate: float = 0.0,
                         max_functions: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve helper functions from the toolbox.
        
        Args:
            category: Filter by category
            min_success_rate: Minimum success rate filter
            max_functions: Maximum number of functions to return
            
        Returns:
            List of helper function dictionaries
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT name, description, code, category, usage_count, success_rate,
                       complexity_score, dependencies, source_task_id
                FROM helper_functions 
                WHERE success_rate >= ?
            """
            params = [min_success_rate]
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            # Order by success rate and usage count
            query += " ORDER BY success_rate DESC, usage_count DESC"
            
            if max_functions:
                query += f" LIMIT {max_functions}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            functions = []
            for row in rows:
                functions.append({
                    'name': row[0],
                    'description': row[1], 
                    'code': row[2],
                    'category': row[3],
                    'usage_count': row[4],
                    'success_rate': row[5],
                    'complexity_score': row[6],
                    'dependencies': json.loads(row[7]),
                    'source_task_id': row[8]
                })
            
            return functions
    
    def update_function_usage(self, function_name: str, task_id: str, 
                            success: bool, execution_time: float = None):
        """
        Record usage of a function and update its success rate.
        
        Args:
            function_name: Name of the function used
            task_id: Task where it was used
            success: Whether the usage was successful
            execution_time: Time taken for execution
        """
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            # Record the usage
            cursor.execute("""
                INSERT INTO function_usage 
                (function_name, task_id, success, execution_time, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (function_name, task_id, success, execution_time, timestamp))
            
            # Update the function's statistics
            cursor.execute("""
                UPDATE helper_functions 
                SET usage_count = usage_count + 1,
                    success_rate = (
                        SELECT AVG(CAST(success AS REAL)) 
                        FROM function_usage 
                        WHERE function_name = ?
                    ),
                    updated_at = ?
                WHERE name = ?
            """, (function_name, timestamp, function_name))
            
            conn.commit()
    
    def record_task_attempt(self, task_id: str, attempt_number: int, 
                          success: bool, functions_used: List[str]):
        """
        Record a task attempt with functions used.
        
        Args:
            task_id: Task identifier
            attempt_number: Attempt number for this task
            success: Whether the attempt was successful
            functions_used: List of function names used in this attempt
        """
        timestamp = datetime.now().isoformat()
        functions_json = json.dumps(functions_used)
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO task_attempts 
                (task_id, attempt_number, success, functions_used, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (task_id, attempt_number, success, functions_json, timestamp))
            
            conn.commit()
    
    def get_failed_tasks(self) -> List[str]:
        """
        Get list of task IDs that have failed in previous attempts.
        
        Returns:
            List of task IDs that could be retried
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT task_id 
                FROM task_attempts 
                WHERE task_id NOT IN (
                    SELECT task_id 
                    FROM task_attempts 
                    WHERE success = 1
                )
                ORDER BY timestamp DESC
            """)
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_task_history(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get attempt history for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of attempt records
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT attempt_number, success, functions_used, timestamp
                FROM task_attempts 
                WHERE task_id = ?
                ORDER BY attempt_number
            """, (task_id,))
            
            attempts = []
            for row in cursor.fetchall():
                attempts.append({
                    'attempt_number': row[0],
                    'success': row[1],
                    'functions_used': json.loads(row[2]),
                    'timestamp': row[3]
                })
            
            return attempts
    
    def suggest_functions_for_task(self, task_data: Dict[str, Any], 
                                 max_suggestions: int = 10) -> List[Dict[str, Any]]:
        """
        Suggest relevant helper functions for a task based on patterns and history.
        
        Args:
            task_data: ARC task data
            max_suggestions: Maximum number of functions to suggest
            
        Returns:
            List of suggested functions ordered by relevance
        """
        # Simple heuristic-based suggestion for now
        # In the future, this could use more sophisticated pattern matching
        
        functions = self.get_all_functions(min_success_rate=0.3, max_functions=max_suggestions)
        
        # Sort by success rate and usage count
        functions.sort(key=lambda f: (f['success_rate'], f['usage_count']), reverse=True)
        
        return functions[:max_suggestions]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get toolbox statistics.
        
        Returns:
            Dictionary with toolbox metrics
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            # Count functions by category
            cursor.execute("""
                SELECT category, COUNT(*) as count, AVG(success_rate) as avg_success
                FROM helper_functions 
                GROUP BY category
            """)
            categories = {row[0]: {'count': row[1], 'avg_success': row[2]} 
                         for row in cursor.fetchall()}
            
            # Total functions
            cursor.execute("SELECT COUNT(*) FROM helper_functions")
            total_functions = cursor.fetchone()[0]
            
            # Total usage
            cursor.execute("SELECT COUNT(*) FROM function_usage")
            total_usage = cursor.fetchone()[0]
            
            # Failed tasks available for retry
            failed_tasks = len(self.get_failed_tasks())
            
            return {
                'total_functions': total_functions,
                'total_usage_records': total_usage,
                'categories': categories,
                'failed_tasks_available': failed_tasks
            }
    
    def export_toolbox(self, output_path: str = None):
        """
        Export the entire toolbox to a JSON file in the run directory.
        
        Args:
            output_path: Path to export file (defaults to toolbox_export.json in run dir)
        """
        if output_path is None:
            output_path = os.path.join(self.run_output_dir, "toolbox_export.json")
            
        functions = self.get_all_functions()
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'run_output_dir': self.run_output_dir,
            'statistics': self.get_statistics(),
            'functions': functions
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Run toolbox exported to {output_path}")
        return output_path


def initialize_default_toolbox(toolbox: PersistentToolbox):
    """
    Initialize the toolbox with some basic helper functions.
    
    Args:
        toolbox: PersistentToolbox instance
    """
    default_functions = [
        {
            'name': 'rotate_90',
            'description': 'Rotate grid 90 degrees clockwise',
            'code': '''def rotate_90(grid):
    """Rotate grid 90 degrees clockwise."""
    if not grid or not grid[0]:
        return grid
    return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]''',
            'category': 'grid_operations',
            'complexity_score': 2
        },
        {
            'name': 'flip_horizontal',
            'description': 'Flip grid horizontally',
            'code': '''def flip_horizontal(grid):
    """Flip grid horizontally."""
    return [row[::-1] for row in grid]''',
            'category': 'grid_operations',
            'complexity_score': 1
        },
        {
            'name': 'flip_vertical',
            'description': 'Flip grid vertically',
            'code': '''def flip_vertical(grid):
    """Flip grid vertically."""
    return grid[::-1]''',
            'category': 'grid_operations',
            'complexity_score': 1
        },
        {
            'name': 'find_objects',
            'description': 'Find connected components of the same color',
            'code': '''def find_objects(grid, color=None):
    """Find connected components in grid."""
    if not grid:
        return []
    
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    objects = []
    
    def dfs(r, c, target_color):
        if (r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or 
            visited[r][c] or grid[r][c] != target_color):
            return []
        
        visited[r][c] = True
        cells = [(r, c)]
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            cells.extend(dfs(r + dr, c + dc, target_color))
        
        return cells
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if not visited[r][c] and (color is None or grid[r][c] == color):
                if grid[r][c] != 0:  # Skip background
                    obj = dfs(r, c, grid[r][c])
                    if obj:
                        objects.append(obj)
    
    return objects''',
            'category': 'pattern_detection',
            'complexity_score': 4
        },
        {
            'name': 'count_colors',
            'description': 'Count occurrences of each color in grid',
            'code': '''def count_colors(grid):
    """Count occurrences of each color in grid."""
    counts = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts''',
            'category': 'analysis',
            'complexity_score': 2
        },
        {
            'name': 'get_grid_bounds',
            'description': 'Get bounding box of non-zero elements',
            'code': '''def get_grid_bounds(grid):
    """Get bounding box of non-zero elements."""
    non_zero_rows = []
    non_zero_cols = []
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0:
                non_zero_rows.append(r)
                non_zero_cols.append(c)
    
    if not non_zero_rows:
        return None
    
    return {
        'min_row': min(non_zero_rows),
        'max_row': max(non_zero_rows),
        'min_col': min(non_zero_cols),
        'max_col': max(non_zero_cols)
    }''',
            'category': 'analysis',
            'complexity_score': 3
        }
    ]
    
    for func in default_functions:
        toolbox.add_helper_function(**func)
    
    print(f"Initialized toolbox with {len(default_functions)} default functions")


if __name__ == "__main__":
    # Example usage
    toolbox = PersistentToolbox()
    initialize_default_toolbox(toolbox)
    
    stats = toolbox.get_statistics()
    print("Toolbox Statistics:")
    print(f"Total functions: {stats['total_functions']}")
    print(f"Categories: {list(stats['categories'].keys())}")