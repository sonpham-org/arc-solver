"""Simple SQLite storage for Gemini responses.

This module creates a SQLite database at db/responses.db and exposes
init_db() and insert_response(...) to record each run.
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any

ROOT = os.path.dirname(os.path.dirname(__file__))
DB_DIR = os.path.join(ROOT, "db")
DB_PATH = os.path.join(DB_DIR, "responses.db")

# separate DB for run-level metrics
RUNS_DB_PATH = os.path.join(DB_DIR, "runs.db")

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT,
    start_timestamp TEXT,
    end_timestamp TEXT,
    duration_seconds REAL,
    total_queries INTEGER,
    total_input_tokens INTEGER,
    total_output_tokens INTEGER,
    total_cost REAL
);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_hash TEXT,
    reasoning TEXT,
    final_json TEXT,
    challenges TEXT,
    solutions TEXT,
    challenge_id TEXT,
    json_index INTEGER,
    run_name TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    apply_output TEXT,
    score_train REAL,
    score_test REAL,
    tokens INTEGER,
    cost_estimate REAL
);
"""

PROMPTS_TABLE = """
CREATE TABLE IF NOT EXISTS prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _ensure_dir():
    os.makedirs(DB_DIR, exist_ok=True)


def init_db():
    """Initialize the database and create tables if needed."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        # Improve concurrent write throughput
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            # Best-effort: if PRAGMA not supported, continue
            pass
        conn.executescript(SCHEMA)
        conn.executescript(PROMPTS_TABLE)
        # migration: ensure `challenges`, `solutions` and `prompt_hash` columns exist for older DBs
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(responses)")
        cols = [r[1] for r in cur.fetchall()]
        if 'challenges' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN challenges TEXT")
            conn.commit()
        if 'solutions' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN solutions TEXT")
            conn.commit()
        if 'challenge_id' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN challenge_id TEXT")
            conn.commit()
        if 'apply_output' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN apply_output TEXT")
            conn.commit()
        if 'prompt_hash' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN prompt_hash TEXT")
            conn.commit()
        if 'run_name' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN run_name TEXT")
            conn.commit()
        if 'input_tokens' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN input_tokens INTEGER")
            conn.commit()
        if 'output_tokens' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN output_tokens INTEGER")
            conn.commit()
        if 'cost_estimate' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN cost_estimate REAL")
            conn.commit()
        if 'train_scores' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN train_scores TEXT")
            conn.commit()
        if 'json_index' not in cols:
            cur.execute("ALTER TABLE responses ADD COLUMN json_index INTEGER")
            conn.commit()
        conn.commit()
        # also ensure runs DB/table exists
        try:
            rconn = sqlite3.connect(RUNS_DB_PATH)
            try:
                rconn.executescript(RUNS_SCHEMA)
                rconn.commit()
            finally:
                rconn.close()
        except Exception:
            pass
    finally:
        conn.close()


def update_response_add_cost(response_id: int, additional_cost: float):
    """Add additional_cost (USD) to the cost_estimate for a response row."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE responses SET cost_estimate = COALESCE(cost_estimate, 0) + ? WHERE id = ?",
            (float(additional_cost), response_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_response_train_scores(response_id: int, train_scores_json: str):
    """Store a JSON string containing the list of per-training-example scores."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE responses SET train_scores = ? WHERE id = ?",
            (train_scores_json, response_id),
        )
        conn.commit()
    finally:
        conn.close()


def insert_response(
    model_name: str,
    prompt_hash: Optional[str],
    reasoning: Optional[str],
    final_json: Optional[str],
    challenges: Optional[str],
    solutions: Optional[str],
    challenge_id: Optional[str],
    score_train: Optional[float],
    score_test: Optional[float],
    cost_estimate: Optional[float] = None,
    tokens: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    json_index: Optional[int] = None,
):
    """Insert a response record into the DB.

    Returns the inserted row id.
    """
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO responses
            (timestamp, model_name, prompt_hash, reasoning, final_json, challenges, solutions, challenge_id, json_index, run_name, input_tokens, output_tokens, score_test, tokens, cost_estimate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                model_name,
                prompt_hash,
                reasoning,
                final_json,
                challenges,
                solutions,
                challenge_id,
                json_index,
                run_name,
                input_tokens,
                output_tokens,
                score_test,
                tokens,
                cost_estimate,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def insert_prompt(hash_value: str, prompt_text: str, created_at: Optional[str] = None):
    """Insert a prompt into the prompts table if it doesn't already exist."""
    if created_at is None:
        created_at = datetime.utcnow().isoformat() + "Z"
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # use INSERT OR IGNORE to avoid duplicates
        cur.execute(
            "INSERT OR IGNORE INTO prompts (hash, prompt, created_at) VALUES (?, ?, ?)",
            (hash_value, prompt_text, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def update_response_scores(response_id: int, score_test: Optional[float]):
    """Update score_test for an existing response row."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE responses SET score_test = ? WHERE id = ?",
            (score_test, response_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_response_train_score(response_id: int, score_train: Optional[float]):
    """Update score_train for an existing response row."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE responses SET score_train = ? WHERE id = ?",
            (score_train, response_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_response_tokens(response_id: int, extra_tokens: int):
    """Increment the tokens column for a response by extra_tokens."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # increment output_tokens specifically (newer schema)
        cur.execute(
            "UPDATE responses SET output_tokens = COALESCE(output_tokens, 0) + ? WHERE id = ?",
            (extra_tokens, response_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_response_apply_output(response_id: int, apply_json: str):
    """Store the parsed apply output (JSON string) for a response."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE responses SET apply_output = ? WHERE id = ?",
            (apply_json, response_id),
        )
        conn.commit()
    finally:
        conn.close()


def insert_run_metrics(
    run_name: str,
    start_timestamp: str,
    end_timestamp: str,
    duration_seconds: float,
    total_queries: int,
    total_input_tokens: int,
    total_output_tokens: int,
    total_cost: float | None = None,
):
    """Insert a run-level metrics row into the runs DB."""
    _ensure_dir()
    conn = sqlite3.connect(RUNS_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (run_name, start_timestamp, end_timestamp, duration_seconds, total_queries, total_input_tokens, total_output_tokens, total_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                start_timestamp,
                end_timestamp,
                float(duration_seconds),
                int(total_queries),
                int(total_input_tokens),
                int(total_output_tokens),
                float(total_cost) if total_cost is not None else None,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()
