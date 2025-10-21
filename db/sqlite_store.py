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
    tokens: Optional[int],
    cost_estimate: Optional[float],
    timestamp: Optional[str] = None,
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
            (timestamp, model_name, prompt_hash, reasoning, final_json, challenges, solutions, challenge_id, score_train, score_test, tokens, cost_estimate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                score_train,
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


def update_response_scores(response_id: int, score_train: Optional[float], score_test: Optional[float]):
    """Update score_train and score_test for an existing response row."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE responses SET score_train = ?, score_test = ? WHERE id = ?",
            (score_train, score_test, response_id),
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
        cur.execute(
            "UPDATE responses SET tokens = COALESCE(tokens, 0) + ? WHERE id = ?",
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
