#!/usr/bin/env python3
"""Simple viewer for arc-solver/db/responses.db

Usage:
    python view_db.py --limit 10
    python view_db.py --json --limit 5
"""
import os
import sqlite3
import argparse
import json

ROOT = os.path.dirname(__file__)
DB_PATH = os.path.join(ROOT, "db", "responses.db")


def fetch_rows(limit: int = 20):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, timestamp, model_name, prompt_hash, reasoning, final_json, score_train, score_test, tokens, cost_estimate"
            " FROM responses ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return cols, rows
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="View arc-solver responses DB")
    parser.add_argument("--limit", "-n", type=int, default=20, help="Number of rows to show")
    parser.add_argument("--json", action="store_true", help="Output rows as JSON array")
    args = parser.parse_args()

    try:
        cols, rows = fetch_rows(limit=args.limit)
    except FileNotFoundError as e:
        print(e)
        return

    if args.json:
        out = []
        for r in rows:
            out.append({cols[i]: r[i] for i in range(len(cols))})
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        # table-like output
        TRUNC = 10
        def _shorten(v):
            if v is None:
                return "NULL"
            # replace newlines with spaces so truncation is more readable
            s = str(v).replace("\n", " ")
            if len(s) <= TRUNC:
                return s
            return s[:TRUNC-1] + "…"

        widths = [max(len(str(c)), 12) for c in cols]
        for r in rows:
            for i, v in enumerate(r):
                widths[i] = max(widths[i], len(_shorten(v)))

        # header
        header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
        sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
        print(header)
        print(sep)

        for r in rows:
            print(" | ".join((_shorten(r[i])).ljust(widths[i]) for i in range(len(cols))))


if __name__ == "__main__":
    main()
