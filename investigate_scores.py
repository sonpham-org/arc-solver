#!/usr/bin/env python3
"""investigate_scores.py

Read db/responses.db and compute average score_test grouped by timestamp.

Usage:
    python investigate_scores.py [--db db/responses.db] [--granularity date|hour|minute] [--csv out.csv]

By default groups by date (YYYY-MM-DD) and prints group, count, avg, stddev.
"""
import sqlite3
import argparse
from collections import defaultdict
from datetime import datetime
import math
import csv

DB_DEFAULT = "db/responses.db"


def parse_args():
    parser = argparse.ArgumentParser(description="Investigate average score_test grouped by timestamp")
    parser.add_argument("--db", default=DB_DEFAULT, help="Path to responses.db")
    parser.add_argument(
        "--granularity",
        choices=["timestamp", "date", "hour", "minute"],
        default="timestamp",
        help="Grouping granularity (timestamp/date/hour/minute)",
    )
    parser.add_argument("--csv", help="Optional CSV output path")
    return parser.parse_args()


def truncate_ts(ts: str, granularity: str) -> str:
    # expect ISO timestamp like 2025-10-21T12:34:56Z or similar
    try:
        # strip trailing Z for fromisoformat
        s = ts.rstrip("Z")
        dt = datetime.fromisoformat(s)
    except Exception:
        # fallback: try to parse common formats
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            # as last resort, return raw timestamp
            return ts

    if granularity == "date":
        return dt.date().isoformat()
    if granularity == "hour":
        return dt.strftime("%Y-%m-%dT%H:00")
    if granularity == "minute":
        return dt.strftime("%Y-%m-%dT%H:%M")
    if granularity == "timestamp":
        # return the original timestamp string (preserve full precision)
        return ts
    return ts


def mean_and_std(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def main():
    args = parse_args()
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    # Try to select token & cost columns if present; fallback to minimal columns if not
    try:
        cur.execute("SELECT timestamp, score_test, challenge_id, input_tokens, output_tokens, cost_estimate FROM responses WHERE score_test IS NOT NULL")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # older DB without token/cost columns
        cur.execute("SELECT timestamp, score_test, challenge_id FROM responses WHERE score_test IS NOT NULL")
        rows3 = cur.fetchall()
        # normalize rows to 6-tuples (ts, score, cid, itok, otok, cost)
        rows = [(ts, score, cid, None, None, None) for ts, score, cid in rows3]
    groups = defaultdict(list)
    challenge_sets = defaultdict(set)
    token_sums_in = defaultdict(int)
    token_sums_out = defaultdict(int)
    cost_sums = defaultdict(float)
    for ts, score, cid, itok, otok, cost in rows:
        key = truncate_ts(ts, args.granularity)
        try:
            val = float(score)
        except Exception:
            continue
        groups[key].append(val)
        if cid is not None:
            challenge_sets[key].add(str(cid))
        try:
            token_sums_in[key] += int(itok) if itok is not None else 0
        except Exception:
            pass
        try:
            token_sums_out[key] += int(otok) if otok is not None else 0
        except Exception:
            pass
        try:
            cost_sums[key] += float(cost) if cost is not None else 0.0
        except Exception:
            pass

    results = []
    for k in sorted(groups.keys()):
        vals = groups[k]
        mean, std = mean_and_std(vals)
        # percentage of perfect scores (== 1.0)
        perfect = sum(1 for v in vals if abs(v - 1.0) < 1e-9)
        pct_perfect = (perfect / len(vals)) * 100 if vals else 0.0
        num_challenges = len(challenge_sets.get(k, set()))
        sum_in = token_sums_in.get(k, 0)
        sum_out = token_sums_out.get(k, 0)
        cost_sum = cost_sums.get(k, 0.0)
        results.append((k, len(vals), mean, std, pct_perfect, num_challenges, sum_in, sum_out, cost_sum))

    # print table
    print(f"Grouping by: {args.granularity} (total groups: {len(results)})")
    headers = ["group", "count", "mean", "stddev", "pct_perfect", "num_challenges", "sum_input_tokens", "sum_output_tokens", "cost_estimate"]
    # prepare string rows
    str_rows = []
    for k, cnt, mean, std, pct, nc, sum_in, sum_out, cost_sum in results:
        mean_s = f"{mean:.4f}" if mean is not None else ""
        std_s = f"{std:.4f}" if std is not None else ""
        pct_s = f"{pct:.2f}%"
        str_rows.append([str(k), str(cnt), mean_s, std_s, pct_s, str(nc), str(sum_in), str(sum_out), f"${cost_sum:.6f}"])

    # compute column widths
    cols = list(zip(*([headers] + str_rows))) if str_rows else [[h] for h in headers]
    widths = [max(len(cell) for cell in col) for col in cols]

    # header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)
    for row in str_rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "count", "mean", "stddev", "pct_perfect", "num_challenges", "sum_input_tokens", "sum_output_tokens", "cost_estimate"])
            for k, cnt, mean, std, pct, nc, sum_in, sum_out, cost_sum in results:
                w.writerow([k, cnt, f"{mean:.6f}", f"{std:.6f}", f"{pct:.4f}", nc, f"{sum_in}", f"{sum_out}", f"{cost_sum:.6f}"])
        print(f"Wrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
