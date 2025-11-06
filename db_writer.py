"""Background DB writer that batches/serializes non-critical sqlite_store operations.

This module provides a simple worker thread that consumes a queue of DB
operations and executes them against the existing `sqlite_store` module.

Usage:
    from db_writer import DBWriter
    writer = DBWriter()
    writer.start()
    writer.enqueue('insert_prompt', prompt_hash, prompt)
    writer.stop_and_flush()
"""
from __future__ import annotations

import threading
import queue
import time
from typing import Any, Tuple

from db import sqlite_store


class DBWriter:
    def __init__(self, batch_size: int = 50, flush_interval: float = 0.5):
        self._q: queue.Queue[Tuple[str, Tuple[Any, ...]]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.batch_size = batch_size
        self.flush_interval = flush_interval

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, op_name: str, *args: Any) -> None:
        """Enqueue a sqlite_store operation by name and args.

        The worker will call getattr(sqlite_store, op_name)(*args). This
        is intended for non-critical writes where the caller does not need
        an immediate return value.
        """
        self._q.put((op_name, args))

    def _run(self) -> None:
        buffer = []
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=self.flush_interval)
                buffer.append(item)
                # drain up to batch_size quickly
                while len(buffer) < self.batch_size:
                    try:
                        item = self._q.get(block=False)
                        buffer.append(item)
                    except queue.Empty:
                        break
            except queue.Empty:
                # timeout: flush whatever we have
                pass

            if not buffer:
                continue

            # execute the buffered operations
            for op_name, args in buffer:
                try:
                    fn = getattr(sqlite_store, op_name, None)
                    if fn is None:
                        # unknown op name; skip
                        continue
                    fn(*args)
                except Exception:
                    # swallow errors to avoid crashing the worker; writes are
                    # best-effort. In a more robust system we'd log or
                    # push to a dead-letter queue.
                    pass

            buffer.clear()

        # flush any remaining items on stop
        while True:
            try:
                op = self._q.get(block=False)
                try:
                    fn = getattr(sqlite_store, op[0], None)
                    if fn:
                        fn(*op[1])
                except Exception:
                    pass
            except queue.Empty:
                break

    def stop_and_flush(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)


_global_writer: DBWriter | None = None


def get_global_writer() -> DBWriter:
    global _global_writer
    if _global_writer is None:
        _global_writer = DBWriter()
        _global_writer.start()
    return _global_writer


def enqueue(op_name: str, *args: Any) -> None:
    get_global_writer().enqueue(op_name, *args)
