from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import json
import sqlite3

from backend.app.domain.run import RunStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    request_json TEXT NOT NULL,
                    progress_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    pid INTEGER,
                    run_dir TEXT,
                    adapter_dir TEXT,
                    config_path TEXT,
                    log_path TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_status_created "
                "ON runs(status, created_at)"
            )

    def create(self, run_id: str, request: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(id, status, created_at, request_json) "
                "VALUES (?, ?, ?, ?)",
                (run_id, RunStatus.QUEUED, now, json.dumps(request)),
            )
        run = self.get(run_id)
        assert run is not None
        return run

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def claim_next(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM runs WHERE status = ? "
                "ORDER BY created_at ASC LIMIT 1",
                (RunStatus.QUEUED,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None

            run_id = row["id"]
            connection.execute(
                "UPDATE runs SET status = ?, started_at = ? "
                "WHERE id = ? AND status = ?",
                (RunStatus.RUNNING, utc_now(), run_id, RunStatus.QUEUED),
            )
            connection.execute("COMMIT")
        return self.get(run_id)

    def set_runtime_paths(
        self,
        run_id: str,
        *,
        run_dir: Path,
        adapter_dir: Path,
        config_path: Path,
        log_path: Path,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET run_dir = ?, adapter_dir = ?, config_path = ?, log_path = ?
                WHERE id = ?
                """,
                (
                    str(run_dir),
                    str(adapter_dir),
                    str(config_path),
                    str(log_path),
                    run_id,
                ),
            )

    def set_pid(self, run_id: str, pid: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET pid = ? WHERE id = ?",
                (pid, run_id),
            )

    def update_progress(self, run_id: str, progress: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET progress_json = ? WHERE id = ?",
                (json.dumps(progress), run_id),
            )

    def finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if status not in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            raise ValueError(f"非法终态: {status}")

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?, result_json = ?, error = ?
                WHERE id = ?
                """,
                (
                    status,
                    utc_now(),
                    json.dumps(result) if result is not None else None,
                    error,
                    run_id,
                ),
            )

    def request_cancel(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                return None

            status = RunStatus(row["status"])
            if status == RunStatus.QUEUED:
                connection.execute(
                    "UPDATE runs SET status = ?, cancel_requested = 1, "
                    "finished_at = ? WHERE id = ?",
                    (RunStatus.CANCELLED, utc_now(), run_id),
                )
            elif status == RunStatus.RUNNING:
                connection.execute(
                    "UPDATE runs SET cancel_requested = 1 WHERE id = ?",
                    (run_id,),
                )
            connection.execute("COMMIT")
        return self.get(run_id)

    def recover_interrupted(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?,
                    error = COALESCE(error, '后端进程重启，训练任务失去进程所有权')
                WHERE status = ?
                """,
                (RunStatus.FAILED, utc_now(), RunStatus.RUNNING),
            )
        return cursor.rowcount

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        run = dict(row)
        run["request"] = json.loads(run.pop("request_json"))
        progress = run.pop("progress_json")
        result = run.pop("result_json")
        run["progress"] = json.loads(progress) if progress else None
        run["result"] = json.loads(result) if result else None
        run["cancel_requested"] = bool(run["cancel_requested"])
        return run
