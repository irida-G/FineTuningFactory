from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
import asyncio
import json
import shutil
import subprocess

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.schemas import PathInspectionRequest, TrainingRunRequest
from backend.app.application.training_service import TrainingService
from backend.app.config import Settings
from backend.app.domain.run import RunStatus
from backend.app.infrastructure.database import RunRepository
from backend.app.workers.training_worker import TrainingWorker


TERMINAL_STATUSES = {
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


def _read_log_chunk(
    log_path: str | None,
    offset: int,
    *,
    initial_tail_bytes: int = 64 * 1024,
) -> tuple[str, int]:
    if not log_path:
        return "", offset
    path = Path(log_path)
    if not path.is_file():
        return "", offset

    size = path.stat().st_size
    if offset == 0 and size > initial_tail_bytes:
        offset = size - initial_tail_bytes
    if offset > size:
        offset = 0

    with path.open("rb") as file:
        file.seek(offset)
        content = file.read(128 * 1024)
        new_offset = file.tell()
    return content.decode("utf-8", errors="replace"), new_offset


def _read_log_tail(log_path: str | None, lines: int) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.is_file():
        return ""
    from collections import deque

    with path.open("r", encoding="utf-8", errors="replace") as file:
        return "".join(deque(file, maxlen=lines))


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or Settings.from_env()
    repository = RunRepository(active_settings.database_path)
    service = TrainingService(repository)
    worker = TrainingWorker(repository, active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings.ensure_directories()
        repository.initialize()
        repository.recover_interrupted()
        worker.start()
        app.state.settings = active_settings
        app.state.repository = repository
        app.state.training_service = service
        app.state.worker = worker
        yield
        worker.stop()

    app = FastAPI(
        title="FineTuningFactory API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/system")
    def system_info() -> dict[str, object]:
        cli = active_settings.llamafactory_cli
        resolved_cli = shutil.which(cli) if not Path(cli).is_absolute() else cli
        cli_available = bool(resolved_cli and Path(resolved_cli).is_file())
        version = None
        if cli_available:
            try:
                completed = subprocess.run(
                    [str(resolved_cli), "version"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                output = completed.stdout + completed.stderr
                for line in output.splitlines():
                    if "version" in line.lower():
                        version = line.strip("| ")
                        break
            except (OSError, subprocess.TimeoutExpired):
                pass
        return {
            "llamafactory_cli": str(resolved_cli or cli),
            "llamafactory_available": cli_available,
            "llamafactory_version": version,
            "runs_dir": str(active_settings.runs_dir),
            "worker_mode": "single-gpu",
        }

    @app.post("/api/resources/model/inspect")
    def inspect_model(body: PathInspectionRequest) -> dict[str, object]:
        try:
            return service.inspect_model(body.path)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/resources/datasets/inspect")
    def inspect_datasets(body: PathInspectionRequest) -> dict[str, object]:
        try:
            return service.inspect_dataset_dir(body.path)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    def create_run(body: TrainingRunRequest) -> dict[str, object]:
        try:
            run = service.create_run(body)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        worker.notify()
        return run

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, object]]:
        return repository.list(limit)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        run = repository.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        return run

    @app.get("/api/runs/{run_id}/log")
    def get_run_log(
        run_id: str,
        lines: int = Query(default=300, ge=1, le=5000),
    ) -> dict[str, str]:
        run = repository.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        return {"content": _read_log_tail(run["log_path"], lines)}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, object]:
        run = worker.cancel(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        if RunStatus(run["status"]) in TERMINAL_STATUSES and not run["cancel_requested"]:
            raise HTTPException(status_code=409, detail="训练任务已经结束")
        return run

    @app.get("/api/runs/{run_id}/events")
    async def run_events(request: Request, run_id: str) -> StreamingResponse:
        if repository.get(run_id) is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")

        async def event_stream() -> AsyncIterator[str]:
            offset = 0
            previous_snapshot = ""
            event_id = 0
            terminal_idle_rounds = 0

            while not await request.is_disconnected():
                run = await asyncio.to_thread(repository.get, run_id)
                if run is None:
                    break

                snapshot = json.dumps(run, ensure_ascii=False, sort_keys=True)
                if snapshot != previous_snapshot:
                    event_id += 1
                    yield (
                        f"id: {event_id}\n"
                        "event: snapshot\n"
                        f"data: {json.dumps(run, ensure_ascii=False)}\n\n"
                    )
                    previous_snapshot = snapshot

                log_content, offset = await asyncio.to_thread(
                    _read_log_chunk,
                    run["log_path"],
                    offset,
                )
                if log_content:
                    event_id += 1
                    yield (
                        f"id: {event_id}\n"
                        "event: log\n"
                        f"data: {json.dumps({'content': log_content}, ensure_ascii=False)}\n\n"
                    )

                if RunStatus(run["status"]) in TERMINAL_STATUSES:
                    terminal_idle_rounds += 1
                    if terminal_idle_rounds >= 2:
                        break
                else:
                    terminal_idle_rounds = 0

                await asyncio.sleep(active_settings.poll_interval)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    if active_settings.frontend_dist.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=active_settings.frontend_dist, html=True),
            name="frontend",
        )

    return app


app = create_app()
