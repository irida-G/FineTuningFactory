from __future__ import annotations

from pathlib import Path
from typing import Any
import threading

from backend.app.api.schemas import TrainingRunRequest
from backend.app.config import Settings
from backend.app.domain.run import RunStatus
from backend.app.infrastructure.database import RunRepository
from backend.app.infrastructure.llamafactory.cli_runner import (
    TrainingCancelled,
    train_lora_with_cli,
)


class TrainingWorker:
    def __init__(self, repository: RunRepository, settings: Settings):
        self.repository = repository
        self.settings = settings
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="single-gpu-training-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        with self._lock:
            for cancel_event in self._active_cancel_events.values():
                cancel_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def notify(self) -> None:
        self._wake_event.set()

    def cancel(self, run_id: str) -> dict[str, Any] | None:
        run = self.repository.request_cancel(run_id)
        if run is None:
            return None
        with self._lock:
            cancel_event = self._active_cancel_events.get(run_id)
        if cancel_event is not None:
            cancel_event.set()
        self._wake_event.set()
        return self.repository.get(run_id)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            run = self.repository.claim_next()
            if run is None:
                self._wake_event.wait(self.settings.poll_interval)
                self._wake_event.clear()
                continue
            if self._stop_event.is_set():
                self.repository.finish(
                    run["id"],
                    RunStatus.CANCELLED,
                    error="后端正在关闭，训练任务未启动",
                )
                break
            self._execute(run)

    def _execute(self, run: dict[str, Any]) -> None:
        run_id = run["id"]
        request = TrainingRunRequest.model_validate(run["request"])
        cancel_event = threading.Event()
        with self._lock:
            self._active_cancel_events[run_id] = cancel_event

        if self._stop_event.is_set():
            cancel_event.set()

        current = self.repository.get(run_id)
        if current and current["cancel_requested"]:
            self.repository.finish(
                run_id,
                RunStatus.CANCELLED,
                error=f"训练已取消: {run_id}",
            )
            with self._lock:
                self._active_cancel_events.pop(run_id, None)
            return

        run_dir = self.settings.runs_dir / run_id
        adapter_dir = run_dir / "adapter"
        config_path = run_dir / "train.yaml"
        log_path = run_dir / "console.log"
        self.repository.set_runtime_paths(
            run_id,
            run_dir=run_dir,
            adapter_dir=adapter_dir,
            config_path=config_path,
            log_path=log_path,
        )

        try:
            result = train_lora_with_cli(
                llamafactory_cli=self.settings.llamafactory_cli,
                model_name_or_path=request.model_path,
                dataset_dir=request.dataset_dir,
                dataset_names=request.dataset_names,
                template=request.template,
                runs_root=str(self.settings.runs_dir),
                gpu_id=request.gpu_id,
                use_qlora=request.method == "qlora",
                lora_rank=request.lora_rank,
                lora_dropout=request.lora_dropout,
                cutoff_len=request.cutoff_len,
                learning_rate=request.learning_rate,
                num_train_epochs=request.num_train_epochs,
                per_device_train_batch_size=request.per_device_train_batch_size,
                gradient_accumulation_steps=request.gradient_accumulation_steps,
                logging_steps=request.logging_steps,
                save_steps=request.save_steps,
                warmup_ratio=request.warmup_ratio,
                mixed_precision=request.mixed_precision,
                trust_remote_code=request.trust_remote_code,
                run_id=run_id,
                on_progress=lambda progress: self.repository.update_progress(
                    run_id, progress
                ),
                on_process_started=lambda pid: self.repository.set_pid(run_id, pid),
                cancel_event=cancel_event,
            )
        except TrainingCancelled as error:
            self.repository.finish(
                run_id,
                RunStatus.CANCELLED,
                error=str(error),
            )
        except Exception as error:
            self.repository.finish(
                run_id,
                RunStatus.FAILED,
                error=str(error),
            )
        else:
            self.repository.finish(
                run_id,
                RunStatus.SUCCEEDED,
                result=result,
            )
        finally:
            with self._lock:
                self._active_cancel_events.pop(run_id, None)
