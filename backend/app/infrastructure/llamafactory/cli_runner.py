from collections import deque
from pathlib import Path
from typing import Any, Callable, Literal
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
import uuid

import yaml


ProgressCallback = Callable[[dict[str, Any]], None]
ProcessStartedCallback = Callable[[int], None]


class TrainingCancelled(RuntimeError):
    """Raised when a running LlamaFactory process is cancelled."""

def read_log_tail(log_path: Path, line_count: int = 50) -> str:

    if not log_path.exists():
        return "日志文件不存在"

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        return "".join(deque(file, maxlen=line_count))

def read_latest_progress(
        progress_path: Path,
) -> dict[str, Any] | None:
    if not progress_path.is_file():
        return None

    with progress_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        recent_lines = deque(file, maxlen=5)

    for line in reversed(recent_lines):
        line = line.strip()

        if not line:
            continue

        try:
            progress = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(progress, dict):
            return progress

    return None

def monitor_training_progress(
    progress_path: Path,
    stop_event: threading.Event,
    poll_interval: float = 0.5,
    on_progress: ProgressCallback | None = None,
) -> None:
    last_progress: dict[str, Any] | None = None

    while not stop_event.is_set():
        progress = read_latest_progress(progress_path)

        if progress is not None and progress != last_progress:
            if on_progress is not None:
                try:
                    on_progress(progress)
                except Exception as error:
                    print(f"[进度回调失败] {error}", flush=True)

            current_steps = progress.get("current_steps", "?")
            total_steps = progress.get("total_steps", "?")
            percentage = progress.get("percentage", "?")
            loss = progress.get("loss", "?")
            epoch = progress.get("epoch", "?")
            remaining_time = progress.get("remaining_time", "?")

            print(
                "[训练进度] "
                f"step={current_steps}/{total_steps} | "
                f"进度={percentage}% | "
                f"epoch={epoch} | "
                f"loss={loss} | "
                f"剩余时间={remaining_time}",
                flush=True,
            )

            last_progress = progress

        stop_event.wait(poll_interval)

    final_progress = read_latest_progress(progress_path)

    if final_progress is not None and final_progress != last_progress:
        if on_progress is not None:
            try:
                on_progress(final_progress)
            except Exception as error:
                print(f"[最终进度回调失败] {error}", flush=True)

        print(f"[最终训练进度] {final_progress}", flush=True)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return

    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)

    if process.poll() is None:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()

def train_lora_with_cli(
        *,
        llamafactory_cli: str,
        model_name_or_path: str,
        dataset_dir: str,
        dataset_names: list[str],
        template: str,
        runs_root: str,
        gpu_id: int | None = None,
        use_qlora: bool = True,
        lora_rank: int = 8,
        cutoff_len: int = 2048,
        learning_rate: float = 1e-4,
        num_train_epochs: float = 3.0,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        logging_steps: int = 10,
        save_steps: int = 100,
        warmup_ratio: float = 0.1,
        lora_dropout: float = 0.0,
        mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16",
        trust_remote_code: bool = False,
        run_id: str | None = None,
        on_progress: ProgressCallback | None = None,
        on_process_started: ProcessStartedCallback | None = None,
        cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """启动一次LlamaFactory LoRA/QLoRA并返回Adapter信息"""

    # --- 1. 检查外部输入---

    dataset_path = Path(dataset_dir).expanduser().resolve()
    runs_path = Path(runs_root).expanduser().resolve()

    #数据集处理
    if not isinstance(dataset_names, list) or not dataset_names:
        raise ValueError("dataset_names必须是非空列表")

    normalized_dataset_names: list[str] = []

    for name in dataset_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('每个数据集名称必须是非空字符串')

        normalized_name = name.strip()

        if "," in normalized_name:
            raise ValueError(
                f"数据集名称不能包含逗号: {normalized_name}"
            )
        normalized_dataset_names.append(normalized_name)

    if len(set(normalized_dataset_names)) != len(normalized_dataset_names):
        raise ValueError("dataset_names中不能包含重复名称")

    if not dataset_path.is_dir():
        raise ValueError(f"dataset_dir不存在: {dataset_path}")

    dataset_info_path = dataset_path / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise ValueError(
            f"dataset_dir中没有dataset_info.json: {dataset_info_path}"
        )

    try:
        with dataset_info_path.open("r", encoding="utf-8") as file:
            dataset_info = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"dataset_info.json不是合法JSON: {error}"
        ) from error

    if not isinstance(dataset_info, dict):
        raise ValueError("dataset_info.json的顶层必须是JSON对象")

    for name in normalized_dataset_names:
        if name not in dataset_info:
            raise ValueError(
                f"dataset_info.json中没有注册数据集: {name}"
            )

        dataset_config = dataset_info[name]

        if not isinstance(dataset_config, dict):
            raise ValueError(
                f"数据集配置必须是JSON对象: {name}"
            )

        local_file_name = dataset_config.get("file_name")

        if not isinstance(local_file_name, str) or not local_file_name.strip():
            raise ValueError(
                f"本地数据集必须配置非空file_name: {name}"
            )

        local_file_path = (
            dataset_path / local_file_name.strip()
        ).resolve()

        try:
            local_file_path.relative_to(dataset_path)
        except ValueError as error:
            raise ValueError(
                f"数据集文件不能位于dataset_dir之外: {local_file_path}"
            ) from error

        if not local_file_path.is_file():
            raise ValueError(
                f"数据集文件不存在: {local_file_path}"
            )

    cli_path = shutil.which(llamafactory_cli)
    if cli_path is None:
        raise RuntimeError(
            f"找不到LlamaFacroty CLI: {llamafactory_cli}\n"
            "请传入llamafactory-cli命令, 或其绝对路径"
        )

    if mixed_precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(
            "mixed_precision必须是bf16, fp16或fp32"
        )

    # --- 2. 为该次训练建立独立目录 ---

    if run_id is None:
        run_id = uuid.uuid4().hex
    elif re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id) is None:
        raise ValueError("run_id只能包含字母、数字、下划线和连字符")

    run_dir = runs_path / run_id
    adapter_dir = run_dir / "adapter"
    config_path = run_dir / "train.yaml"
    log_path = run_dir / "console.log"

    run_dir.mkdir(parents=True, exist_ok=False)

    # --- 3. 构造LlamaFactory配置 ---

    config: dict[str, Any] = {
        #基础模型
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": trust_remote_code,

        #训练目标和参数更新方法
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_target": "all",
        "lora_dropout": lora_dropout,

        #数据集
        "dataset_dir": str(dataset_path),
        "dataset": ",".join(normalized_dataset_names),
        "mix_strategy": "concat",
        "template": template,
        "cutoff_len": cutoff_len,
        "preprocessing_num_workers": 4,
        "dataloader_num_workers": 2,

        #输出
        "output_dir": str(adapter_dir),
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": 2,
        "plot_loss": True,
        "report_to": "none",
        "overwrite_output_dir": False,

        #训练超参数
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup_ratio,
        # "bf16": True,

        "gradient_checkpointing": True,
    }

    if use_qlora:
        config["quantization_bit"] = 4

    if mixed_precision == "bf16":
        config["bf16"] = True
    elif mixed_precision == "fp16":
        config["fp16"] = True

    # --- 4. 把Python字典写成YAML ---

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    # --- 5. 准备子进程 ---

    command = [
        cli_path,
        "train",
        str(config_path),
    ]

    child_env = os.environ.copy()

    child_env["PYTHONUNBUFFERED"] = "1"

    if gpu_id is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # --- 6. 启动训练并记录日志 ---

    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled(f"训练已取消: {run_id}")

    progress_path = adapter_dir / "trainer_log.jsonl"
    stop_event = threading.Event()

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        progress_thread = threading.Thread(
            target=monitor_training_progress,
            kwargs={
                "progress_path": progress_path,
                "stop_event": stop_event,
                "poll_interval": 0.5,
                "on_progress": on_progress,
            },
            name=f"progress-monitor-{run_id}",
            daemon=False,
        )

        progress_thread.start()

        process_done = threading.Event()
        cancel_thread: threading.Thread | None = None

        if on_process_started is not None:
            try:
                on_process_started(process.pid)
            except Exception as error:
                print(f"[PID回调失败] {error}", flush=True)

        if cancel_event is not None:
            def watch_cancellation() -> None:
                while not process_done.is_set():
                    if cancel_event.wait(0.2):
                        _terminate_process_group(process)
                        return

            cancel_thread = threading.Thread(
                target=watch_cancellation,
                name=f"cancel-monitor-{run_id}",
                daemon=False,
            )
            cancel_thread.start()

        try:
            print(f"训练已经启动, run_id:{run_id}")
            print(f"训练进程PID: {process.pid}")
            print(f"训练日志: {log_path}")

            assert process.stdout is not None

            for line in process.stdout:
                print(line, end="", flush=True)

                log_file.write(line)
                log_file.flush()

            return_code = process.wait()

        finally:
            process_done.set()
            if cancel_thread is not None:
                cancel_thread.join()
            stop_event.set()
            progress_thread.join()
            if process.stdout is not None:
                process.stdout.close()

    # --- 7. 检查进程退出结果 ---

    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled(f"训练已取消: {run_id}")

    if return_code != 0:
        log_tail =  read_log_tail(log_path)

        raise RuntimeError(
            f"LlamaFactory训练失败, 退出码: {return_code}\n"
            f"日志位置: {log_path}\n"
            f"日志最后50行: \n{log_tail}"
        )

    # --- 8. 检查Adapter是否真正生成 ---

    adapter_config_path = adapter_dir / "adapter_config.json"

    weight_files = [
        *adapter_dir.glob("adapter_model*.safetensors"),
        *adapter_dir.glob("adapter_model*.bin"),
    ]
    weight_files = [
        path for path in weight_files
        if path.is_file() and path.stat().st_size > 0
    ]
    if not adapter_config_path.is_file():
        raise RuntimeError(
            "训练进程成功退出, 但没有生成adapter_config.json"
        )

    if not weight_files:
        raise RuntimeError(
            "训练进程成功退出, 但没有生成有效的Adapter权重"
        )

    # --- 9. 读取Adapter配置和训练指标 ---

    with adapter_config_path.open("r", encoding="utf-8") as file:
        adapter_config = json.load(file)

    metrics_path = adapter_dir / "train_results.json"
    metrics: dict[str, Any] = {}

    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

    return {
        "run_id": run_id,
        "status": "SUCCEEDED",
        "pid": process.pid,
        "adapter_dir": str(adapter_dir),
        "adapter_config": adapter_config,
        "adapter_weights":[
            str(path) for path in weight_files
        ],
        "metrics": metrics,
        "config_path": str(config_path),
        "log_path": str(log_path),
    }

if __name__ == "__main__":
    result = train_lora_with_cli(
        # 同一环境可直接写llamafactory-cli;
        # 独立环境可写其绝对路径
        llamafactory_cli=("llamafactory-cli"),
        model_name_or_path="/absolute/path/to/base-model",
        dataset_dir="/absolute/path/to/dataset",
        dataset_names=[
            "dataset_name1",
            "dataset_name2",
            ],
        template="model_specific_template",
        runs_root = "/absolute/path/to/runs",
        gpu_id=0,
        use_qlora=True,
        lora_rank=8,
        cutoff_len=2048,
        learning_rate=1e-4,
        num_train_epochs=3.0,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
