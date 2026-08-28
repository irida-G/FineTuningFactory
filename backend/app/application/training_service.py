from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import uuid

from backend.app.api.schemas import TrainingRunRequest
from backend.app.infrastructure.database import RunRepository


_TEMPLATE_BY_MODEL_TYPE = {
    "qwen2": "qwen",
    "qwen3": "qwen3",
    "llama": "llama3",
    "mistral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma",
    "chatglm": "chatglm3",
    "internlm2": "intern2",
    "baichuan": "baichuan2",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek3",
}


class TrainingService:
    def __init__(self, repository: RunRepository):
        self.repository = repository

    def create_run(self, request: TrainingRunRequest) -> dict[str, Any]:
        model = self.inspect_model(request.model_path)
        datasets = self.inspect_dataset_dir(request.dataset_dir)
        available = {item["name"] for item in datasets["datasets"]}
        missing = [name for name in request.dataset_names if name not in available]
        if missing:
            raise ValueError(
                "dataset_info.json中没有可用的本地数据集: " + ", ".join(missing)
            )

        payload = request.model_dump()
        payload["model_path"] = model["path"]
        payload["dataset_dir"] = datasets["path"]
        run_id = uuid.uuid4().hex
        return self.repository.create(run_id, payload)

    @staticmethod
    def inspect_model(raw_path: str) -> dict[str, Any]:
        model_path = Path(raw_path).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError(f"模型目录不存在: {model_path}")

        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"模型目录中缺少config.json: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"模型config.json不是合法JSON: {error}") from error

        if not isinstance(config, dict):
            raise ValueError("模型config.json顶层必须是JSON对象")

        model_type = config.get("model_type")
        suggested_template = _TEMPLATE_BY_MODEL_TYPE.get(str(model_type), "")
        weight_files = sorted(
            path.name
            for pattern in ("*.safetensors", "*.bin")
            for path in model_path.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        )
        if not weight_files:
            raise ValueError("模型目录中没有非空的.safetensors或.bin权重文件")

        return {
            "path": str(model_path),
            "name": model_path.name,
            "model_type": model_type,
            "architectures": config.get("architectures", []),
            "torch_dtype": config.get("torch_dtype"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "vocab_size": config.get("vocab_size"),
            "weight_file_count": len(weight_files),
            "suggested_template": suggested_template,
        }

    @staticmethod
    def inspect_dataset_dir(raw_path: str) -> dict[str, Any]:
        dataset_dir = Path(raw_path).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise ValueError(f"数据集目录不存在: {dataset_dir}")

        info_path = dataset_dir / "dataset_info.json"
        if not info_path.is_file():
            raise ValueError(f"数据集目录中缺少dataset_info.json: {info_path}")

        try:
            with info_path.open("r", encoding="utf-8") as file:
                dataset_info = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"dataset_info.json不是合法JSON: {error}") from error

        if not isinstance(dataset_info, dict):
            raise ValueError("dataset_info.json顶层必须是JSON对象")

        datasets: list[dict[str, Any]] = []
        for name, entry in dataset_info.items():
            if not isinstance(entry, dict):
                continue
            file_name = entry.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                continue

            data_path = (dataset_dir / file_name.strip()).resolve()
            try:
                data_path.relative_to(dataset_dir)
            except ValueError:
                continue
            if not data_path.is_file():
                continue

            datasets.append(
                {
                    "name": name,
                    "file_name": file_name,
                    "size_bytes": data_path.stat().st_size,
                    "format": entry.get("format", "alpaca"),
                    "columns": entry.get("columns", {}),
                    "ranking": bool(entry.get("ranking", False)),
                }
            )

        if not datasets:
            raise ValueError("dataset_info.json中没有指向现有文件的本地数据集")

        return {
            "path": str(dataset_dir),
            "dataset_info_path": str(info_path),
            "datasets": datasets,
        }
