from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PathInspectionRequest(BaseModel):
    path: str = Field(min_length=1)


class TrainingRunRequest(BaseModel):
    model_path: str = Field(min_length=1)
    dataset_dir: str = Field(min_length=1)
    dataset_names: list[str] = Field(min_length=1)
    template: str = Field(min_length=1)
    method: Literal["lora", "qlora"] = "qlora"
    gpu_id: int | None = Field(default=0, ge=0)
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    lora_rank: int = Field(default=8, ge=1, le=1024)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    cutoff_len: int = Field(default=2048, ge=128, le=262144)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    num_train_epochs: float = Field(default=3.0, gt=0, le=1000)
    per_device_train_batch_size: int = Field(default=1, ge=1, le=1024)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=65536)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    trust_remote_code: bool = False

    @field_validator("model_path", "dataset_dir", "template")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value

    @field_validator("dataset_names")
    @classmethod
    def normalize_dataset_names(cls, values: list[str]) -> list[str]:
        names = [value.strip() for value in values if value.strip()]
        if not names:
            raise ValueError("至少选择一个数据集")
        if len(names) != len(set(names)):
            raise ValueError("数据集名称不能重复")
        return names
