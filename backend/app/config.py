from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    runs_dir: Path
    database_path: Path
    llamafactory_cli: str
    frontend_dist: Path
    poll_interval: float = 0.5

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(
            os.getenv("FTF_DATA_DIR", str(PROJECT_ROOT / ".data"))
        ).expanduser().resolve()
        runs_dir = data_dir / "runs"
        cli = os.getenv("FTF_LLAMAFACTORY_CLI")
        if not cli:
            cli = shutil.which("llamafactory-cli") or str(
                PROJECT_ROOT / ".venv" / "bin" / "llamafactory-cli"
            )

        return cls(
            data_dir=data_dir,
            runs_dir=runs_dir,
            database_path=data_dir / "finetuningfactory.sqlite3",
            llamafactory_cli=cli,
            frontend_dist=PROJECT_ROOT / "frontend" / "dist",
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
