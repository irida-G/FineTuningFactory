from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys
import textwrap
import time
import unittest

import yaml
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


class TrainingApiIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model_dir = self.root / "model"
        self.dataset_dir = self.root / "datasets"
        self.fake_cli = self.root / "fake_llamafactory_cli.py"
        self.model_dir.mkdir()
        self.dataset_dir.mkdir()

        (self.model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2",
                    "architectures": ["Qwen2ForCausalLM"],
                    "torch_dtype": "bfloat16",
                    "hidden_size": 1024,
                    "num_hidden_layers": 24,
                    "vocab_size": 151936,
                }
            ),
            encoding="utf-8",
        )
        (self.model_dir / "model.safetensors").write_bytes(b"fake model weights")
        (self.dataset_dir / "math.json").write_text(
            json.dumps([{"instruction": "1+1", "output": "2"}]),
            encoding="utf-8",
        )
        (self.dataset_dir / "code.json").write_text(
            json.dumps([{"instruction": "print", "output": "print('ok')"}]),
            encoding="utf-8",
        )
        (self.dataset_dir / "dataset_info.json").write_text(
            json.dumps(
                {
                    "math_sft": {"file_name": "math.json"},
                    "code_sft": {"file_name": "code.json"},
                }
            ),
            encoding="utf-8",
        )
        self.fake_cli.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import sys
                import time
                from pathlib import Path
                import yaml

                config_path = Path(sys.argv[2])
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                output_dir = Path(config["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                progress_path = output_dir / "trainer_log.jsonl"
                for step in range(1, 4):
                    progress = {{
                        "current_steps": step,
                        "total_steps": 3,
                        "loss": round(1 / step, 4),
                        "lr": 0.0001,
                        "epoch": round(step / 3, 2),
                        "percentage": round(step / 3 * 100, 2),
                        "remaining_time": f"0:00:0{{3 - step}}",
                    }}
                    with progress_path.open("a", encoding="utf-8") as log:
                        log.write(json.dumps(progress) + "\\n")
                    print(f"fake progress: {{step}}/3", flush=True)
                    time.sleep(0.2)

                (output_dir / "adapter_config.json").write_text(
                    json.dumps({{"peft_type": "LORA"}}), encoding="utf-8"
                )
                (output_dir / "adapter_model.safetensors").write_bytes(b"adapter")
                (output_dir / "train_results.json").write_text(
                    json.dumps({{"train_loss": 0.3333}}), encoding="utf-8"
                )
                print("training complete", flush=True)
                """
            ),
            encoding="utf-8",
        )
        os.chmod(self.fake_cli, 0o755)

        data_dir = self.root / "state"
        settings = Settings(
            data_dir=data_dir,
            runs_dir=data_dir / "runs",
            database_path=data_dir / "runs.sqlite3",
            llamafactory_cli=str(self.fake_cli),
            frontend_dist=self.root / "missing-frontend",
            poll_interval=0.05,
        )
        self.client_context = TestClient(create_app(settings))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary_directory.cleanup()

    def request_payload(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_dir),
            "dataset_dir": str(self.dataset_dir),
            "dataset_names": ["math_sft", "code_sft"],
            "template": "qwen",
            "method": "qlora",
            "gpu_id": None,
            "mixed_precision": "fp32",
            "lora_rank": 16,
            "logging_steps": 1,
        }

    def wait_for_terminal(self, run_id: str, timeout: float = 8) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(response.status_code, 200)
            run = response.json()
            if run["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                return run
            time.sleep(0.05)
        self.fail("训练任务未在超时时间内结束")

    def test_local_qlora_run_streams_progress_and_produces_adapter(self) -> None:
        model_response = self.client.post(
            "/api/resources/model/inspect",
            json={"path": str(self.model_dir)},
        )
        self.assertEqual(model_response.status_code, 200)
        self.assertEqual(model_response.json()["suggested_template"], "qwen")

        dataset_response = self.client.post(
            "/api/resources/datasets/inspect",
            json={"path": str(self.dataset_dir)},
        )
        self.assertEqual(dataset_response.status_code, 200)
        self.assertEqual(
            {item["name"] for item in dataset_response.json()["datasets"]},
            {"math_sft", "code_sft"},
        )

        create_response = self.client.post("/api/runs", json=self.request_payload())
        self.assertEqual(create_response.status_code, 202)
        run_id = create_response.json()["id"]
        run = self.wait_for_terminal(run_id)

        self.assertEqual(run["status"], "SUCCEEDED", run.get("error"))
        self.assertEqual(run["progress"]["current_steps"], 3)
        self.assertTrue(Path(run["adapter_dir"]).joinpath("adapter_config.json").is_file())

        config = yaml.safe_load(Path(run["config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(config["dataset"], "math_sft,code_sft")
        self.assertEqual(config["quantization_bit"], 4)
        self.assertEqual(config["lora_rank"], 16)

        log_response = self.client.get(f"/api/runs/{run_id}/log")
        self.assertIn("fake progress: 3/3", log_response.json()["content"])

        with self.client.stream("GET", f"/api/runs/{run_id}/events") as response:
            stream = "".join(response.iter_text())
        self.assertIn("event: snapshot", stream)
        self.assertIn("event: log", stream)
        self.assertIn('"status": "SUCCEEDED"', stream)

    def test_queued_or_running_run_can_be_cancelled(self) -> None:
        create_response = self.client.post("/api/runs", json=self.request_payload())
        self.assertEqual(create_response.status_code, 202)
        run_id = create_response.json()["id"]

        cancel_response = self.client.post(f"/api/runs/{run_id}/cancel")
        self.assertEqual(cancel_response.status_code, 200)

        run = self.wait_for_terminal(run_id)
        self.assertEqual(run["status"], "CANCELLED")
        self.assertTrue(run["cancel_requested"])


if __name__ == "__main__":
    unittest.main()
