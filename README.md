# FineTuningFactory

FineTuningFactory 是一个围绕 LlamaFactory CLI 构建的本地单 GPU 微调控制台。它直接读取本机模型目录和 LlamaFactory 数据集目录，创建 LoRA 或 QLoRA SFT 任务，并实时展示训练状态、结构化指标和控制台日志。

## 已实现功能

- 检查本地 Hugging Face 格式模型目录，包括 `config.json` 和非空权重文件。
- 读取本地 `dataset_info.json`，一次选择一个或多个已注册数据集。
- 配置 LoRA、QLoRA 4-bit、精度、rank、dropout、序列长度、学习率、epoch、batch size、梯度累积、日志与保存间隔。
- 使用 SQLite 持久化 `QUEUED -> RUNNING -> SUCCEEDED | FAILED | CANCELLED` 任务状态。
- 单 GPU Worker 串行执行训练，API 请求不会被长时间训练阻塞。
- 使用 SSE 实时发送任务快照和新增控制台日志。
- 保存每次训练的 YAML、日志、进度文件、指标和 Adapter 权重路径。
- 支持取消排队中或运行中的任务，并终止 LlamaFactory 子进程组。

## 项目结构

```text
backend/app/
├── api/                         请求模型
├── application/                 本地资源检查与任务创建
├── domain/                      Run状态机
├── infrastructure/
│   ├── database.py              SQLite仓库
│   └── llamafactory/cli_runner.py
├── workers/training_worker.py   单GPU Worker
└── main.py                      FastAPI和SSE

frontend/
└── src/                         React训练控制台
```

## 环境准备

需要 Python 3.12+、Node.js、npm，以及与目标硬件匹配的 LlamaFactory 运行环境。

```bash
uv sync
cd frontend
npm install
cd ..
```

当前项目验证使用 LlamaFactory `0.9.5`。macOS 可用于界面、API、假 CLI 和小模型流程验证；QLoRA 4-bit 的正式训练通常需要 Linux、NVIDIA GPU、CUDA 及兼容的 bitsandbytes 环境。

## 启动开发环境

```bash
./scripts/dev.sh
```

打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)。Vite 会把 `/api` 转发到 `127.0.0.1:8000`。

也可以分别启动：

```bash
.venv/bin/python -m backend.app.serve
```

```bash
cd frontend
npm run dev
```

## 单端口运行

先构建前端：

```bash
cd frontend
npm run build
cd ..
.venv/bin/python -m backend.app.serve
```

构建产物存在时，FastAPI 会直接托管前端。打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

## 本地资源格式

模型目录至少需要：

```text
/path/to/model/
├── config.json
├── tokenizer.json
└── model-00001-of-00002.safetensors
```

数据集目录使用 LlamaFactory 的注册方式：

```text
/path/to/datasets/
├── dataset_info.json
├── math.json
└── code.json
```

示例 `dataset_info.json`：

```json
{
  "math_sft": { "file_name": "math.json" },
  "code_sft": { "file_name": "code.json" }
}
```

前端填写的是数据集目录，检查成功后再勾选 `math_sft`、`code_sft` 等注册名。

## 配置

默认运行状态位于项目的 `.data/`：

```text
.data/
├── finetuningfactory.sqlite3
└── runs/<run_id>/
    ├── train.yaml
    ├── console.log
    └── adapter/
```

可使用环境变量覆盖：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `FTF_DATA_DIR` | `<project>/.data` | SQLite和Run产物根目录 |
| `FTF_LLAMAFACTORY_CLI` | 当前环境中的CLI | `llamafactory-cli`绝对路径或命令 |
| `FTF_HOST` | `127.0.0.1` | API监听地址 |
| `FTF_PORT` | `8000` | API端口 |

## 测试

端到端测试使用临时本地模型、两个本地数据集和假 CLI，不需要 GPU：

```bash
.venv/bin/python -W error::ResourceWarning -m unittest -v backend.tests.test_training_api
cd frontend && npm run build
```

## 安全边界

这是本地单用户 MVP，不包含登录、权限隔离、配额或容器沙箱。后端会读取请求中提供的本地路径，`trust_remote_code` 还可能执行模型仓库中的 Python 代码。因此默认只监听回环地址，不要直接暴露到公网，也不要对不可信用户开放。
