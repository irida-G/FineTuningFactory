#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "缺少项目环境，请先执行: uv sync"
  exit 1
fi

if [[ ! -d "frontend/node_modules" ]]; then
  echo "缺少前端依赖，请先执行: cd frontend && npm install"
  exit 1
fi

.venv/bin/python -m backend.app.serve &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd frontend
npm run dev
