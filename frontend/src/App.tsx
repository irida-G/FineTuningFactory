import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Ban,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  Copy,
  Cpu,
  Database,
  FileCode2,
  FolderSearch,
  Gauge,
  HardDrive,
  LoaderCircle,
  Play,
  RefreshCw,
  ScrollText,
  Settings2,
  SquareTerminal,
} from "lucide-react";

type Status = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";

type SystemInfo = {
  llamafactory_cli: string;
  llamafactory_available: boolean;
  llamafactory_version: string | null;
  runs_dir: string;
  worker_mode: string;
};

type ModelInfo = {
  path: string;
  name: string;
  model_type: string | null;
  architectures: string[];
  torch_dtype: string | null;
  hidden_size: number | null;
  num_hidden_layers: number | null;
  vocab_size: number | null;
  weight_file_count: number;
  suggested_template: string;
};

type DatasetInfo = {
  name: string;
  file_name: string;
  size_bytes: number;
  format: string;
  ranking: boolean;
};

type Progress = {
  current_steps?: number;
  total_steps?: number;
  step?: number;
  loss?: number;
  eval_loss?: number;
  lr?: number;
  epoch?: number;
  percentage?: number;
  elapsed_time?: string;
  remaining_time?: string;
};

type RunRequest = {
  model_path: string;
  dataset_dir: string;
  dataset_names: string[];
  template: string;
  method: "lora" | "qlora";
  gpu_id: number | null;
  mixed_precision: "bf16" | "fp16" | "fp32";
  lora_rank: number;
  lora_dropout: number;
  cutoff_len: number;
  learning_rate: number;
  num_train_epochs: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  logging_steps: number;
  save_steps: number;
  warmup_ratio: number;
  trust_remote_code: boolean;
};

type Run = {
  id: string;
  status: Status;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  request: RunRequest;
  progress: Progress | null;
  result: Record<string, unknown> | null;
  error: string | null;
  pid: number | null;
  run_dir: string | null;
  adapter_dir: string | null;
  config_path: string | null;
  log_path: string | null;
  cancel_requested: boolean;
};

const initialForm: RunRequest = {
  model_path: "",
  dataset_dir: "",
  dataset_names: [],
  template: "",
  method: "qlora",
  gpu_id: null,
  mixed_precision: "bf16",
  lora_rank: 8,
  lora_dropout: 0,
  cutoff_len: 2048,
  learning_rate: 0.0001,
  num_train_epochs: 3,
  per_device_train_batch_size: 1,
  gradient_accumulation_steps: 8,
  logging_steps: 10,
  save_steps: 100,
  warmup_ratio: 0.1,
  trust_remote_code: false,
};

const terminalStatuses = new Set<Status>(["SUCCEEDED", "FAILED", "CANCELLED"]);

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `请求失败 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function formatDate(value: string | null): string {
  if (!value) return "--";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function statusLabel(status: Status): string {
  return {
    QUEUED: "排队中",
    RUNNING: "训练中",
    SUCCEEDED: "已完成",
    FAILED: "失败",
    CANCELLED: "已取消",
  }[status];
}

function StatusBadge({ status }: { status: Status }) {
  return <span className={`status status-${status.toLowerCase()}`}>{statusLabel(status)}</span>;
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [form, setForm] = useState<RunRequest>(initialForm);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [log, setLog] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const selectedIdRef = useRef<string | null>(null);

  const loadRuns = useCallback(async () => {
    const data = await api<Run[]>("/api/runs");
    setRuns(data);
    const selectedId = selectedIdRef.current;
    if (selectedId) {
      const updated = data.find((run) => run.id === selectedId);
      if (updated) setSelectedRun(updated);
    }
  }, []);

  useEffect(() => {
    Promise.all([api<SystemInfo>("/api/system"), api<Run[]>("/api/runs")])
      .then(([systemData, runData]) => {
        setSystem(systemData);
        setRuns(runData);
        if (runData[0]) setSelectedRun(runData[0]);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void loadRuns().catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [loadRuns]);

  useEffect(() => {
    selectedIdRef.current = selectedRun?.id ?? null;
    if (!selectedRun) return;
    setLog("");
    let terminal = terminalStatuses.has(selectedRun.status);
    const source = new EventSource(`/api/runs/${selectedRun.id}/events`);
    source.addEventListener("snapshot", (event) => {
      const updated = JSON.parse((event as MessageEvent).data) as Run;
      terminal = terminalStatuses.has(updated.status);
      setSelectedRun(updated);
      setRuns((current) => {
        const exists = current.some((run) => run.id === updated.id);
        const next = exists
          ? current.map((run) => (run.id === updated.id ? updated : run))
          : [updated, ...current];
        return next;
      });
    });
    source.addEventListener("log", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as { content: string };
      setLog((current) => `${current}${payload.content}`.slice(-200_000));
    });
    source.onerror = () => {
      if (terminal) source.close();
    };
    return () => source.close();
  }, [selectedRun?.id]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "nearest" });
  }, [log]);

  async function inspectModel() {
    setBusy("model");
    setError(null);
    try {
      const info = await api<ModelInfo>("/api/resources/model/inspect", {
        method: "POST",
        body: JSON.stringify({ path: form.model_path }),
      });
      setModelInfo(info);
      setForm((current) => ({
        ...current,
        model_path: info.path,
        template: current.template || info.suggested_template,
      }));
    } catch (reason) {
      setModelInfo(null);
      setError((reason as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function inspectDatasets() {
    setBusy("datasets");
    setError(null);
    try {
      const info = await api<{ path: string; datasets: DatasetInfo[] }>(
        "/api/resources/datasets/inspect",
        { method: "POST", body: JSON.stringify({ path: form.dataset_dir }) },
      );
      setDatasets(info.datasets);
      setForm((current) => ({
        ...current,
        dataset_dir: info.path,
        dataset_names: info.datasets.map((dataset) => dataset.name),
      }));
    } catch (reason) {
      setDatasets([]);
      setError((reason as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy("submit");
    setError(null);
    try {
      const run = await api<Run>("/api/runs", {
        method: "POST",
        body: JSON.stringify(form),
      });
      setRuns((current) => [run, ...current]);
      setSelectedRun(run);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function cancelRun() {
    if (!selectedRun) return;
    setBusy("cancel");
    setError(null);
    try {
      const run = await api<Run>(`/api/runs/${selectedRun.id}/cancel`, { method: "POST" });
      setSelectedRun(run);
      await loadRuns();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(null);
    }
  }

  function updateNumber<K extends keyof RunRequest>(key: K, value: string) {
    setForm((current) => ({ ...current, [key]: Number(value) }));
  }

  function copyPath(value: string | null) {
    if (value) void navigator.clipboard.writeText(value);
  }

  const percentage = useMemo(() => {
    if (!selectedRun) return 0;
    if (selectedRun.status === "SUCCEEDED") return 100;
    const progress = selectedRun.progress;
    if (typeof progress?.percentage === "number") return progress.percentage;
    const current = progress?.current_steps ?? progress?.step;
    if (current && progress?.total_steps) return (current / progress.total_steps) * 100;
    return 0;
  }, [selectedRun]);

  const canSubmit = Boolean(
    system?.llamafactory_available &&
      modelInfo &&
      datasets.length &&
      form.dataset_names.length &&
      form.template &&
      !busy,
  );

  const artifactPaths = selectedRun
    ? [
        { label: "Adapter", value: selectedRun.adapter_dir, Icon: HardDrive },
        { label: "训练配置", value: selectedRun.config_path, Icon: FileCode2 },
        { label: "控制台日志", value: selectedRun.log_path, Icon: ScrollText },
      ]
    : [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Gauge size={19} /></div>
          <div>
            <h1>FineTuningFactory</h1>
            <p>本地 Adapter 训练控制台</p>
          </div>
        </div>
        <div className="system-strip">
          <span className={`system-dot ${system?.llamafactory_available ? "online" : "offline"}`} />
          <span>{system?.llamafactory_available ? "LlamaFactory 就绪" : "CLI 不可用"}</span>
          <span className="system-separator" />
          <Cpu size={15} />
          <span>单 GPU Worker</span>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <CircleAlert size={17} />
          <span>{error}</span>
          <button className="icon-button" type="button" title="关闭错误" onClick={() => setError(null)}>×</button>
        </div>
      )}

      <main className="workspace">
        <form className="config-pane" onSubmit={submit}>
          <section className="section-block">
            <div className="section-heading">
              <HardDrive size={17} />
              <h2>训练资源</h2>
            </div>

            <Field label="本地模型目录" hint="目录内需要config.json和模型权重">
              <div className="input-command">
                <input
                  value={form.model_path}
                  onChange={(event) => {
                    setForm({ ...form, model_path: event.target.value });
                    setModelInfo(null);
                  }}
                  placeholder="/data/models/Qwen3-4B"
                />
                <button className="icon-command" type="button" title="检查模型目录" onClick={inspectModel} disabled={busy === "model"}>
                  {busy === "model" ? <LoaderCircle className="spin" size={17} /> : <FolderSearch size={17} />}
                  检查
                </button>
              </div>
            </Field>

            {modelInfo && (
              <div className="inspection-row">
                <Check size={15} />
                <strong>{modelInfo.name}</strong>
                <span>{modelInfo.model_type ?? "未知架构"}</span>
                <span>{modelInfo.weight_file_count} 个权重文件</span>
                {modelInfo.torch_dtype && <span>{modelInfo.torch_dtype}</span>}
              </div>
            )}

            <Field label="本地数据集目录" hint="目录内需要dataset_info.json及其注册的数据文件">
              <div className="input-command">
                <input
                  value={form.dataset_dir}
                  onChange={(event) => {
                    setForm({ ...form, dataset_dir: event.target.value, dataset_names: [] });
                    setDatasets([]);
                  }}
                  placeholder="/data/sft-datasets"
                />
                <button className="icon-command" type="button" title="读取数据集注册信息" onClick={inspectDatasets} disabled={busy === "datasets"}>
                  {busy === "datasets" ? <LoaderCircle className="spin" size={17} /> : <Database size={17} />}
                  读取
                </button>
              </div>
            </Field>

            {datasets.length > 0 && (
              <div className="dataset-list">
                {datasets.map((dataset) => (
                  <label className="dataset-row" key={dataset.name}>
                    <input
                      type="checkbox"
                      checked={form.dataset_names.includes(dataset.name)}
                      onChange={(event) => {
                        const names = event.target.checked
                          ? [...form.dataset_names, dataset.name]
                          : form.dataset_names.filter((name) => name !== dataset.name);
                        setForm({ ...form, dataset_names: names });
                      }}
                    />
                    <span className="dataset-name">{dataset.name}</span>
                    <span>{dataset.format}</span>
                    <span>{formatBytes(dataset.size_bytes)}</span>
                    <span className="dataset-file">{dataset.file_name}</span>
                  </label>
                ))}
              </div>
            )}
          </section>

          <section className="section-block">
            <div className="section-heading">
              <Settings2 size={17} />
              <h2>微调配置</h2>
            </div>

            <div className="method-control" role="group" aria-label="微调方法">
              <button type="button" className={form.method === "lora" ? "active" : ""} onClick={() => setForm({ ...form, method: "lora" })}>LoRA</button>
              <button type="button" className={form.method === "qlora" ? "active" : ""} onClick={() => setForm({ ...form, method: "qlora" })}>QLoRA · 4-bit</button>
            </div>

            <div className="form-grid">
              <Field label="对话模板">
                <input value={form.template} onChange={(event) => setForm({ ...form, template: event.target.value })} placeholder="qwen / llama3" />
              </Field>
              <Field label="计算精度">
                <select value={form.mixed_precision} onChange={(event) => setForm({ ...form, mixed_precision: event.target.value as RunRequest["mixed_precision"] })}>
                  <option value="bf16">BF16</option>
                  <option value="fp16">FP16</option>
                  <option value="fp32">FP32</option>
                </select>
              </Field>
              <Field label="LoRA Rank">
                <input type="number" min="1" value={form.lora_rank} onChange={(event) => updateNumber("lora_rank", event.target.value)} />
              </Field>
              <Field label="LoRA Dropout">
                <input type="number" min="0" max="0.99" step="0.01" value={form.lora_dropout} onChange={(event) => updateNumber("lora_dropout", event.target.value)} />
              </Field>
              <Field label="截断长度">
                <input type="number" min="128" step="128" value={form.cutoff_len} onChange={(event) => updateNumber("cutoff_len", event.target.value)} />
              </Field>
              <Field label="学习率">
                <input type="number" min="0" step="0.00001" value={form.learning_rate} onChange={(event) => updateNumber("learning_rate", event.target.value)} />
              </Field>
              <Field label="训练轮数">
                <input type="number" min="0.1" step="0.5" value={form.num_train_epochs} onChange={(event) => updateNumber("num_train_epochs", event.target.value)} />
              </Field>
              <Field label="设备 Batch Size">
                <input type="number" min="1" value={form.per_device_train_batch_size} onChange={(event) => updateNumber("per_device_train_batch_size", event.target.value)} />
              </Field>
              <Field label="梯度累积步数">
                <input type="number" min="1" value={form.gradient_accumulation_steps} onChange={(event) => updateNumber("gradient_accumulation_steps", event.target.value)} />
              </Field>
              <Field label="日志间隔">
                <input type="number" min="1" value={form.logging_steps} onChange={(event) => updateNumber("logging_steps", event.target.value)} />
              </Field>
              <Field label="保存间隔">
                <input type="number" min="1" value={form.save_steps} onChange={(event) => updateNumber("save_steps", event.target.value)} />
              </Field>
              <Field label="Warmup Ratio">
                <input type="number" min="0" max="1" step="0.01" value={form.warmup_ratio} onChange={(event) => updateNumber("warmup_ratio", event.target.value)} />
              </Field>
              <Field label="GPU 设备">
                <select
                  value={form.gpu_id ?? "auto"}
                  onChange={(event) => setForm({
                    ...form,
                    gpu_id: event.target.value === "auto" ? null : Number(event.target.value),
                  })}
                >
                  <option value="auto">自动选择</option>
                  <option value="0">GPU 0</option>
                  <option value="1">GPU 1</option>
                </select>
              </Field>
            </div>
            <label className="switch-row">
              <span>
                <strong>信任模型自定义代码</strong>
                <small>仅对可信本地模型启用 trust_remote_code</small>
              </span>
              <input
                type="checkbox"
                checked={form.trust_remote_code}
                onChange={(event) => setForm({ ...form, trust_remote_code: event.target.checked })}
              />
            </label>
          </section>

          <div className="submit-bar">
            <div>
              <strong>{form.method === "qlora" ? "QLoRA 4-bit" : "LoRA"}</strong>
              <span>{form.dataset_names.length} 个数据集 · rank {form.lora_rank} · 有效 batch {form.per_device_train_batch_size * form.gradient_accumulation_steps}</span>
            </div>
            <button className="primary-button" type="submit" disabled={!canSubmit}>
              {busy === "submit" ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />}
              创建训练任务
            </button>
          </div>
        </form>

        <aside className="runs-pane">
          <div className="pane-heading">
            <div><h2>训练任务</h2><span>{runs.length} 条记录</span></div>
            <button className="icon-button" type="button" title="刷新任务" onClick={() => void loadRuns()}><RefreshCw size={16} /></button>
          </div>
          <div className="run-list">
            {runs.length === 0 && <div className="empty-state"><Clock3 size={24} /><span>还没有训练任务</span></div>}
            {runs.map((run) => (
              <button key={run.id} type="button" className={`run-item ${selectedRun?.id === run.id ? "selected" : ""}`} onClick={() => setSelectedRun(run)}>
                <div className="run-item-top">
                  <StatusBadge status={run.status} />
                  <span>{formatDate(run.created_at)}</span>
                </div>
                <strong>{run.request.model_path.split("/").filter(Boolean).pop()}</strong>
                <div className="run-item-meta">
                  <span>{run.request.method.toUpperCase()}</span>
                  <span>{run.request.dataset_names.length} 数据集</span>
                  {run.pid && <span>PID {run.pid}</span>}
                </div>
                <ChevronRight size={16} />
              </button>
            ))}
          </div>
        </aside>

        <section className="detail-pane">
          {!selectedRun ? (
            <div className="empty-detail"><Activity size={30} /><span>选择一个任务查看训练状态</span></div>
          ) : (
            <>
              <div className="detail-header">
                <div>
                  <div className="detail-title"><StatusBadge status={selectedRun.status} /><code>{selectedRun.id}</code></div>
                  <p>{selectedRun.request.model_path}</p>
                </div>
                {(selectedRun.status === "QUEUED" || selectedRun.status === "RUNNING") && (
                  <button className="danger-button" type="button" onClick={cancelRun} disabled={busy === "cancel"}>
                    {busy === "cancel" ? <LoaderCircle className="spin" size={16} /> : <Ban size={16} />}
                    取消任务
                  </button>
                )}
              </div>

              <div className="progress-band">
                <div className="progress-summary">
                  <strong>{Math.max(0, Math.min(100, percentage)).toFixed(1)}%</strong>
                  <span>
                    step {selectedRun.progress?.current_steps ?? selectedRun.progress?.step ?? "--"}
                    /{selectedRun.progress?.total_steps ?? "--"}
                  </span>
                </div>
                <div className="progress-track"><div style={{ width: `${Math.max(0, Math.min(100, percentage))}%` }} /></div>
                <div className="metric-strip">
                  <div><span>Loss</span><strong>{selectedRun.progress?.loss ?? selectedRun.progress?.eval_loss ?? "--"}</strong></div>
                  <div><span>Epoch</span><strong>{selectedRun.progress?.epoch ?? "--"}</strong></div>
                  <div><span>学习率</span><strong>{selectedRun.progress?.lr ?? "--"}</strong></div>
                  <div><span>剩余时间</span><strong>{selectedRun.progress?.remaining_time ?? "--"}</strong></div>
                </div>
              </div>

              {selectedRun.error && <div className="run-error"><CircleAlert size={16} /><span>{selectedRun.error}</span></div>}

              <div className="detail-columns">
                <div className="log-panel">
                  <div className="panel-heading"><span><SquareTerminal size={16} />训练日志</span><span className="live-indicator">实时</span></div>
                  <pre>{log || "等待训练进程输出..."}<div ref={logEndRef} /></pre>
                </div>
                <div className="artifact-panel">
                  <div className="panel-heading"><span><ScrollText size={16} />运行信息</span></div>
                  <dl>
                    <div><dt>状态</dt><dd>{statusLabel(selectedRun.status)}</dd></div>
                    <div><dt>开始时间</dt><dd>{formatDate(selectedRun.started_at)}</dd></div>
                    <div><dt>结束时间</dt><dd>{formatDate(selectedRun.finished_at)}</dd></div>
                    <div><dt>模板</dt><dd>{selectedRun.request.template}</dd></div>
                    <div><dt>精度</dt><dd>{selectedRun.request.mixed_precision.toUpperCase()}</dd></div>
                  </dl>
                  <div className="path-list">
                    {artifactPaths.map(({ label, value, Icon }) => (
                      <div className="path-row" key={label}>
                        <Icon size={15} />
                        <div><span>{label}</span><code>{value ?? "尚未生成"}</code></div>
                        <button type="button" className="icon-button" title={`复制${label}路径`} disabled={!value} onClick={() => copyPath(value)}><Copy size={14} /></button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
