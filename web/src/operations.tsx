/** 总览、实验任务和数据索引页面；真实运行按保存配置创建隔离快照。 */
import { useState } from "react";
import { Activity, Database, FlaskConical, Server } from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { api, items, type RecordData } from "./api";
import {
  Card,
  Empty,
  ErrorBox,
  Json,
  Pager,
  Refresh,
  State,
  format,
  perform,
  useApi,
  statusLabel,
} from "./ui";
/** 总览展示真实健康、资源曲线与最近任务；navigate 切换业务页。 */
export function Overview({ navigate }: { navigate: (page: string) => void }) {
  const health = useApi("/health", 10000);
  const services = useApi("/services", 1000);
  const [startLoading, setStartLoading] = useState<Record<string, boolean>>({});
  const [startErrors, setStartErrors] = useState<Record<string, string>>({});
  const [range, setRange] = useState(15);
  const resources = useApi(
    `/resources?since=${Math.floor(Date.now() / 60000) * 60 - range * 60}&limit=600`,
    5000,
  );
  const tasks = useApi("/tasks", 4000);
  const runs = useApi("/runs", 5000);
  /** 请求启动指定服务；name 为后端允许的 vllm 或 chroma 名称。 */
  async function startService(name: "vllm" | "chroma") {
    setStartLoading((current) => ({ ...current, [name]: true }));
    setStartErrors((current) => ({ ...current, [name]: "" }));
    try {
      await api(`/services/${name}/start`, {});
      await services.refresh();
    } catch (error) {
      setStartErrors((current) => ({ ...current, [name]: String(error) }));
      await services.refresh();
    } finally {
      setStartLoading((current) => ({ ...current, [name]: false }));
    }
  }
  const points = items(resources.data).map((row) => ({
    ...row,
    time: new Date(row.timestamp * 1000).toLocaleTimeString(),
    cpu: row.cpu_percent,
    memory:
      row.wsl_memory_swap?.used_bytes === undefined
        ? null
        : row.wsl_memory_swap.used_bytes / 2 ** 30,
    gpu: row.nvidia_smi_device_wide
      ? Number(row.nvidia_smi_device_wide.split(",")[2])
      : null,
  }));
  const latest = items(resources.data).at(-1);
  const gpu = latest?.nvidia_smi_device_wide?.split(",").map(Number);
  return (
    <>
      <div className="metrics">
        {[
          ["backend", "Web API", Server],
          ["vllm", "生成服务", Activity],
          ["chroma", "向量数据库", Database],
        ].map(([key, label, Icon]) => {
          const value = health.data?.[String(key)];
          const Graphic = Icon as typeof Server;
          if (key === "backend")
            return (
              <div className="metric" key={String(key)}>
                <Graphic size={21} />
                <span>{String(label)}</span>
                <strong>
                  {value?.status
                    ? statusLabel(value.status)
                    : health.loading
                      ? "读取中"
                      : "未连接"}
                </strong>
                {value?.error && (
                  <small className="danger">{String(value.error)}</small>
                )}
              </div>
            );
          const service = services.data?.services?.[String(key)];
          const active = ["starting", "running"].includes(service?.status);
          const serviceLabel: Record<string, string> = {
            stopped: "已停止",
            unavailable: "不可达",
            starting: "启动中",
            running: "运行中",
            failed: "启动失败",
          };
          const stageLabel: Record<string, string> = {
            not_started: "未启动",
            validating: "校验配置",
            launching: "启动服务进程",
            waiting_for_health: "等待健康检查",
            healthy: "健康",
            driver_exited: "启动进程已退出",
            process_exited: "服务进程已退出",
            failed: "启动失败",
            health_failed: "健康检查失败",
            external_process: "检测到外部服务进程",
          };
          const serviceError =
            startErrors[String(key)] ||
            service?.error ||
            service?.health?.error;
          return (
            <div className="metric" key={String(key)}>
              <Graphic size={21} />
              <span>{String(label)}</span>
              <strong>
                {serviceLabel[service?.status] ??
                  (services.loading ? "读取中" : "不可达")}
              </strong>
              {service?.stage && (
                <small>
                  阶段：
                  {stageLabel[service.stage] ?? String(service.stage)}
                </small>
              )}
              {service?.elapsed_seconds != null && (
                <small>
                  {service.status === "starting" ? "已等待：" : "启动耗时："}
                  {format(service.elapsed_seconds, " 秒")}
                </small>
              )}
              {key === "vllm" &&
                (health.data?.vllm?.restart_required ||
                  service?.restart_required ||
                  service?.configuration_matches === false) && (
                  <div className="danger">
                    {String(
                      service?.message ??
                        "运行配置与已保存配置不一致，需重启 vLLM。",
                    )}
                    {(health.data?.vllm?.configuration_differences?.length ??
                      service?.configuration_differences?.length) > 0 && (
                      <pre>
                        {JSON.stringify(
                          health.data?.vllm?.configuration_differences?.length
                            ? health.data.vllm.configuration_differences
                            : service.configuration_differences,
                          null,
                          2,
                        )}
                      </pre>
                    )}
                    <details>
                      <summary>查看实际运行参数与启动配置</summary>
                      <pre>
                        {JSON.stringify(
                          service?.running_configuration ??
                            service?.snapshot ??
                            health.data?.vllm?.runtime?.processes,
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                  </div>
                )}
              {serviceError && (
                <details className="service-error">
                  <summary>
                    原始错误：{String(serviceError).split("\n", 1)[0]}
                  </summary>
                  <pre>{String(serviceError)}</pre>
                </details>
              )}
              <div className="toolbar">
                <button
                  disabled={active || startLoading[String(key)]}
                  onClick={() => void startService(key as "vllm" | "chroma")}
                >
                  {startLoading[String(key)] ? "提交中…" : "启动"}
                </button>
                {service?.log_source && (
                  <button
                    className="secondary compact"
                    onClick={() =>
                      navigate(
                        `logs?source=${encodeURIComponent(service.log_source)}`,
                      )
                    }
                  >
                    查看日志
                  </button>
                )}
              </div>
            </div>
          );
        })}
        <button
          className="metric metric-action"
          onClick={() => navigate("tasks")}
        >
          <FlaskConical size={21} />
          <span>实验任务</span>
          <strong>
            {
              items(tasks.data).filter((t) =>
                ["queued", "running"].includes(t.status),
              ).length
            }{" "}
            项运行中
          </strong>
        </button>
      </div>
      <State state={health} />
      <State state={services} />
      {latest && (
        <Card title="最新资源采样">
          <div className="resource-values">
            <div>
              <span>CPU · WSL 系统</span>
              <strong>{format(latest.cpu_percent, " %")}</strong>
            </div>
            <div>
              <span>内存 · WSL 系统</span>
              <strong>
                {format(
                  latest.wsl_memory_swap?.used_bytes === undefined
                    ? null
                    : latest.wsl_memory_swap.used_bytes / 2 ** 30,
                  " GiB",
                )}
              </strong>
            </div>
            <div>
              <span>GPU · 整卡利用率</span>
              <strong>{format(gpu?.[2], " %")}</strong>
            </div>
            <div>
              <span>显存 · 整卡已用 / 总量</span>
              <strong>
                {format(gpu?.[1], " MiB")} / {format(gpu?.[0], " MiB")}
              </strong>
            </div>
          </div>
          <p className="muted">
            采样时间 {new Date(latest.timestamp * 1000).toLocaleString()}
          </p>
          {Object.entries(latest)
            .filter(
              ([key]) => key.endsWith("_error") || key.endsWith("_status"),
            )
            .map(([key, value]) => (
              <p className="warning" key={key}>
                采样状态 {key}：
                {typeof value === "string" ? value : JSON.stringify(value)}
              </p>
            ))}
        </Card>
      )}
      <Card
        title="主机资源观测"
        actions={
          <select
            aria-label="资源时间范围"
            value={range}
            onChange={(e) => setRange(Number(e.target.value))}
          >
            <option value={15}>最近 15 分钟</option>
            <option value={60}>最近 1 小时</option>
            <option value={360}>最近 6 小时</option>
          </select>
        }
      >
        <State state={resources} />
        {points.length ? (
          <>
            <div className="chart">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={points}>
                  <defs>
                    <linearGradient id="cpuFill" x1="0" y1="0" x2="0" y2="1">
                      <stop stopColor="#459C83" stopOpacity={0.3} />
                      <stop offset="1" stopColor="#459C83" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="time" minTickGap={60} />
                  <YAxis yAxisId="percent" unit="%" />
                  <YAxis yAxisId="memory" orientation="right" unit="GiB" />
                  <Tooltip />
                  <Legend />
                  <Area
                    yAxisId="percent"
                    name="CPU %"
                    dataKey="cpu"
                    stroke="#459C83"
                    fill="url(#cpuFill)"
                    isAnimationActive={false}
                  />
                  <Area
                    yAxisId="percent"
                    name="GPU %"
                    dataKey="gpu"
                    stroke="#6597C5"
                    fill="transparent"
                    isAnimationActive={false}
                  />
                  <Area
                    yAxisId="memory"
                    name="WSL 内存 GiB"
                    dataKey="memory"
                    stroke="#DFB35D"
                    fill="transparent"
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <Json value={resources.data} label="CPU / 内存 / GPU 采样" />
          </>
        ) : (
          !resources.loading && <Empty>此时间范围暂无采样。</Empty>
        )}
      </Card>
      <Card title="近期实验" actions={<Refresh onClick={tasks.refresh} />}>
        <State state={tasks} />
        <TaskTable
          rows={items(tasks.data).slice(0, 5)}
          select={(id) => {
            sessionStorage.setItem("memmgr-task", id);
            navigate("tasks");
          }}
        />
        <State state={runs} />
        {items(runs.data)
          .slice(0, 3)
          .map((run) => (
            <RecentRunMetrics key={run.id} id={run.id} name={run.name} />
          ))}
        {!runs.loading && !items(runs.data).length && (
          <p className="muted">暂无运行指标。</p>
        )}
      </Card>
    </>
  );
}
/** 最近运行指标；id 为真实运行 ID，name 为目录名称，由后端复用原统计口径。 */
function RecentRunMetrics({ id, name }: { id: string; name: string }) {
  const state = useApi(`/runs/${encodeURIComponent(id)}/metrics`, 10000);
  const metrics = state.data?.metrics;
  return (
    <section className="recent-run">
      <h3>
        <a href={`#reports?run=${id}`}>{name} · 结果 →</a>
      </h3>
      <State state={state} />
      {metrics ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>指标</th>
                <th>有效 n</th>
                <th>均值</th>
                <th>单位</th>
              </tr>
            </thead>
            <tbody>
              {[
                "e2e_total_seconds",
                "full_retrieval_seconds",
                "recall_at_k",
                "ndcg_at_k",
              ].map((key) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>{format(metrics[key]?.n)}</td>
                  <td>{format(metrics[key]?.mean)}</td>
                  <td>{metrics[key]?.unit ?? "null"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !state.loading && <p className="muted">该运行暂无可读取指标。</p>
      )}
      <p className="muted">formal · ok</p>
    </section>
  );
}
/** 工作区选择；value 为隔离名称，change 更新选择，workspaces 为真实发现结果。 */
export function Workspace({
  value,
  change,
  workspaces,
}: {
  value: string;
  change: (name: string) => void;
  workspaces: RecordData[];
}) {
  return (
    <label>
      工作区
      <select
        aria-label="工作区"
        value={value}
        onChange={(e) => change(e.target.value)}
      >
        <option value="formal">formal · 正式数据（只读）</option>
        {workspaces
          .filter((w) => w.name !== "formal")
          .map((w) => (
            <option key={w.name} value={w.name}>
              {w.name}
            </option>
          ))}
      </select>
    </label>
  );
}
/** 任务表格；rows 为真实任务，select 打开详情。 */
function TaskTable({
  rows,
  select,
}: {
  rows: RecordData[];
  select: (id: string) => void;
}) {
  return rows.length ? (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>任务 / ID</th>
            <th>工作区</th>
            <th>状态</th>
            <th>创建时间</th>
            <th>详情</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <tr key={t.id}>
              <td>
                {t.kind}
                <small>{t.id}</small>
              </td>
              <td>
                {t.workspace} / {t.tier}
              </td>
              <td>
                <span className={`badge ${t.status}`}>
                  {statusLabel(t.status)}
                </span>
              </td>
              <td>{new Date(t.created_at * 1000).toLocaleString()}</td>
              <td>
                <button className="link" onClick={() => select(t.id)}>
                  查看任务 →
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <Empty>暂无实验任务。</Empty>
  );
}
/** 任务管理；dataMode 限定数据准备、索引及重建。 */
export function Tasks({ dataMode = false }: { dataMode?: boolean }) {
  const tasks = useApi("/tasks", 2500);
  const spaces = useApi("/workspaces", 5000);
  const config = useApi("/config");
  const [workspace, setWorkspace] = useState("formal");
  const [name, setName] = useState("");
  const [kind, setKind] = useState(dataMode ? "prepare" : "retrieval");
  const [tier, setTier] = useState("small");
  const [concurrency, setConcurrency] = useState(1);
  const [selected, setSelected] = useState(
    sessionStorage.getItem("memmgr-task") ?? "",
  );
  const [error, setError] = useState("");
  const [offset, setOffset] = useState(0);
  const detail = useApi(
    selected ? `/tasks/${encodeURIComponent(selected)}` : "/tasks",
    2000,
  );
  const rows = items(tasks.data).filter(
    (t) => !dataMode || ["prepare", "index", "rebuild"].includes(t.kind),
  );
  const space = items(spaces.data).find((s) => s.name === workspace);
  const runs = useApi("/runs", 5000);
  const elapsed = detail.data?.started_at
    ? Math.max(
        0,
        (detail.data.finished_at ?? Date.now() / 1000) - detail.data.started_at,
      )
    : null;
  /** 保存所选任务，以便刷新页面后恢复同一任务详情；id 为真实任务标识。 */
  function selectTask(id: string) {
    setSelected(id);
    sessionStorage.setItem("memmgr-task", id);
  }
  /** 创建真实任务；重建前确认隔离索引的具体影响范围。 */
  async function create() {
    if (workspace === "formal")
      throw new Error("请选择隔离工作区，正式数据只读");
    const destructive =
      kind === "rebuild" || (kind === "prepare" && space?.prepared);
    if (
      destructive &&
      !window.confirm(
        kind === "rebuild"
          ? `重建将删除并重新生成 ${workspace} / ${tier} 的索引，仅影响该工作区。继续？`
          : `重新准备 ${workspace} 会覆盖全部 small / medium / full 语料及查询数据，并使旧 index manifest 失效，需要重新索引。继续？`,
      )
    )
      return;
    const task = await api("/tasks", {
      kind,
      workspace,
      tier,
      concurrency,
      confirm: destructive,
    });
    selectTask(task.id);
    await tasks.refresh();
  }
  return (
    <>
      <Card title="工作区与运行配置">
        <div className="toolbar">
          <Workspace
            value={workspace}
            change={setWorkspace}
            workspaces={items(spaces.data)}
          />
          <label>
            新隔离工作区
            <input
              aria-label="新工作区名称"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="例如 browser-check"
              pattern="[A-Za-z0-9_-]+"
            />
          </label>
          <button
            disabled={!name}
            onClick={() =>
              perform(async () => {
                await api("/workspaces", { name });
                setWorkspace(name);
                setName("");
                await spaces.refresh();
              }, setError)
            }
          >
            创建工作区
          </button>
        </div>
        <State state={spaces} />
        <State state={config} />
        <Json value={config.data?.baseline} label="已保存 baseline" />
        {dataMode && space && (
          <>
            <p>
              数据准备：{space.prepared ? "已完成" : "尚未准备"} ·{" "}
              {space.readonly ? "只读正式数据" : "隔离可写工作区"}
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>层级</th>
                    <th>语料</th>
                    <th>索引状态</th>
                    <th>集合</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(space.tiers ?? {}).map(([key, value]) => {
                    const info = value as RecordData;
                    return (
                      <tr key={key}>
                        <td>{key}</td>
                        <td>{info.corpus_exists ? "已准备" : "未准备"}</td>
                        <td>{info.indexed ? "已建立" : "尚未建立"}</td>
                        <td>{info.collection ?? "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <Json value={space} label="数据来源与索引状态" />
          </>
        )}
        <ErrorBox error={error} />
      </Card>
      <Card title={dataMode ? "数据准备与索引操作" : "启动实验任务"}>
        <div className="toolbar">
          <label>
            任务类型
            <select
              aria-label="任务类型"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
            >
              {(dataMode
                ? ["prepare", "index", "rebuild"]
                : ["retrieval", "rag", "baseline"]
              ).map((k) => (
                <option key={k} value={k}>
                  {
                    (
                      {
                        prepare: "准备数据",
                        index: "建立索引",
                        rebuild: "重建索引",
                        retrieval: "检索实验",
                        rag: "问答实验",
                        baseline: "完整基线",
                      } as Record<string, string>
                    )[k]
                  }
                  （{k}）
                </option>
              ))}
            </select>
          </label>
          <label>
            语料层级
            <select
              aria-label="语料层级"
              value={tier}
              onChange={(e) => setTier(e.target.value)}
            >
              {["small", "medium", "full"].map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          {!dataMode && (
            <label>
              客户端并发
              <input
                aria-label="客户端并发"
                type="number"
                min={1}
                max={4}
                value={concurrency}
                onChange={(e) => setConcurrency(Number(e.target.value))}
              />
            </label>
          )}
          <button
            disabled={
              workspace === "formal" || concurrency < 1 || concurrency > 4
            }
            onClick={() => perform(create, setError)}
          >
            提交{dataMode ? "数据" : "实验"}任务
          </button>
        </div>
      </Card>
      <Card title="任务历史" actions={<Refresh onClick={tasks.refresh} />}>
        <State state={tasks} />
        <TaskTable rows={rows.slice(offset, offset + 10)} select={selectTask} />
        <Pager
          offset={offset}
          size={10}
          count={rows.slice(offset, offset + 10).length}
          change={setOffset}
        />
      </Card>
      {selected && (
        <Card
          title={`任务详情 · ${selected}`}
          actions={<Refresh onClick={detail.refresh} />}
        >
          <State state={detail} />
          {detail.data && (
            <>
              <div className="toolbar">
                <span className={`badge ${detail.data.status}`}>
                  {statusLabel(detail.data.status)}
                </span>
                <span>耗时：{format(elapsed, " 秒")}</span>
                {["running", "queued"].includes(detail.data.status) && (
                  <button
                    className="danger-button"
                    onClick={() =>
                      perform(async () => {
                        if (
                          window.confirm(
                            "确认取消此任务？已写入的隔离数据将保留，详情记录取消状态。",
                          )
                        ) {
                          await api(`/tasks/${selected}/cancel`, {});
                          await detail.refresh();
                          await tasks.refresh();
                        }
                      }, setError)
                    }
                  >
                    取消任务
                  </button>
                )}
                <a href={`#logs?source=task-${selected}`}>查看关联日志 →</a>
              </div>
              <p className="muted">
                当前阶段：{detail.data.progress?.stage ?? detail.data.status} ·
                子进程 PID {detail.data.pid ?? "尚未创建"} · 正式{" "}
                {detail.data.progress?.request_counts?.formal ?? 0} / 预热{" "}
                {detail.data.progress?.request_counts?.warmup ?? 0} / 成功{" "}
                {detail.data.progress?.request_counts?.ok ?? 0} / 失败{" "}
                {detail.data.progress?.request_counts?.failed ?? 0}
              </p>
              {(
                detail.data.run_ids ??
                items(runs.data)
                  .filter((r) => r.path?.startsWith(detail.data?.output))
                  .map((r) => r.id)
              ).map((id: string) => (
                <p key={id}>
                  <a href={`#reports?run=${id}`}>运行结果 {id} →</a>
                </p>
              ))}
              <ErrorBox error={detail.data.error ?? ""} />
              <Json value={detail.data.snapshot} label="运行配置快照" />
              <Json value={detail.data.output} label="结果与产物" />
              <Json value={detail.data} label="完整任务记录" />
            </>
          )}
        </Card>
      )}
      {!dataMode && (
        <Card title="全部运行（含正式历史）">
          <State state={runs} />
          {items(runs.data).length ? (
            items(runs.data).map((run) => (
              <div className="history-row" key={run.id}>
                <a href={`#reports?run=${run.id}`}>{run.name} →</a>
                <Json value={run.summary} label="运行摘要" />
              </div>
            ))
          ) : (
            <Empty>尚无可读取的原始实验运行目录。</Empty>
          )}
        </Card>
      )}
    </>
  );
}
