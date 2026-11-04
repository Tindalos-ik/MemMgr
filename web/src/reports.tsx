/** 真实 Markdown、Mermaid、GFM 数值图表与独立运行比较，保持单位、n 和 null。 */
import { useEffect, useId, useRef, useState, type ComponentProps } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
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
} from "./ui";
const colors = [
  "#459C83",
  "#6597C5",
  "#DFB35D",
  "#A58CC8",
  "#DA8D80",
  "#75B7BE",
  "#8AA95B",
];
mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  theme: "neutral",
  fontFamily: "Microsoft YaHei, sans-serif",
});
/** 渲染真实 Mermaid 围栏；source 为报告原文图定义，渲染错误持久显示。 */
function Mermaid({ source }: { source: string }) {
  const id = useId().replace(/:/g, "");
  const box = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    mermaid
      .render(`diagram${id}`, source)
      .then(({ svg }) => {
        if (live && box.current) {
          box.current.innerHTML = svg;
          setError("");
        }
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    return () => {
      live = false;
    };
  }, [source, id]);
  return (
    <>
      <ErrorBox error={error} />
      <div className="mermaid" ref={box} />
    </>
  );
}
/** 以稳定组件渲染 Markdown code；className 标识 Mermaid，children 为原始围栏内容。 */
function ReportCode({ className, children, ...props }: ComponentProps<"code">) {
  return className === "language-mermaid" ? (
    <Mermaid source={String(children).trimEnd()} />
  ) : (
    <code className={className} {...props}>
      {children}
    </code>
  );
}
/** 提取 GFM 表格；markdown 为后端正文，只采纳独立数值且保留列名单位。 */
function parseTables(markdown: string) {
  const lines = markdown.split("\n");
  const tables: { title: string; headers: string[]; rows: RecordData[] }[] = [];
  let title = "报告数据";
  for (let i = 0; i < lines.length; i++) {
    if (/^#{1,6} /.test(lines[i])) title = lines[i].replace(/^#+ /, "");
    if (
      lines[i].trim().startsWith("|") &&
      /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] ?? "")
    ) {
      const headers = lines[i]
        .trim()
        .slice(1, -1)
        .split("|")
        .map((v) => v.trim());
      const rows: RecordData[] = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const values = lines[i]
          .trim()
          .slice(1, -1)
          .split("|")
          .map((v) => v.trim());
        const row: RecordData = { name: values[0] };
        headers.slice(1).forEach((h, j) => {
          const value = values[j + 1];
          row[h] =
            value && /^-?\d+(\.\d+)?%?$/.test(value)
              ? Number(value.replace("%", ""))
              : null;
        });
        rows.push(row);
        i++;
      }
      if (
        rows.length &&
        headers.slice(1).some((h) => rows.some((r) => r[h] !== null))
      )
        tables.push({ title, headers, rows });
    }
  }
  return tables;
}
/** 单指标多组图表；table 为真实表格解析结果，分组颜色按条目区分。 */
function MetricChart({
  table,
}: {
  table: { headers: string[]; rows: RecordData[] };
}) {
  const metrics = table.headers
    .slice(1)
    .filter((h) => table.rows.some((r) => r[h] !== null));
  const [metric, setMetric] = useState(metrics[0]);
  return (
    <>
      <label>
        图表指标
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {metrics.map((h) => (
            <option key={h}>{h}</option>
          ))}
        </select>
      </label>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={table.rows}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" />
            <YAxis
              label={{
                value: metric,
                angle: -90,
                position: "insideLeft",
                style: { fontSize: 10 },
              }}
            />
            <Tooltip formatter={(value) => [format(value), metric]} />
            <Legend />
            <Bar dataKey={metric} isAnimationActive={false}>
              {table.rows.map((r, i) => (
                <Cell key={r.name} fill={colors[i % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
/** 展示完整原始运行，包括配置、摘要、请求样例和资源；id 为后端运行 ID。 */
function RunDetail({ id }: { id: string }) {
  const [offset, setOffset] = useState(0);
  const state = useApi(
    `/runs/${encodeURIComponent(id)}?offset=${offset}&limit=10`,
  );
  return (
    <Card title={`运行结果 · ${id}`}>
      <State state={state} />
      {state.data && (
        <>
          <p>{state.data.name}</p>
          <Json value={state.data.summary} label="统计与计时边界" />
          <Json value={state.data.config} label="实际运行配置" />
          <Json value={state.data.requests} label="原始正式 / 预热请求样例" />
          <Pager
            offset={offset}
            size={10}
            count={state.data.requests?.length ?? 0}
            change={setOffset}
          />
          <Json value={state.data.resources} label="资源样本" />
        </>
      )}
    </Card>
  );
}
/** 比较同一字段但不混合统计；runs 保留各自有效样本 n 和单位，details 打开完整 run。 */
function Comparison({
  value,
  details,
}: {
  value: RecordData;
  details: (id: string) => void;
}) {
  const runs: RecordData[] = value.runs ?? [];
  const metrics = Array.from(
    new Set(runs.flatMap((r) => Object.keys(r.metrics ?? {}))),
  );
  const [metric, setMetric] = useState(
    metrics.find((key) => runs.some((run) => run.metrics?.[key]?.n > 0)) ??
      metrics[0] ??
      "",
  );
  const rows = runs.map((run) => ({
    name: run.name ?? run.id,
    value: run.metrics?.[metric]?.mean ?? null,
    id: run.id,
  }));
  const unit = runs[0]?.metrics?.[metric]?.unit ?? "null";
  return (
    <>
      <div className="warning">条件或统计口径不同不可合并。</div>
      <Json value={value.config_differences} label="逐项配置差异" />
      <Json value={value.warnings} label="后端原始统计说明" />
      <label>
        比较指标
        <select
          aria-label="比较指标"
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
        >
          {metrics.map((key) => (
            <option key={key}>{key}</option>
          ))}
        </select>
      </label>
      {metrics.length > 0 && (
        <>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" />
                <YAxis
                  label={{
                    value: unit,
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 10 },
                  }}
                />
                <Tooltip
                  formatter={(value) => [format(value, ` ${unit}`), metric]}
                />
                <Bar
                  dataKey="value"
                  name={`${metric} · 均值（${unit}）`}
                  isAnimationActive={false}
                >
                  {rows.map((row, i) => (
                    <Cell key={row.id} fill={colors[i % colors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>运行</th>
                  <th>正式成功 n</th>
                  <th>指标有效 n</th>
                  <th>单位</th>
                  <th>均值</th>
                  <th>中位数</th>
                  <th>P95</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const m = run.metrics?.[metric];
                  return (
                    <tr key={run.id}>
                      <td>
                        <button
                          className="link"
                          onClick={() => details(run.id)}
                        >
                          {run.name ?? run.id}
                        </button>
                      </td>
                      <td>{format(run.n)}</td>
                      <td>{format(m?.n)}</td>
                      <td>{m?.unit ?? "null"}</td>
                      <td>{format(m?.mean)}</td>
                      <td>{format(m?.median)}</td>
                      <td>{format(m?.p95)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="muted">
            统计总体：
            {runs[0]?.metrics?.[metric]?.population ?? "null"}
          </p>
        </>
      )}
      {runs.map((run) => (
        <section key={run.id}>
          <h3>{run.name ?? run.id}</h3>
          <Json value={run.metrics} label="全部指标" />
          <Json value={run.config} label="实际配置" />
          <Json
            value={run.batch_aggregate}
            label="批次墙钟吞吐 · 请求/秒及 token/秒"
          />
          <Json value={run.batches} label="每次重复统计" />
          <Json value={run.resources} label="资源采样" />
          <Json value={run.examples} label="请求与回答样例" />
          <button className="secondary" onClick={() => details(run.id)}>
            查看原始请求样例
          </button>
        </section>
      ))}
    </>
  );
}
/** 报告与运行浏览；hash run 参数用于任务结果真实链接。 */
export function Reports() {
  const reports = useApi("/reports");
  const runs = useApi("/runs", 5000);
  const [name, setName] = useState("");
  const report = useApi(
    name ? `/reports/${encodeURIComponent(name)}` : "/reports",
  );
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<RecordData>();
  const [error, setError] = useState("");
  const [runId, setRunId] = useState(
    new URLSearchParams(location.hash.split("?")[1]).get("run") ?? "",
  );
  useEffect(() => {
    /** 更新同页结果链接；读取 hash 中的真实 run ID。 */ function changed() {
      setRunId(
        new URLSearchParams(location.hash.split("?")[1]).get("run") ?? "",
      );
    }
    window.addEventListener("hashchange", changed);
    return () => window.removeEventListener("hashchange", changed);
  }, []);
  return (
    <>
      <Card title="实验报告" actions={<Refresh onClick={reports.refresh} />}>
        <State state={reports} />
        <div className="toolbar">
          {items(reports.data).map((r) => (
            <button
              key={r.name}
              className={name === r.name ? "" : "secondary"}
              onClick={() => {
                if (name === r.name) void report.refresh();
                else setName(r.name);
              }}
            >
              {r.name}
            </button>
          ))}
        </div>
        {!items(reports.data).length && !reports.loading && (
          <Empty>服务端尚无报告文档。</Empty>
        )}
      </Card>
      {name && (
        <>
          <State state={report} />
          {report.data?.markdown && (
            <>
              <Card title={name} actions={<Refresh onClick={report.refresh} />}>
                <article className="markdown">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{ code: ReportCode }}
                  >
                    {report.data.markdown}
                  </ReactMarkdown>
                </article>
              </Card>
              {parseTables(report.data.markdown).map((table, index) => (
                <Card key={index} title={`${table.title} · 图表`}>
                  <MetricChart table={table} />
                </Card>
              ))}
            </>
          )}
        </>
      )}
      <Card title="运行比较" actions={<Refresh onClick={runs.refresh} />}>
        <State state={runs} />
        {items(runs.data).map((run) => (
          <div className="run-choice" key={run.id}>
            <input
              aria-label={`选择运行 ${run.name ?? run.id}`}
              type="checkbox"
              checked={selected.includes(run.id)}
              onChange={(e) =>
                setSelected(
                  e.target.checked
                    ? [...selected, run.id]
                    : selected.filter((id) => id !== run.id),
                )
              }
            />
            <button className="link" onClick={() => setRunId(run.id)}>
              {run.name ?? run.id}
            </button>
            <small>{run.id}</small>
            <Json value={run.summary} label="运行摘要" />
          </div>
        ))}
        {!items(runs.data).length && !runs.loading && (
          <Empty>暂无运行产物。</Empty>
        )}
        <button
          disabled={selected.length < 2 || selected.length > 8}
          onClick={() =>
            perform(
              async () =>
                setComparison(await api("/compare", { run_ids: selected })),
              setError,
            )
          }
        >
          比较所选 {selected.length} 个运行（2–8 个）
        </button>
        <ErrorBox error={error} />
        {comparison && <Comparison value={comparison} details={setRunId} />}
      </Card>
      {runId && <RunDetail key={runId} id={runId} />}
    </>
  );
}
