/** 真实 SSE 问答和字节游标日志；手动上滚暂停自动跟随，缓冲区有界。 */
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, ask, items, type RecordData } from "./api";
import {
  Card,
  Empty,
  ErrorBox,
  Json,
  Refresh,
  State,
  perform,
  useApi,
  format,
  statusLabel,
} from "./ui";
import { Workspace } from "./operations";
/** 问答界面，检索证据、prompt、阶段耗时均使用服务端事件记录。 */
export function QuestionAnswer() {
  const spaces = useApi("/workspaces");
  const history = useApi("/requests", 5000);
  const [workspace, setWorkspace] = useState("formal");
  const [question, setQuestion] = useState("");
  const [tier, setTier] = useState("small");
  const [topK, setTopK] = useState(3);
  const [budget, setBudget] = useState(1024);
  const [answer, setAnswer] = useState("");
  const [events, setEvents] = useState<RecordData>({});
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [follow, setFollow] = useState(true);
  const box = useRef<HTMLDivElement>(null);
  const control = useRef<AbortController | null>(null);
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [answer, follow]);
  /** 提交真实问答，并逐个保留 SSE 阶段事件。 */
  async function submit() {
    setError("");
    setAnswer("");
    setEvents({});
    setRunning(true);
    setFollow(true);
    control.current = new AbortController();
    try {
      await ask(
        { question, workspace, tier, top_k: topK, context_tokens: budget },
        (event, data) => {
          if (event === "delta") setAnswer((old) => old + data.text);
          else {
            setEvents((old) => ({ ...old, [event]: data }));
            if (event === "error") setError(data.detail);
          }
        },
        control.current.signal,
      );
      await history.refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }
  /** 读取历史请求；id 为后端持久请求标识。 */
  async function load(id: string) {
    const record = await api(`/requests/${id}`);
    setEvents({
      retrieval: record.retrieval,
      prompt: record.prompt,
      done: record,
    });
    setAnswer(record.answer ?? record.response ?? "");
    setQuestion(record.input?.question ?? record.input?.query ?? "");
    setWorkspace(record.input?.workspace ?? "formal");
    setTier(record.input?.tier ?? "small");
    setTopK(record.input?.top_k ?? 3);
    setBudget(record.input?.context_tokens ?? 1024);
  }
  return (
    <>
      <Card title="交互式问答">
        <div className="toolbar">
          <Workspace
            value={workspace}
            change={setWorkspace}
            workspaces={items(spaces.data)}
          />
          <label>
            层级
            <select value={tier} onChange={(e) => setTier(e.target.value)}>
              {["small", "medium", "full"].map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          <label>
            候选块 Top K
            <input
              aria-label="候选块 Top K"
              type="number"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
          <label>
            完整输入预算 / token
            <input
              aria-label="完整输入预算"
              type="number"
              min={128}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
            />
          </label>
        </div>
        <label>
          科学论断或问题
          <textarea
            aria-label="问题"
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="输入问题"
          />
        </label>
        <div className="toolbar">
          <button disabled={running || !question.trim()} onClick={submit}>
            {running ? "正在接收响应…" : "发送问题"}
          </button>
          {running && (
            <button
              className="secondary"
              onClick={() => control.current?.abort()}
            >
              停止接收
            </button>
          )}
        </div>
        <State state={spaces} />
        <ErrorBox error={error} />
      </Card>
      <Card
        title="模型回答"
        actions={
          <button
            className="secondary compact"
            onClick={() => setFollow(!follow)}
          >
            {follow ? "暂停跟随" : "恢复跟随"}
          </button>
        }
      >
        <div
          className="stream answer"
          ref={box}
          onScroll={(e) => {
            const el = e.currentTarget;
            if (el.scrollHeight - el.scrollTop - el.clientHeight > 45)
              setFollow(false);
          }}
        >
          {answer ? (
            <ReactMarkdown>{answer}</ReactMarkdown>
          ) : (
            <Empty>{running ? "等待回答…" : "暂无回答。"}</Empty>
          )}
        </div>
        <p className="muted" role="status">
          {running ? "生成中" : events.done ? "请求完成" : "等待请求"} ·{" "}
          {follow ? "自动跟随" : "跟随已暂停"}
        </p>
      </Card>
      {Object.entries(events)
        .filter(([, data]) => data)
        .map(([event, data]) => (
          <Card
            key={event}
            title={
              (
                {
                  retrieval: "检索证据 · cosine distance 越小越相关",
                  prompt: "Prompt 与输入预算",
                  done: "用量、引用与阶段时长",
                  request: "请求标识",
                  error: "服务端错误",
                } as Record<string, string>
              )[event] ?? event
            }
          >
            {event === "prompt" && (
              <>
                <div className="resource-values">
                  <div>
                    <span>完整输入预算</span>
                    <strong>{data.budget} token</strong>
                  </div>
                  <div>
                    <span>实际本地输入</span>
                    <strong>{data.input_tokens_local} token</strong>
                  </div>
                  <div>
                    <span>证据 token</span>
                    <strong>{data.evidence_tokens} token</strong>
                  </div>
                  <div>
                    <span>预算裁切</span>
                    <strong>{data.truncated ? "已裁切" : "未裁切"}</strong>
                  </div>
                </div>
                <p>
                  实际纳入文献：
                  {(data.included_doc_ids ?? []).join("、") || "无"}
                </p>
                <details>
                  <summary>展开实际 rendered prompt</summary>
                  <pre>{data.rendered_prompt}</pre>
                </details>
              </>
            )}
            {event === "done" && (
              <>
                <p>
                  {statusLabel(data.status)} · 结束原因{" "}
                  {data.finish_reason ?? "null"} · 内容块{" "}
                  {format(data.stream_chunks)}
                </p>
                <p>
                  服务端输入 {format(data.usage?.prompt_tokens, " token")} ·
                  输出 {format(data.usage?.completion_tokens, " token")}
                </p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>阶段</th>
                        <th>耗时（秒）</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ["查询编码", data.retrieval?.query_embedding_seconds],
                        ["数据库检索", data.retrieval?.retrieval_seconds],
                        ["上下文组织", data.prompt?.context_assembly_seconds],
                        ["端到端首次内容", data.e2e_ttft_seconds],
                        ["模型首次内容", data.model_ttft_seconds],
                        ["生成阶段", data.model_total_seconds],
                        ["端到端总耗时", data.e2e_total_seconds],
                      ].map(([label, value]) => (
                        <tr key={String(label)}>
                          <td>{label}</td>
                          <td>{format(value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="muted">
                  回答引用：
                  {(data.answer?.match(/\\[DOC\\s+[^\\]]+\\]/g) ?? []).join(
                    "、",
                  ) || "正文未出现 [DOC id] 引用"}
                </p>
              </>
            )}
            {event === "retrieval" &&
              (data.hits ?? []).map((hit: RecordData, i: number) => (
                <details key={hit.id ?? i} open={i === 0}>
                  <summary>
                    #{i + 1} · {hit.id ?? hit.doc_id} · distance{" "}
                    {String(hit.distance ?? "null")}
                  </summary>
                  <p className="long-text">{hit.text ?? hit.document}</p>
                  <Json value={hit.metadata} label="metadata" />
                </details>
              ))}
            <Json value={data} label="完整记录" />
          </Card>
        ))}
      <Card title="请求历史" actions={<Refresh onClick={history.refresh} />}>
        <State state={history} />
        {items(history.data).length ? (
          items(history.data).map((r) => (
            <button
              className="history-row"
              key={r.id}
              onClick={() => perform(() => load(r.id), setError)}
            >
              <span>
                {r.input?.question ?? r.input?.query ?? r.question ?? r.id}
              </span>
              <small>
                {statusLabel(r.status)} ·{" "}
                {new Date(r.created_at * 1000).toLocaleString()}
              </small>
            </button>
          ))
        ) : (
          <Empty>暂无请求记录。</Empty>
        )}
      </Card>
    </>
  );
}
/** 日志查看器；source 为后端日志源，cursor 为字节偏移，缓冲保留最后 128 KiB。 */
export function Logs() {
  const tasks = useApi("/tasks");
  const [source, setSource] = useState(
    new URLSearchParams(location.hash.split("?")[1]).get("source") ?? "backend",
  );
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [poll, setPoll] = useState(true);
  const [follow, setFollow] = useState(true);
  const [notice, setNotice] = useState("");
  const cursor = useRef(0);
  const identity = useRef<string | null>(null);
  const box = useRef<HTMLPreElement>(null);
  useEffect(() => {
    cursor.current = 0;
    identity.current = null;
    setText("");
    setError("");
    setNotice("");
  }, [source]);
  useEffect(() => {
    let live = true;
    let busy = false;
    /** 按游标读取真实日志，源不可用时显示明确错误。 */ async function readLog() {
      if (busy) return;
      busy = true;
      try {
        const value = await api(
          `/logs/${encodeURIComponent(source)}?cursor=${cursor.current}&limit=32768${identity.current ? `&identity=${encodeURIComponent(identity.current)}` : ""}`,
        );
        if (!live) return;
        if (!value.available) throw new Error(value.reason ?? "日志源不可用");
        cursor.current = value.cursor;
        identity.current = value.identity ?? null;
        if (value.reset)
          setNotice("日志文件已轮换或游标重置，当前缓冲从新来源重新读取。");
        setText((old) => {
          const combined = (value.reset ? "" : old) + value.text;
          if (combined.length > 131072)
            setNotice("缓冲区已裁去较早文本；最多保留 128 KiB 字符。");
          return combined.slice(-131072);
        });
        setError("");
      } catch (e) {
        if (live) setError(String(e));
      } finally {
        busy = false;
      }
    }
    void readLog();
    if (poll) {
      const timer = setInterval(readLog, 1500);
      return () => {
        live = false;
        clearInterval(timer);
      };
    }
    return () => {
      live = false;
    };
  }, [source, poll]);
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [text, follow]);
  return (
    <Card title="服务与任务日志">
      <div className="toolbar">
        <label>
          日志来源
          <select
            aria-label="日志来源"
            value={source}
            onChange={(e) => setSource(e.target.value)}
          >
            {[
              "backend",
              "chroma",
              "vllm",
              "service-vllm",
              "service-chroma",
              ...items(tasks.data).map((t) => `task-${t.id}`),
            ].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <button className="secondary" onClick={() => setPoll(!poll)}>
          {poll ? "暂停轮询" : "恢复轮询"}
        </button>
        <button className="secondary" onClick={() => setFollow(!follow)}>
          {follow ? "暂停跟随" : "恢复跟随"}
        </button>
        <span className="muted">
          游标 {cursor.current} bytes · {poll ? "轮询中" : "轮询暂停"}
        </span>
      </div>
      <ErrorBox error={error} />
      <pre
        className="stream log"
        ref={box}
        onScroll={(e) => {
          const el = e.currentTarget;
          if (el.scrollHeight - el.scrollTop - el.clientHeight > 45)
            setFollow(false);
        }}
      >
        {text || "当前没有日志文本。"}
      </pre>
      <p className="muted">
        {notice} {follow ? "自动跟随" : "跟随已暂停"}
      </p>
    </Card>
  );
}
