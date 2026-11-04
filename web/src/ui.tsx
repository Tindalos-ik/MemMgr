/** MemMgr 通用卡片、数据展开、分页和真实请求状态组件。 */
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { RefreshCw, AlertCircle } from "lucide-react";
import { api, type RecordData } from "./api";
/** 订阅 API；path 为请求路径，interval 为可选轮询毫秒，错误保留至成功刷新。 */
export function useApi(path: string, interval = 0) {
  const [data, setData] = useState<RecordData>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const currentPath = useRef(path);
  currentPath.current = path;
  const [loadedPath, setLoadedPath] = useState("");
  const inFlight = useRef(new Map<string, Promise<void>>());
  const refresh = useCallback(() => {
    const pending = inFlight.current.get(path);
    if (pending) return pending;
    let request: Promise<void>;
    request = (async () => {
      try {
        const value = await api(path);
        if (currentPath.current !== path) return;
        setLoadedPath(path);
        setData(value);
        setError("");
      } catch (e) {
        if (currentPath.current !== path) return;
        setError(String(e));
      } finally {
        inFlight.current.delete(path);
        if (currentPath.current === path) setLoading(false);
      }
    })();
    inFlight.current.set(path, request);
    return request;
  }, [path]);
  useEffect(() => {
    setData(undefined);
    setLoading(true);
    void refresh();
    if (interval) {
      const timer = setInterval(refresh, interval);
      return () => clearInterval(timer);
    }
  }, [refresh, interval]);
  return {
    data: loadedPath === path ? data : undefined,
    error,
    loading: loading || (loadedPath !== path && !error),
    refresh,
  };
}
/** 显示卡片；title 为标题，actions 为操作区，children 为正文。 */
export function Card({
  title,
  actions,
  children,
  className = "",
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      <div className="card-head">
        <h2>{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}
/** 显示持久错误；error 为异常文本。 */
export function ErrorBox({ error }: { error: string }) {
  return error ? (
    <div className="error" role="alert">
      <AlertCircle size={18} />
      <span>{error}</span>
    </div>
  ) : null;
}
/** 显示异步状态；state 为 useApi 返回值。 */
export function State({
  state,
}: {
  state: { loading: boolean; error: string; data?: RecordData };
}) {
  return (
    <>
      <ErrorBox error={state.error} />
      {state.loading && (
        <p className="muted" role="status">
          加载中…
        </p>
      )}
    </>
  );
}
/** 展开原始记录；value 为服务端数据，label 为摘要。 */
export function Json({
  value,
  label = "查看完整记录",
}: {
  value: unknown;
  label?: string;
}) {
  return (
    <details>
      <summary>{label}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
/** 显示无数据状态；children 为操作指导。 */
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
/** 显示刷新按钮；onClick 为实际刷新函数。 */
export function Refresh({ onClick }: { onClick: () => void }) {
  return (
    <button className="secondary compact" onClick={onClick}>
      <RefreshCw size={15} />
      刷新
    </button>
  );
}
/** 格式化值；value 为原始值，unit 为单位，缺失保留 null。 */
export function format(value: unknown, unit = ""): string {
  if (value === undefined || value === null) return "null · 未采集";
  if (typeof value === "number")
    return `${Number.isInteger(value) ? value : value.toFixed(3)}${unit}`;
  return String(value);
}
/** 显示分页；offset 为起始项，size 为每页数量，count 为当前条目数，change 修改偏移。 */
export function Pager({
  offset,
  size,
  count,
  change,
}: {
  offset: number;
  size: number;
  count: number;
  change: (value: number) => void;
}) {
  return (
    <div className="pager">
      <button
        className="secondary"
        disabled={!offset}
        onClick={() => change(Math.max(0, offset - size))}
      >
        上一页
      </button>
      <span>
        第 {offset / size + 1} 页 · 当前 {count} 条
      </span>
      <button
        className="secondary"
        disabled={count < size}
        onClick={() => change(offset + size)}
      >
        下一页
      </button>
    </div>
  );
}
/** 统一异步操作；action 为操作函数，setError 接收持久错误。 */
export async function perform(
  action: () => Promise<unknown>,
  setError: (message: string) => void,
) {
  try {
    await action();
    setError("");
  } catch (e) {
    setError(String(e));
  }
}
/** 显示中文状态及后端原始状态码；value 为服务或任务状态。 */
export function statusLabel(value: string | undefined) {
  const labels: Record<string, string> = {
    ok: "正常",
    queued: "排队中",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
    interrupted: "已中断",
    usage_missing: "用量缺失",
    unavailable: "不可用",
    error: "错误",
  };
  return value ? `${labels[value] ?? value}（${value}）` : "尚未读取";
}
