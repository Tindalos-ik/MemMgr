/** MemMgr HTTP 与 POST SSE 调用；错误保留服务端 detail，全部请求连接真实 API。 */
export type RecordData = Record<string, any>;
/** 请求 JSON；path 为 API 相对路径，body 为可选 JSON 对象，method 为 HTTP 方法。 */
export async function api<T = RecordData>(
  path: string,
  body?: unknown,
  method = body === undefined ? "GET" : "POST",
): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method,
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail ?? data),
    );
  return data;
}
/** 提取列表；data 为后端统一的 items 响应。 */
export function items(data: RecordData | undefined): RecordData[] {
  return data?.items ?? [];
}
/** 流式问答；body 为问答参数，onEvent 接收事件名和真实内容，signal 允许取消网络读取。 */
export async function ask(
  body: unknown,
  onEvent: (event: string, data: RecordData) => void,
  signal: AbortSignal,
) {
  const response = await fetch("/api/qa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(JSON.stringify(data.detail ?? data));
  }
  if (!response.body) throw new Error("浏览器未提供 SSE 响应流");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { value, done } = await reader.read();
    pending += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = pending.indexOf("\n\n")) >= 0) {
      const block = pending.slice(0, boundary);
      pending = pending.slice(boundary + 2);
      const event =
        block
          .split("\n")
          .find((line) => line.startsWith("event:"))
          ?.slice(6)
          .trim() ?? "message";
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(event, JSON.parse(data));
    }
    if (done) break;
  }
}
