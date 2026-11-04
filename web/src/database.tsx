/** Chroma 集合分页、语义检索及独立 namespace 长期记忆操作。 */
import { useState } from "react";
import { api, items, type RecordData } from "./api";
import {
  Card,
  Empty,
  ErrorBox,
  Json,
  Pager,
  Refresh,
  State,
  useApi,
} from "./ui";
/** 以原文、ID、metadata 展示记录；rows 为服务端条目。 */
function Records({ rows }: { rows: RecordData[] }) {
  return rows.length ? (
    <div className="records">
      {rows.some((row) => row.distance !== undefined) && (
        <p className="muted">cosine distance 越小越相关</p>
      )}
      {rows.map((row, index) => (
        <details key={row.id ?? index}>
          <summary>
            {row.id ?? `记录 ${index + 1}`}{" "}
            {row.distance !== undefined && `· distance ${row.distance}`}
          </summary>
          <p className="long-text">
            {row.text ?? row.document ?? row.documents}
          </p>
          <Json value={row.metadata} label="metadata" />
          <Json value={row} label="完整条目" />
        </details>
      ))}
    </div>
  ) : (
    <Empty>当前查询没有记录。</Empty>
  );
}
/** 数据库页面，索引集合只读浏览，长期记忆使用独立命名空间。 */
export function DatabasePage() {
  const [collectionOffset, setCollectionOffset] = useState(0);
  const collections = useApi(
    `/collections?limit=10&offset=${collectionOffset}`,
  );
  const [collection, setCollection] = useState("");
  const [offset, setOffset] = useState(0);
  const [idFilter, setIdFilter] = useState("");
  const [textFilter, setTextFilter] = useState("");
  const [search, setSearch] = useState({ id: "", text: "" });
  const [semantic, setSemantic] = useState("");
  const [topK, setTopK] = useState(3);
  const [workspace, setWorkspace] = useState("formal");
  const [tier, setTier] = useState("small");
  const [queryResult, setQueryResult] = useState<RecordData>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  /** 执行数据库操作；action 为真实请求，report 接收持久错误，完成前禁用重复提交。 */
  async function act(
    action: () => Promise<unknown>,
    report: (message: string) => void,
  ) {
    setBusy(true);
    setQueryResult(undefined);
    setMemoryResult(undefined);
    report("");
    try {
      await action();
      report("");
    } catch (e) {
      report(String(e));
    } finally {
      setBusy(false);
    }
  }
  const records = useApi(
    collection
      ? `/collections/${encodeURIComponent(collection)}?limit=10&offset=${offset}&id=${encodeURIComponent(search.id)}&text=${encodeURIComponent(search.text)}`
      : `/collections?limit=10&offset=${collectionOffset}`,
  );
  const [namespace, setNamespace] = useState("");
  const namespaces = useApi("/memory", 5000);
  const [activeNamespace, setActiveNamespace] = useState("");
  const [memoryOffset, setMemoryOffset] = useState(0);
  const [memoryId, setMemoryId] = useState("");
  const [memoryText, setMemoryText] = useState("");
  const [metadata, setMetadata] = useState("{}");
  const [editing, setEditing] = useState(false);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memoryResult, setMemoryResult] = useState<RecordData>();
  const memory = useApi(
    activeNamespace
      ? `/memory/${encodeURIComponent(activeNamespace)}?limit=10&offset=${memoryOffset}`
      : "/memory",
  );
  /** 展示写 API 已从数据库核验的指定 ID 读回，分页列表独立刷新。 */
  async function saveMemory() {
    const body = {
      id: memoryId,
      text: memoryText,
      metadata: JSON.parse(metadata),
    };
    const result = await api(
      `/memory/${encodeURIComponent(activeNamespace)}${editing ? `/${encodeURIComponent(memoryId)}` : ""}`,
      body,
      editing ? "PUT" : "POST",
    );
    setMemoryResult(result);
    await memory.refresh();
    await namespaces.refresh();
  }
  return (
    <>
      {busy && (
        <p className="muted" role="status">
          正在处理…
        </p>
      )}
      <ErrorBox error={error} />
      <Card
        title="向量集合 · 只读"
        actions={<Refresh onClick={collections.refresh} />}
      >
        <State state={collections} />
        <div className="collection-grid">
          {items(collections.data).map((c) => (
            <button
              className={`collection ${collection === c.name ? "selected" : ""}`}
              key={c.name}
              onClick={() => {
                setCollection(c.name);
                setOffset(0);
                setQueryResult(undefined);
              }}
            >
              <strong>{c.name}</strong>
              <span>{c.count ?? "null"} 条记录</span>
              <small>{c.metadata?.namespace ?? ""}</small>
            </button>
          ))}
        </div>
        {!collections.loading && !items(collections.data).length && (
          <Empty>暂无集合。</Empty>
        )}
        <Pager
          offset={collectionOffset}
          size={10}
          count={items(collections.data).length}
          change={setCollectionOffset}
        />
      </Card>
      {collection && (
        <Card title={`集合记录 · ${collection} · 只读`}>
          <div className="toolbar">
            <label>
              ID 精确过滤
              <input
                aria-label="记录 ID 过滤"
                value={idFilter}
                onChange={(e) => setIdFilter(e.target.value)}
              />
            </label>
            <label>
              文本包含
              <input
                aria-label="记录文本过滤"
                value={textFilter}
                onChange={(e) => setTextFilter(e.target.value)}
              />
            </label>
            <button
              onClick={() => {
                setOffset(0);
                setSearch({ id: idFilter, text: textFilter });
              }}
            >
              查询记录
            </button>
          </div>
          <State state={records} />
          <Records rows={items(records.data)} />
          <Pager
            offset={offset}
            size={10}
            count={items(records.data).length}
            change={setOffset}
          />
          <hr />
          <h3>语义 Top K</h3>
          <div className="toolbar">
            <label>
              工作区
              <input
                value={workspace}
                onChange={(e) => setWorkspace(e.target.value)}
              />
            </label>
            <label>
              tier
              <select value={tier} onChange={(e) => setTier(e.target.value)}>
                {["small", "medium", "full"].map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </label>
            <label>
              Top K
              <input
                aria-label="数据库 Top K"
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </label>
          </div>
          <label>
            语义检索问题
            <input
              aria-label="语义检索问题"
              value={semantic}
              onChange={(e) => setSemantic(e.target.value)}
            />
          </label>
          <button
            disabled={!semantic || busy}
            onClick={() =>
              act(
                async () =>
                  setQueryResult(
                    await api(
                      `/collections/${encodeURIComponent(collection)}/query`,
                      { query: semantic, workspace, tier, top_k: topK },
                    ),
                  ),
                setError,
              )
            }
          >
            执行语义检索
          </button>
          {queryResult && (
            <>
              <Records rows={queryResult.hits ?? items(queryResult)} />
              <Json value={queryResult} />
            </>
          )}
        </Card>
      )}
      <Card title="长期记忆 · namespace">
        <State state={namespaces} />
        <div className="toolbar">
          {items(namespaces.data).map((space) => (
            <button
              className="secondary"
              key={space.namespace}
              onClick={() => {
                setNamespace(space.namespace);
                setActiveNamespace(space.namespace);
                setMemoryOffset(0);
                setMemoryResult(undefined);
              }}
            >
              {space.namespace} · {space.count} 条
            </button>
          ))}
        </div>
        <div className="toolbar">
          <label>
            namespace
            <input
              aria-label="记忆 namespace"
              value={namespace}
              onChange={(e) => setNamespace(e.target.value)}
              placeholder="browser-check"
            />
          </label>
          <button
            disabled={!namespace}
            onClick={() => {
              setActiveNamespace(namespace);
              setMemoryOffset(0);
              setMemoryResult(undefined);
            }}
          >
            打开记忆空间
          </button>
        </div>
        {activeNamespace && (
          <>
            <State state={memory} />
            <h3>{activeNamespace}</h3>
            <div className="toolbar">
              <label>
                记忆 ID
                <input
                  aria-label="记忆 ID"
                  value={memoryId}
                  onChange={(e) => setMemoryId(e.target.value)}
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={editing}
                  onChange={(e) => setEditing(e.target.checked)}
                />
                更新已有 ID
              </label>
            </div>
            <label>
              原文
              <textarea
                aria-label="记忆原文"
                value={memoryText}
                onChange={(e) => setMemoryText(e.target.value)}
                rows={3}
              />
            </label>
            <label>
              metadata · JSON 对象
              <textarea
                aria-label="记忆 metadata"
                value={metadata}
                onChange={(e) => setMetadata(e.target.value)}
                rows={2}
              />
            </label>
            <div className="toolbar">
              <button
                disabled={!memoryId || !memoryText || busy}
                onClick={() => act(saveMemory, setError)}
              >
                {editing ? "保存更新" : "新增记忆"}
              </button>
              <button
                className="danger-button"
                disabled={!memoryId || busy}
                onClick={() =>
                  act(async () => {
                    if (
                      window.confirm(
                        `永久删除 ${activeNamespace} / ${memoryId}？`,
                      )
                    ) {
                      await api(
                        `/memory/${encodeURIComponent(activeNamespace)}/${encodeURIComponent(memoryId)}?confirm=true`,
                        undefined,
                        "DELETE",
                      );
                      await memory.refresh();
                      setMemoryResult(
                        await api(
                          `/memory/${encodeURIComponent(activeNamespace)}?limit=20&offset=0`,
                        ),
                      );
                    }
                  }, setError)
                }
              >
                删除指定 ID
              </button>
              <Refresh onClick={memory.refresh} />
            </div>
            <Records rows={items(memory.data)} />
            <Pager
              offset={memoryOffset}
              size={10}
              count={items(memory.data).length}
              change={setMemoryOffset}
            />
            <label>
              记忆语义查询
              <input
                aria-label="记忆语义查询"
                value={memoryQuery}
                onChange={(e) => setMemoryQuery(e.target.value)}
              />
            </label>
            <button
              disabled={!memoryQuery || busy}
              onClick={() =>
                act(
                  async () =>
                    setMemoryResult(
                      await api(
                        `/memory/${encodeURIComponent(activeNamespace)}/query`,
                        { query: memoryQuery, top_k: topK },
                      ),
                    ),
                  setError,
                )
              }
            >
              检索记忆
            </button>
            {memoryResult && (
              <>
                <Records
                  rows={
                    memoryResult.item
                      ? [memoryResult.item]
                      : (memoryResult.hits ?? items(memoryResult))
                  }
                />
                <Json value={memoryResult} label="操作结果" />
              </>
            )}
          </>
        )}
      </Card>
    </>
  );
}
