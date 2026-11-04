/** 三份关联配置结构化编辑；消费服务端只读 schema，完整提交后重新读取。 */
import { useEffect, useState } from "react";
import { api, type RecordData } from "./api";
import { Card, ErrorBox, Refresh, State, perform, useApi } from "./ui";
const descriptions: Record<string, string> = {
  chunk_tokens: "token 上限",
  chunk_overlap: "token 数，须小于 chunk_tokens",
  query_instruction: "仅用于查询编码",
  port: "与 llm.base_url 一致",
  base_url: "与 server host / port 一致",
  max_tokens: "输出 token 上限",
  max_model_len: "上下文 token 上限，baseline / server 须一致",
  timeout: "秒",
  context_tokens: "输入 token 预算，含模板、问题与证据",
  concurrency: "1–4",
  gpu_memory_utilization: "0.1–0.95",
};
/** 递归字段；value 为对象，change 更新对象，prefix 为完整路径，readonly 为服务器禁止修改的路径。 */
function Fields({
  value,
  change,
  prefix,
  readonly,
}: {
  value: RecordData;
  change: (value: RecordData) => void;
  prefix: string;
  readonly: string[];
}) {
  return (
    <div className="config-fields">
      {Object.entries(value).map(([key, entry]) => {
        const path = `${prefix}.${key}`;
        const disabled = readonly.some(
          (p) => path === p || path.startsWith(`${p}.`),
        );
        if (entry && typeof entry === "object" && !Array.isArray(entry))
          return (
            <fieldset key={key}>
              <legend>{key}</legend>
              <Fields
                value={entry}
                prefix={path}
                readonly={readonly}
                change={(next) => change({ ...value, [key]: next })}
              />
            </fieldset>
          );
        return (
          <label key={key} className="config-field">
            <span>
              {key} {disabled && <span className="badge">只读</span>}
            </span>
            {typeof entry === "boolean" ? (
              <select
                disabled={disabled}
                aria-label={path}
                value={String(entry)}
                onChange={(e) =>
                  change({ ...value, [key]: e.target.value === "true" })
                }
              >
                <option>true</option>
                <option>false</option>
              </select>
            ) : (
              <input
                disabled={disabled}
                aria-label={path}
                type={typeof entry === "number" ? "number" : "text"}
                step={
                  typeof entry === "number" && Number.isInteger(entry)
                    ? 1
                    : "any"
                }
                min={
                  typeof entry === "number" && key !== "seed" ? 0 : undefined
                }
                max={
                  key === "gpu_memory_utilization"
                    ? 1
                    : key === "concurrency"
                      ? 4
                      : undefined
                }
                required
                value={
                  Array.isArray(entry)
                    ? JSON.stringify(entry)
                    : String(entry ?? "")
                }
                onChange={(e) => {
                  const raw = e.target.value;
                  change({
                    ...value,
                    [key]:
                      typeof entry === "number"
                        ? raw === ""
                          ? ""
                          : Number(raw)
                        : raw,
                  });
                }}
              />
            )}
            {!disabled && descriptions[key] && (
              <small>{descriptions[key]}</small>
            )}
          </label>
        );
      })}
    </div>
  );
}
/** 校验并恢复数字数组类型；value 为编辑值，original 为后端真实原值。 */
function typed(value: RecordData, original: RecordData): RecordData {
  return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => {
      const before = original[key];
      if (Array.isArray(before)) {
        const parsed = typeof entry === "string" ? JSON.parse(entry) : entry;
        if (
          !Array.isArray(parsed) ||
          parsed.some((v) => typeof v !== "number" || !Number.isInteger(v))
        )
          throw new Error(`${key} 必须为整数数组`);
        return [key, parsed];
      }
      if (before && typeof before === "object")
        return [key, typed(entry, before)];
      if (
        typeof before === "number" &&
        (typeof entry !== "number" || !Number.isFinite(entry))
      )
        throw new Error(`${key} 必须为有限数字`);
      return [key, entry];
    }),
  );
}
/** 编辑全部三配置，切换标签不丢弃未保存字段，提交整组由后端验证关联约束。 */
export function Configuration() {
  const state = useApi("/config/schema");
  const [name, setName] = useState("baseline");
  const [draft, setDraft] = useState<RecordData>({});
  const [error, setError] = useState("");
  const [result, setResult] = useState<RecordData>();
  useEffect(() => {
    if (state.data?.values) setDraft(structuredClone(state.data.values));
  }, [state.data]);
  return (
    <>
      <Card title="配置" actions={<Refresh onClick={state.refresh} />}>
        <p className="muted">
          服务启动参数或生成模型快照变更需重启 vLLM；embedding
          身份变更需重新索引。
        </p>
        <State state={state} />
        <div className="tabs">
          {["baseline", "models", "server"].map((tab) => (
            <button
              key={tab}
              className={name === tab ? "active" : "secondary"}
              onClick={() => setName(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
      </Card>
      <Card title={name}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            setResult(undefined);
            setError("");
            void perform(async () => {
              setResult(
                await api(
                  "/config",
                  typed(draft, state.data?.values ?? {}),
                  "PUT",
                ),
              );
              await state.refresh();
            }, setError);
          }}
        >
          <Fields
            value={draft[name] ?? {}}
            change={(value) => setDraft({ ...draft, [name]: value })}
            prefix={name}
            readonly={state.data?.readonly ?? []}
          />
          <ErrorBox error={error} />
          <button type="submit" disabled={state.loading}>
            保存全部三份配置
          </button>
        </form>
        {result && (
          <div className="success" role="status">
            已保存。
          </div>
        )}
      </Card>
    </>
  );
}
