# 多轮检索问答

WebUI 问答页选择工作区、索引层级后点击“新建会话”。继续发送问题会使用该会话最近完成的问答；会话列表可恢复先前会话。“结束会话”释放历史引用，已有请求记录仍可查阅。

每轮重新检索当前证据。历史仅包含问题和回答，历史回答用于理解上下文；本轮事实与引用以本轮实际证据为准。历史最多占完整输入预算的一半，按完整轮次保留最近内容，最多检查最近 32 轮。页面显示采用及省略的历史轮数。当前问题本身超出预算时请求失败。

## WebAPI

创建会话：

```bash
curl -sS http://127.0.0.1:8765/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"formal","tier":"small"}'
```

将返回的 `id` 填入后续 `/api/qa` 请求的 `session_id`：

```bash
curl -N http://127.0.0.1:8765/api/qa \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"填写返回的32位id","workspace":"formal","tier":"small","question":"这项研究支持什么结论？","context_tokens":1536,"top_k":3}'
```

`query` 可显式指定检索文本，`question` 为当前问题。只传 `question` 时，检索文本包含保留历史中最近三轮的问题与当前问题。实际检索文本保存于请求的 `retrieval_query`。省略 `session_id` 保持单轮问答。

| 操作 | 接口 |
|---|---|
| 分页会话列表 | `GET /api/sessions?limit=20&offset=0` |
| 状态、完成轮次引用 | `GET /api/sessions/{id}` |
| 取消当前生成 | `POST /api/sessions/{id}/cancel` |
| 结束会话 | `DELETE /api/sessions/{id}` |
| 完整单次请求 | `GET /api/requests/{request_id}` |

取消会等待生成任务关闭上游连接；关闭浏览器响应流也会触发取消。取消、失败与不完整响应均不加入会话历史。生成中的会话结束操作和共享资源争用返回 409。等待生成取消后再结束会话。

会话绑定工作区、索引层级、模型版本及数值模式。切换这些设置后创建新会话。已完成历史保存于 `.agent/web/sessions`，后端重启后可继续；服务启动时将遗留运行请求标记为中断。

## 证据与 KV

请求记录保存本轮候选证据、实际纳入文献、历史请求 ID、完整 prompt、token 数及输入摘要。`citations` 记录引用 ID 与允许的文献 ID，出现证据中没有的引用会使请求失败。缺少引用单独记录；引用内容是否充分支持结论仍需核对原文。

正式 vLLM 接入方式见[分层 KV 配置](kv-cache.md)。所有消息按实际 token 输入引擎，前缀变化由真实块身份决定复用范围。会话结束释放对历史的引用；GPU、CPU、SSD 中已经生成的缓存继续遵守所选缓存策略和容量限制。
