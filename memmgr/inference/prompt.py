"""使用真实生成 tokenizer 组装聊天证据，并核验完整输入预算。

候选文档按检索顺序加入。达到预算时取当前文档的字符前缀，
最后对整个聊天模板重新计数，保持实际采用的文档 ID 与证据一致。
"""

from __future__ import annotations


def chat_input_ids(tokenizer, messages):
    """渲染聊天模板并返回一维输入 token ID 列表。

    模板包含生成起始标记，编码时关闭额外特殊 token。
    对列表、嵌套列表及张量返回值统一取首个样本。
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape"):
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]
        return input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return input_ids


def assemble_prompt(tokenizer, query, candidates, budget):
    """在完整输入预算内组装问题和按排名排列的候选证据。

    参数：
        tokenizer：生成模型的 tokenizer，用于完整模板计数。
        query：原始问题文本。
        candidates：按检索顺序排列的 (doc_id, text) 序列。
        budget：包含系统提示、问题、引用标记和生成模板的输入上限。

    返回：
        消息列表、实际纳入文档 ID、证据文本、输入 ID 和截断标记。

    边界：
        当前文档超预算时对字符前缀二分裁切，空前缀文档省略。
        连空证据提示词也放不下时抛出 ValueError。
    """
    system = (
        "Answer in one or two concise sentences using only the supplied evidence. "
        "If any part is supported, begin with the exact supporting "
        "[DOC document_id] citation and state only what that evidence supports; "
        "if the full answer is not established, say which part is uncertain. "
        "If no evidence supports an answer, say the evidence is insufficient. "
        "Never invent or cite a document absent from the evidence."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {query}\nEvidence:\n"},
    ]
    included_doc_ids = []
    evidence = ""
    truncated = False

    for doc_id, text in candidates:
        complete_evidence = evidence + f"[DOC {doc_id}] {text}\n"
        messages[1]["content"] = f"Question: {query}\nEvidence:\n{complete_evidence}"
        if len(chat_input_ids(tokenizer, messages)) > budget:
            truncated = True
            prefix_low = 0
            prefix_high = len(text)
            # 字符前缀便于保留原文；每个候选仍通过真实 tokenizer 核验总预算。
            while prefix_low < prefix_high:
                prefix_length = (prefix_low + prefix_high + 1) // 2
                candidate_evidence = evidence + f"[DOC {doc_id}] {text[:prefix_length]}\n"
                messages[1]["content"] = (
                    f"Question: {query}\nEvidence:\n{candidate_evidence}"
                )
                if len(chat_input_ids(tokenizer, messages)) <= budget:
                    prefix_low = prefix_length
                else:
                    prefix_high = prefix_length - 1
            if prefix_low:
                evidence += f"[DOC {doc_id}] {text[:prefix_low]}\n"
                included_doc_ids.append(doc_id)
            break
        evidence = complete_evidence
        included_doc_ids.append(doc_id)

    # 统一恢复实际采用的证据，避免二分试探留下末次候选文本。
    messages[1]["content"] = f"Question: {query}\nEvidence:\n{evidence}"
    input_ids = chat_input_ids(tokenizer, messages)
    if len(input_ids) > budget:
        raise ValueError(
            f"Chat prompt uses {len(input_ids)} tokens; budget is {budget}. "
            "Even the empty-evidence prompt does not fit."
        )
    return messages, included_doc_ids, evidence, input_ids, truncated
