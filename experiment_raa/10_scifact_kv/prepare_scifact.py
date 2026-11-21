"""用 SciFact 测试集、独立 Chroma 和项目聊天模板冻结回放输入。"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from transformers import AutoTokenizer

from memmgr.inference.models import local_snapshot
from memmgr.inference.prompt import assemble_prompt, chat_input_ids
from memmgr.long_term.index import index_tier, make_embedder, sha256_file
from memmgr.long_term.retrieval import load_queries_and_qrels, open_collection, score_retrieval
from memmgr.short_term.config import local_path

ROOT = Path(__file__).resolve().parents[2]
TIER = "full"
BLOCK = 64
SEED = 1010
SEQUENCES = 6
DISTINCT = 8
REVISITS = (7, 6, 7, 5, 6, 4, 5, 7, 4, 7)
SOURCE = ROOT / "data/scifact-baseline/scifact"


def read(path):
    """读取 path 中的 JSON 数据。"""
    return json.loads(path.read_bytes())


def save(path, value):
    """在新文件 path 中保存 value，拒绝覆盖已有记录。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def token_digest(ids):
    """按连接器的 JSON 编码计算输入 ids 的摘要。"""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def lookup(collection, embed, query, top_k):
    """用 embed 编码 query，在 collection 查询 top_k 个文本块并保留原始排名。"""
    result = collection.query(query_embeddings=[embed([query], query=True)[0].tolist()],
                              n_results=top_k, include=["documents", "metadatas", "distances"])
    fields = (result[key][0] for key in ("ids", "documents", "metadatas", "distances"))
    rows = [dict(chunk_id=key, text=text, doc_id=str(meta["doc_id"]), distance=distance)
            for key, text, meta, distance in zip(*fields, strict=True)]
    if len(rows) != top_k:
        raise ValueError(f"Expected {top_k} Chroma results, got {len(rows)}")
    return rows


def candidates(chunks):
    """将查询返回的 chunks 按首个命中块去重，保留文档排名。"""
    by_document = {}
    for row in chunks:
        by_document.setdefault(row["doc_id"], row["text"])
    return list(by_document.items())


def tokenizer_for(config):
    """按 config 的固定生成模型 revision 加载本地 tokenizer。"""
    llm = config["llm"]
    return AutoTokenizer.from_pretrained(local_snapshot(llm["tokenizer"], llm["revision"]),
                                         local_files_only=True)


def chosen_sequences(qrels):
    """用测试标签 qrels 的问题 ID 固定抽样，构造互不相交的首次访问和固定复访。"""
    chosen = random.Random(SEED).sample(sorted(qrels), SEQUENCES * DISTINCT)
    groups = [chosen[start:start + DISTINCT] for start in range(0, len(chosen), DISTINCT)]
    return [{"id": f"seq-{i:02}", "distinct": group,
             "revisits": [group[pos] for pos in REVISITS]} for i, group in enumerate(groups)]


def replay_plan(inputs, sequence):
    """将已冻结 inputs 的 sequence 展开为请求，并计算此前出现过的最长完整前缀。"""
    seen = set()
    rows = []
    last_visit = {}
    for phase, query_ids in (("distinct", sequence["distinct"]), ("revisit", sequence["revisits"])):
        for query_id in query_ids:
            index = len(rows)
            tokens = inputs["requests"][query_id]["prompt_token_ids"]
            prefixes = [tuple(tokens[:end]) for end in range(BLOCK, len(tokens) + 1, BLOCK)]
            rows.append(dict(index=index, query_id=query_id, phase=phase,
                             request_gap=index - last_visit[query_id] if query_id in last_visit else None,
                             prior_prefix_tokens=BLOCK * sum(prefix in seen for prefix in prefixes),
                             prompt_token_ids=tokens, prompt_sha256=token_digest(tokens)))
            seen.update(prefixes)
            last_visit[query_id] = index
    return rows


def verify(directory):
    """离线核对 directory 的语料、抽样、证据与逐 token 输入，返回冻结数据。"""
    inputs = read(directory / "inputs.json")
    config = inputs["config"]
    dataset = dict(config, dataset=dict(config["dataset"], root=str(directory / "dataset")))
    root, queries, qrels = load_queries_and_qrels(dataset, TIER)
    if (len(queries) != 1109 or len(qrels) != 300 or inputs["sequences"] != chosen_sequences(qrels)
            or inputs["index"]["documents"] != 5183
            or inputs["source_sha256"] != {name: sha256_file(root / name) for name in
                                             ("corpus.jsonl", "queries.jsonl", "qrels.tsv")}):
        raise ValueError("SciFact provenance or sample mismatch")
    tokenizer = tokenizer_for(config)
    expected = {query_id for sequence in inputs["sequences"] for query_id in sequence["distinct"]}
    top_k = config["experiment"]["top_k"]
    budget = config["experiment"]["context_tokens"]
    if inputs["requests"].keys() != expected or config["embedding"]["device"] != "cpu":
        raise ValueError("Prepared requests or BGE device mismatch")
    for query_id, record in inputs["requests"].items():
        chunks = record["retrieved_chunks"]
        documents = candidates(chunks)
        messages, included, _, ids, truncated = assemble_prompt(tokenizer, queries[query_id], documents, budget)
        recall, ndcg = score_retrieval([doc_id for doc_id, _ in documents], qrels[query_id], top_k)
        if (not included or len(chunks) != top_k or record["query"] != queries[query_id]
                or record["qrels"] != sorted(qrels[query_id]) or record["recall"] != recall
                or record["ndcg"] != ndcg or record["messages"] != messages
                or record["included_doc_ids"] != included or record["truncated"] is not truncated
                or ids != chat_input_ids(tokenizer, record["messages"]) or ids != record["prompt_token_ids"]):
            raise ValueError(f"Retrieved evidence or chat prompt mismatch: {query_id}")
    return inputs


def prepare(output, port):
    """通过隔离 Chroma 的 port 导入全量 SciFact，在新目录 output 冻结检索记录。"""
    root = output / "dataset" / TIER
    root.mkdir(parents=True, exist_ok=False)
    paths = {"corpus.jsonl": SOURCE / "corpus.jsonl", "queries.jsonl": SOURCE / "queries.jsonl",
             "qrels.tsv": SOURCE / "qrels/test.tsv"}
    for name, path in paths.items():
        shutil.copyfile(path, root / name)
    config = read(ROOT / "config/baseline.json")
    config["dataset"]["root"] = str(root.parent)
    config["embedding"]["snapshot_path"] = str(local_snapshot(config["embedding"]["model"],
                                                             config["embedding"]["revision"]))
    config["chroma"].update(port=port, collection_prefix="memmgr-kv-scifact-10")
    manifest = index_tier(config, TIER)
    collection, reopened = open_collection(config, TIER)
    if manifest != reopened:
        raise ValueError("Reopened Chroma index differs from imported index")
    _, queries, qrels = load_queries_and_qrels(config, TIER)
    sequences = chosen_sequences(qrels)
    tokenizer = tokenizer_for(config)
    _, embed = make_embedder(config)
    top_k = config["experiment"]["top_k"]
    budget = config["experiment"]["context_tokens"]
    requests = {}
    for sequence in sequences:
        for query_id in sequence["distinct"]:
            query = queries[query_id]
            chunks = lookup(collection, embed, query, top_k)
            documents = candidates(chunks)
            messages, included, _, ids, truncated = assemble_prompt(tokenizer, query, documents, budget)
            recall, ndcg = score_retrieval([doc_id for doc_id, _ in documents], qrels[query_id], top_k)
            requests[query_id] = dict(query=query, qrels=sorted(qrels[query_id]), retrieved_chunks=chunks,
                                      recall=recall, ndcg=ndcg, messages=messages, included_doc_ids=included,
                                      prompt_token_ids=ids, truncated=truncated)
    save(output / "inputs.json", dict(config=config, source_sha256={name: sha256_file(path)
         for name, path in paths.items()}, index=manifest, sequences=sequences, requests=requests))


def main():
    """读取输出目录与独立 Chroma 端口，准备真实请求并从保存的证据复核输入。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--chroma-port", type=int, default=18110)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = local_path(args.output, ROOT)
    if not args.verify_only:
        if args.chroma_port == 8001:
            parser.error("Use a dedicated Chroma port, not the project port 8001")
        output.mkdir(parents=True, exist_ok=False)
        prepare(output, args.chroma_port)
    inputs = verify(output)
    print(f"已核对 {inputs['index']['documents']} 篇索引文档，{len(inputs['requests'])} 个真实检索输入，"
          f"{len(inputs['sequences'])} 条请求序列：{output}", flush=True)


if __name__ == "__main__":
    main()
