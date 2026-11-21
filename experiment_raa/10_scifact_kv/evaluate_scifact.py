"""汇总已逐 token 复核的独立回放，并估计配对 TTFT 差值的区间。"""

import argparse
from itertools import combinations
import json
import math

import numpy as np

from memmgr.short_term.config import local_path

from .prepare_scifact import BLOCK, ROOT, save
from .run_scifact import ORDERS, POLICIES
from .verify_scifact import TIMINGS, verify_run

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 1010
PHASES = ("distinct", "revisit", "all")
CLOCKS = ("client", "server")


def quantiles(values):
    """计算 values 的中位数和最近秩 P95；空集合保留样本数和 null。"""
    if not len(values):
        return dict(n=0, median=None, p95=None)
    ordered = sorted(values)
    return dict(n=len(ordered), median=float(np.median(ordered)),
                p95=float(ordered[math.ceil(0.95 * len(ordered)) - 1]))


def paired_interval(clustered, draws):
    """以序列为簇，用抽样索引 draws 估计 clustered 配对差值的 95% 百分位区间。"""
    values = np.asarray(clustered, dtype=float)
    samples = values[draws].reshape(len(draws), -1)
    rank = math.ceil(0.95 * samples.shape[1]) - 1
    medians = np.median(samples, axis=1)
    tails = np.partition(samples, rank, axis=1)[:, rank]
    return dict(**quantiles(values.ravel()),
                median_ci95=np.quantile(medians, (0.025, 0.975)).tolist(),
                p95_ci95=np.quantile(tails, (0.025, 0.975)).tolist())


def latency(row, clock):
    """从 row 取 clock 指定的客户端或服务端 TTFT，以秒返回。"""
    return row["client_ttft_seconds"] if clock == "client" else row["metrics"]["ttft"]


def paired(groups, sequences, phase, clock, base, treatment, draws):
    """在同序列、同顺序和同请求位置配对 base 与 treatment，按 phase 和 clock 汇总。"""
    by_epoch = {epoch: [] for epoch in ORDERS}
    clustered = []
    for sequence in sequences:
        differences = []
        for epoch, policies in groups.items():
            current = [(latency(b, clock) - latency(a, clock)) * 1000
                       for a, b in zip(policies[base][sequence], policies[treatment][sequence], strict=True)
                       if phase == "all" or a["phase"] == phase]
            differences.extend(current)
            by_epoch[epoch].extend(current)
        clustered.append(differences)
    return dict(delta_ms=paired_interval(clustered, draws),
                by_order_ms={epoch: quantiles(values) for epoch, values in by_epoch.items()})


def summarize_policy(rows):
    """汇总 rows 的复用、缓存动作和每请求策略计时；未执行的阶段计零。"""
    metrics = [row["metrics"] for row in rows]
    timing = [row["timing"] for row in rows]
    counts = sorted({row["prior_prefix_tokens"] for row in rows})
    return dict(
        requests=len(rows),
        prior_prefix_tokens_histogram={str(count): sum(row["prior_prefix_tokens"] == count for row in rows)
                                       for count in counts},
        requests_with_gpu_hits=sum(item["gpu"] > 0 for item in metrics),
        requests_with_cpu_hits=sum(item["cpu"] > 0 for item in metrics),
        requests_reusing_beyond_first_block=sum(item["gpu"] + item["cpu"] > BLOCK for item in metrics),
        tokens={name: int(sum(item[name] for item in metrics)) for name in ("queries", "gpu", "cpu", "computed")},
        migration_bytes={name: int(sum(item[name] for item in metrics)) for name in ("store_bytes", "load_bytes")},
        evicted_blocks=sum(row["evicted"] for row in rows), evicted_unread_blocks=sum(row["evicted_unread"] for row in rows),
        admitted_blocks=sum(row["admitted_blocks"] for row in rows),
        retain_decisions={"accepted": sum(row["retained"] is True for row in rows),
                          "rejected": sum(row["retained"] is False for row in rows)},
        ttft_ms={clock: quantiles([latency(row, clock) * 1000 for row in rows]) for clock in CLOCKS},
        overhead_us={name.removesuffix("_ns"): quantiles([item[name] / 1000 for item in timing]) for name in TIMINGS},
        measured_decision_total_us=quantiles([sum(item[field] for field in TIMINGS) / 1000 for item in timing]),
        timing_calls={name: sum(item[name] for item in timing)
                      for name in ("scored_rows", "score_calls", "select_calls", "evict_calls")},
    )


def evaluate(config, inputs, groups, context):
    """用 config、inputs、groups 和 context 汇总正式配对结果及固定实验身份。"""
    sequences = [sequence["id"] for sequence in inputs["sequences"]]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(sequences), (BOOTSTRAP_DRAWS, len(sequences)))
    phase_groups = {}
    for phase in PHASES:
        phase_groups[phase] = dict(
            policies={policy: summarize_policy([row for policies in groups.values() for rows in policies[policy].values()
                                                for row in rows if phase == "all" or row["phase"] == phase])
                      for policy in POLICIES},
            paired={f"{treatment}-{base}": {clock: paired(groups, sequences, phase, clock, base, treatment, draws)
                                            for clock in CLOCKS}
                    for base, treatment in combinations(POLICIES, 2)},
        )
    one_policy = phase_groups["all"]["policies"]["lru"]["requests"]
    baseline_inputs = sum(len(rows) for rows in groups["forward"]["lru"].values())
    output_tokens = sum(len(row["generated_token_ids"]) for policies in groups.values() for sequences_ in policies.values()
                        for rows in sequences_.values() for row in rows)
    retrieval = list(inputs["requests"].values())
    return dict(
        inputs_sha256=config["inputs_sha256"], model_hashes=config["model_hashes"], threshold=config["model_threshold"],
        retrieval={"documents": inputs["index"]["documents"], "chunks": inputs["index"]["chunks"],
                   "questions": len(retrieval), "mean_recall_at_3": sum(row["recall"] for row in retrieval) / len(retrieval)},
        pairing={"independent_sequences": sequences, "orders": config["orders"], "positions_per_order": baseline_inputs,
                 "requests_verified_token_by_token": one_policy * len(POLICIES), "output_tokens_verified": output_tokens},
        capacities={key: context[key] for key in ("gpu", "gpu_blocks", "cpu_blocks", "gpu_bytes")}
        | {"cpu_bytes": config["cpu_bytes"], "block_tokens": BLOCK, "block_bytes": config["block_bytes"]},
        bootstrap=dict(method="paired sequence-cluster percentile", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                       level=0.95, orders_kept_together=True),
        phases=phase_groups,
    )


def write_requests(path, groups):
    """在 path 写入 groups 的逐请求计时、配对键与复用资料，保留生成原记录所在阶段。"""
    with path.open("x", encoding="utf-8") as stream:
        for epoch, policies in groups.items():
            for policy, sequences in policies.items():
                for sequence, rows in sequences.items():
                    for row in rows:
                        record = {key: row[key] for key in ("index", "query_id", "phase", "request_gap", "prompt_tokens",
                                  "prior_prefix_tokens", "client_ttft_seconds", "timing", "admitted_blocks", "retained",
                                  "evicted", "evicted_unread")}
                        record.update(epoch=epoch, policy=policy, sequence=sequence, server_ttft_seconds=latency(row, "server"),
                                      **{key: int(row["metrics"][key]) for key in ("gpu", "cpu", "computed", "load_bytes", "store_bytes")})
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    """对 root 的完整回放做独立离线复核，在新目录 output 保存汇总和逐请求数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    root, output = (local_path(path, ROOT) for path in (args.root, args.output))
    config, inputs, groups, context = verify_run(root)
    summary = evaluate(config, inputs, groups, context)
    output.mkdir(parents=True, exist_ok=False)
    write_requests(output / "per_request.jsonl", groups)
    save(output / "review.json", summary)
    print(f"已核对 {summary['pairing']['requests_verified_token_by_token']} 条请求并输出：{output}", flush=True)


if __name__ == "__main__":
    main()
