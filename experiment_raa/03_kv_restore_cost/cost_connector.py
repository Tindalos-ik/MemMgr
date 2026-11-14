"""为 vLLM 0.29.0 原生 OffloadingConnector 增加可审计的读回准入。"""

import importlib.metadata
import json
from pathlib import Path
import time

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.kv_cache_interface import FullAttentionSpec

from cost_policy import CostPolicy, geometry


class CostOffloadingConnector(OffloadingConnector):
    """继承原生 CPU 写入、LRU 和 DMA 路径，仅控制可恢复前缀的读取。"""

    def __init__(self, vllm_config, role, kv_cache_config):
        """用 vllm_config 和 kv_cache_config 初始化 role 对应角色；scheduler 读取策略并打开审计日志。"""
        if importlib.metadata.version("vllm") != "0.29.0":
            raise ValueError("This experiment requires vLLM 0.29.0")
        groups = kv_cache_config.kv_cache_groups
        if (len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec
                or groups[0].kv_cache_spec.block_size != 64
                or vllm_config.parallel_config.world_size != 1
                or vllm_config.scheduler_config.max_num_seqs != 1):
            raise ValueError("Require one full-attention group, 64-token blocks and one sequence/worker")
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.mode = extra["cost_mode"]
        if self.mode not in ("always", "never", "gate"):
            raise ValueError("Unknown cost mode")
        self.policy = CostPolicy(extra["cost_profile"], extra["cost_margin_ms"]) if self.mode == "gate" else None
        if self.policy is not None and self.policy.context != extra["cost_context"]:
            raise ValueError("Calibration and runtime context differ")
        super().__init__(vllm_config, role, kv_cache_config)
        self.audit = None
        if role == KVConnectorRole.SCHEDULER:
            if self.connector_scheduler.config.supports_partial_tail:
                raise ValueError("Partial-tail state is outside this experiment")
            self.audit = Path(extra["cost_decisions"]).open("x", encoding="utf-8", buffering=1)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """查找 request 的 CPU 候选，依据 num_computed_tokens 保留 GPU 命中，拒绝时返回零外部命中。"""
        started = time.perf_counter()
        candidates, asynchronous = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if candidates is None:
            return candidates, asynchronous
        geometry(request.num_prompt_tokens, num_computed_tokens, candidates)
        estimate = self.policy.decide(request.num_prompt_tokens, num_computed_tokens, candidates) if self.policy and candidates else None
        admit = candidates > 0 and (estimate["admit"] if estimate else self.mode == "always")
        if candidates and not admit:
            # 原生查找尚未创建 load job；撤回外部命中，随后按原生 prefill 路径计算。
            status = self.connector_scheduler._req_status[request.request_id]
            if status.transfer_jobs or status.partial_tail_boundary is not None:
                raise RuntimeError("Unexpected transfer state before admission")
            status.update_num_hit_chunks(num_computed_tokens)
        record = dict(request_id=request.request_id, mode=self.mode,
                      prompt_tokens=request.num_prompt_tokens, gpu_tokens=num_computed_tokens,
                      missing_tokens=request.num_prompt_tokens - num_computed_tokens,
                      cpu_tokens=candidates, admitted_tokens=candidates if admit else 0,
                      remaining_tokens=request.num_prompt_tokens - num_computed_tokens - (candidates if admit else 0),
                      reason="no-candidate" if not candidates else ("cost" if estimate else self.mode),
                      estimate=estimate, profile_sha256=self.policy.sha256 if self.policy else None,
                      lookup_and_policy_seconds=time.perf_counter() - started)
        self.audit.write(json.dumps(record) + "\n")
        return (candidates, asynchronous) if admit else (0, False)

    def shutdown(self):
        """停止原生 connector 并关闭 scheduler 的审计文件。"""
        try:
            super().shutdown()
        finally:
            if self.audit is not None:
                self.audit.close()
