"""仅为实验 09 在原生 SSD 提升预约前决定是否回迁。"""

import os
from pathlib import Path

from memmgr.short_term.connector import ManagedConnector, ManagedTieringManager, ManagedTieringSpec
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

BLOCK = 64
PROBE_PREFIX = "cmpl-ssd09-probe-"


def geometry(prompt, gpu, cpu_keys, ssd_keys):
    """按 prompt、gpu token 数与两层候选块数标识实测几何。"""
    return f"{prompt}:{gpu}:{len(cpu_keys)}:{len(ssd_keys)}"


class SSDCostManager(ManagedTieringManager):
    """保留 07 的 LRU、写入和 DMA，仅为指定请求屏蔽 SSD 查找。"""

    def __init__(self, native, extra):
        """接管 native 的实际三级资源，extra 提供实验模式和冻结成本表。"""
        super().__init__(native, extra)
        self.denied = set()
        self.probed = set()
        self.mode = extra["ssd_cost_mode"]
        if self.mode not in {"restore", "recompute", "gate"}:
            raise ValueError(self.mode)
        self.profile = extra.get("ssd_cost_profile")
        if (self.profile is not None) != (self.mode == "gate"):
            raise ValueError("Only gate mode accepts the frozen cost profile")
        if len(self.secondary_tiers) != 1 or self.control.options["policy"] != "lru":
            raise ValueError("Experiment 09 requires one SSD tier and LRU")

    def decide(self, status, gpu):
        """在 status 首轮原生查找前检查 GPU 后的 CPU 状态和 SSD 文件元数据。"""
        request, ctx = status.req, status.req_context
        if not ctx.req_id.startswith(PROBE_PREFIX) or ctx.req_id in self.probed:
            return
        if gpu % BLOCK or request.skip_reading_prefix_cache or status.transfer_jobs:
            raise ValueError("Unexpected probe scheduler state")
        self._maybe_process_finished_jobs()
        status.update_offload_keys()
        cpu, ssd, layers = [], [], []
        tier = self.secondary_tiers[0]
        for key in status.group_states[0].offload_keys[gpu // BLOCK:(request.num_prompt_tokens - 1) // BLOCK]:
            resident = self.primary_tier.lookup(key, ctx)
            if resident is LookupResult.HIT:
                cpu.append(key.hex())
                layers.append("cpu")
                continue
            if resident is not LookupResult.MISS:
                raise RuntimeError("Probe entered before CPU transfers drained")
            path = Path(tier.file_mapper.get_file_name(key))
            if not path.is_file():
                break
            if path.stat().st_size != tier._block_size:
                raise RuntimeError(f"Incomplete SSD candidate: {path}")
            ssd.append(key.hex())
            layers.append("ssd")
        layout = geometry(request.num_prompt_tokens, gpu, cpu, ssd)
        simple = layers == ["cpu"] * len(cpu) + ["ssd"] * len(ssd)
        row = self.profile["rows"].get(layout) if self.profile is not None and simple else None
        if self.mode == "recompute" and not simple:
            raise ValueError("Experiment recompute requires the CPU prefix before the SSD suffix")
        if not ssd:
            admit, reason = False, "no-ssd"
        elif self.mode == "gate":
            admit, reason = (row["admit"], "calibrated") if row is not None else (True, "outside-profile")
        else:
            admit, reason = self.mode == "restore", self.mode
        if not admit:
            self.denied.add(ctx.req_id)
        self.probed.add(ctx.req_id)
        self.control.emit("ssd_cost_decision", request_id=ctx.req_id, geometry=layout,
                          prompt=request.num_prompt_tokens, gpu=gpu, cpu_keys=cpu, ssd_keys=ssd,
                          layers=layers, admit=admit, reason=reason)

    def lookup(self, key, req_context, *, exclude_tier_idx=None):
        """为 req_context 查找 key；拒绝时只排除 SSD，保留原生 CPU 查找。"""
        if req_context.req_id in self.denied:
            exclude_tier_idx = 0
        return super().lookup(key, req_context, exclude_tier_idx=exclude_tier_idx)

    def _initiate_promotion(self, tier_idx, key, req_context):
        """为 req_context 从 tier_idx 预约 key；拒绝请求进入此处即终止实验。"""
        if req_context.req_id in self.denied:
            raise RuntimeError("Rejected SSD request reached promotion")
        return super()._initiate_promotion(tier_idx, key, req_context)

    def on_request_finished(self, req_context, *, exclude_tier_idx=None):
        """清理 req_context 的准入状态，并保留原生结束流程和层排除参数。"""
        super().on_request_finished(req_context, exclude_tier_idx=exclude_tier_idx)
        self.denied.discard(req_context.req_id)
        self.probed.discard(req_context.req_id)


class SSDCostSpec(ManagedTieringSpec):
    """复用 07 的 SSD 布局和 GPU worker，选择实验专用调度管理器。"""

    def get_manager(self):
        """延迟创建原生资源，将其交给实验 09 的管理器。"""
        if self._manager is None:
            self._manager = SSDCostManager(TieringOffloadingSpec.get_manager(self), self.extra_config)
        return self._manager


class SSDCostConnector(ManagedConnector):
    """在固定的 07 环境上安装 SSD 实验策略。"""

    def __init__(self, vllm_config, role, kv_cache_config):
        """用 vllm_config、role 与 kv_cache_config 创建 connector，核对成本表环境。"""
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            manager = self.connector_scheduler.manager
            self.control.emit("ssd_cost_process", pid=os.getpid())
            if manager.profile is not None and manager.profile["context"] != self.context:
                raise ValueError("Calibration context differs from current vLLM context")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """先为 request 决定 SSD 准入，再用 num_computed_tokens 调用原生调度。"""
        scheduler = self.connector_scheduler
        scheduler.manager.decide(scheduler._req_status[request.request_id], num_computed_tokens)
        return super().get_num_new_matched_tokens(request, num_computed_tokens)
