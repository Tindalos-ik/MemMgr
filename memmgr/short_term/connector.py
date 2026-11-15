"""在 vLLM 原生 KV 传输路径上执行连续前缀准入、淘汰与成本门槛。"""

import hashlib
import fcntl
import importlib.metadata
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import time
import uuid

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.base import OffloadPolicy, RequestOffloadingContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from .cost import CostPolicy
from .reuse import History, Predictor


class StrictAuditHandler(RotatingFileHandler):
    """限制审计日志大小，保留一次轮转，写入错误直接传播。"""

    def handleError(self, record):
        """传播当前 record 写入期间的原始异常。"""
        raise


class ManagedLRU(LRUCachePolicy):
    """保留原生块状态和引用保护，为 learned 模式提供叶块优先淘汰。"""

    def __init__(self, cache_capacity):
        """初始化 cache_capacity 个块，controller 在管理器创建时绑定。"""
        super().__init__(cache_capacity)
        self.controller = None

    def plan(self, n, protected):
        """为 n 个空位选择低复用叶块，跳过 protected 与在途块，保持父块连续性。"""
        controller = self.controller
        controller.refresh_scores(self.blocks)
        remaining = set(self.blocks)
        chosen = []
        ranks = {key: rank for rank, key in enumerate(self.evictable_blocks)}
        while len(chosen) < n:
            parents = {controller.parents[key] for key in remaining}
            candidates = [key for key in self.evictable_blocks
                          if key in remaining and key not in protected and key not in parents]
            if not candidates:
                return None
            key = min(candidates, key=lambda key: (controller.scores[key], ranks[key]))
            if self.blocks[key].ref_cnt != 0:
                raise RuntimeError("Pinned block entered eviction candidates")
            chosen.append((key, self.blocks[key]))
            remaining.remove(key)
        return chosen

    def evict(self, n, protected):
        """原子淘汰 n 个块；protected 保留调用者正在使用的块。"""
        if self.controller.options["policy"] != "learned":
            return super().evict(n, protected)
        candidates = self.plan(n, protected | self.controller.protected)
        if candidates is not None:
            for key, _ in candidates:
                self.remove(key)
        return candidates


class PolicyController:
    """为 CPU 管理器提供有界统计、冻结准入与事件记录。"""

    def __init__(self, primary, extra):
        """绑定 primary 的原生块状态；extra 提供固定配置及模型身份。"""
        self.primary = primary
        if primary._num_blocks + extra["max_model_len"] // 64 > 4096:
            raise ValueError("CPU cache and request span exceed the bounded prefix history")
        self.options = extra["policy_config"]
        self.identity = dict(model_revision=extra["model_revision"], dtype=extra["dtype"],
                             block_tokens=64, features="block-chain-v1", vllm="0.29.0")
        self.history = History()
        self.parents, self.active, self.scores = {}, {}, {}
        self.protected = set()
        self.session = uuid.uuid4().hex
        self.path = Path(self.options["audit_path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600))
        self.handler = StrictAuditHandler(self.path, maxBytes=64 * 2**20, backupCount=1, encoding="utf-8")
        self.predictor = (Predictor(self.options["model_dir"], self.identity)
                          if self.options["policy"] != "lru" else None)
        primary._policy.controller = self
        self.emit("init", identity=self.identity, policy=self.options["policy"],
                  capacity=primary._num_blocks, model_sha256=self.predictor.sha256 if self.predictor else None)

    def emit(self, kind, **fields):
        """将 kind 和 fields 追加为带 session 的单个完整事件。"""
        value = json.dumps(dict(session=self.session, time_ns=time.monotonic_ns(), kind=kind, **fields))
        self.handler.emit(logging.LogRecord("kv", logging.INFO, __file__, 0, value, (), None))

    def refresh_scores(self, keys):
        """为当前时钟下尚未评分的 keys 批量评分，覆盖排队期间新驻留的块。"""
        if self.predictor is not None:
            missing = [key for key in keys if key not in self.scores]
            self.scores.update(zip(missing, self.predictor.score([self.history.features(key) for key in missing])))

    def begin(self, request):
        """在 request 查找前观察所有完整前缀块，每个请求只更新一次时钟。"""
        started = time.perf_counter()
        keys = [make_offload_key(value, 0) for value in request.block_hashes]
        if len(keys) != request.num_prompt_tokens // 64:
            raise ValueError("Unexpected complete prompt hash count")
        protected = set(self.primary._policy.blocks)
        protected.update(key for state in self.active.values() for key in state["keys"])
        self.history.observe(keys, 64, protected)
        for index, key in enumerate(keys):
            self.parents[key] = keys[index - 1] if index else None
        self.parents = {key: parent for key, parent in self.parents.items() if key in self.history.entries}
        self.scores = {}
        self.refresh_scores(list(dict.fromkeys([*keys, *self.primary._policy.blocks])))
        admitted = []
        for key in keys:
            if self.predictor and self.scores[key] < self.predictor.threshold:
                break
            admitted.append(key)
        self.active[request.request_id] = dict(keys=keys, admitted=admitted)
        self.emit("begin", request_id=request.request_id, clock=self.history.clock,
                  prompt_tokens=request.num_prompt_tokens, keys=[key.hex() for key in keys],
                  features=[self.history.features(key) for key in keys],
                  prompt_sha256=hashlib.sha256(json.dumps(request.prompt_token_ids).encode()).hexdigest(),
                  admitted=[key.hex() for key in admitted],
                  policy_seconds=time.perf_counter() - started)

    def select(self, keys, req_context):
        """按 req_context 的冻结决定筛选 keys，并保护当前请求全部前缀块。"""
        state = self.active[req_context.req_id]
        if not set(keys) <= set(state["keys"]):
            raise RuntimeError("Decode or unknown block offered for prompt-only storage")
        self.protected = set(state["keys"])
        if "selected" not in state:
            selected = state["admitted"]
            if self.options["policy"] == "learned":
                self.refresh_scores(list(dict.fromkeys([*state["keys"], *self.primary._policy.blocks])))
                missing = [key for key in selected if key not in self.primary._policy.blocks]
                required = max(0, len(missing) - self.primary._get_num_free_blocks())
                victims = self.primary._policy.plan(required, self.protected)
                if victims is None:
                    raise RuntimeError("Cannot reserve current prefix with pinned blocks")
                incoming = sum(self.scores[key] for key in missing)
                displaced = sum(self.scores[key] for key, _ in victims)
                if required and incoming <= displaced:
                    selected = []
                self.emit("retain", request_id=req_context.req_id, incoming=incoming,
                          displaced=displaced, required=required,
                          victims=[key.hex() for key, _ in victims], accepted=bool(selected))
            state["selected"] = set(selected)
        return [key for key in keys if key in state["selected"]]


class ManagedCPUManager(CPUOffloadingManager):
    """控制写入与淘汰，沿用原生 CPU 内存分配、DMA 和引用计数。"""

    def __init__(self, spec):
        """从 spec 初始化容量、固定策略和策略控制器。"""
        super().__init__(spec.num_blocks, cache_policy="ManagedLRU",
                         cache_policy_module_path=__name__)
        self.control = PolicyController(self, spec.extra_config)

    def on_new_request(self, req_context):
        """为 req_context 提供完整 prompt 的再次写入机会。"""
        return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)

    def prepare_store(self, keys, req_context):
        """筛选 req_context 的 keys，安排原生写入并记录实际淘汰。"""
        selected = self.control.select(keys, req_context)
        result = super().prepare_store(selected, req_context)
        self.control.protected = set()
        if result is None:
            raise RuntimeError("CPU KV allocation failed")
        self.control.emit("store", request_id=req_context.req_id,
                          keys=[key.hex() for key in result.keys_to_store],
                          evicted=[key.hex() for key in result.evicted_keys])
        return result

    def complete_store(self, keys, req_context, success=True):
        """确认 req_context 的 keys 已完成写入，传输失败立即终止。"""
        if not success:
            raise RuntimeError("GPU to CPU transfer failed")
        super().complete_store(keys, req_context, success)
        self.control.emit("stored", request_id=req_context.req_id, keys=[key.hex() for key in keys])

    def complete_load(self, keys, req_context):
        """确认 req_context 的 keys 已完成 CPU 到 GPU 回迁。"""
        super().complete_load(keys, req_context)
        self.control.emit("loaded", request_id=req_context.req_id, keys=[key.hex() for key in keys])

    def shutdown(self):
        """释放原生管理器并关闭本次审计文件。"""
        try:
            super().shutdown()
        finally:
            self.control.handler.close()


class ManagedCPUSpec(CPUOffloadingSpec):
    """只替换 scheduler 管理器，保留原生 GPU worker。"""

    def get_manager(self):
        """创建并复用本 spec 唯一的 ManagedCPUManager。"""
        if self._manager is None:
            self._manager = ManagedCPUManager(self)
        return self._manager


class ManagedFileSystemTier(FileSystemTierManager):
    """使用原生 O_DIRECT 文件传输，失败时立即报错。"""

    def __init__(self, offloading_spec, primary_kv_view, tier_type, **options):
        """用 offloading_spec、primary_kv_view、tier_type 和 options 初始化文件层。"""
        self.lease = None
        self.jobs = {}
        self.control = None
        Path(options["root_dir"]).mkdir(parents=True, exist_ok=True, mode=0o700)
        super().__init__(offloading_spec, primary_kv_view, tier_type, **options)
        if not self._use_o_direct:
            self.shutdown()
            raise RuntimeError("SSD tier requires a filesystem supporting O_DIRECT")
        self.limit = offloading_spec.extra_config["policy_config"]["ssd_bytes"]
        self.paths = {str(path) for path in Path(options["root_dir"]).rglob("*.bin")}
        self.allocated = sum(Path(path).stat().st_size for path in self.paths)
        if self.allocated > self.limit:
            self.shutdown()
            raise ValueError("Existing SSD cache exceeds configured capacity")
        self.lease = (Path(options["root_dir"]) / ".memmgr.lock").open("a")
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.shutdown()
            raise

    def bind(self, control):
        """为文件层绑定 control，记录实际 I/O 布局与容量。"""
        self.control = control
        control.emit("fs_init", direct_io=True, block_bytes=self._block_size,
                     capacity=self.limit, existing_bytes=self.allocated)

    def submit_store(self, job_metadata):
        """为 job_metadata 的新文件预留 SSD 配额，容量不足时立即终止。"""
        paths = {str(self.file_mapper.get_file_name(key)) for key in job_metadata.keys}
        new_paths = paths - self.paths
        requested = len(new_paths) * self._block_size
        if self.allocated + requested > self.limit:
            raise RuntimeError("SSD KV capacity exhausted")
        self.paths.update(new_paths)
        self.allocated += requested
        self.jobs[job_metadata.job_id] = job_metadata
        self.control.emit("fs_submit", direction="store", job=job_metadata.job_id,
                          request_id=job_metadata.req_context.req_id,
                          keys=[key.hex() for key in job_metadata.keys])
        super().submit_store(job_metadata)

    def submit_load(self, job_metadata):
        """提交 job_metadata 的原生 SSD 到 CPU 读取并记录块身份。"""
        self.jobs[job_metadata.job_id] = job_metadata
        self.control.emit("fs_submit", direction="load", job=job_metadata.job_id,
                          request_id=job_metadata.req_context.req_id,
                          keys=[key.hex() for key in job_metadata.keys])
        super().submit_load(job_metadata)

    def get_finished_jobs(self):
        """返回完成的真实传输，任一失败立即报错。"""
        result = list(super().get_finished_jobs())
        for completed in result:
            job = self.jobs.pop(completed.job_id)
            self.control.emit("fs_complete", direction="load" if job.is_promotion else "store",
                              job=completed.job_id, success=completed.success,
                              request_id=job.req_context.req_id, keys=[key.hex() for key in job.keys],
                              transfer_seconds=completed.transfer_time)
            if not completed.success:
                raise RuntimeError("SSD KV transfer failed")
        return result

    def shutdown(self):
        """停止真实文件传输并释放当前缓存目录的独占锁。"""
        try:
            super().shutdown()
        finally:
            if self.lease is not None:
                self.lease.close()


class ManagedTieringManager(TieringOffloadingManager):
    """保留原生 SSD 提升，使用同一控制器筛选 GPU 写入并保护缓存块。"""

    def __init__(self, native, extra):
        """接管 native 尚未使用的原生资源；extra 提供策略配置。"""
        super().__init__(native.primary_tier, native.secondary_tiers)
        self.control = PolicyController(self.primary_tier, extra)
        for tier in self.secondary_tiers:
            tier.bind(self.control)

    def on_new_request(self, req_context, **options):
        """登记 req_context 和 options，使完整 prompt 都有写入机会。"""
        super().on_new_request(req_context, **options)
        return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)

    def prepare_store(self, keys, req_context):
        """按 req_context 的冻结准入筛选 keys，交给原生三级管理器。"""
        selected = self.control.select(keys, req_context)
        result = super().prepare_store(selected, req_context)
        self.control.protected = set()
        if result is None:
            raise RuntimeError("CPU KV allocation failed")
        self.control.emit("store", request_id=req_context.req_id,
                          keys=[key.hex() for key in result.keys_to_store],
                          evicted=[key.hex() for key in result.evicted_keys])
        return result

    def complete_store(self, keys, req_context, success=True):
        """确认 req_context 的 keys 完成 GPU 到 CPU 写入，失败立即报错。"""
        if not success:
            raise RuntimeError("GPU to CPU transfer failed")
        super().complete_store(keys, req_context, success)
        self.control.emit("stored", request_id=req_context.req_id, keys=[key.hex() for key in keys])

    def complete_load(self, keys, req_context):
        """确认 req_context 的 keys 完成原生 CPU 到 GPU 回迁。"""
        super().complete_load(keys, req_context)
        self.control.emit("loaded", request_id=req_context.req_id, keys=[key.hex() for key in keys])

    def _initiate_promotion(self, tier_idx, key, req_context):
        """从 tier_idx 为 req_context 提升 key，记录实际 CPU 淘汰。"""
        before = set(self.primary_tier._policy.blocks)
        result = super()._initiate_promotion(tier_idx, key, req_context)
        self.control.emit("promotion", request_id=req_context.req_id, key=key.hex(), accepted=result,
                          evicted=[item.hex() for item in before - self.primary_tier._policy.blocks.keys()])
        if not result:
            raise RuntimeError("CPU allocation failed during SSD promotion")
        return result

    def shutdown(self):
        """排空原生三级传输，释放资源并关闭本次审计文件。"""
        try:
            super().shutdown()
        finally:
            self.control.handler.close()


class ManagedTieringSpec(TieringOffloadingSpec):
    """复用原生三级资源创建流程，配置固定的缓存策略类。"""

    def __init__(self, config):
        """从 config 创建三级 spec，并指定可绑定控制器的 ManagedLRU。"""
        super().__init__(config)
        self.eviction_policy = "ManagedLRU"
        self.cache_policy_module_path = __name__

    def get_manager(self):
        """创建原生共享内存与文件层后交由 ManagedTieringManager 管理。"""
        if self._manager is None:
            self._manager = ManagedTieringManager(super().get_manager(), self.extra_config)
        return self._manager


class ManagedConnector(OffloadingConnector):
    """对固定 vLLM 接口执行策略，并在创建回迁任务前应用成本门槛。"""

    def __init__(self, vllm_config, role, kv_cache_config):
        """按 vllm_config、role 和 kv_cache_config 初始化，拒绝未验证的布局。"""
        groups = kv_cache_config.kv_cache_groups
        if (importlib.metadata.version("vllm") != "0.29.0" or len(groups) != 1
                or not envs.VLLM_BATCH_INVARIANT
                or type(groups[0].kv_cache_spec) is not FullAttentionSpec
                or groups[0].kv_cache_spec.block_size != 64
                or str(vllm_config.model_config.dtype) != "torch.float16"
                or vllm_config.parallel_config.world_size != 1
                or vllm_config.scheduler_config.max_num_seqs != 1):
            raise ValueError("Require vLLM 0.29.0, one full-attention group, worker and sequence, 64-token blocks")
        super().__init__(vllm_config, role, kv_cache_config)
        self.control = self.connector_scheduler.manager.control if role == KVConnectorRole.SCHEDULER else None
        self.cost = None
        if self.control is not None:
            native = Path(importlib.import_module("vllm").__file__).parent
            names = ("distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
                     "v1/kv_offload/cpu/manager.py")
            self.context = dict(identity=self.control.identity,
                gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
                packages={name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
                native={name: hashlib.sha256((native / name).read_bytes()).hexdigest() for name in names},
                cpu_blocks=self.control.primary._num_blocks,
                gpu_blocks=kv_cache_config.num_blocks,
                layout="LBHNC", enforce_eager=vllm_config.model_config.enforce_eager,
                batch_invariant=envs.VLLM_BATCH_INVARIANT,
                gpu_memory_utilization=vllm_config.cache_config.gpu_memory_utilization,
                gpu_bytes=vllm_config.kv_transfer_config.kv_connector_extra_config["gpu_bytes"],
                max_model_len=vllm_config.model_config.max_model_len)
            self.context.update(policy=self.control.options["policy"],
                                model_sha256=self.control.predictor.sha256 if self.control.predictor else None,
                                source={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in Path(__file__).parent.glob("*.py")})
            self.control.emit("context", context=self.context)
            if self.connector_scheduler.config.supports_partial_tail:
                raise ValueError("Partial-tail KV state has not been validated")
            options = self.control.options
            if options["restore"] == "gate":
                self.cost = CostPolicy(options["cost_profile"], self.context, options["margin_ms"])

    def on_new_request(self, request):
        """在原生调度前为 request 记录完整前缀块访问。"""
        self.control.begin(request)
        super().on_new_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """按 request 和 num_computed_tokens 查找候选并应用显式成本决定。"""
        candidates, asynchronous = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if candidates is None:
            return candidates, asynchronous
        options = self.control.options
        if self.cost is not None and len(self.control.active) != 1:
            raise ValueError("Calibrated cost gating requires serial requests")
        estimate = self.cost.decide(request.num_prompt_tokens, num_computed_tokens, candidates) if self.cost and candidates else None
        admit = candidates > 0 and (estimate["admit"] if estimate else options["restore"] == "always")
        if candidates and not admit:
            status = self.connector_scheduler._req_status[request.request_id]
            if status.transfer_jobs or status.partial_tail_boundary is not None:
                raise RuntimeError("Unexpected transfer state before cost admission")
            status.update_num_hit_chunks(num_computed_tokens)
        self.control.emit("lookup", request_id=request.request_id,
                          prompt=request.num_prompt_tokens, gpu=num_computed_tokens,
                          external=candidates, admitted=candidates if admit else 0, estimate=estimate)
        return (candidates, asynchronous) if admit else (0, False)

    def request_finished(self, request, block_ids):
        """按原生接口结束 request 和 block_ids，并释放本次策略状态。"""
        result = super().request_finished(request, block_ids)
        self.control.active.pop(request.request_id, None)
        return result

    def request_finished_all_groups(self, request, block_ids):
        """结束 request 的全部 block_ids 组，并释放本次策略状态。"""
        result = super().request_finished_all_groups(request, block_ids)
        self.control.active.pop(request.request_id, None)
        return result
