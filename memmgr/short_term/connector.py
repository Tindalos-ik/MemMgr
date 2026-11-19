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
from vllm.v1.kv_offload.base import (
    LookupResult, OffloadPolicy, OffloadingCounterMetadata, RequestOffloadingContext, make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.fs.io import batch_store_block
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import JobMetadata, TieringOffloadingManager
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from .cost import CostPolicy
from .prefix_fit import fit_prefix
from .retention import RetentionProfile, best_prefix, filesystem_device, retention_context
from .reuse import History, Predictor


class StrictAuditHandler(RotatingFileHandler):
    """限制审计日志大小，保留一次轮转，写入错误直接传播。"""

    def handleError(self, record):
        """传播当前 record 写入期间的原始异常。"""
        raise


class ManagedLRU(LRUCachePolicy):
    """保留原生引用保护，为学习策略提供连续前缀的叶块优先淘汰。"""

    def __init__(self, cache_capacity):
        """初始化 cache_capacity 个块，controller 在管理器创建时绑定。"""
        super().__init__(cache_capacity)
        self.controller = None

    def plan(self, n, protected, *, partial=False):
        """为 n 个空位选择最低价值的叶块；protected 不可动，partial 允许返回较短的计划。"""
        if not n:
            return []
        controller = self.controller
        if controller.retention is None:
            controller.refresh_scores(self.blocks)
            scores = controller.scores
        else:
            scores = {key: value for key, (_, _, value) in
                      controller.retention_values(self.evictable_blocks.keys() - protected).items()}
        remaining = set(self.blocks)
        chosen = []
        ranks = {key: rank for rank, key in enumerate(self.evictable_blocks)}
        while len(chosen) < n:
            parents = {controller.parents[key] for key in remaining}
            candidates = [key for key in self.evictable_blocks
                          if key in remaining and key not in protected and key not in parents]
            if not candidates:
                return chosen if partial else None
            key = min(candidates, key=lambda key: (scores[key], ranks[key]))
            if self.blocks[key].ref_cnt != 0:
                raise RuntimeError("Pinned block entered eviction candidates")
            chosen.append((key, self.blocks[key]))
            remaining.remove(key)
        return chosen

    def evict(self, n, protected):
        """原子淘汰 n 个块；protected 保留调用者正在使用的块。"""
        if self.controller.options["policy"] not in {"learned", "time_value"}:
            return super().evict(n, protected | self.controller.protected)
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
        self.retention = self.retention_tier = None
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

    def bind_retention(self, context, tiers):
        """核对 profile 和实际 context；tiers 提供只读的下层文件索引。"""
        if self.options["policy"] != "time_value":
            return
        tier = self.options["tier"]
        device = None
        if tier == "ssd":
            self.retention_tier, = tiers
            device = dict(mount=filesystem_device(self.options["ssd_root"]),
                          capacity=self.options["ssd_bytes"], write=self.options["ssd_write"])
        self.retention = RetentionProfile(self.options["retention_profile"], retention_context(context, tier, device))
        span = (context["max_model_len"] - 1) // self.identity["block_tokens"]
        required = span if tier == "ssd" else min(span, context["cpu_blocks"])
        if self.retention.max_depth < required:
            raise ValueError(f"Retention profile does not cover the configured {required}-block span")
        self.emit("retention_profile", sha256=self.retention.sha256, max_depth=self.retention.max_depth)

    def retention_values(self, keys):
        """估计 keys 的连续复用概率、当前非 CPU 替代路径及其加权时间差。"""
        blocks = self.primary._policy.blocks
        tier = self.retention_tier
        ancestors = set(keys)
        for key in keys:
            parent = self.parents[key]
            while parent in self.history.entries and parent not in ancestors:
                ancestors.add(parent)
                parent = self.parents[parent]
        self.refresh_scores(ancestors)
        probabilities, copies, reach = {}, {}, {None: True}

        def probability(key):
            """给定 key 及其祖先的冻结分数，保证复用概率随前缀加长不升高。"""
            if key not in probabilities:
                parent = self.parents[key]
                upper = 1.0 if parent is None else probability(parent) if parent in ancestors else 0.0
                probabilities[key] = min(self.scores[key], upper)
            return probabilities[key]

        def copied(key):
            """查询 key 是否已被文件层发布为可恢复副本；不触发提升。"""
            if key not in copies:
                copies[key] = tier.lookup(key, None) is LookupResult.HIT
            return copies[key]

        def reachable(key):
            """检查沿 key 到根的 CPU 或 SSD 路径是否连续可读。"""
            if key not in reach:
                cpu = blocks.get(key)
                reach[key] = (key in self.parents and (bool(cpu and cpu.is_ready) or copied(key))
                              and reachable(self.parents[key]))
            return reach[key]

        values = {}
        for key in keys:
            lower = bool(tier and copied(key) and reachable(self.parents[key]))
            p = probability(key)
            depth = self.history.entries[key][4] // self.identity["block_tokens"]
            values[key] = (p, "ssd" if lower else "recompute", p * self.retention.gain(depth, lower))
        return values

    def time_value_select(self, selected, request_id):
        """在容量允许的 selected 中选正净收益最高的前缀，为 request_id 记录计价。"""
        profile = self.retention
        blocks = self.primary._policy.blocks
        missing = [key for key in selected if key not in blocks]
        if not missing:
            return selected
        free = self.primary._get_num_free_blocks()
        victims = [key for key, _ in self.primary._policy.plan(max(0, len(missing) - free), self.protected,
                                                              partial=True)]
        incoming = self.retention_values(missing)
        outgoing = self.retention_values(victims)
        tier = self.retention_tier
        writes = {key: profile.cpu_write + (profile.ssd_write if tier and tier.needs_write(key) else 0.0)
                  for key in missing}
        count, benefit, written, displaced = best_prefix([(incoming[key][2], writes[key]) for key in missing],
                                                         [outgoing[key][2] for key in victims], free)
        stop = selected.index(missing[count]) if count < len(missing) else len(selected)
        self.emit("retain_value", request_id=request_id, proposed=len(missing), written=count,
                  free=free, benefit_seconds=benefit, write_seconds=written,
                  displaced_seconds=displaced, net_seconds=benefit - written - displaced,
                  incoming=[(key.hex(), *incoming[key], writes[key]) for key in missing],
                  victims=[(key.hex(), *outgoing[key]) for key in victims],
                  planned_evictions=max(0, count - free))
        return selected[:stop]

    def observe(self, request_id, keys):
        """为 request_id 观察已排序的 keys 并保存冻结准入；可复用真实审计的键链。"""
        protected = set(self.primary._policy.blocks)
        protected.update(key for state in self.active.values() for key in state["keys"])
        self.history.observe(keys, 64, protected)
        for index, key in enumerate(keys):
            self.parents[key] = keys[index - 1] if index else None
        self.parents = {key: parent for key, parent in self.parents.items() if key in self.history.entries}
        self.scores = {}
        self.refresh_scores(dict.fromkeys([*keys, *self.primary._policy.blocks]))
        admitted = []
        for key in keys:
            if self.predictor and self.scores[key] < self.predictor.threshold:
                break
            admitted.append(key)
        self.active[request_id] = dict(keys=keys, admitted=admitted)
        return admitted

    def begin(self, request):
        """在 request 查找前观察所有完整前缀块，每个请求只更新一次时钟。"""
        started = time.perf_counter()
        keys = [make_offload_key(value, 0) for value in request.block_hashes]
        if len(keys) != request.num_prompt_tokens // 64:
            raise ValueError("Unexpected complete prompt hash count")
        if self.retention is not None and keys:
            fingerprint = hashlib.sha256(json.dumps(request.prompt_token_ids[:64]).encode()).hexdigest()
            if fingerprint in self.retention.calibration_prefixes:
                raise ValueError("Retention workload shares a first block with calibration")
        admitted = self.observe(request.request_id, keys)
        self.emit("begin", request_id=request.request_id, clock=self.history.clock,
                  prompt_tokens=request.num_prompt_tokens, keys=[key.hex() for key in keys],
                  features=[self.history.features(key) for key in keys],
                  prompt_sha256=hashlib.sha256(json.dumps(request.prompt_token_ids).encode()).hexdigest(),
                  admitted=[key.hex() for key in admitted],
                  policy_seconds=time.perf_counter() - started)

    def select(self, keys, req_context):
        """按 req_context 的准入和现有容量筛选本批 keys，保护已驻留的连续前缀。"""
        state = self.active[req_context.req_id]
        offered = set(keys)
        if not offered <= set(state["keys"]):
            raise RuntimeError("Decode or unknown block offered for prompt-only storage")
        learned = self.options["policy"] == "learned"
        admitted = state["admitted"]
        if learned and "selected" in state:
            admitted = admitted[:len(state["selected"])]
        selected, self.protected = fit_prefix(
            self.primary, state["keys"], admitted, offered, self.parents if learned else None)
        if "selected" not in state:
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
        if self.retention is not None:
            selected = self.time_value_select(selected, req_context.req_id)
        return [key for key in selected if key in offered]


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


class _SSDCachePolicy(ManagedLRU):
    """沿用 CPU 策略的牺牲块选择，在槽进入原生空闲列表前通知 SSD 管理器。"""

    def evict(self, n, protected):
        """按 CPU 策略淘汰 n 个非 protected 块，并交接原槽给 SSD。"""
        victims = super().evict(n, protected)
        if victims is None:
            req_context, reason = self.tiering.allocation
            if self.tiering._wait_for_eager_slots(req_context, reason):
                victims = super().evict(n, protected)
        if victims:
            self.tiering.before_reuse(victims)
        return victims


class ManagedFileSystemTier(FileSystemTierManager):
    """同一独占目录内，只发布成功写完的 O_DIRECT 文件。"""

    def __init__(self, offloading_spec, primary_kv_view, tier_type, **options):
        """用 offloading_spec 的配额与 primary_kv_view 初始化 tier_type 文件层；options 指定目录和线程。"""
        root = Path(options["root_dir"])
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.limit = offloading_spec.extra_config["policy_config"]["ssd_bytes"]
        self.jobs, self.pending = {}, set()
        self.failed = False
        self.lease = (root / ".memmgr.lock").open("a")
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            paths = list(root.rglob("*.bin"))
            self.paths = {str(path) for path in paths}
            self.allocated = sum(path.stat().st_size for path in paths)
            if self.allocated > self.limit:
                raise ValueError("Existing SSD cache exceeds configured capacity")
            super().__init__(offloading_spec, primary_kv_view, tier_type, **options)
            if not self._use_o_direct:
                super().shutdown()
                raise RuntimeError("SSD tier requires a filesystem supporting O_DIRECT")
        except BaseException:
            self.lease.close()
            raise

    def bind(self, control, write_mode):
        """为文件层绑定 control，并记录 write_mode 及实际容量。"""
        self.control = control
        control.emit("fs_init", direct_io=True, block_bytes=self._block_size,
                     capacity=self.limit, existing_bytes=self.allocated, ssd_write=write_mode)

    def needs_write(self, key):
        """仅对尚未成功写入且不在途的 key 发起写入。"""
        path = self.file_mapper.get_file_name(key)
        return path not in self.paths and path not in self.pending

    def lookup(self, key, req_context):
        """为 req_context 查询 key；作业成功前不允许用未发布的文件提升。"""
        path = self.file_mapper.get_file_name(key)
        if path in self.pending:
            return LookupResult.RETRY
        return LookupResult.HIT if path in self.paths else LookupResult.MISS

    def check_capacity(self, blocks):
        """检查 blocks 个新文件能否在现有 SSD 配额内预约。"""
        if self.allocated + blocks * self._block_size > self.limit:
            self.failed = True
            raise RuntimeError("SSD KV capacity exhausted")

    def _audit_job(self, kind, job, reason, **fields):
        """记录 kind 的 job、原生块槽和触发 reason，fields 补充完成结果。"""
        self.control.emit(kind, direction="load" if job.is_promotion else "store", job=job.job_id,
                          request_id=job.req_context.req_id, keys=[key.hex() for key in job.keys],
                          slots=[int(slot) for slot in job.block_ids], reason=reason, **fields)

    def _begin_job(self, job, reason):
        """为 job 登记 reason 和源文件写入配额；重复或超额提交立即失败。"""
        paths = [] if job.is_promotion else [self.file_mapper.get_file_name(key) for key in job.keys]
        if paths:
            if len(set(paths)) != len(paths) or any(path in self.paths or path in self.pending for path in paths):
                raise RuntimeError("Duplicate SSD KV store")
            self.check_capacity(len(paths))
            self.allocated += len(paths) * self._block_size
            self.pending.update(paths)
        self.jobs[job.job_id] = (job, reason, paths)
        self._audit_job("fs_submit", job, reason)
        return paths

    def _finish_job(self, result):
        """发布 result 中确认完成的文件并记录结果；失败文件不进入查找索引。"""
        job, reason, paths = self.jobs.pop(result.job_id)
        self.pending.difference_update(paths)
        if result.success:
            self.paths.update(paths)
        else:
            self.failed = True
        self._audit_job("fs_complete", job, reason, success=result.success,
                        transfer_seconds=result.transfer_time)

    def submit_store(self, job_metadata):
        """异步写入 job_metadata，由原生 CPU 引用计数保护源槽。"""
        self._begin_job(job_metadata, "eager")
        super().submit_store(job_metadata)

    def store_before_reuse(self, job, reason):
        """从尚未归还的 CPU 槽写入 job；reason 指明 GPU 写或 SSD 提升。"""
        paths = self._begin_job(job, reason)
        started = time.perf_counter()
        try:
            batch_store_block(paths, self._primary_kv_view,
                              [slot * self._block_size for slot in job.block_ids],
                              self._block_size, self._use_o_direct)
        except Exception:
            self._finish_job(JobResult(job.job_id, False, transfer_time=time.perf_counter() - started))
            raise
        result = JobResult(job.job_id, True, transfer_time=time.perf_counter() - started)
        self._finish_job(result)
        return result

    def submit_load(self, job_metadata):
        """提交 job_metadata 的原生 SSD 到 CPU 读取并记录目标槽。"""
        self._begin_job(job_metadata, "promotion")
        super().submit_load(job_metadata)

    def get_finished_jobs(self):
        """交付真实结果；管理器结算引用与预约后再传播失败。"""
        results = list(super().get_finished_jobs())
        for result in results:
            self._finish_job(result)
        return results

    def shutdown(self):
        """结束文件传输线程后释放目录的独占锁。"""
        try:
            super().shutdown()
        finally:
            self.lease.close()


class ManagedTieringManager(TieringOffloadingManager):
    """保留 CPU 决策与 SSD 提升；以原生引用或同步槽交接保护写盘。"""

    SLOT_WAIT = "vllm:kv_offload_ssd_slot_wait_seconds"

    def __init__(self, native, extra):
        """接管 native 的原生资源；extra 提供固定的 CPU 策略与 SSD 写入模式。"""
        super().__init__(native.primary_tier, native.secondary_tiers)
        if len(self.secondary_tiers) != 1 or not isinstance(self.secondary_tiers[0], ManagedFileSystemTier):
            raise ValueError("Managed SSD requires exactly one managed filesystem tier")
        self.ssd = self.secondary_tiers[0]
        self.control = PolicyController(self.primary_tier, extra)
        self.write_mode = self.control.options.get("ssd_write", "eager")
        self.ssd.bind(self.control, self.write_mode)
        self.primary_tier._policy.tiering = self
        self.allocation = None
        self.evicted = ()
        self.slot_wait = 0.0

    def before_reuse(self, victims):
        """在原生分配器收到 victims 的槽之前，写完尚未入盘的块。"""
        req_context, reason = self.allocation
        self.evicted = victims
        if self.write_mode != "on_evict":
            return
        keys, slots = [], []
        for key, block in victims:
            if block.ref_cnt != 0:
                raise RuntimeError("CPU eviction selected an in-flight KV block")
            if self.ssd.needs_write(key):
                keys.append(key)
                slots.append(block.block_id)
        if not keys:
            return
        started = time.perf_counter()
        job = TransferJob(self._next_job_id(), keys, slots, False, req_context)
        metadata = JobMetadata(job, 0)
        self._metrics.on_job_registered(metadata)
        try:
            result = self.ssd.store_before_reuse(job, reason)
        except Exception:
            self._metrics.on_job_finished(metadata, JobResult(job.job_id, False))
            raise
        self._metrics.on_job_finished(metadata, result)
        self._record_wait(req_context, reason, len(slots), started)

    def _record_wait(self, req_context, reason, slots, started):
        """记录 req_context 此次 reason 从 started 起为保护 slots 个源槽付出的墙钟时间。"""
        seconds = time.perf_counter() - started
        self.slot_wait += seconds
        self.control.emit("ssd_slot_wait", request_id=req_context.req_id, reason=reason,
                          slots=slots, seconds=seconds)

    def _wait_for_eager_slots(self, req_context, reason):
        """当 req_context 的 reason 无法分配时，等待在途提前写并释放对应 CPU 引用。"""
        slots = len(self.ssd.pending)
        if not slots:
            return False
        started = time.perf_counter()
        self.ssd.drain_jobs()
        self._process_finished_jobs()
        self._record_wait(req_context, reason, slots, started)
        return True

    def _process_finished_jobs(self):
        """释放所有已完成作业的 CPU 引用或预约，再传播盘层失败。"""
        super()._process_finished_jobs()
        if self.ssd.failed:
            raise RuntimeError("SSD KV transfer failed")

    def on_new_request(self, req_context, **options):
        """登记 req_context 和 options，使完整 prompt 都有写入机会。"""
        super().on_new_request(req_context, **options)
        return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)

    def prepare_store(self, keys, req_context):
        """按 req_context 的 CPU 准入分配 keys，并审计实际的 CPU 槽。"""
        self._maybe_process_finished_jobs()
        self.allocation = (req_context, "gpu_store")
        try:
            selected = self.control.select(keys, req_context)
            result = super().prepare_store(selected, req_context)
            if result is None and self._wait_for_eager_slots(*self.allocation):
                result = super().prepare_store(selected, req_context)
        finally:
            self.allocation = None
            self.evicted = ()
            self.control.protected = set()
        if result is None:
            raise RuntimeError("CPU KV allocation failed")
        self.control.emit("store", request_id=req_context.req_id,
                          keys=[key.hex() for key in result.keys_to_store],
                          slots=[int(slot) for slot in result.store_spec.block_ids],
                          evicted=[key.hex() for key in result.evicted_keys])
        return result

    def complete_store(self, keys, req_context, success=True):
        """确认 req_context 的 keys 已写进 CPU；仅 eager 在此提交未写过的块。"""
        if not success:
            raise RuntimeError("GPU to CPU transfer failed")
        self.primary_tier.complete_store(keys, req_context)
        self.control.emit("stored", request_id=req_context.req_id, keys=[key.hex() for key in keys])
        if self.write_mode == "eager":
            missing = [key for key in keys if self.ssd.needs_write(key)]
            if missing:
                self.ssd.check_capacity(len(missing))
                self.ssd.submit_store(self.create_store_job(missing, req_context))
        state = self._req_state[req_context.req_id]
        assert state.pending_primary_stores > 0
        state.pending_primary_stores -= 1
        self._maybe_finalize_request(req_context.req_id)

    def complete_load(self, keys, req_context):
        """确认 req_context 的 keys 完成原生 CPU 到 GPU 回迁。"""
        super().complete_load(keys, req_context)
        self.control.emit("loaded", request_id=req_context.req_id, keys=[key.hex() for key in keys])

    def _maybe_finalize_request(self, req_id, exclude_tier_idx=None):
        """为 req_id 确认最后一批 SSD 任务已提交；exclude_tier_idx 沿用原生排除规则。"""
        super()._maybe_finalize_request(req_id, exclude_tier_idx)
        if req_id not in self._req_state:
            self.control.emit("ssd_request_closed", request_id=req_id)

    def _initiate_promotion(self, tier_idx, key, req_context):
        """从 tier_idx 为 req_context 提升 key，确保写完牺牲块才预约目标槽。"""
        self.allocation = (req_context, "promotion")
        try:
            if not self.primary_tier._get_num_free_blocks() and not self.primary_tier._num_evictable_cache_blocks:
                self._wait_for_eager_slots(*self.allocation)
            if not super()._initiate_promotion(tier_idx, key, req_context):
                raise RuntimeError("CPU allocation failed during SSD promotion")
            pending = self._pending_load_submissions[tier_idx][req_context.req_id]
            self.control.emit("promotion", request_id=req_context.req_id, key=key.hex(), accepted=True,
                              slot=int(pending.block_ids[-1]),
                              evicted=[victim.hex() for victim, _ in self.evicted])
            return True
        finally:
            self.allocation = None
            self.evicted = ()

    def get_stats(self):
        """在原生指标中累加实际的 CPU 槽分配等待时间。"""
        stats = super().get_stats()
        stats.increase_counter(self.SLOT_WAIT, self.slot_wait)
        self.slot_wait = 0.0
        return stats

    def shutdown(self):
        """先提交并排空盘层作业，处理结果后释放 mmap、目录锁和审计文件。"""
        try:
            try:
                self._flush_pending_promotions()
                self.ssd.drain_jobs()
                self._process_finished_jobs()
                self._metrics.assert_idle()
                assert not (self._jobs or self.ssd.jobs or self.ssd.pending)
                self.control.emit("ssd_shutdown", success=True)
            finally:
                super().shutdown()
        finally:
            self.control.handler.close()


class ManagedTieringSpec(TieringOffloadingSpec):
    """复用原生三级资源创建流程，配置固定的缓存策略类。"""

    def __init__(self, config):
        """从 config 创建只在 SSD 层接管原生槽交接的三级 spec。"""
        super().__init__(config)
        self.eviction_policy = "_SSDCachePolicy"
        self.cache_policy_module_path = __name__
        options = self.extra_config["policy_config"]
        if options.get("ssd_write", "eager") not in ("eager", "on_evict") or options["restore"] != "always":
            raise ValueError("Invalid managed SSD write or restore policy")

    @classmethod
    def build_metric_definitions(cls, extra_config):
        """扩充 extra_config 的原生指标定义，公开源槽等待累计秒数。"""
        metrics = super().build_metric_definitions(extra_config)
        metrics[ManagedTieringManager.SLOT_WAIT] = OffloadingCounterMetadata(
            documentation="Scheduler wall time waiting to safely reuse CPU KV slots for SSD offload, in seconds."
        )
        return metrics

    def get_manager(self):
        """创建原生共享内存与文件层后交由 ManagedTieringManager 管理。"""
        if self._manager is None:
            native = super().get_manager()
            try:
                self._manager = ManagedTieringManager(native, self.extra_config)
            except BaseException:
                self._manager = None
                native.shutdown()
                raise
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
            self.control.bind_retention(self.context, getattr(self.connector_scheduler.manager, "secondary_tiers", ()))
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
