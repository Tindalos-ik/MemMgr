"""在 vLLM 原生 CPU KV 传输路径上执行 GBDT 写入准入和前缀保留策略。"""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.base import OffloadPolicy, RequestOffloadingContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from reuse_features import History, Predictor


class ReusePolicy(LRUCachePolicy):
    """按前缀复用概率选择淘汰对象，同分沿用 LRU，前缀内部优先淘汰尾块。"""

    def __init__(self, capacity, manager):
        """建立 capacity 块缓存；manager 提供访问历史、所有者和冻结模型。"""
        super().__init__(capacity)
        self.manager = manager

    def plan(self, n, protected, current_root):
        """为 n 个块生成只读淘汰方案，跳过 protected 和 current_root 前缀，返回候选块与评分。"""
        if n == 0:
            return [], []
        manager = self.manager
        groups = {}
        for rank, key in enumerate(self.evictable_blocks):
            root, position = manager.owners[key]
            if key not in protected and root != current_root:
                if self.blocks[key].ref_cnt != 0:
                    raise RuntimeError("Eviction candidate is pinned")
                groups.setdefault(root, []).append((key, position, rank))
        if sum(map(len, groups.values())) < n:
            return None
        roots = list(groups)
        features = [manager.history.features(root) for root in roots]
        scores = manager.predictor.score(features)
        ranked = sorted(zip(roots, scores), key=lambda item: (item[1], min(k[2] for k in groups[item[0]])))
        candidates = [(key, self.blocks[key]) for root, _ in ranked
                      for key, _, _ in sorted(groups[root], key=lambda item: -item[1])][:n]
        details = [dict(root=root, features=vector, score=score,
                        keys=[key.hex() for key, _, _ in groups[root]])
                   for root, vector, score in zip(roots, features, scores)]
        return candidates, details

    def evict(self, n, protected):
        """按只读方案原子淘汰 n 个空闲块，跳过 protected 和本次写入前缀。"""
        started = time.perf_counter()
        manager = self.manager
        plan = self.plan(n, protected, manager.store_root)
        if plan is None:
            return None
        candidates, details = plan
        for key, _ in candidates:
            self.remove(key)
        manager.emit("evict", request_id=manager.store_request, n=n,
                     protected=[key.hex() for key in protected],
                     candidates=details,
                     keys=[key.hex() for key, _ in candidates],
                     policy_seconds=time.perf_counter() - started)
        return candidates


class ReuseManager(CPUOffloadingManager):
    """继承原生容量、引用保护和异步传输管理，增加真实请求采集与写入筛选。"""

    def __init__(self, spec):
        """从 spec 读取容量、模式、模型目录和审计文件路径。"""
        extra = spec.extra_config
        self.mode = extra["reuse_mode"]
        if self.mode not in ("lru", "admission", "learned"):
            raise ValueError("Unknown reuse mode")
        if spec.blocks_per_chunk != 1 or spec.tokens_per_hash != 64:
            raise ValueError("Require one 64-token GPU block per CPU chunk/hash")
        super().__init__(spec.num_blocks, cache_policy="lru")
        self.history = History()
        self.predictor = Predictor(extra["reuse_model"]) if self.mode != "lru" else None
        self.audit = Path(extra["reuse_audit"]).open("x", encoding="utf-8", buffering=1)
        self.active = {}
        self.owners = {}
        self.store_root = None
        self.store_request = None
        if self.mode == "learned":
            self._policy = ReusePolicy(spec.num_blocks, self)
        self.emit("init", mode=self.mode, capacity=spec.num_blocks,
                  bytes_per_block=spec.kv_bytes_per_chunk,
                  model_sha256=self.predictor.sha256 if self.predictor else None)

    def emit(self, kind, **fields):
        """将 kind 类型的 fields 审计记录追加到文件，记录当前请求时钟。"""
        self.audit.write(json.dumps(dict(kind=kind, clock=self.history.clock, **fields)) + "\n")

    def begin(self, request):
        """在 request 的 CPU 查找前观察完整 prompt，保留独立于 CPU 驻留状态的特征。"""
        started = time.perf_counter()
        tokens = request.prompt_token_ids
        if (len(tokens) % 64 or len(tokens) > 3072
                or len(request.block_hashes) != len(tokens) // 64):
            raise ValueError("Require aligned complete prompt hashes, at most 3072 tokens")
        keys = [make_offload_key(value, 0) for value in request.block_hashes]
        root = hashlib.sha256(json.dumps(tokens[:64]).encode()).hexdigest()
        features = self.history.observe(root, len(tokens))
        score = self.predictor.score([features])[0] if self.predictor else None
        accepted = self.predictor is None or score >= self.predictor.threshold
        positions = {key: index for index, key in enumerate(keys)}
        self.active[request.request_id] = dict(root=root, keys=keys, positions=positions,
                                               accepted=accepted, score=score)
        resident = [key.hex() for key in keys
                    if (block := self._policy.get(key)) is not None and block.is_ready]
        self.emit("begin", request_id=request.request_id, root=root,
                  prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                  keys=[key.hex() for key in keys], features=features, score=score,
                  accepted=accepted, resident=resident, history_size=len(self.history.entries),
                  policy_seconds=time.perf_counter() - started)

    def on_new_request(self, req_context):
        """为 req_context 请求提供完整 prompt 的再次写入机会，包括 GPU 已命中的块。"""
        if req_context.req_id not in self.active:
            raise RuntimeError("Request was not observed before offloading")
        return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)

    def prepare_store(self, keys, req_context):
        """依据 req_context 的冻结准入决定筛选 keys，交给原生管理器分配和安排真实写入。"""
        state = self.active[req_context.req_id]
        keys = list(keys)
        if any(key not in state["positions"] for key in keys):
            raise RuntimeError("Decode or unknown block offered for prompt-only store")
        if self.mode == "learned" and "write_accepted" not in state:
            started = time.perf_counter()
            missing = [key for key in state["keys"] if self._policy.get(key) is None]
            required = max(0, len(missing) - self._get_num_free_blocks())
            candidates, details = [], []
            if state["accepted"] and required:
                plan = self._policy.plan(required, set(state["keys"]), state["root"])
                if plan is None:
                    raise RuntimeError("Cannot reserve complete prefix with current pinned blocks")
                candidates, details = plan
            scores = {item["root"]: item["score"] for item in details}
            incoming = len(missing) * state["score"]
            displaced = math.fsum(scores[self.owners[key][0]] for key, _ in candidates)
            state["write_accepted"] = state["accepted"] and (not required or incoming > displaced)
            self.emit("retain", request_id=req_context.req_id, missing=[key.hex() for key in missing],
                      required_blocks=required, candidates=details,
                      victims=[key.hex() for key, _ in candidates],
                      incoming_value=incoming, displaced_value=displaced,
                      accepted=state["write_accepted"],
                      reason="reuse-threshold" if not state["accepted"] else (
                          "free-capacity" if not required else "replacement-value"),
                      policy_seconds=time.perf_counter() - started)
        accepted = state["write_accepted"] if self.mode == "learned" else state["accepted"]
        selected = keys if accepted else []
        self.store_root, self.store_request = state["root"], req_context.req_id
        try:
            output = super().prepare_store(selected, req_context)
        finally:
            self.store_root = self.store_request = None
        if output is None:
            raise RuntimeError("Native CPU allocation failed")
        for key in output.evicted_keys:
            del self.owners[key]
        for key in output.keys_to_store:
            self.owners[key] = (state["root"], state["positions"][key])
        if len(self.owners) > self._num_blocks or len(self.owners) != len(self._policy.blocks):
            raise RuntimeError("CPU block ownership or capacity mismatch")
        self.emit("store", request_id=req_context.req_id,
                  offered=[key.hex() for key in keys], accepted=accepted,
                  keys=[key.hex() for key in output.keys_to_store],
                  evicted=[key.hex() for key in output.evicted_keys], resident_blocks=len(self.owners))
        return output

    def complete_store(self, keys, req_context, success=True):
        """确认 req_context 的 keys 已完成真实写入；传输失败立即退出。"""
        if not success:
            raise RuntimeError("CPU store failed")
        super().complete_store(keys, req_context, success)
        self.emit("stored", request_id=req_context.req_id, keys=[key.hex() for key in keys])

    def prepare_load(self, keys, req_context):
        """为 req_context 锁定 keys，沿用原生 CPU 到 GPU 传输描述。"""
        output = super().prepare_load(keys, req_context)
        self.emit("load", request_id=req_context.req_id, keys=[key.hex() for key in keys])
        return output

    def complete_load(self, keys, req_context):
        """完成 req_context 的 keys 回迁后解除原生引用保护。"""
        super().complete_load(keys, req_context)
        self.emit("loaded", request_id=req_context.req_id, keys=[key.hex() for key in keys])

    def on_request_finished(self, req_context):
        """结束 req_context 的准入状态；异步完成回调继续由原生引用计数管理。"""
        del self.active[req_context.req_id]

    def shutdown(self):
        """关闭本次 manager 的审计文件。"""
        self.emit("end", resident_blocks=len(self.owners), active_requests=len(self.active))
        self.audit.close()


class ReuseSpec(CPUOffloadingSpec):
    """保留原生 GPU worker 和 DMA，仅替换 scheduler 端管理器。"""

    def get_manager(self):
        """为当前 spec 创建并复用唯一 ReuseManager。"""
        if self._manager is None:
            self._manager = ReuseManager(self)
        return self._manager


class ReuseConnector(OffloadingConnector):
    """在原生请求生命周期中采集完整前缀，并审计最终可恢复 token 数。"""

    def __init__(self, vllm_config, role, kv_cache_config):
        """校验 vllm_config、kv_cache_config 的实验边界，初始化 role 对应的原生角色。"""
        groups = kv_cache_config.kv_cache_groups
        if (importlib.metadata.version("vllm") != "0.29.0" or len(groups) != 1
                or type(groups[0].kv_cache_spec) is not FullAttentionSpec
                or groups[0].kv_cache_spec.block_size != 64
                or vllm_config.parallel_config.world_size != 1
                or vllm_config.scheduler_config.max_num_seqs != 1):
            raise ValueError("Require vLLM 0.29.0, one full-attention group and one worker/sequence")
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER and self.connector_scheduler.config.supports_partial_tail:
            raise ValueError("Partial-tail state is outside this experiment")

    def on_new_request(self, request):
        """在原生 scheduler 登记 request 之前采集全量候选。"""
        self.connector_scheduler.manager.begin(request)
        super().on_new_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """沿用 request 的回迁规则，记录 num_computed_tokens 和实际 CPU 连续命中。"""
        tokens, asynchronous = super().get_num_new_matched_tokens(request, num_computed_tokens)
        self.connector_scheduler.manager.emit("lookup", request_id=request.request_id,
                                              gpu_tokens=num_computed_tokens, cpu_tokens=tokens)
        return tokens, asynchronous
