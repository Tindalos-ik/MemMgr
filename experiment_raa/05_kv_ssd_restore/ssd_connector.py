"""审计 vLLM 原生三级 KV 传输，严格检查 SSD I/O 与完成状态。"""

import importlib.metadata
import json
from pathlib import Path
import time

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def emit(path, kind, **fields):
    """将 kind 和 fields 作为单个完整事件追加至 path。"""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(kind=kind, time_ns=time.monotonic_ns(), **fields)) + "\n")


def keys_hex(keys):
    """返回 keys 中原生 OffloadKey 的十六进制表示。"""
    return [key.hex() for key in keys]


class StrictFileSystemTier(FileSystemTierManager):
    """沿用原生文件布局与 O_DIRECT I/O，传输失败立即报错。"""

    def __init__(self, offloading_spec, primary_kv_view, tier_type, **options):
        """用 offloading_spec、primary_kv_view、tier_type 和 options 创建文件层。"""
        super().__init__(offloading_spec, primary_kv_view, tier_type, **options)
        self.audit = offloading_spec.extra_config["audit_path"]
        self.jobs = {}
        if not self._use_o_direct:
            self.shutdown()
            raise RuntimeError("SSD experiments require O_DIRECT on the selected filesystem")
        emit(self.audit, "fs_init", direct_io=True, block_bytes=self._block_size,
             root_dir=options["root_dir"], run_config=self.file_mapper.get_run_config())

    def submit_store(self, job_metadata):
        """提交 job_metadata 的 CPU 到 SSD 写入并记录块身份。"""
        self.jobs[job_metadata.job_id] = job_metadata
        emit(self.audit, "fs_submit", direction="store", job=job_metadata.job_id,
             request_id=job_metadata.req_context.req_id, keys=keys_hex(job_metadata.keys))
        super().submit_store(job_metadata)

    def submit_load(self, job_metadata):
        """提交 job_metadata 的 SSD 到 CPU 读取并记录块身份。"""
        self.jobs[job_metadata.job_id] = job_metadata
        emit(self.audit, "fs_submit", direction="load", job=job_metadata.job_id,
             request_id=job_metadata.req_context.req_id, keys=keys_hex(job_metadata.keys))
        super().submit_load(job_metadata)

    def get_finished_jobs(self):
        """返回已完成真实传输；失败立即终止，保留失败事件。"""
        results = list(super().get_finished_jobs())
        for result in results:
            job = self.jobs.pop(result.job_id)
            emit(self.audit, "fs_complete", direction="load" if job.is_promotion else "store",
                 job=result.job_id, request_id=job.req_context.req_id,
                 keys=keys_hex(job.keys), success=result.success,
                 transfer_seconds=result.transfer_time)
            if not result.success:
                raise RuntimeError(f"SSD transfer failed: job {result.job_id}")
        return results


class AuditTieringManager(TieringOffloadingManager):
    """在原生管理器边界记录 CPU 淘汰、SSD 提升和 GPU 回迁。"""

    def __init__(self, primary_tier, secondary_tiers, audit):
        """复用 primary_tier、secondary_tiers；audit 为事件输出路径。"""
        super().__init__(primary_tier, secondary_tiers)
        self.audit = audit
        emit(audit, "manager_init", capacity=primary_tier._num_blocks)

    def prepare_store(self, keys, req_context):
        """为 req_context 的 keys 安排原生写入并记录淘汰块。"""
        result = super().prepare_store(keys, req_context)
        if result is None:
            raise RuntimeError("CPU allocation failed during GPU store")
        emit(self.audit, "cpu_store", request_id=req_context.req_id,
             keys=keys_hex(result.keys_to_store), evicted=keys_hex(result.evicted_keys))
        return result

    def _initiate_promotion(self, tier_idx, key, req_context):
        """为 req_context 从 tier_idx 提升 key，记录提升时释放的 CPU 块。"""
        before = set(self.primary_tier._policy.blocks)
        result = super()._initiate_promotion(tier_idx, key, req_context)
        emit(self.audit, "promotion", request_id=req_context.req_id, key=key.hex(),
             accepted=result, evicted=keys_hex(before - self.primary_tier._policy.blocks.keys()))
        if not result:
            raise RuntimeError("CPU allocation failed during SSD promotion")
        return result

    def complete_store(self, keys, req_context, success=True):
        """确认 req_context 的 keys 已从 GPU 写入 CPU，失败立即报错。"""
        if not success:
            raise RuntimeError("GPU to CPU transfer failed")
        emit(self.audit, "cpu_stored", request_id=req_context.req_id, keys=keys_hex(keys))
        super().complete_store(keys, req_context, success)

    def prepare_load(self, keys, req_context):
        """为 req_context 锁定 keys，交给原生 CPU 到 GPU 传输。"""
        result = super().prepare_load(keys, req_context)
        emit(self.audit, "gpu_load", request_id=req_context.req_id, keys=keys_hex(keys))
        return result

    def complete_load(self, keys, req_context):
        """确认 req_context 的 keys 完成 CPU 到 GPU 回迁。"""
        super().complete_load(keys, req_context)
        emit(self.audit, "gpu_loaded", request_id=req_context.req_id, keys=keys_hex(keys))


class AuditTieringSpec(TieringOffloadingSpec):
    """沿用原生共享内存和 worker，创建带事件记录的 scheduler 管理器。"""

    def get_manager(self):
        """首次创建原生资源后交由 AuditTieringManager 管理。"""
        if self._manager is None:
            native = super().get_manager()
            self._manager = AuditTieringManager(native.primary_tier, native.secondary_tiers,
                                                self.extra_config["audit_path"])
        return self._manager


class AuditConnector(OffloadingConnector):
    """固定实验版本，使用原生 OffloadingConnector 的实际调度和 DMA。"""

    def __init__(self, vllm_config, role, kv_cache_config):
        """用 vllm_config、role、kv_cache_config 初始化并检查版本。"""
        if importlib.metadata.version("vllm") != "0.29.0":
            raise ValueError("Require vLLM 0.29.0")
        super().__init__(vllm_config, role, kv_cache_config)
