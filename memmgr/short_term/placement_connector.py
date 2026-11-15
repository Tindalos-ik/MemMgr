"""在原生传输、引用计数和块池上执行有界 SSD 缓存与联合放置。"""

from collections import OrderedDict
from pathlib import Path
import os
import time

from vllm.v1.kv_offload.base import LookupResult, OffloadPolicy, RequestOffloadingContext, make_offload_key

from .connector import (ManagedConnector, ManagedFileSystemTier, ManagedTieringManager,
                        ManagedTieringSpec, PolicyController)
from .placement import PlacementCosts, PlacementDecision, cost_context


class BoundedFileSystemTier(ManagedFileSystemTier):
    """用独占目录的内存索引执行真实文件查找、容量淘汰及原生 O_DIRECT 传输。"""

    def __init__(self, *args, **kwargs):
        """按 args/kwargs 创建原生文件层，并核对当前命名空间的完整块文件。"""
        super().__init__(*args, **kwargs)
        self.files = OrderedDict()
        directory = Path(f"{self.file_mapper.base_path}_r{self.file_mapper.rank}")
        for path in sorted(directory.rglob("*.bin"), key=lambda item: (item.stat().st_mtime_ns, str(item))):
            key = make_offload_key(bytes.fromhex(path.stem), int(path.parent.name.split("_g")[1]))
            if str(path) != self.file_mapper.get_file_name(key) or path.stat().st_size != self._block_size:
                raise ValueError("Invalid existing SSD block identity or size")
            self.files[key] = str(path)
        if set(self.files.values()) != self.paths:
            raise ValueError("SSD root contains blocks outside the selected model namespace")

    def bind(self, control):
        """绑定 control；联合策略从空缓存开始，保证所有驻留块都有访问历史。"""
        super().bind(control)
        self.control.ssd = self
        if control.options["policy"] == "placement" and self.files:
            raise ValueError("Placement requires an empty SSD cache at session start")
        if (control.primary._num_blocks + self.limit // self._block_size
                + self._offloading_spec.extra_config["max_model_len"] // 64 > 4096):
            raise ValueError("Tier capacities exceed the bounded prefix history")

    def on_new_request(self, req_context):
        """为 req_context 请求已驻留 CPU 块的放置机会。"""
        policy = OffloadPolicy.REQUEST_LEVEL if self.control.options["policy"] == "placement" else OffloadPolicy.BLOCK_LEVEL
        return RequestOffloadingContext(policy=policy)

    def lookup(self, key, req_context):
        """按独占索引查找 key；req_context 保留原生请求与传输语义。"""
        if key in self.files:
            self.files.move_to_end(key)
            return LookupResult.HIT
        if any(key in job.keys for job in self.jobs.values() if not job.is_promotion):
            return LookupResult.RETRY
        return LookupResult.MISS

    def touch(self, keys, req_context):
        """将 req_context 实际访问的 keys 移到 SSD LRU 队尾。"""
        for key in reversed(list(keys)):
            if key in self.files:
                self.files.move_to_end(key)

    def plan(self, keys, protected):
        """为待写入 keys 只读规划 SSD 淘汰，排除 protected 和在途块。"""
        incoming = set(keys)
        reserved = {key for job in self.jobs.values() for key in job.keys}
        missing = incoming - self.files.keys() - reserved
        required = max(0, (self.allocated + len(missing) * self._block_size - self.limit + self._block_size - 1)
                       // self._block_size)
        remaining, victims = set(self.files), []
        while len(victims) < required:
            candidates = [key for key in self.files if key in remaining and key not in protected
                          and key not in incoming and key not in reserved]
            if self.control.options["policy"] == "placement":
                parents = {self.control.parents[key] for key in remaining}
                candidates = [key for key in candidates if key not in parents]
                self.control.refresh_scores(candidates)
                candidates.sort(key=lambda key: self.control.storage_value(key))
            if not candidates:
                return None
            victims.append(candidates[0])
            remaining.remove(candidates[0])
        return victims

    def submit_store(self, job_metadata):
        """为 job_metadata 原子检查空间，删除已规划的空闲文件，再提交原生写入。"""
        protected = {key for state in self.control.active.values() for key in state["keys"]}
        victims = self.plan(job_metadata.keys, protected)
        if victims is None:
            raise RuntimeError("SSD cache has no unpinned capacity for the selected placement")
        for key in victims:
            path = self.files.pop(key)
            Path(path).unlink()
            self.paths.remove(path)
            self.allocated -= self._block_size
        self.control.emit("fs_evict", keys=[key.hex() for key in victims], allocated=self.allocated)
        self.control.emit("fs_reserve", job=job_metadata.job_id,
                          keys=[key.hex() for key in job_metadata.keys if self.file_mapper.get_file_name(key) not in self.paths])
        super().submit_store(job_metadata)

    def get_finished_jobs(self):
        """确认真实作业完成后更新文件索引，保留失败即终止的原生边界。"""
        pending = dict(self.jobs)
        result = super().get_finished_jobs()
        for completed in result:
            job = pending[completed.job_id]
            if not job.is_promotion:
                for key in job.keys:
                    self.files[key] = self.file_mapper.get_file_name(key)
        return result


class PlacementController(PolicyController):
    """用冻结路径成本和复用概率选择 CPU、SSD 或丢弃。"""

    def __init__(self, primary, extra):
        """绑定 primary/extra，区分 CPU 保留块与尚未释放的 SSD 中转块。"""
        super().__init__(primary, extra)
        self.transient_keys = set()
        self.delays = {}
        if self.options["policy"] == "placement" and self.predictor.delay_model is None:
            raise ValueError("Placement requires the frozen reuse interval model")

    def refresh_scores(self, keys):
        """为 keys 批量更新复用概率及条件下次复用间隔。"""
        super().refresh_scores(keys)
        if self.options["policy"] == "placement":
            missing = [key for key in keys if key not in self.delays]
            self.delays.update(zip(missing, self.predictor.delay([self.history.features(key) for key in missing])))

    def release_cpu(self, keys, reason):
        """释放 keys 的空闲原生 CPU 槽位，并记录 reason；任何在途引用均禁止释放。"""
        for key in keys:
            block = self.primary._policy.get(key)
            if block is None or block.ref_cnt != 0:
                raise RuntimeError("Cannot release an absent or pinned CPU block")
            self.primary._policy.remove(key)
            self.primary._num_evictable_cache_blocks -= 1
            self.primary._free_block(block)
            self.transient_keys.discard(key)
        if keys:
            self.emit("cpu_release", keys=[key.hex() for key in keys], reason=reason)

    def resident_keys(self):
        """保护 CPU 与 SSD 驻留块的访问统计。"""
        return super().resident_keys() | set(self.ssd.files)

    def begin(self, request):
        """观察 request；联合策略要求串行，拒绝未经校准的排队状态。"""
        if self.options["policy"] == "placement" and any(not state.get("finished") for state in self.active.values()):
            raise ValueError("Placement currently requires serial requests")
        self.delays = {}
        super().begin(request)

    def storage_value(self, key):
        """返回 SSD 中 key 的预计恢复收益密度，写入成本已经发生。"""
        return max(0.0, self.scores[key] * (self.costs.unit("compute")
                   - self.costs.unit("cpu_read") - self.costs.unit("ssd_read"))) / self.delays[key]

    def eviction_value(self, key):
        """返回 CPU 中 key 的预计收益；已有 SSD 副本时只计算速度差。"""
        if self.options["policy"] != "placement":
            return super().eviction_value(key)
        alternative = (self.costs.unit("cpu_read") + self.costs.unit("ssd_read")
                       if key in self.ssd.files else self.costs.unit("compute"))
        return max(0.0, self.scores[key] * (alternative - self.costs.unit("cpu_read"))) / self.delays[key]

    def select(self, keys, req_context):
        """为 req_context 冻结一次完整前缀的放置，再筛选当前批次 keys。"""
        if self.options["policy"] != "placement":
            return super().select(keys, req_context)
        state = self.active[req_context.req_id]
        if not set(keys) <= set(state["keys"]):
            raise RuntimeError("Unknown or decode block offered for placement")
        self.protected = set(state["keys"])
        if "selected" not in state:
            started = time.perf_counter()
            if len(state["keys"]) > self.costs.max_blocks:
                raise ValueError("Prompt exceeds the calibrated placement range")
            resident = self.primary._policy.blocks
            retained = set(resident) - self.transient_keys
            missing = [key for key in state["keys"] if key not in retained]
            self.refresh_scores(list(dict.fromkeys([*missing, *resident, *self.ssd.files])))
            positions = [index + 1 for index, key in enumerate(state["keys"]) if key in missing]
            probabilities = [self.scores[key] for key in missing]
            interval = (sum(self.delays[key] for key in missing) / len(missing)) if missing else 0.0
            # 留出一个最大前缀的真实 CPU 槽位，供 SSD 中转使用。
            retention = self.primary._num_blocks - self.costs.max_blocks
            required = max(0, len(retained) + len(missing) - retention)
            cpu_victims = self.primary._policy.plan(required, self.protected)
            ssd_victims = self.ssd.plan(missing, self.protected)
            physical_missing = [key for key in missing if key not in resident]
            ssd_missing = [key for key in physical_missing if key not in self.ssd.files]
            staging_required = max(0, len(ssd_missing) - self.primary._get_num_free_blocks())
            staging_victims = self.primary._policy.plan(staging_required, self.protected)
            candidates = {"drop": 0.0}
            raw = {}
            if cpu_victims is not None and retention > 0:
                raw["cpu"] = self.costs.gain(probabilities, positions, "cpu", len(physical_missing))
                candidates["cpu"] = raw["cpu"] - interval * sum(self.eviction_value(key) for key, _ in cpu_victims)
            if ssd_victims is not None and len(missing) <= self.primary._num_blocks:
                writes = len(set(missing) - self.ssd.files.keys())
                raw["ssd"] = self.costs.gain(probabilities, positions, "ssd", len(ssd_missing), writes)
                # SSD 写入同样占用 CPU 中转空间，计入被挤出块的损失。
                if staging_victims is not None:
                    candidates["ssd"] = (raw["ssd"] - interval * sum(self.storage_value(key) for key in ssd_victims)
                                         - interval * sum(self.eviction_value(key) for key, _ in staging_victims))
            target = max(candidates, key=candidates.get)
            if candidates[target] <= self.options["margin_ms"] / 1000:
                target = "drop"
            targets = {key: ("cpu" if key in retained else "ssd" if target == "drop" and key in self.ssd.files else target)
                       for key in state["keys"]}
            if target == "cpu":
                self.release_cpu([key for key, _ in cpu_victims], "placement-replacement")
            decision = PlacementDecision(targets=targets)
            req_context.set_state(decision)
            state["selected"] = {key for key, destination in targets.items()
                                 if destination == "cpu" or destination == "ssd" and key not in self.ssd.files}
            self.emit("placement", request_id=req_context.req_id, target=target,
                      targets={key.hex(): value for key, value in targets.items()},
                      probabilities={key.hex(): self.scores[key] for key in state["keys"]},
                      reuse_intervals={key.hex(): self.delays[key] for key in state["keys"]}, incoming_interval=interval,
                      candidates_seconds=candidates, raw_seconds=raw,
                      missing=[key.hex() for key in missing], positions=positions,
                      cpu_writes=len(physical_missing), ssd_staging_writes=len(ssd_missing), ssd_writes=len(set(missing) - self.ssd.files.keys()),
                      cpu_loss=interval * sum(self.eviction_value(key) for key, _ in cpu_victims) if cpu_victims is not None else None,
                      staging_loss=interval * sum(self.eviction_value(key) for key, _ in staging_victims) if staging_victims is not None else None,
                      ssd_loss=interval * sum(self.storage_value(key) for key in ssd_victims) if ssd_victims is not None else None,
                      cpu_victims=[key.hex() for key, _ in cpu_victims] if cpu_victims is not None else None,
                      ssd_victims=[key.hex() for key in ssd_victims] if ssd_victims is not None else None,
                      cpu_retention_blocks=retention,
                      policy_seconds=time.perf_counter() - started)
        return [key for key in keys if key in state["selected"]]


class PlacementTieringManager(ManagedTieringManager):
    """沿用原生任务和引用保护，只为 SSD 目标级联，并安全释放中转块。"""

    controller_type = PlacementController

    def __init__(self, native, extra):
        """从 native/extra 创建管理器，记录尚未释放的 SSD 中转块。"""
        super().__init__(native, extra)
        self.staging = {}
        self.snapshot_clock = -2

    def prepare_store(self, keys, req_context):
        """先回收完成作业，再按 req_context 的决定准备当前 keys。"""
        self._maybe_process_finished_jobs()
        result = super().prepare_store(keys, req_context)
        self.control.transient_keys.difference_update(result.evicted_keys)
        decision = req_context.get_state(PlacementDecision)
        if decision is not None:
            for key, target in decision.targets.items():
                if target == "ssd" and key in self.primary_tier._policy.blocks:
                    self.staging[key] = decision
                    self.control.transient_keys.add(key)
                elif target == "cpu":
                    self.staging.pop(key, None)
                    self.control.transient_keys.discard(key)
        return result

    def _initiate_promotion(self, tier_idx, key, req_context):
        """为 req_context 提升 tier_idx 中的 key，先标为 CPU 中转块。"""
        result = super()._initiate_promotion(tier_idx, key, req_context)
        if self.control.options["policy"] == "placement" and result:
            self.control.transient_keys.add(key)
        return result

    def _ssd_keys(self, keys, req_context):
        """筛选 req_context 中尚无完成或在途 SSD 副本的 keys。"""
        decision = req_context.get_state(PlacementDecision)
        busy = {key for job in self.control.ssd.jobs.values() if not job.is_promotion for key in job.keys}
        return [key for key in keys if decision.targets[key] == "ssd"
                and key not in self.control.ssd.files and key not in busy]

    def prepare_load(self, keys, req_context):
        """为 req_context 钉住真实 keys，并记录 GPU 回迁开始。"""
        result = super().prepare_load(keys, req_context)
        self.control.emit("load", request_id=req_context.req_id, keys=[key.hex() for key in keys])
        return result

    def complete_store(self, keys, req_context, success=True):
        """完成 keys 的 GPU 写入；req_context 决定 CPU 保留或真实 SSD 级联。"""
        if self.control.options["policy"] != "placement":
            return super().complete_store(keys, req_context, success)
        if not success:
            raise RuntimeError("GPU to CPU placement transfer failed")
        self.primary_tier.complete_store(keys, req_context, success)
        self.control.emit("stored", request_id=req_context.req_id, keys=[key.hex() for key in keys])
        selected = self._ssd_keys(keys, req_context)
        if selected:
            self.control.ssd.submit_store(self.create_store_job(selected, req_context, 0))
        state = self._req_state[req_context.req_id]
        if state.pending_primary_stores <= 0:
            raise RuntimeError("Unbalanced primary placement store completion")
        state.pending_primary_stores -= 1
        self._maybe_finalize_request(req_context.req_id)

    def _cascade_existing_blocks_to_request_level_tiers(self, keys, req_context, request_level_tiers):
        """仅将 req_context 选择的 SSD keys 交给原生 request-level 级联。"""
        selected = self._ssd_keys(keys, req_context)
        if selected:
            super()._cascade_existing_blocks_to_request_level_tiers(selected, req_context, request_level_tiers)

    def on_request_finished(self, req_context, **options):
        """结束 req_context，延迟释放仍有真实传输引用的中转块。"""
        decision = req_context.get_state(PlacementDecision)
        if decision is not None:
            decision.finished = True
        super().on_request_finished(req_context, **options)
        self.control.active.pop(req_context.req_id, None)
        self.control.emit("request_done", request_id=req_context.req_id)
        self._settle()

    def has_pending_work(self):
        """保留最后一次真实结束调度与 SSD 中转清理所需的引擎步。"""
        return (super().has_pending_work() or bool(self.staging)
                or any(state.get("finished") for state in self.control.active.values()))

    def on_schedule_end(self, context):
        """完成 context 的原生调度后，仅释放已写入 SSD 且无引用的中转块。"""
        super().on_schedule_end(context)
        self._settle()

    def _process_finished_jobs(self):
        """消费原生完成队列后释放真实中转槽位，覆盖最后一个请求的空闲阶段。"""
        super()._process_finished_jobs()
        self._settle()

    def get_stats(self):
        """发布原生统计前记录已经完成传输的驻留状态。"""
        self._settle()
        return super().get_stats()

    def _settle(self):
        """仅在引用已释放时归还中转槽位，并为真实空闲状态记录一次快照。"""
        for key, decision in list(self.staging.items()):
            block = self.primary_tier._policy.get(key)
            if block is None:
                del self.staging[key]
                self.control.transient_keys.discard(key)
            elif decision.finished and block.ref_cnt == 0 and key in self.control.ssd.files:
                self.control.release_cpu([key], "ssd-staging-complete")
                del self.staging[key]
        if (not self._jobs and not self.control.active and not self.staging
                and all(block.ref_cnt == 0 for block in self.primary_tier._policy.blocks.values())
                and self.snapshot_clock != self.control.history.clock):
            self.snapshot_clock = self.control.history.clock
            self.control.emit("residency", clock=self.snapshot_clock,
                              cpu=[key.hex() for key in self.primary_tier._policy.blocks],
                              ssd=[key.hex() for key in self.control.ssd.files],
                              allocated=self.control.ssd.allocated,
                              cpu_free=self.primary_tier._get_num_free_blocks())


class PlacementTieringSpec(ManagedTieringSpec):
    """创建同容量 LRU、预测策略与联合策略共用的原生三级资源。"""

    def get_manager(self):
        """为当前 spec 只创建一次带有界文件层的放置管理器。"""
        if self._manager is None:
            from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

            self._manager = PlacementTieringManager(TieringOffloadingSpec.get_manager(self), self.extra_config)
        return self._manager


class PlacementConnector(ManagedConnector):
    """核对真实三级资源后加载冻结成本，供原生 scheduler 执行联合放置。"""

    def __init__(self, *args, **kwargs):
        """将 args/kwargs 传给现有 connector，并为 scheduler 核对 SSD 与成本身份。"""
        super().__init__(*args, **kwargs)
        if self.control is not None:
            tier = self.control.ssd
            self.context.update(ssd_bytes=tier.limit, block_bytes=tier._block_size,
                                filesystem_device=os.stat(tier.file_mapper.base_path).st_dev)
            self.control.emit("placement_context", context=cost_context(self.context))
            if self.control.options["policy"] == "placement":
                self.control.costs = PlacementCosts(self.control.options["placement_profile"], self.context)
                self.control.emit("placement_costs", sha256=self.control.costs.sha256)

    def request_finished(self, request, block_ids):
        """结束 request/block_ids；冻结状态保留到原生最后一批写入已提交。"""
        result = self.connector_scheduler.request_finished(request)
        state = self.control.active.get(request.request_id)
        if state is not None:
            state["finished"] = True
        context = self.connector_scheduler._req_status.get(request.request_id)
        decision = context.req_context.get_state(PlacementDecision) if context is not None else None
        if decision is not None:
            decision.finished = True
        return result

    def request_finished_all_groups(self, request, block_ids):
        """按原生语义结束 request 的全部 block_ids 组。"""
        return self.request_finished(request, block_ids)
