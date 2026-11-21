"""在支持 O_DIRECT 的 WSL/ Linux 文件系统上验证原生 CPU 槽与 SSD 作业。"""

import errno
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from memmgr.short_term.connector import ManagedTieringSpec
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig, OffloadingConfig, OffloadingGroupConfig, OffloadingModelConfig, OffloadingParallelConfig,
)


class SSDWritebackTest(unittest.TestCase):
    """使用实际共享 mmap 和文件读写；无需启动模型服务。"""

    def setUp(self):
        """在项目 SSD 文件系统上分配本次测试独占的缓存目录及块数据。"""
        agent = Path(".agent").resolve()
        agent.mkdir(exist_ok=True)
        temporary = TemporaryDirectory(dir=agent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.keys = tuple(make_offload_key(hashlib.sha256(bytes([i])).digest(), 0) for i in range(5))
        self.payload = {key: os.urandom(4096) for key in self.keys}

    def manager(self, mode, slots=2):
        """创建 mode 对应的真实管理器，CPU 仅分配 slots 个 4 KiB 槽。"""
        extra = dict(cpu_bytes_to_use=slots * 4096, max_model_len=64, model_revision="native-test", dtype="float16",
                     policy_config=dict(policy="lru", ssd_write=mode, ssd_bytes=2**20, restore="always",
                                        audit_path=str(self.root / "audit.jsonl")),
                     secondary_tiers=[dict(type="ManagedFileSystemTier", module_path="memmgr.short_term.connector",
                                           root_dir=str(self.root / "ssd"), n_read_threads=1, n_write_threads=1)])
        spec = ManagedTieringSpec(OffloadingConfig(
            (OffloadingGroupConfig(64, ("layer",)),), 4096, False, extra, uuid4().hex,
            OffloadingModelConfig("native-test", "float16"), OffloadingCacheConfig(64, 1),
            OffloadingParallelConfig(0, 1, 1, 1, 1, 1, 0, 1, None, True)))
        self.assertEqual(spec.num_blocks, slots)
        return spec.get_manager()

    def begin(self, manager, key):
        """为 manager 登记含 key 的原生卸载请求。"""
        request = ReqContext(uuid4().hex)
        manager.on_new_request(request)
        manager.control.observe(request.req_id, [key])
        return request

    def put(self, manager, key):
        """通过 manager 预约 key 的槽，直接写入共享 mmap，再回报完成。"""
        request = self.begin(manager, key)
        result = manager.prepare_store([key], request)
        self.assertEqual(result.keys_to_store, [key])
        slot = int(result.store_spec.block_ids[0])
        self.slot(manager, slot)[:] = self.payload[key]
        manager.complete_store([key], request)
        manager.on_request_finished(request)
        return slot, result.evicted_keys

    def slot(self, manager, slot):
        """返回 manager 的 slot 对应的原生 mmap 字节视图。"""
        return manager.primary_tier.get_kv_memoryview().cast("B")[slot * 4096:(slot + 1) * 4096]

    def events(self):
        """读取本次真实 SSD 运行的审计记录。"""
        with (self.root / "audit.jsonl").open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]

    def test_eviction_writes_original_slot_and_promotion_evicts(self):
        """验证首次淘汰与提升淘汰的盘面字节、并行 CPU 引用、去重及关闭排空。"""
        a, b, c, d, e = self.keys
        manager = self.manager("on_evict")
        try:
            self.put(manager, a)
            slot_b, _ = self.put(manager, b)
            self.assertFalse(manager.ssd.paths)
            pin = self.begin(manager, a)
            manager.prepare_load([a], pin)
            self.assertEqual(self.put(manager, c), (slot_b, [b]))
            self.assertEqual(Path(manager.ssd.file_mapper.get_file_name(b)).read_bytes(), self.payload[b])
            promote = self.begin(manager, b)
            self.assertEqual(manager.lookup(b, promote), LookupResult.HIT_PENDING)
            self.assertEqual(manager.primary_tier._policy.blocks[a].ref_cnt, 1)
            self.assertEqual(Path(manager.ssd.file_mapper.get_file_name(c)).read_bytes(), self.payload[c])
            manager._flush_pending_promotions()
            manager.ssd.drain_jobs()
            manager._process_finished_jobs()
            restored = manager.prepare_load([b], promote)
            self.assertEqual(int(restored.block_ids[0]), slot_b)
            self.assertEqual(self.slot(manager, slot_b), self.payload[b])
            manager.complete_load([b], promote)
            manager.on_request_finished(promote)
            manager.complete_load([a], pin)
            manager.on_request_finished(pin)
            self.assertGreater(manager.slot_wait, 0.0)
            self.put(manager, d)
            self.put(manager, e)
            late = self.begin(manager, b)
            self.assertEqual(manager.lookup(b, late), LookupResult.HIT_PENDING)
            self.assertTrue(manager._pending_load_submissions)
        finally:
            manager.shutdown()
        events = self.events()
        self.assertEqual([event["kind"] for event in events[-2:]], ["fs_complete", "ssd_shutdown"])
        self.assertEqual(events[-2]["direction"], "load")
        self.assertEqual(events[-2]["keys"], [b.hex()])
        writes = [key for event in events if event["kind"] == "fs_complete" and event["direction"] == "store"
                  for key in event["keys"]]
        self.assertEqual(writes.count(b.hex()), 1)

    def test_eager_write_releases_only_its_own_reference(self):
        """同槽被 CPU→GPU 和 SSD 同时读取时，SSD 完成仅释放自己的引用。"""
        a, b, *_ = self.keys
        manager = self.manager("eager")
        try:
            self.put(manager, a)
            request = self.begin(manager, a)
            manager.prepare_load([a], request)
            block = manager.primary_tier._policy.blocks[a]
            self.assertEqual(block.ref_cnt, 2)
            manager.ssd.drain_jobs()
            manager._process_finished_jobs()
            self.assertEqual(block.ref_cnt, 1)
            manager.complete_load([a], request)
            manager.on_request_finished(request)
            self.put(manager, b)
        finally:
            manager.shutdown()
        self.assertEqual(Path(manager.ssd.file_mapper.get_file_name(b)).read_bytes(), self.payload[b])
        self.assertEqual(self.events()[-1]["kind"], "ssd_shutdown")

    def test_eager_promotion_waits_for_pinned_slot(self):
        """单槽仍被提前写引用时，提升必须等待并记录实际等待。"""
        a, b, *_ = self.keys
        manager = self.manager("eager", slots=1)
        try:
            self.put(manager, a)
            manager.ssd.drain_jobs()
            manager._process_finished_jobs()
            self.assertEqual(self.put(manager, b), (0, [a]))
            restore = self.begin(manager, a)
            self.assertEqual(manager.lookup(a, restore), LookupResult.HIT_PENDING)
            self.assertGreater(manager.slot_wait, 0.0)
        finally:
            manager.shutdown()
        writes = [event for event in self.events() if event["kind"] == "fs_complete" and event["direction"] == "store"]
        self.assertEqual([event["keys"] for event in writes], [[a.hex()], [b.hex()]])
        self.assertEqual(Path(manager.ssd.file_mapper.get_file_name(b)).read_bytes(), self.payload[b])

    def test_failed_eviction_does_not_reuse_the_source_slot(self):
        """用真实 ENOTDIR/EEXIST 验证延写失败不会归还或覆盖牺牲块的槽。"""
        a, b, c, *_ = self.keys
        manager = self.manager("on_evict")
        try:
            path = Path(manager.ssd.file_mapper.get_file_name(a))
            path.parent.parent.mkdir(parents=True)
            path.parent.touch()
            slot, _ = self.put(manager, a)
            self.put(manager, b)
            with self.assertRaises(OSError) as failed:
                self.put(manager, c)
            self.assertIn(failed.exception.errno, (errno.ENOTDIR, errno.EEXIST))
            self.assertNotIn(slot, manager.primary_tier._free_list)
            self.assertNotIn(c, manager.primary_tier._policy.blocks)
            self.assertEqual(self.slot(manager, slot), self.payload[a])
        finally:
            with self.assertRaisesRegex(RuntimeError, "SSD KV transfer failed"):
                manager.shutdown()
        self.assertTrue(manager.ssd.lease.closed)
        manager._metrics.assert_idle()
        self.assertFalse(self.events()[-1]["success"])

    def test_failed_promotion_releases_unreadable_destination(self):
        """已有文件丢失时传播真实读失败，释放 CPU 预约并在关闭时报告失败。"""
        a, b, c, *_ = self.keys
        manager = self.manager("on_evict")
        try:
            self.put(manager, a)
            self.put(manager, b)
            self.put(manager, c)
            Path(manager.ssd.file_mapper.get_file_name(a)).unlink()
            restore = self.begin(manager, a)
            self.assertEqual(manager.lookup(a, restore), LookupResult.HIT_PENDING)
            manager._flush_pending_promotions()
            manager.ssd.drain_jobs()
            with self.assertRaisesRegex(RuntimeError, "SSD KV transfer failed"):
                manager._process_finished_jobs()
            self.assertNotIn(a, manager.primary_tier._policy.blocks)
            self.assertFalse(manager._jobs)
        finally:
            with self.assertRaisesRegex(RuntimeError, "SSD KV transfer failed"):
                manager.shutdown()
        self.assertTrue(manager.ssd.lease.closed)
        manager._metrics.assert_idle()
