"""用真实 vLLM 请求、分配器和引用计数检查连续前缀写入。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import Request

from memmgr.short_term.connector import ManagedCPUManager, PolicyController
from memmgr.short_term.prefix_fit import fit_prefix


class PrefixCapacity(unittest.TestCase):
    """直接驱动 scheduler 侧管理器，不启动 GPU 服务。"""

    def setUp(self):
        """为每个用例创建独立审计目录和原生哈希器。"""
        directory = self.enterContext(TemporaryDirectory())
        self.audit = Path(directory) / "audit.jsonl"
        init_none_hash(sha256)
        self.hasher = get_request_block_hasher(64, sha256)

    def manager(self, capacity):
        """创建 capacity 个槽的 CPU 管理器，绑定正式 LRU 控制器。"""
        # CPU spec 依赖 GPU 布局；此处调用正式管理器使用的同一原生构造函数。
        manager = ManagedCPUManager.__new__(ManagedCPUManager)
        CPUOffloadingManager.__init__(manager, capacity, cache_policy="ManagedLRU",
                                      cache_policy_module_path=ManagedCPUManager.__module__)
        manager.control = PolicyController(manager, dict(
            max_model_len=2048, dtype="float16", model_revision="test",
            policy_config=dict(policy="lru", audit_path=self.audit)))
        self.addCleanup(manager.shutdown)
        return manager

    def begin(self, manager, name, *tokens):
        """向 manager 登记 name；每个 tokens 元素构成一个完整块，末尾留一个 token。"""
        request = Request(name, [token for token in tokens for _ in range(64)] + [0],
                          SamplingParams(max_tokens=1), None, block_hasher=self.hasher)
        context = ReqContext(request.request_id)
        manager.control.begin(request)
        manager.on_new_request(context)
        return context, [make_offload_key(value, 0) for value in request.block_hashes]

    def test_oversize_prefix_does_not_reuse_pending_write_slots(self):
        """1,025 token 缩到八块；并发写入不得覆盖尚未完成的八个槽。"""
        manager = self.manager(8)
        warm, keys = self.begin(manager, "warm", *range(1, 17))
        first = manager.prepare_store(keys, warm)
        self.assertEqual(first.keys_to_store, keys[:8])
        self.assertEqual(len(set(first.store_spec.block_ids)), 8)
        other, unrelated = self.begin(manager, "other", 99)
        self.assertEqual(manager.prepare_store(unrelated, other).keys_to_store, [])
        self.assertTrue(all(manager.lookup(key, warm) is LookupResult.HIT_PENDING for key in keys[:8]))
        manager.complete_store(first.keys_to_store, warm)
        self.assertTrue(all(manager.lookup(key, warm) is LookupResult.HIT for key in keys[:8]))
        self.assertTrue(all(manager.lookup(key, warm) is LookupResult.MISS for key in keys[8:]))

    def test_split_offer_keeps_unoffered_hit_and_read_reference(self):
        """后续批次保留已命中父块；只能淘汰未被 DMA 读取的槽。"""
        manager = self.manager(3)
        prompt, keys = self.begin(manager, "prompt", 1, 2, 3)
        self.assertEqual(manager.prepare_store(keys[1:], prompt).keys_to_store, [])
        first = manager.prepare_store(keys[:1], prompt)
        manager.complete_store(first.keys_to_store, prompt)
        other, unrelated = self.begin(manager, "other", 9, 10)
        second = manager.prepare_store(unrelated, other)
        manager.complete_store(second.keys_to_store, other)
        pinned = manager.prepare_load(unrelated[:1], other)

        extension = manager.prepare_store(keys[1:], prompt)
        self.assertEqual(extension.keys_to_store, keys[1:2])
        self.assertEqual(extension.evicted_keys, unrelated[1:])
        self.assertEqual(extension.store_spec.block_ids, second.store_spec.block_ids[1:])
        self.assertEqual(manager.lookup(keys[0], prompt), LookupResult.HIT)
        self.assertTrue(set(extension.store_spec.block_ids).isdisjoint(pinned.block_ids))

        manager.complete_store(extension.keys_to_store, prompt)
        manager.complete_load(unrelated[:1], other)
        tail = manager.prepare_store(keys[2:], prompt)
        self.assertEqual(tail.keys_to_store, keys[2:])
        self.assertEqual(tail.evicted_keys, unrelated[:1])
        self.assertEqual(tail.store_spec.block_ids, pinned.block_ids)

    def test_leaf_only_capacity_excludes_ancestors_of_pinned_blocks(self):
        """在途子块阻止 learned 淘汰其父块；完成读取后，两块均可淘汰。"""
        manager = self.manager(2)
        old, resident = self.begin(manager, "old", 1, 2)
        store = manager.prepare_store(resident, old)
        manager.complete_store(store.keys_to_store, old)
        manager.prepare_load(resident[1:], old)
        _, incoming = self.begin(manager, "incoming", 3, 4)
        offered = set(incoming)
        parents = manager.control.parents

        self.assertEqual(fit_prefix(manager, incoming, incoming, offered, parents)[0], [])
        manager.complete_load(resident[1:], old)
        self.assertEqual(fit_prefix(manager, incoming, incoming, offered, parents)[0], incoming)

    def test_failed_dma_still_propagates(self):
        """原生写入失败传递到调用方，不被解释为前缀容量不足。"""
        manager = self.manager(1)
        request, keys = self.begin(manager, "request", 1, 2)
        result = manager.prepare_store(keys, request)
        with self.assertRaisesRegex(RuntimeError, "GPU to CPU transfer failed"):
            manager.complete_store(result.keys_to_store, request, success=False)
