"""验证数据消费者、分块参数与持久化命名的行为边界。

测试使用临时文件，覆盖官方 qrels、语料内容签名和命名空间隔离。
"""
import tempfile
import unittest
from pathlib import Path

from memmgr.long_term.data import _qrels
from memmgr.long_term.index import validate_chunk_config, index_collection_name, index_identity
from memmgr.long_term.memory import collection_name


class ConsumerContracts(unittest.TestCase):
    """验证上层调用依赖的数据格式、参数边界和隔离契约。"""

    def test_official_three_column_qrels(self):
        """验证官方三列 qrels 的正相关与零相关分值均按原始 ID 保留。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'test.tsv'
            path.write_text(
                'query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n',
                encoding='utf-8',
            )
            self.assertEqual(_qrels(path), {'q1': {'d1': 1, 'd2': 0}})

    def test_index_identity_changes_with_corpus_bytes(self):
        """在相同路径改写语料字节，验证索引身份随真实内容摘要变化。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / 'corpus.jsonl'
            path.write_text('{"_id":"a"}\\n', encoding='utf-8')
            embedding_config = {'model': 'BAAI/bge-small-en-v1.5', 'revision': 'fixed'}
            first = index_identity(root, 'small', embedding_config, 384, 32)
            path.write_text('{"_id":"b"}\\n', encoding='utf-8')
            second = index_identity(root, 'small', embedding_config, 384, 32)
            self.assertNotEqual(first['corpus_sha256'], second['corpus_sha256'])

    def test_chunk_overlap_boundaries(self):
        """覆盖合法重叠的两端边界及非法块长、负重叠、步长为零等错误。"""
        self.assertIsNone(validate_chunk_config(384, 0))
        self.assertIsNone(validate_chunk_config(384, 383))
        for size, overlap in ((0, 0), (-1, 0), (384, -1), (384, 384), (8, 9)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                validate_chunk_config(size, overlap)

    def test_namespace_names_do_not_collide_on_sanitization(self):
        """验证相似字符命名空间与长前缀仍保留不同摘要，集合名称满足长度限制。"""
        self.assertNotEqual(
            collection_name('m', 'tenant/a'),
            collection_name('m', 'tenant_a'),
        )
        self.assertNotEqual(
            collection_name('m', 'a:b'),
            collection_name('m', 'a_b'),
        )
        long_prefix = 'p' * 100
        first_namespace_name = collection_name(long_prefix, 'namespace/a')
        second_namespace_name = collection_name(long_prefix, 'namespace_a')
        self.assertLessEqual(len(first_namespace_name), 63)
        self.assertLessEqual(len(second_namespace_name), 63)
        self.assertNotEqual(first_namespace_name, second_namespace_name)
        self.assertTrue(
            first_namespace_name.endswith(
                __import__('hashlib').sha256(b'namespace/a').hexdigest()[:24],
            ),
        )
        first_name = index_collection_name(long_prefix, 'small', 'abc123def456')
        second_name = index_collection_name(long_prefix, 'small', '789abc123def')
        self.assertLessEqual(len(first_name), 63)
        self.assertLessEqual(len(second_name), 63)
        self.assertTrue(first_name.endswith('abc123def456'))
        self.assertTrue(second_name.endswith('789abc123def'))
if __name__ == '__main__':
    unittest.main()
