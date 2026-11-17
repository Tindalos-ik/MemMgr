"""使用固定版本的真实 Qwen tokenizer 验证聊天预算和证据纳入边界。

测试加载共享环境中的 Transformers 和固定缓存中的 tokenizer。
"""

import unittest

from transformers import AutoTokenizer

from memmgr.inference.prompt import assemble_prompt


class RealQwenPromptBudget(unittest.TestCase):
    """覆盖真实聊天模板的裁切、超长问题与空证据边界。"""

    @classmethod
    def setUpClass(cls):
        """按固定 revision 加载 Qwen tokenizer，供本组预算测试共用。"""
        cls.tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-1.5B-Instruct',
            revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        )

    def test_prompt_budget_truncates_evidence_and_keeps_question_and_source(self):
        """验证长证据被裁切后仍保留问题、来源标记，最终 token 数落在预算内。"""
        query = 'What does the study support?'
        messages, included_doc_ids, evidence, input_ids, truncated = assemble_prompt(
            self.tokenizer,
            query,
            [('scifact-123', 'supported study detail. ' * 400)],
            180,
        )
        self.assertTrue(truncated)
        self.assertEqual(included_doc_ids, ['scifact-123'])
        self.assertIn(query, messages[1]['content'])
        self.assertIn('[DOC scifact-123]', messages[1]['content'])
        self.assertLessEqual(len(input_ids), 180)

    def test_overlong_question_fails_clearly(self):
        """验证问题自身超出空证据预算时给出明确异常，避免生成隐式溢出请求。"""
        with self.assertRaisesRegex(
            ValueError,
            'empty-evidence prompt does not fit',
        ):
            assemble_prompt(
                self.tokenizer,
                'long question ' * 200,
                [('unused', 'source text')],
                32,
            )

    def test_document_is_not_included_when_marker_has_no_budget(self):
        """验证预算仅够基础模板时，文档既不进入提示词也不进入 included ID 列表。"""
        query = 'q'
        _, _, _, base_ids, _ = assemble_prompt(self.tokenizer, query, [], 180)
        messages, included_doc_ids, evidence, input_ids, truncated = assemble_prompt(
            self.tokenizer,
            query,
            [('no-room', 'evidence')],
            len(base_ids),
        )
        self.assertTrue(truncated)
        self.assertEqual(included_doc_ids, [])
        self.assertNotIn('[DOC no-room]', messages[1]['content'])
        self.assertLessEqual(len(input_ids), len(base_ids))

    def test_history_and_current_evidence_share_the_complete_budget(self):
        """历史与本轮证据一起经过真实 tokenizer 预算，历史保持完整消息对。"""
        history = [{"role": "user", "content": "Question: Which protein was discussed?"},
                   {"role": "assistant", "content": "DdrB was discussed."}]
        messages, included, evidence, ids, truncated = assemble_prompt(
            self.tokenizer, "What does the current study establish?",
            [("new-source", "current study detail. " * 400)], 256, history=history,
        )
        self.assertEqual(messages[1:3], history)
        self.assertEqual(included, ["new-source"])
        self.assertIn("[DOC new-source]", messages[-1]["content"])
        self.assertTrue(truncated)
        self.assertLessEqual(len(ids), 256)

    def test_changed_current_evidence_replaces_earlier_sources(self):
        """固定真实历史、替换当前证据，核对当前消息与实际 token 序列随之改变。"""
        history = [{"role": "user", "content": "Question: Summarize the study."},
                   {"role": "assistant", "content": "Earlier answer [DOC old-source]."}]
        first = assemble_prompt(self.tokenizer, "What is supported now?", [("old-source", "old finding")], 256, history=history)
        second = assemble_prompt(self.tokenizer, "What is supported now?", [("new-source", "new finding")], 256, history=history)
        self.assertEqual(first[0][:-1], second[0][:-1])
        self.assertNotEqual(first[3], second[3])
        self.assertNotIn("old-source", second[0][-1]["content"])
        self.assertEqual(second[1], ["new-source"])
if __name__ == '__main__':
    unittest.main()
