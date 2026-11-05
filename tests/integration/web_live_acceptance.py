"""对已启动的真实 MemMgr WebAPI 执行安全、分页与配置事务验收，不替代任何实际服务。

运行：MEMMGR_WEB_URL=http://127.0.0.1:8765 MEMMGR_TEST_WORKSPACE=<已准备并索引的隔离工作区>
python tests/integration/web_live_acceptance.py。测试必须独占后端，会临时修改关联配置及所选隔离清单后恢复。
"""

from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import unittest
import uuid
import httpx


class LiveWebAcceptance(unittest.TestCase):
    """验证真实服务响应、安全拒绝和保存事务，不使用模拟对象。"""

    @classmethod
    def setUpClass(cls):
        """按环境变量为 cls 创建真实 HTTP 客户端，地址必须显式指定。"""
        cls.client = httpx.Client(
            base_url=os.environ["MEMMGR_WEB_URL"], timeout=30, trust_env=False
        )
        cls.workspace = os.environ["MEMMGR_TEST_WORKSPACE"]
        if cls.workspace == "formal":
            raise ValueError("Tests require an explicitly selected isolated workspace")

    @classmethod
    def tearDownClass(cls):
        """关闭 cls 持有的实际连接池。"""
        cls.client.close()

    def test_host_and_origin(self):
        """拒绝任意 Host 和跨站写请求，在路径校验前终止操作。"""
        response = self.client.get("/api/config", headers={"Host": "untrusted.example"})
        self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post(
            "/api/workspaces",
            json={"name": "forbidden"},
            headers={"Origin": "https://untrusted.example"},
        )
        self.assertEqual(response.status_code, 403, response.text)

    def test_allowlist_and_types(self):
        """用真实配置提交非法数据/输出路径、块重叠和布尔数字类型，核实全部拒绝且未保存。"""
        saved = self.client.get("/api/config").raise_for_status().json()
        for section, key, value in [
            ("dataset", "root", "../../outside"),
            ("experiment", "output_dir", "/tmp/outside"),
            (
                "embedding",
                "chunk_overlap",
                saved["baseline"]["embedding"]["chunk_tokens"],
            ),
            ("embedding", "chunk_overlap", 1.5),
            ("experiment", "repeats", True),
        ]:
            candidate = copy.deepcopy(saved)
            candidate["baseline"][section][key] = value
            response = self.client.put("/api/config", json=candidate)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.client.get("/api/config").json(), saved)
        response = self.client.post(
            "/api/tasks", json={"kind": "prepare", "workspace": "formal"}
        )
        self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post("/api/workspaces", json={"name": "../outside"})
        self.assertEqual(response.status_code, 422, response.text)
        candidate = copy.deepcopy(saved)
        candidate["server"]["port"] = 18002
        response = self.client.put("/api/config", json=candidate)
        self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post(
            "/api/tasks",
            json={
                "kind": "rebuild",
                "workspace": self.workspace,
                "confirm": False,
            },
        )
        self.assertEqual(response.status_code, 422, response.text)
        response = self.client.delete("/api/memory/absent/item?confirm=false")
        self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post(
            "/api/tasks",
            json={"kind": "retrieval", "workspace": self.workspace, "pid": 1},
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(
            self.client.post("/api/tasks/" + uuid.uuid4().hex + "/cancel").status_code,
            404,
        )

    def test_atomic_linked_config_save(self):
        """整体改变 server.port 与 baseline.llm.base_url 并真实保存重读，结束后恢复原配置。"""
        saved = self.client.get("/api/config").raise_for_status().json()
        candidate = copy.deepcopy(saved)
        new_port = 18000 if saved["server"]["port"] != 18000 else 18001
        candidate["server"]["port"] = new_port
        candidate["baseline"]["llm"]["base_url"] = (
            f"http://{candidate['server']['host']}:{new_port}/v1"
        )
        try:
            response = self.client.put("/api/config", json=candidate)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.client.get("/api/config").json(), candidate)
            health = self.client.get("/api/health").raise_for_status().json()
            self.assertEqual(health["vllm"]["status"], "unavailable")
            self.assertIn("error", health["vllm"])
            response = self.client.post(
                "/api/qa",
                json={
                    "question": "Is DdrB involved in DNA repair?",
                    "workspace": "formal",
                },
            )
            self.assertGreaterEqual(response.status_code, 400, response.text)
            self.assertIn("detail", response.json())
        finally:
            response = self.client.put("/api/config", json=saved)
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get("/api/config").json(), saved)

    def test_real_isolated_identity_guard(self):
        """暂改真实隔离清单 identity，验证查询立即拒绝不匹配索引，随后恢复原始文件字节。"""
        workspace = self.workspace
        self.assertNotEqual(workspace, "formal")
        item = next(
            row
            for row in self.client.get("/api/workspaces")
            .raise_for_status()
            .json()["items"]
            if row["name"] == workspace
        )
        name = item["tiers"]["small"]["collection"]
        manifest_path = Path(item["path"]) / "index-manifests" / f"{name}.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        self.assertIn(
            "/.agent/web/workspaces/" + workspace + "/", manifest["identity"]["dataset"]
        )
        try:
            manifest["identity"]["corpus_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            response = self.client.post(
                "/api/collections/" + name + "/query",
                json={
                    "query": "DNA repair",
                    "workspace": workspace,
                    "tier": "small",
                    "top_k": 1,
                },
            )
            self.assertEqual(response.status_code, 502, response.text)
            self.assertIn("identity mismatch", response.json()["detail"])
        finally:
            manifest_path.write_bytes(original)

    def test_unknown_memory_read_is_nonmutating(self):
        """读取不存在的 namespace，验证404且真实 Chroma 集合数保持不变。"""
        before = (
            self.client.get("/api/collections?limit=1")
            .raise_for_status()
            .json()["total"]
        )
        response = self.client.get("/api/memory/absent_" + uuid.uuid4().hex)
        self.assertEqual(response.status_code, 404, response.text)
        after = (
            self.client.get("/api/collections?limit=1")
            .raise_for_status()
            .json()["total"]
        )
        self.assertEqual(before, after)

    def test_real_resource_and_report(self):
        """核实真实资源字段有效边界、服务端分页上限和未知报告拒绝行为。"""
        response = self.client.get("/api/resources?limit=2").raise_for_status().json()
        self.assertLessEqual(len(response["items"]), 2)
        for sample in response["items"]:
            self.assertGreater(sample["timestamp"], 0)
            self.assertIsInstance(sample["cpu_percent"], (int, float))
            self.assertGreaterEqual(sample["cpu_percent"], 0)
            self.assertLessEqual(sample["cpu_percent"], 100)
            memory = sample["wsl_memory_swap"]
            self.assertGreaterEqual(memory["used_bytes"], 0)
            self.assertLessEqual(memory["used_bytes"], memory["total_bytes"])
        response = (
            self.client.get("/api/collections?limit=1&offset=0")
            .raise_for_status()
            .json()
        )
        self.assertLessEqual(len(response["items"]), 1)
        self.assertEqual(self.client.get("/api/reports/unknown.md").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
