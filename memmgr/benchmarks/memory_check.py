"""实现显式请求的记忆重启验收，保存重启前状态并核对重启后持久化结果。

验收命名空间使用随机标识隔离；before/after 两阶段由命令行调用，已有集合和卷保持原状态。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..long_term import memory


def check(config, phase, artifact=None):
    """执行 before 建立隔离记忆状态，或执行 after 比对保存的状态。

    before 在输出目录写验收 JSON，包含更新/删除 ID 和易混淆命名空间；after 要求 artifact
    路径，验证计数、更新内容、删除缺失和语义查询。检查失败抛出 RuntimeError，返回值为 None，结果同时打印。
    """
    if phase == 'before':
        root = Path(config['experiment'].get('output_dir', '.agent/runs'))
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"memory-check-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
        nonce = uuid.uuid4().hex
        namespace = f'acceptance-{nonce}/user'
        collision_namespace = namespace.replace('/', '_')
        updated_id = memory.add(
            config,
            namespace,
            'initial note about cats',
            f'{nonce}-one',
        )
        deleted_id = memory.add(
            config,
            namespace,
            'temporary note about planets',
            f'{nonce}-two',
        )
        memory.update(
            config,
            namespace,
            updated_id,
            'updated note about cats and persistent memory',
        )
        memory.delete(config, namespace, deleted_id)
        isolated_id = memory.add(
            config,
            collision_namespace,
            'must remain isolated',
            f'{nonce}-collision',
        )
        expected_state = {
            'phase': 'before',
            'namespace': namespace,
            'collision_namespace': collision_namespace,
            'updated_id': updated_id,
            'deleted_id': deleted_id,
            'collision_id': isolated_id,
            'expected_count': 1,
            'expected_updated_text': 'updated note about cats and persistent memory',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'instructions': (
                'Restart Chroma server/container normally, then invoke function-check --phase'
                ' after --artifact with this path. Do not delete collections or volumes.'
            ),
        }
        path.write_text(json.dumps(expected_state, indent=2), encoding='utf-8')
        print(json.dumps(expected_state, indent=2))
        return
    if not artifact:
        raise ValueError('--artifact is required for --phase after')
    state = json.loads(Path(artifact).read_text(encoding='utf-8'))
    namespace = state['namespace']
    actual = memory.list_items(config, namespace)
    ids = actual.get('ids', [])
    texts = actual.get('documents', [])
    updated = memory.get(config, namespace, state['updated_id'])
    deleted = memory.get(config, namespace, state['deleted_id'])
    cross = memory.get(config, state['collision_namespace'], state['collision_id'])
    result = {
        'phase': 'after',
        'namespace_count': len(ids),
        'expected_count': state['expected_count'],
        'updated_text': (updated.get('documents') or [None])[0],
        'deleted_absent': not deleted.get('ids'),
        'cross_namespace_isolated': bool(cross.get('ids')),
        'pass': (
            len(ids) == 1
            and bool(updated.get('ids'))
            and state['expected_updated_text'] in (updated.get('documents') or [''])[0]
            and not deleted.get('ids')
            and bool(cross.get('ids'))
        ),
    }
    query_result = memory.query(
        config,
        namespace,
        'updated note cats persistent memory',
        k=5,
    )
    result['query_ids'] = query_result.get('ids', [[]])[0]
    result['query_updated_present'] = state['updated_id'] in result['query_ids']
    result['query_deleted_absent'] = state['deleted_id'] not in result['query_ids']
    result['pass'] &= result['query_updated_present'] and result['query_deleted_absent']
    print(json.dumps(result, indent=2))
    if not result['pass']:
        raise RuntimeError('Memory restart acceptance check failed')
