"""提供 MemMgr 命令行入口：参数解析、单条问答、实验矩阵与记忆命令派发。

数据/索引、RAG 批次和检索批次分别由各职责模块处理；重模型依赖保留按需加载。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .data import load_config, prepare
from .index import index_tier
from .rag_batch import run_rag_benchmark
from .retrieval_batch import run_retrieval_benchmark


def build_parser():
    """构建并返回 CLI 参数解析器，定义子命令、选项、必选参数和默认值。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    index_parser = sub.add_parser('index')
    index_parser.add_argument('--tier', default='small')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--tier', default='small')
    run_parser.add_argument('--concurrency', type=int, default=1)
    ask_parser = sub.add_parser('ask')
    ask_parser.add_argument('--query', required=True)
    ask_parser.add_argument('--tier', default='full')
    ask_parser.add_argument('--concurrency', type=int, default=1)
    retrieval_parser = sub.add_parser('retrieval')
    retrieval_parser.add_argument('--tier', default='small')
    retrieval_parser.add_argument('--concurrency', type=int, default=1)
    matrix_parser = sub.add_parser('matrix')
    matrix_parser.add_argument(
        '--tiers',
        nargs='+',
        default=['small', 'medium'],
    )
    matrix_parser.add_argument(
        '--concurrencies',
        nargs='+',
        type=int,
        default=[1, 2, 4],
    )
    memory_parser = sub.add_parser('memory')
    memory_actions = memory_parser.add_subparsers(dest='memory_cmd', required=True)
    for action in ('add', 'get', 'query', 'update', 'delete'):
        action_parser = memory_actions.add_parser(action)
        action_parser.add_argument('--namespace', required=True)
        action_parser.add_argument('--id')
        action_parser.add_argument('--text')
        action_parser.add_argument('--k', type=int, default=5)
    check_parser = sub.add_parser('function-check')
    check_parser.add_argument(
        '--phase',
        choices=['before', 'after'],
        required=True,
    )
    check_parser.add_argument('--artifact')
    return parser


def write_matrix(config, tiers, requested_concurrencies):
    """根据语料级别与客户端并发生成单因素实验配置，并打印/保存矩阵。

    requested_concurrencies 中大于 1 的值构成并发对照，其他预算与 K
    条件按固定顺序生成。配置逐份深拷贝，矩阵仅写文件；每行包含配置路径和各项实验参数。
    """
    concurrencies = [c for c in requested_concurrencies if c > 1]
    conditions = [
        {'context_tokens': b, 'top_k': 3, 'concurrency': 1}
        for b in (512, 1024, 1536)
    ]
    conditions += [
        {'context_tokens': 1024, 'top_k': top_k, 'concurrency': 1}
        for top_k in (1, 5)
    ]
    conditions += [
        {'context_tokens': 1024, 'top_k': 3, 'concurrency': c}
        for c in concurrencies
    ]
    output_root = Path(config['experiment'].get('output_dir', '.agent/runs'))
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / 'matrix-configs'
    config_dir.mkdir(exist_ok=True)
    matrix_rows = []
    for tier in tiers:
        for condition_index, condition in enumerate(conditions, 1):
            condition_config = json.loads(json.dumps(config))
            condition_config['experiment'].update(
                {
                    parameter_name: parameter_value
                    for parameter_name, parameter_value in condition.items()
                    if parameter_name != 'concurrency'
                },
            )
            config_path = config_dir / f'{tier}-{condition_index:02d}.json'
            config_path.write_text(
                json.dumps(condition_config, indent=2),
                encoding='utf-8',
            )
            matrix_rows.append(
                {'tier': tier, **condition, 'config': str(config_path)},
            )
    (output_root / 'matrix.json').write_text(
        json.dumps(matrix_rows, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(matrix_rows, indent=2))


def dispatch_memory(config, args):
    """按已解析参数调用记忆增删改查接口并输出文本或 JSON 结果。

    缺少 text/id 时抛出 ValueError，命名空间与持久化行为由 memory 模块负责。
    """
    from . import memory
    namespace = args.namespace
    if args.memory_cmd == 'add':
        if args.text is None:
            raise ValueError('--text required')
        print(memory.add(config, namespace, args.text, args.id))
    elif args.memory_cmd == 'update':
        if args.id is None or args.text is None:
            raise ValueError('--id and --text required')
        memory.update(config, namespace, args.id, args.text)
        print('updated')
    elif args.memory_cmd == 'delete':
        if args.id is None:
            raise ValueError('--id required')
        memory.delete(config, namespace, args.id)
        print('deleted')
    elif args.memory_cmd == 'get':
        if args.id:
            print(
                json.dumps(
                    memory.get(config, namespace, args.id),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        else:
            print(
                json.dumps(
                    memory.list_items(config, namespace),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
    elif args.memory_cmd == 'query':
        if args.text is None:
            raise ValueError('--text required')
        print(
            json.dumps(
                memory.query(config, namespace, args.text, args.k),
                ensure_ascii=False,
                indent=2,
            ),
        )


def main():
    """解析配置与子命令并派发到对应应用模块。

    --help 在读取配置和加载模型之前退出；ask 深拷贝实验配置并固定为零预热、一次重复，再交给同一 RAG 路径处理。
    """
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.cmd == 'prepare':
        print(json.dumps(prepare(config), indent=2))
    elif args.cmd == 'index':
        print(json.dumps(index_tier(config, args.tier), indent=2))
    elif args.cmd == 'run':
        asyncio.run(run_rag_benchmark(config, args.tier, args.concurrency))
    elif args.cmd == 'ask':
        ask_cfg = json.loads(json.dumps(config))
        ask_cfg['experiment']['warmups'] = 0
        ask_cfg['experiment']['repeats'] = 1
        asyncio.run(
            run_rag_benchmark(
                ask_cfg,
                args.tier,
                args.concurrency,
                query_override=args.query,
            ),
        )
    elif args.cmd == 'retrieval':
        asyncio.run(
            run_retrieval_benchmark(config, args.tier, args.concurrency),
        )
    elif args.cmd == 'matrix':
        write_matrix(config, args.tiers, args.concurrencies)
    elif args.cmd == 'function-check':
        from .memory_check import check
        check(config, args.phase, args.artifact)
    elif args.cmd == 'memory':
        dispatch_memory(config, args)
