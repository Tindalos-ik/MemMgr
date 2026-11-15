"""构造可重复的真实推理请求序列，覆盖热点、扫描干扰与热点变化。"""

import random


def make_plan(tokenizer, rounds, seed):
    """使用 tokenizer、rounds、seed 生成固定输入；阶段和角色字段仅用于事后统计。"""
    rng = random.Random(seed)
    body = tokenizer.encode("The memory system stores keys and values for attention. " * 1024,
                            add_special_tokens=False)
    prompts, first_blocks = {}, set()
    plan = []

    def append(identity, phase, cycle, hot):
        """将 identity 对应真实输入加入 phase/cycle；hot 标明统计角色。"""
        if identity not in prompts:
            length = rng.choice((512, 1024, 1536)) if not hot else 1024
            nonce = tokenizer.encode(rng.randbytes(32).hex() + "\n", add_special_tokens=False)
            tokens = (nonce + body)[:length]
            first = tuple(tokens[:64])
            if len(tokens) != length or first in first_blocks:
                raise ValueError("Prompt length or prefix independence failed")
            first_blocks.add(first)
            prompts[identity] = tokens
        plan.append(dict(index=len(plan), identity=identity, phase=phase, cycle=cycle,
                         hot=hot, prompt_token_ids=prompts[identity]))

    for cycle in range(rounds):
        for identity in ("a", "b", "a", "b"):
            append(identity, "stable", cycle, True)
        for number in range(4):
            append(f"scan-{cycle}-{number}", "scan", cycle, False)
        for identity in ("a", "b"):
            append(identity, "scan", cycle, True)
        for identity in ("c", "d", "c", "d"):
            append(identity, "shift", cycle, True)
        for number in range(4):
            append(f"shift-{cycle}-{number}", "shift", cycle, False)
        for identity in ("c", "d"):
            append(identity, "shift", cycle, True)
        for identity in ("a", "b"):
            append(identity, "long_gap", cycle, True)
        for number in range(7):
            append(f"gap-{cycle}-{number}", "long_gap", cycle, False)
        for identity in ("a", "b"):
            append(identity, "long_gap", cycle, True)
    return plan
