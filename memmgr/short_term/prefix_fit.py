"""在原生 CPU 分配前计算可保留的最长连续前缀。"""


def fit_prefix(primary, prompt, admitted, offered, parents=None):
    """计算 admitted 中可保留的前缀及其保护集合。

    primary 提供 CPU 槽和引用状态；prompt 是完整键链，offered 是本批 GPU 可写键。
    offered 为 None 时按完整准入前缀规划，实际写入前仍须检查本批键。
    parents 在 learned 策略下给出驻留块的父键，淘汰仅能从叶块开始。
    """
    policy = primary._policy
    blocks = policy.blocks
    protected = set()
    for key in prompt:
        if key not in blocks:
            break
        protected.add(key)

    evictable = policy.evictable_blocks.keys()
    if parents is None:
        reclaimable = evictable - protected
    else:
        # 受保护或在途的后代会阻止其驻留祖先成为可淘汰叶块。
        blocked = set(blocks) - evictable
        blocked.update(protected)
        for key in tuple(blocked):
            parent = parents[key]
            while parent in blocks and parent not in blocked:
                blocked.add(parent)
                parent = parents[parent]
        reclaimable = evictable - blocked

    available = primary._get_num_free_blocks() + len(reclaimable)
    for index, key in enumerate(admitted):
        if offered is not None and key not in blocks and key not in offered:
            return admitted[:index], protected
        if key not in blocks or key in reclaimable:
            if available == 0:
                return admitted[:index], protected
            available -= 1
        protected.add(key)
    return admitted, protected
