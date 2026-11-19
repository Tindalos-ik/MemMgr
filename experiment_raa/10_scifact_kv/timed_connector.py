"""沿用生产 CPU KV 决策与传输，只在实验 scheduler 上计量策略耗时。"""

from time import perf_counter_ns

from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from memmgr.short_term.connector import (
    ManagedConnector, ManagedCPUManager, ManagedCPUSpec, ManagedLRU, PolicyController,
)
from memmgr.short_term.reuse import History


class RequestTiming:
    """保存单个串行请求的独占分段耗时，单位为纳秒。"""

    def __init__(self, request_id):
        """为 request_id 初始化计时器，区分未调用与实际执行。"""
        self.request_id = request_id
        self.read_features = False
        self.feature_ns = self.score_ns = self.audit_ns = 0
        self.select_ns = self.evict_ns = 0
        self.feature_rows = self.score_calls = self.scored_rows = 0
        self.select_calls = self.evict_calls = 0

    def nested_ns(self):
        """返回需要从外层选块区间扣除的特征、评分和审计耗时。"""
        return self.feature_ns + self.score_ns + self.audit_ns


class TimedHistory(History):
    """区分模型输入特征与仅为审计生成的特征。"""

    def __init__(self, controller):
        """使用 controller 提供当前请求的计时器。"""
        super().__init__()
        self.controller = controller

    def observe(self, keys, block_tokens, protected=()):
        """计量按 block_tokens 观察 keys 的耗时；protected 保留驻留统计。"""
        timer = self.controller.timer
        start = perf_counter_ns()
        super().observe(keys, block_tokens, protected)
        timer.feature_ns += perf_counter_ns() - start

    def features(self, key):
        """返回 key 的原始特征，仅在供模型计算时计量读取。"""
        timer = self.controller.timer
        if not timer.read_features:
            return super().features(key)
        start = perf_counter_ns()
        result = super().features(key)
        timer.feature_ns += perf_counter_ns() - start
        timer.feature_rows += 1
        return result


class TimedPredictor:
    """计量冻结模型的批量评分，保留原始特征变换和阈值。"""

    def __init__(self, predictor, controller):
        """使用已加载的 predictor 和 controller，不重新读取冻结模型。"""
        self.predictor = predictor
        self.threshold = predictor.threshold
        self.sha256 = predictor.sha256
        self.controller = controller

    def score(self, vectors):
        """为 vectors 评分；记录真实非空批次的次数、条数和耗时。"""
        if not vectors:
            return self.predictor.score(vectors)
        timer = self.controller.timer
        start = perf_counter_ns()
        result = self.predictor.score(vectors)
        timer.score_ns += perf_counter_ns() - start
        timer.score_calls += 1
        timer.scored_rows += len(vectors)
        return result


class TimedLRU(ManagedLRU):
    """计量底层 CPU 淘汰步骤，与上层保留决定分开。"""

    def evict(self, n, protected):
        """按实际策略淘汰 n 块并保留 protected；独立计量淘汰步骤。"""
        timer = self.controller.timer
        nested = timer.nested_ns()
        start = perf_counter_ns()
        result = super().evict(n, protected)
        timer.evict_ns += perf_counter_ns() - start - (timer.nested_ns() - nested)
        timer.evict_calls += 1
        return result


class TimedController(PolicyController):
    """继承策略决定，仅替换观测器、评分器和计时入口。"""

    def __init__(self, primary, extra):
        """用 primary 和 extra 初始化原始控制器，再绑定实验观测器。"""
        self.timer = None
        super().__init__(primary, extra)
        self.history = TimedHistory(self)
        if self.predictor is not None:
            self.predictor = TimedPredictor(self.predictor, self)

    def begin(self, request):
        """为串行 request 创建计时器，再执行原有访问观察与写入准入。"""
        if self.timer is not None:
            raise RuntimeError("Timing requires only one active request")
        self.timer = RequestTiming(request.request_id)
        return super().begin(request)

    def refresh_scores(self, keys):
        """为 keys 执行原有评分，标记提取的特征将用于模型输入。"""
        timer = self.timer
        timer.read_features = True
        try:
            return super().refresh_scores(keys)
        finally:
            timer.read_features = False

    def select(self, keys, req_context):
        """筛选 req_context 的 keys，计量保留决定并扣除内层特征、评分和审计。"""
        timer = self.timer
        if req_context.req_id != timer.request_id:
            raise RuntimeError("Store decision outside its timed request")
        nested = timer.nested_ns()
        start = perf_counter_ns()
        result = super().select(keys, req_context)
        timer.select_ns += perf_counter_ns() - start - (timer.nested_ns() - nested)
        timer.select_calls += 1
        return result

    def emit(self, kind, **fields):
        """保存原始 kind 事件和 fields，记录写审计本身的耗时。"""
        timer = self.timer
        if timer is None:
            return super().emit(kind, **fields)
        start = perf_counter_ns()
        super().emit(kind, **fields)
        timer.audit_ns += perf_counter_ns() - start

    def finish_timing(self, request_id):
        """发布 request_id 的分段计时，清除当前请求计时器。"""
        timer = self.timer
        if request_id != timer.request_id or timer.read_features:
            raise RuntimeError("Unexpected timing request boundary")
        self.timer = None
        fields = {key: value for key, value in vars(timer).items() if key != "read_features"}
        self.emit("timing", **fields)


class TimedCPUManager(ManagedCPUManager):
    """使用原生 CPU 管理器和项目调度，只替换带时钟的策略对象。"""

    def __init__(self, spec):
        """按 spec 使用原生块池与实验淘汰类，绑定实验控制器。"""
        CPUOffloadingManager.__init__(self, spec.num_blocks, cache_policy="TimedLRU",
                                      cache_policy_module_path=__name__)
        self.control = TimedController(self, spec.extra_config)


class TimedCPUSpec(ManagedCPUSpec):
    """仅替换 scheduler 管理器，保留真实 GPU worker 和传输路径。"""

    def get_manager(self):
        """构造并保存此 spec 的唯一计时管理器。"""
        if self._manager is None:
            self._manager = TimedCPUManager(self)
        return self._manager


class TimedConnector(ManagedConnector):
    """在生产连接器的真实请求结束后写入计时。"""

    def request_finished(self, request, block_ids):
        """结束 request 的 block_ids，并发布该请求分段耗时。"""
        result = super().request_finished(request, block_ids)
        self.control.finish_timing(request.request_id)
        return result

    def request_finished_all_groups(self, request, block_ids):
        """结束 request 的全部 block_ids 组，并发布该请求分段耗时。"""
        result = super().request_finished_all_groups(request, block_ids)
        self.control.finish_timing(request.request_id)
        return result

    def reset_cache(self):
        """通过原生接口清空 CPU 块，并在空闲时重置请求历史。"""
        control = self.control
        if control.timer is not None or control.active or self.connector_scheduler._jobs:
            raise RuntimeError("Cannot reset a non-idle experiment cache")
        clock = control.history.clock
        resident = len(control.primary._policy.blocks)
        result = super().reset_cache()
        if result is not True or control.primary._policy.blocks:
            raise RuntimeError("Native CPU reset did not empty the cache")
        control.history = TimedHistory(control)
        control.parents.clear()
        control.scores.clear()
        control.protected.clear()
        control.emit("reset", prior_clock=clock, cpu_blocks_before=resident,
                     cpu_blocks_after=len(control.primary._policy.blocks))
        return result
