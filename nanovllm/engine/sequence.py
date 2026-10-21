from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

"""
    简单来说：
    Sequence 就是一个状态机 + 一个动态 token 缓冲区 + 一套为 IPC 优化的序列化策略。
    三个计数器（num_tokens、num_cached_tokens、num_scheduled_tokens）精确描述了"处理到哪了"，
    配合 block_table 连接到了 KV cache 管理
"""

# 使用 枚举 表示每个用户请求（序列）的状态
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # LLMEngine.__init__ 里：Sequence.block_size = config.kvcache_block_size
    # 在引擎启动时统一设一次，所有后续创建的 Sequence 自动继承。
    block_size = 256
    # 无限自增迭代器。每次 next(Sequence.counter) 返回一个全局唯一的整数，用作 seq_id
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 使用 next方法 生成全局唯一 ID
        self.seq_id = next(Sequence.counter)
        print(f"序列{self.seq_id}被创建了")
        # 请求初始状态 ：等待调度
        self.status = SequenceStatus.WAITING
        # token_ids 会动态增长 prefill 结束、decode 每生成一个 token，都 append_token 进去
        # 浅拷贝 prompt token 列表
        self.token_ids = copy(token_ids)
        # 记录最后一个 token ID 以便 decode 时快速访问 优化性能
        self.last_token = token_ids[-1]
        # 当前 token 总数 （不断增长）
        self.num_tokens = len(self.token_ids)
        # prompt token 总数（保持不变）
        self.num_prompt_tokens = len(token_ids)
        # 以命中的 prefix cache 的 token 数
        self.num_cached_tokens = 0
        # 本轮要新增计算、写入 KV Cache 的 token 数
        # ChunkedPrefill 分轮次跑 每轮跑完，会把本轮 num_scheduled_tokens 并入 num_cached_tokens
        # Decode 阶段：标准自回归单步推理，每轮调度固定 num_scheduled_tokens = 1
        self.num_scheduled_tokens = 0
        # 当前处于 prefill 还是 decode
        self.is_prefill = True
        # 持有的 KV cache block ID 列表
        self.block_table = []
        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    """
        num_cached_tokens 和 num_scheduled_tokens — 调度状态机

        这是理解 chunked prefill 和 prefix cache 的关键。

        Prompt: [A, B, C, D, E, F, G, H]  (8 tokens, block_size=3)

        假设 prefix cache 命中了前 4 个 token [A,B,C,D]：
        num_cached_tokens = 4 此时 waiting

        第一次调度（token budget 只够 2 个 token）：
        num_scheduled_tokens = 2 此时 waiting -> running
        实际计算 token [E, F]  (从 index 4 开始，长度 2)

        执行后：num_cached_tokens = 4 + 2 = 6

        第二次调度（剩余 [G, H]）：
        num_scheduled_tokens = 2
        实际计算 token [G, H]  (从 index 6 开始，长度 2)

        执行后：num_cached_tokens = 6 + 2 = 8
        num_cached_tokens == num_tokens → prefill 结束 → 进入 decode

        三个变量精确描述了"这条序列处理到哪了"：
        - num_tokens：总 token 数（一直增长，decode 时每步 +1）
        - num_cached_tokens：KV cache 里已经存了多少 token 
        - num_scheduled_tokens：本轮要算多少 token

        当 num_cached_tokens + num_scheduled_tokens == num_tokens 时，本序列的 prefill 完成
    """

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    # 计算已生成的 token 数 x = token 总数 - prompt token 数
    # 将 token_ids 序列切片 可以得到 prompt 部分 与 completion 部分
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    # token_ids 序列切片 得到 prompt 部分
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    # token_ids 序列切片 得到 completion 部分
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    # 根据 token 总数 和 块大小 计算需要多少块（向上取整）
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    # 计算最后一个块里有多少有效 token 用于 decode 时计算 slot_mapping：新 token 写到最后一个 block 的哪个位置
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 获取第 i 个 block 对应的 token ID 切片 
    # 用于 prefix cache 匹配 — BlockManager 调用此方法来计算每个 block 的 hash
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # decode 阶段每生成一个 token 调用一次。
    # last_token 单独存一份，因为 ModelRunner.prepare_decode 只需要最后一个 token 作为模型输入
    # 因为 KV Cache 已经缓存了所有历史 token 的 KV，仅输入最新 token 即可完成计算。
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 跨进程传输时 需要把 python 对象序列化为 元组， 反序列化时调用 __setstate__(state)，把元组还原成对象
    # 如果不写 __getstate__ 则会默认序列化对象全部属性
    # 场景背景：多 TP 分布式（tensor_parallel_size > 1）
    def __getstate__(self):
        # 分支1：prefill阶段 → last_state = 完整self.token_ids列表
        # 分支2：decode阶段 → last_state = 单个int：self.last_token
        last_state = self.last_token if not self.is_prefill else self.token_ids
        # 只打包推理必须的最小字段，丢弃无用冗余属性 给 worker 计算时使用
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # 反序列化
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
