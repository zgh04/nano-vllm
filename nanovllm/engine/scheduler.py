from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        print("Scheduler 初始化")
        self.max_num_seqs = config.max_num_seqs # 单步最多处理多少序列
        self.max_num_batched_tokens = config.max_num_batched_tokens # 单批总 token 上限
        self.eos = config.eos   # 结束符 token id
        self.block_size = config.kvcache_block_size # 块大小
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque() # 等待队列：未完成 prefill 或 被强占
        self.running: deque[Sequence] = deque() # 运行队列：已完成 prefill 正在进行 decode

    # 引擎是否已无请求待执行，用于判断引擎退出
    def is_finished(self):
        return not self.waiting and not self.running

    # 外部收到新请求，加入等待队列，等待 prefill 处理
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 核心调度逻辑，先 prefill 后 decode
    # return：本轮要送入 GPU 计算的请求序列列表，以及是 prefill 还是 decode（不会出现二者混合）
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        # 本轮已经占用的 token 总数
        num_batched_tokens = 0

        # prefill（计算密集，token 量大，必须批量优化）
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 只看队首，不 pop，因为长序列可能需要 chunked prefill
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens    # # 本批次还能容纳多少 token
            if remaining == 0:
                break
            # 判断序列是否为新加入序列，即块表为空，无 KVcache 缓存
            if not seq.block_table:
                # 计算需要分配多少 KV 块 内存不足返回 -1
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq，why？
                break
            # 为新加入序列块分配 KVcache 块
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本序列实际编码的 token 数，取 num_tokens 与 remaining 的较小值
            # 累加到 num_batched_tokens（本调度轮次实际编码的 token 总数）
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 判断成立说明序列 prefill 阶段已完成 加入 running 队列 
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                print(f"序列{seq.seq_id} prefill 完成，加入 running 队列")
                self.waiting.popleft()
                self.running.append(seq)
            # 加入返回值列表
            scheduled_seqs.append(seq)
        # 只要本轮调度到 prefill 请求，就立即返回，不再调度 decode
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode：次只生成 1 个 token，访存密集，优先级低于 Prefill
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 循环检查：当前序列是否有空闲 KV 块用来追加1个token
            # why while ？why not if ？
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())    # 抢占队列末尾的序列
                else:
                    self.preempt(seq)   # 抢占自己
                    break
            # 内存充足，调度该序列生成一个 token
            # （while 未被 break 打断时会执行 else）
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 调度完的序列放入队列末尾
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

        
    # 抢占函数，seq 是被强占的请求
    """
        为什么只有 decode 阶段可能触发 preempt？
        GPU 算子层面：
            prefill 是大算子，不可拆分，prefill一旦启动就要跑完
            decode 是迭代执行小算子，执行之间的空隙可以用来抢占
        显存开销层面：
            prefill 的上下文大 恢复开销大
            decode 单步只新增一个 token 的 KV，换入换出开销小
    """
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING # 加入等待队列
        seq.is_prefill = True   # 被强占的请求需要重新开始 prefill
        self.block_manager.deallocate(seq)  # 回收占用的 KVcache Block
        self.waiting.appendleft(seq)    # 加入等待队列队首（而不是队尾）

    # 后处理，推理内核计算结束后调用
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            # print(seq.seq_id, seq.num_cached_tokens,seq.num_scheduled_tokens, seq.num_tokens)
            seq.num_scheduled_tokens = 0
            # prefill 未完整跑完，直接跳过
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 追加模型生成的 token 到序列的生成结果列表
            seq.append_token(token_id)
            # 请求终止条件：生成 eos 或 到达 最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
