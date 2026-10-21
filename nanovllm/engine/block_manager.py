from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id    # 物理块号
        self.ref_count = 0  # 引用计数，不是读写保护，只用来判断块能不能被回收
        self.hash = -1  # 块哈希 初始化为 -1
        self.token_ids = [] # 块对应的 token 列表，防止哈希碰撞导致错误的 KV cache 共享

    # 更新 hash 值 与 token_ids
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 重置
    def reset(self):
        self.ref_count = 1  # 刚分配时引用计数为 1 而不是 0
        self.hash = -1
        self.token_ids = []

# 块 的 全局管理器
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # prefix 的核心 存储 hash → block_id 的映射 给定内容hash，快速找到对应的 block 块
        self.hash_to_block_id: dict[int, int] = dict()  
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块队列
        # 已分配的块集合（而非队列），便于快速判断某个 block 是否已使用
        # can_allocate 方法里有这个判断：if block_id in self.used_block_ids:
        self.used_block_ids: set[int] = set()   

    @classmethod
    # prefix 参数 使 Block i 的 hash 依赖于 Block i-1 的 hash，即 链式哈希
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    # 从空闲链表分配出一个新物理块，返回块 id
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id
    
    # 将块归还到空闲链表
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 注意：只是进行预检查，预估内存开销，实际 allocate 可以使用未被占用块，allocate 使用未被占用块时是从free 池里拿，本质上消耗了一个空闲块
    # num_cached_blocks：核心用途是指导 Chunked Prefill 跳过重复 KV 计算，
    # 只和 token 内容相关，和内存占用状态无关 因此即使块未被占用，仍然会执行 num_cached_blocks += 1
    # 块空闲只代表不能长期分配给本请求，但仍然可被用来 节省计算开销
    # num_new_blocks：统计本次必须新申请的物理 KV 块数量
    # 使用空闲的匹配块时本质上会消耗空闲块数量，因此 不会执行 num_new_blocks-=1
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        # 返回值，即请求已被缓存的 KV cache 块（可直接复用）
        num_cached_blocks = 0
        # 记录本次真正要分配的 block 数
        num_new_blocks = seq.num_blocks
        # 不会检查最后一个块，因为最后一个块可能没填满，此时一定不会缓存命中
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 哈希不存在 或 发生 哈希碰撞
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 如果这个块正在被其他序列占用，则可以复用（没被占用说明即将被释放或覆盖）
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 空闲块数不够分配，调用方 Scheduler 据此判断是否需要触发抢占（preemption）
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        # 阶段 1：复用已缓存的 block
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            # 处理被占用块
            if block_id in self.used_block_ids:
                block.ref_count += 1
            # 处理空闲块 
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 阶段 2:分配新块 
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 释放请求占用的块
    # 释放时不从 hash_to_block_id 中删除。
    # 这意味着即使 block被归还到空闲列表，它的内容哈希仍然保存在全局索引中。后续的新序列仍能匹配到这个 block（通过can_allocate）
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 块引用计数为 0 时才归还到 free_block_ids
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # decode 阶段判断是否能进行本次追加
    # 整体判断逻辑：空闲块总数 >= 需要新增的块数量
    # (len(seq) % self.block_size == 1) 值为 True 时代表需要分配新物理块
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 根据是否需要新分配块来分配或什么都不做
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())


    # seq.num_cached_tokens：该序列已经完成缓存、算好哈希的 token 总数；
    # seq.num_scheduled_tokens：本次新增待调度 / 待计算哈希的 token 数量；
    def hash_blocks(self, seq: Sequence):
        # 1. 计算哈希需要更新的块区间 [start, end)
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # 新增token没有跨越任何块，无块需要更新哈希，直接返回
        if start == end: 
            return
        # 2. 获取前一块的哈希作为滚动哈希初始值
        # start>0：取前一块(start-1)已计算好的复合哈希
        # start=0：当前是第一个块，无前驱哈希，初始值-1
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            # 计算滚动链式哈希
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
