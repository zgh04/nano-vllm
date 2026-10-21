import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 过滤合法配置参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 设置全局 KV 块大小，所有 sequence 共用
        Sequence.block_size = config.kvcache_block_size
        # 多进程容器：保存TP子进程，进程同步事件
        self.ps = []
        self.events = []
        # 创建张量并行多进程 spwan
        # spawn，相比 fork 不会复制父进程 CUDA 上下文，避免多卡显存冲突、cuda 上下文报错
        ctx = mp.get_context("spawn")
        # rank 从 1 开始 rank 0 是当前主进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 主进程模型执行器
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载分词器，配置全局 eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 实例化调度器，用来管理 sequence，block manager
        self.scheduler = Scheduler(config)
        # 无论程序正常结束、异常崩溃，都会自动执行exit()函数，回收多进程，避免显卡占用、僵尸进程残留
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # 新增用户请求
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq) # 加入调度器 waiting 队列

    # 单轮推理核心循环，一次 GPU 前向 = 一次 step()
    # schedule -> run -> postprocess -> outputs
    def step(self):
        # 调度器分配本轮要计算的序列，（区分prefill与decode）
        seqs, is_prefill = self.scheduler.schedule()
        # 计算本轮 token 数量，用于吞吐统计，正负号用来区分prefill与decode
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 调用多卡模型执行前向推理，返回新生成的 token_id
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 调度后处理
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 筛选本轮执行完成（EOS / max_len）的序列，返回 seq_id 与 token 结果
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # 上层主入口
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 初始化进度条，总长度 = 请求数
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # ？
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 批量添加请求到调度队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # key:seq_id, value:完整输出token ids
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()  # 高精度计时器，计算 prefill/decode 吞吐速度
            output, num_tokens = self.step()
            # 根据 num_tokens 正负区分 prefill 还是 decode
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            # 进度条实时展示吞吐
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            # 保存已完成结果，进度条 +1
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 按请求提交顺序排序输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # token_ids 转回文本，封装返回
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
