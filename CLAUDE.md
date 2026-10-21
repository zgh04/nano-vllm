# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Nano-vLLM is a lightweight vLLM-compatible LLM inference engine (~1,200 lines of Python). It achieves comparable throughput to vLLM while being dramatically simpler. The public API mirrors vLLM's `LLM` and `SamplingParams`.

**Note:** inline comments throughout the codebase are written in Chinese. Match that style (or at least don't "fix" it) when editing existing files.

**Dependencies:** `torch>=2.4.0`, `triton>=3.0.0`, `transformers>=4.51.0`, `flash-attn`, `xxhash`. Python 3.10–3.12. GPU-only (CUDA).

## Commands

```bash
# Install (editable)
pip install -e .

# Run the example (interactive generation)
python example.py

# Run the benchmark (256 sequences, Qwen3-0.6B)
python bench.py
```

There is no test suite, linter config, or Makefile in this repo.

Both `example.py` and `bench.py` hardcode the model path `~/huggingface/Qwen3-0.6B/` — edit it to point at your local model before running. To download the weights (from README):

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

**Dependency gotcha:** `tqdm` is imported by `llm_engine.py` but not declared in `pyproject.toml` — it arrives transitively via `transformers`.

## Architecture

### Entry point

`LLM` (`nanovllm/llm.py`) is a thin alias for `LLMEngine`. The public API is:

```python
from nanovllm import LLM, SamplingParams
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
# outputs is a list of {"text": str, "token_ids": list[int]} dicts, in request order
```

`generate` returns a list of dicts (`{"text": ..., "token_ids": ...}`), not bare strings. `prompts` may be `list[str]` or `list[list[int]]`; `sampling_params` may be a single `SamplingParams` (broadcast to all prompts) or a per-prompt list.

### Engine layer (`nanovllm/engine/`)

**`LLMEngine`** — Orchestrates the generation loop: tokenize prompts, add requests, then step (schedule → run model → postprocess) until all sequences finish.

**`Scheduler`** — Decides which sequences to run each step. Prefill has priority over decode. Supports chunked prefill (splitting long prefill across steps). When KV cache is full, preempts running sequences via eviction. Manages `waiting` and `running` deques of `Sequence` objects.

**`BlockManager`** — Paged KV cache allocator using copy-on-write prefix caching. Each `Block` tracks a ref_count, content hash (xxhash), and token_ids. `can_allocate` walks a sequence's blocks looking for hash matches in the cache; matching blocks are shared (ref-counted) rather than re-allocated. Block size is always a multiple of 256.

**`Sequence`** — Tracks a single request's token_ids, status (WAITING/RUNNING/FINISHED), block_table, and scheduling counters (`num_cached_tokens`, `num_scheduled_tokens`). Custom `__getstate__`/`__setstate__` serialize only the last token for decode to minimize IPC overhead. Two class-level globals: `Sequence.block_size` is set once by `LLMEngine.__init__` from the config (affects all future sequences), and `Sequence.counter` is an infinite `itertools.count()` handing out global seq_ids.

**`ModelRunner`** — Owns the model on GPU. Three responsibilities:
1. Prepares inputs for prefill (varlen with `cu_seqlens_q/k`, `slot_mapping`, `block_tables` for prefix cache) or decode (single-token per sequence).
2. Runs the model, dispatching to CUDA graphs for decode batch sizes ≤512 when `enforce_eager=False`.
3. Samples tokens via the `Sampler`.

For tensor parallelism (TP), rank 0 communicates with worker ranks via `multiprocessing.shared_memory.SharedMemory` with pickle serialization and `Event` synchronization. Worker ranks call `loop()` and block on `read_shm()`. TP workers are spawned with the `spawn` context (never `fork` — avoids copying the parent's CUDA context), NCCL initializes on a hardcoded `tcp://localhost:2333` endpoint, and `LLMEngine` registers `atexit` cleanup so worker processes are reaped on crash as well as clean exit.

### Model layer (`nanovllm/models/`)

Only **Qwen3** is implemented (`qwen3.py`). The architecture is standard: `Qwen3ForCausalLM` → `Qwen3Model` → stack of `Qwen3DecoderLayer` → `Qwen3Attention` + `Qwen3MLP`. The `packed_modules_mapping` dict maps HuggingFace weight names to fused layers (QKV, gate+up).

### Layers (`nanovllm/layers/`)

- **`linear.py`** — Tensor-parallel linear types: `ReplicatedLinear`, `ColumnParallelLinear`, `QKVParallelLinear` (fuses Q/K/V), `MergedColumnParallelLinear` (fuses gate+up), `RowParallelLinear` (with `all_reduce`).
- **`attention.py`** — `Attention` module and a Triton `store_kvcache_kernel`. Prefill uses `flash_attn_varlen_func` (with optional `block_table` for prefix cache). Decode uses `flash_attn_with_kvcache`.
- **`sampler.py`** — Gumbel-max sampling (`torch.compile`-d). Greedy sampling (temperature=0) is explicitly disallowed.
- **`rotary_embedding.py`** — Standard RoPE with an LRU-cached `RotaryEmbedding` module (one per head_dim/config, shared across layers).
- **`activation.py`** — `SiluAndMul` (SwiGLU gate).
- **`layernorm.py`** — `RMSNorm` with fused residual-add variant (`add_rms_forward`), both `torch.compile`-d.
- **`embed_head.py`** — `VocabParallelEmbedding` and `ParallelLMHead` with TP-aware sharding and `all_reduce` gather.

### Utils (`nanovllm/utils/`)

- **`loader.py`** — Safetensors weight loader. Resolves packed module mappings (QKV, gate+up) and dispatches to per-parameter `weight_loader` methods.
- **`context.py`** — Module-global singleton `Context` dataclass (not thread-local) used as a side channel to pass CUDA tensors (slot_mapping, block_tables, cu_seqlens) through the model forward pass without threading them through every layer signature. Set by `ModelRunner.prepare_prefill/decode`, consumed by `Attention`, `ParallelLMHead`, and CUDA graph replay. Replaced wholesale by `set_context`/`reset_context` each step.

### KV cache and CUDA graphs

KV cache is a single tensor `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]` allocated to fill remaining GPU memory (respecting `gpu_memory_utilization`). Each attention layer gets a view into its layer slice. Init order matters: `warmup_model()` (a dummy prefill at max batch) runs **before** `allocate_kv_cache()`, because the block count is derived from the warmup's peak memory: `(total * gpu_memory_utilization - used - peak + current) // block_bytes`.

CUDA graphs are captured for decode at batch sizes `[1, 2, 4, 8] + [16, 32, ..., max_bs≤512]`. The smallest captured size ≥ the actual batch size is replayed. All graphs share a single memory pool (`graph_pool`), so any captured size can be replayed without reallocating. Graphs are captured once after KV allocation; `enforce_eager=True` disables them entirely.

### Config (`nanovllm/config.py`)

`Config` dataclass holds all engine settings. Key defaults: `max_num_batched_tokens=16384`, `max_num_seqs=512`, `max_model_len=4096`, `gpu_memory_utilization=0.9`, `kvcache_block_size=256`.

## Key design decisions

- **Tokenizer is optional at request time** — `add_request` accepts either `str` (encoded via the engine's `AutoTokenizer`) or pre-tokenized `list[int]`. The benchmark passes raw token IDs to avoid tokenizer overhead; outputs are decoded back to text only at the very end of `generate`.
- **Single-step scheduling** — `schedule()` returns either prefill OR decode, never both. Chunked prefill is only allowed for the first waiting sequence to avoid starvation.
- **Prefix cache at block granularity** — Hash chain uses xxHash; each block's hash depends on the previous block's hash (like a rolling hash), enabling O(1) match checks per block. Subtlety: `deallocate` never removes entries from `hash_to_block_id`, so a freed block's hash survives and later sequences still skip recomputing it; only blocks in `used_block_ids` count against new allocations.
- **IPC for TP only** — When `tensor_parallel_size=1`, there are no worker processes; `ModelRunner` runs inline in the main process.
