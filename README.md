# flash-attention-mini

A from-scratch, CPU-only implementation of the FlashAttention algorithm
(Dao et al., 2022): tiled attention with online (running) softmax, verified
for correctness against a naive full-matrix baseline and against PyTorch's
own fused `scaled_dot_product_attention`, plus a memory benchmark that
measures the O(S) vs. O(S²) peak-memory claim directly.

**Scope, stated plainly:** this is a teaching/verification implementation
in pure PyTorch (Python-level block loop), not a fused CUDA kernel. It does
not claim a wall-clock speedup — on CPU it is slower than the naive version,
because Python loop overhead dominates. The property it demonstrates is the
one FlashAttention is actually for: attention's peak memory scales with
sequence length, not with sequence length squared, because the full
attention-score matrix is never materialized.

## Why this matters for LLM serving

Naive attention computes and holds the entire `(seq_len, seq_len)` score
matrix before applying softmax. Doubling the context length quadruples that
matrix. This is exactly the constraint that makes long-context serving
(and KV-cache-heavy inference engines such as vLLM and SGLang) memory-bound:
the attention step's intermediate memory, not just the model weights, grows
with context length. FlashAttention's tiling — process one block of keys at
a time, keep a running max and a running normalizer per query row, rescale
the accumulator as a new block's max supersedes the old one — bounds that
intermediate to one block, independent of total sequence length.

## What's implemented

- `flash_attention_mini/naive.py` — reference attention: full
  `Q @ K^T`, softmax, `@ V`. Optional causal masking.
- `flash_attention_mini/tiled.py` — block-wise attention with online
  softmax. Same causal masking, block-by-block, with running `(m, l)`
  statistics and an output accumulator rescaled with `exp(m_old - m_new)`
  as each new block updates the running max. Never allocates a tensor
  larger than `(block_q, block_k)` for scores, regardless of `seq_len`.

## Correctness

```bash
python -m pytest tests/ -v
```

28 cases, all passing, covering:

- **Exact match against the naive baseline** in float64, across sequence
  lengths shorter than, equal to, and longer than one tile, and across
  several block-size combinations (so partial last blocks and
  single-vs-multi-block paths are both exercised), causal and non-causal.
- **Cross-check against `torch.nn.functional.scaled_dot_product_attention`**
  in float32 — the precision this would actually run at — so the
  correctness claim isn't only "matches my own other implementation."
- **A derived property**: under causal masking, row `i`'s output must be
  unchanged if keys/values at positions `> i` are corrupted. This is the
  contract a decode step actually depends on (it must not "see" tokens
  generated after it), independent of whether the mask happens to look
  right in the code.
- **A derived property**: attention weights form a valid probability
  distribution per row, checked by feeding `v = 1` and confirming the
  output (a convex combination of `v`) sums to 1 — reconstructing the
  weights via attention's linearity in `v`, rather than re-deriving
  softmax from the implementation.

## Memory benchmark

```bash
python bench/bench_memory.py
```

Each `(impl, seq_len)` pair runs in its own subprocess and reports that
process's actual peak RSS (`resource.getrusage(...).ru_maxrss`), so
PyTorch's caching allocator from a previous run can't hide how much the
naive version really uses. Batch=1, heads=8, head_dim=64, causal, tile
size 128×128.

| seq_len | naive peak RSS | tiled peak RSS | naive score matrix (theoretical) | tiled score tile (theoretical) |
|--:|--:|--:|--:|--:|
| 256 | 206.8 MB | 210.1 MB | 2.1 MB | 0.52 MB |
| 512 | 239.7 MB | 217.0 MB | 8.4 MB | 0.52 MB |
| 1024 | 280.1 MB | 222.5 MB | 33.6 MB | 0.52 MB |
| 2048 | 485.5 MB | 237.7 MB | 134.2 MB | 0.52 MB |
| 4096 | 1291.9 MB | 262.4 MB | 536.9 MB | 0.52 MB |
| 8192 | 4427.9 MB | 303.5 MB | 2147.5 MB | 0.52 MB |

Going from `seq_len=256` to `seq_len=8192` (32x the sequence length):
naive peak RSS grows **~21x**, tiled peak RSS grows **~1.4x**. Tiled memory
is dominated by the fixed ~200 MB Python/PyTorch process baseline plus
`O(seq_len)` Q/K/V/output buffers, not by the attention computation itself.

The measured naive RSS growth (`4427.9 - 206.8 ≈ 4221 MB`) is roughly 2x the
theoretical raw score-matrix size (`2147.5 MB`) at `seq_len=8192`; softmax
allocates its own output tensor the same size as its input, so the
transient peak is closer to two score matrices, not one. Noted here rather
than smoothed over — the theoretical column is a lower bound on the score
matrix alone, not a prediction of total process RSS.

## Layout

```
flash_attention_mini/   naive.py, tiled.py
tests/                  correctness tests (pytest)
bench/                  run_one.py (single subprocess run), bench_memory.py (driver), results.csv
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m pytest tests/ -v
python bench/bench_memory.py
```
