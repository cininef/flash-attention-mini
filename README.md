# flash-attention-mini

A small, readable, CPU-only implementation of the **FlashAttention algorithm**
(Dao et al., 2022) in plain PyTorch, together with the tests and benchmark
that check its two central claims:

1. **It is exact.** Tiled attention with online softmax returns the same
   numbers as ordinary attention (not an approximation).
2. **It saves memory.** Peak memory grows roughly linearly with sequence
   length instead of quadratically, because the full `S x S` score matrix is
   never built.

> **Scope.** This is a teaching and verification project, not a fast
> implementation. It is a Python loop over blocks, not a fused GPU kernel, and
> it is not faster than naive attention on CPU (in a quick single-run timing on
> my laptop it was about 1.5x slower at `S=1024` and on par at `S=4096`). It
> makes no speed claims. The
> algorithm is not new; the value here is a small codebase where each claim
> above is backed by a check you can run.

## TL;DR

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
python -m pytest tests/ -v          # claim 1: exactness (28 tests, ~2 s)
python bench/bench_memory.py        # claim 2: memory scaling (~15 s)
```

| | naive attention | tiled attention (this repo) |
|---|---|---|
| Intermediate score memory | `S x S` (grows 4x when `S` doubles) | one `128 x 128` tile (constant) |
| Peak process memory, `S` 256 -> 8192 | **~20.7x** larger | **~1.4x** larger |
| Output vs. reference | (is the reference) | matches to `atol=1e-10` in float64 |

## Why this matters for LLM inference

Standard attention computes `softmax(Q K^T / sqrt(d)) V`. The intermediate
`Q K^T` has one entry per (query, key) pair, so doubling the context length
quadruples it. In an LLM serving engine this intermediate competes with the KV
cache and the weights for GPU memory, which is one reason long contexts are
expensive. FlashAttention removes that term: it processes keys/values one
block at a time and never holds the whole score matrix. This repo re-derives
that idea in about 80 lines so it can be read, tested, and measured on a laptop.

## How the algorithm works

The obstacle is softmax: it normalizes over *all* keys, so it seems you need
every score before you can output anything. The fix is to keep three running
quantities **per query row** and update them block by block:

- `m`: the largest score seen so far
- `l`: the sum of `exp(score - m)` seen so far
- `o`: the running (still unnormalized) weighted sum of `V`

For each new block of keys with scores `s` and values `V_blk`:

```
m_new = max(m, max(s))
p     = exp(s - m_new)
l     = exp(m - m_new) * l + sum(p)         # rescale old sum to the new max
o     = exp(m - m_new) * o + p @ V_blk      # rescale old output the same way
m     = m_new
...after the last block:  output = o / l
```

Why this is *exact* and not an approximation: softmax is unchanged if you
subtract the same constant from every score, so `exp(s - m)` for any `m` gives
the same normalized result. The factor `exp(m - m_new)` just re-expresses the
earlier partial sums relative to the newer, larger maximum. Nothing is dropped
or rounded off beyond ordinary floating-point error. This is also what makes
the exactness claim testable with a very tight tolerance.

Code: [`flash_attention_mini/tiled.py`](flash_attention_mini/tiled.py) (the
algorithm) and [`flash_attention_mini/naive.py`](flash_attention_mini/naive.py)
(the reference that builds the full matrix).

## What is verified, and how

The core question is *"how do we know the tiled version computes the right
thing?"* The tests answer it with four independent kinds of evidence, so that
no single mistake in the tiled code can hide behind a matching mistake in the
reference:

| # | Check | What it rules out | Where |
|---|---|---|---|
| 1 | Tiled == naive in **float64**, `atol=1e-10, rtol=1e-8` | Any real algorithmic error. The tolerance is so tight that an approximation could not pass; only an exact reformulation can. Covers sequence lengths 1, 5, 37, 200 and tile sizes 16x16, 32x8, 128x128, causal and not, so single-block, multi-block, and non-divisible last-block paths all run. | `test_tiled_matches_naive` (24 cases) |
| 2 | Tiled == **PyTorch's own** `scaled_dot_product_attention` in float32, `atol=rtol=1e-4` | Both of my implementations sharing the same misunderstanding of attention. The oracle here is not my code. | `test_tiled_matches_torch_sdpa` (2 cases) |
| 3 | **Causal no-leak property**: under a causal mask, corrupt all keys/values after position `i`; outputs at positions `<= i` must not change | A masking bug that still "looks" plausible. This is the contract an autoregressive decoder depends on (a token must not see the future), tested as behavior rather than by reading the mask code. | `test_causal_output_independent_of_future_keys` |
| 4 | **Weights sum to 1**: feed `V = 1`; every output must equal 1 | A normalization bug, e.g. a wrong `l` update or a missing rescale. Uses attention's linearity in `V`, so it does not re-implement softmax. | `test_rows_sum_to_one_softmax_weights` |

The memory claim has its own check (next section), because a correct answer
that still allocates `S x S` would pass every test above.

## Memory benchmark: what is measured and what to trust

`bench/bench_memory.py` runs each `(implementation, seq_len)` in a **separate
subprocess** and records that process's peak resident memory
(`ru_maxrss`). A fresh process per point matters: within one process,
PyTorch's allocator can reuse freed blocks, which would make a later, larger
run look cheaper than it is. Setup: batch 1, 8 heads, head dim 64, float32,
causal, tile 128x128. Raw numbers are in
[`bench/results.csv`](bench/results.csv).

| seq_len | naive peak RSS | tiled peak RSS | naive score matrix (theory) | tiled score tile (theory) |
|--:|--:|--:|--:|--:|
| 256 | 213.9 MB | 210.7 MB | 2.1 MB | 0.52 MB |
| 512 | 232.5 MB | 217.2 MB | 8.4 MB | 0.52 MB |
| 1024 | 278.9 MB | 223.6 MB | 33.6 MB | 0.52 MB |
| 2048 | 489.7 MB | 237.7 MB | 134.2 MB | 0.52 MB |
| 4096 | 1216.3 MB | 259.2 MB | 536.9 MB | 0.52 MB |
| 8192 | 4422.3 MB | 303.4 MB | 2147.5 MB | 0.52 MB |

How to read it:

- **Look at the growth, not the absolute values.** Every row includes a fixed
  ~200 MB baseline (Python + PyTorch import). From 256 to 8192 (32x longer),
  naive grows by ~4.2 GB (about 20.7x total) and tiled by ~93 MB (about 1.4x).
- **The "theory" columns are lower bounds on the score matrix only.** Naive's
  measured growth (~4.2 GB) is about twice the raw score matrix (2.1 GB at
  8192) because `softmax` allocates an output the same size as its input, so
  two such matrices are alive at the peak. This is an explanation from
  reading the code, not something I profiled to confirm.
- **Expect a few percent of run-to-run noise.** Two runs of this benchmark
  differed by up to ~6% at some sequence lengths. The claim is the *shape* of
  the curve (quadratic vs. roughly linear), not the exact megabytes.

## What is not covered

Knowing the limits is part of trusting the results:

- **CPU only, no GPU.** Nothing here says anything about CUDA speed, memory
  hierarchy (SRAM/HBM) effects, or kernel fusion, which are the reasons real
  FlashAttention is fast.
- **Forward pass only.** No backward pass / gradients.
- **float32 and float64 only.** No fp16/bf16, where online-softmax rescaling
  has different numerical behavior.
- **Tests use square attention (`Sq == Sk`).** Cross-attention shapes are not
  tested, and causal masking assumes each query can see at least its own
  position.
- **No dropout, no attention bias, no grouped-query / multi-query layouts.**

## Repository layout

```
flash_attention_mini/
  naive.py       reference attention (builds the full S x S matrix)
  tiled.py       tiled attention with online softmax
tests/
  test_correctness.py   the 28 checks described above
bench/
  run_one.py            one (impl, seq_len) run in a fresh process
  bench_memory.py       drives run_one.py over sequence lengths, writes results.csv
  results.csv           raw benchmark output
```

## References

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re.
  *FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness.* NeurIPS 2022. https://arxiv.org/abs/2205.14135
- Maxim Milakov, Natalia Gimelshein. *Online normalizer calculation for
  softmax.* 2018. https://arxiv.org/abs/1805.02867 (the running max/sum
  recurrence used above).
