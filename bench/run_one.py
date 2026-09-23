"""Run a single attention forward pass and report peak process memory.

Invoked as a fresh subprocess per (impl, seq_len) so PyTorch's caching
allocator from a previous run can't make one measurement look smaller than
it really is -- see bench_memory.py, which drives this file.
"""

import argparse
import platform
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flash_attention_mini import naive_attention, tiled_attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["naive", "tiled"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--block", type=int, default=128)
    args = p.parse_args()

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.heads, args.seq_len, args.dim)
    k = torch.randn(args.batch, args.heads, args.seq_len, args.dim)
    v = torch.randn(args.batch, args.heads, args.seq_len, args.dim)

    if args.impl == "naive":
        out = naive_attention(q, k, v, causal=True)
    else:
        out = tiled_attention(
            q, k, v, causal=True, block_q=args.block, block_k=args.block
        )
    assert out.shape == q.shape

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS/Darwin.
    peak_bytes = peak_kb if platform.system() == "Darwin" else peak_kb * 1024
    print(peak_bytes)


if __name__ == "__main__":
    main()
