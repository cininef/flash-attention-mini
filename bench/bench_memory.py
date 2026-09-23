"""Peak-memory scaling of naive vs. tiled attention, as sequence length grows.

Each (impl, seq_len) pair runs in its own subprocess (run_one.py) so the
measurement is that process's actual peak RSS, not a number PyTorch's
allocator could be reusing across runs. This is CPU wall-clock/memory, not a
CUDA benchmark: the point is the O(S) vs O(S^2) memory curve, not speed.
"""

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ONE = HERE / "run_one.py"

SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
BATCH, HEADS, DIM, BLOCK = 1, 8, 64, 128


def measure(impl: str, seq_len: int) -> int:
    out = subprocess.run(
        [
            sys.executable,
            str(RUN_ONE),
            "--impl",
            impl,
            "--seq-len",
            str(seq_len),
            "--batch",
            str(BATCH),
            "--heads",
            str(HEADS),
            "--dim",
            str(DIM),
            "--block",
            str(BLOCK),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def theoretical_score_matrix_bytes(impl: str, seq_len: int) -> int:
    """The textbook argument, independent of what the OS/allocator does:
    bytes needed to hold the intermediate (Sq, Sk) score tile(s) at any one
    instant, in float32."""
    if impl == "naive":
        return BATCH * HEADS * seq_len * seq_len * 4
    return BATCH * HEADS * BLOCK * BLOCK * 4


def main():
    rows = []
    for seq_len in SEQ_LENS:
        row = {"seq_len": seq_len}
        for impl in ("naive", "tiled"):
            row[f"{impl}_peak_rss_bytes"] = measure(impl, seq_len)
            row[f"{impl}_score_matrix_bytes"] = theoretical_score_matrix_bytes(
                impl, seq_len
            )
        rows.append(row)
        print(
            f"seq_len={seq_len:5d}  "
            f"naive peak RSS={row['naive_peak_rss_bytes']/1e6:8.1f} MB  "
            f"tiled peak RSS={row['tiled_peak_rss_bytes']/1e6:8.1f} MB  "
            f"naive score matrix={row['naive_score_matrix_bytes']/1e6:8.1f} MB  "
            f"tiled score tile={row['tiled_score_matrix_bytes']/1e6:8.3f} MB"
        )

    out_csv = HERE / "results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
