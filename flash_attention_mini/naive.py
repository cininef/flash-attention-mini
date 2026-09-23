import math

import torch


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference scaled-dot-product attention: materializes the full
    (..., Sq, Sk) score matrix.

    This is the correctness baseline for `tiled_attention` and the "why
    flash attention" comparison point for the memory benchmark: its score
    tensor is O(Sq * Sk), quadratic in sequence length, because the whole
    row of attention weights for every query is held in memory at once.

    Shapes: q is (..., Sq, D), k and v are (..., Sk, D); leading dims
    (batch, heads, ...) broadcast together, matching
    `torch.nn.functional.scaled_dot_product_attention`.
    """
    d = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        q_idx = torch.arange(sq, device=q.device).unsqueeze(-1)
        k_idx = torch.arange(sk, device=q.device).unsqueeze(-2)
        scores = scores.masked_fill(k_idx > q_idx, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
