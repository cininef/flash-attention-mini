import math

import torch


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    block_q: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """FlashAttention-style tiled attention with online (running) softmax.

    Never materializes the full (Sq, Sk) score matrix: at any point it holds
    one (block_q, block_k) score tile plus O(Sq) running statistics (m, l)
    and an O(Sq, D) output accumulator, so peak memory for the attention
    computation is O(S) instead of O(S^2). Numerically it targets the same
    result as `naive_attention` / `torch.nn.functional.scaled_dot_product_attention`,
    up to floating-point error -- see `tests/test_correctness.py`.

    This is a plain Python block loop for teaching/benchmarking the memory
    argument, not a fused kernel: it does not claim a wall-clock speedup.

    Shapes: q is (..., Sq, D), k and v are (..., Sk, D).
    """
    d = q.shape[-1]
    d_v = v.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)

    *lead, sq, _ = q.shape
    sk = k.shape[-2]
    batch = math.prod(lead) if lead else 1
    qf = q.reshape(batch, sq, d)
    kf = k.reshape(batch, sk, d)
    vf = v.reshape(batch, sk, d_v)

    out = torch.zeros(batch, sq, d_v, dtype=q.dtype, device=q.device)
    m = torch.full((batch, sq, 1), float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.zeros((batch, sq, 1), dtype=q.dtype, device=q.device)

    for qs in range(0, sq, block_q):
        qe = min(qs + block_q, sq)
        q_blk = qf[:, qs:qe, :]
        q_idx = torch.arange(qs, qe, device=q.device).unsqueeze(-1)
        m_i = m[:, qs:qe, :]
        l_i = l[:, qs:qe, :]
        acc = out[:, qs:qe, :]

        # Causal: keys strictly beyond the last query row in this block
        # cannot be attended to by anyone in the block, so the k loop stops
        # at `qe` instead of `sk`.
        k_end = min(qe, sk) if causal else sk
        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            k_blk = kf[:, ks:ke, :]
            v_blk = vf[:, ks:ke, :]
            s_ij = torch.baddbmm(
                torch.zeros(batch, qe - qs, ke - ks, dtype=q.dtype, device=q.device),
                q_blk,
                k_blk.transpose(-2, -1),
                alpha=scale,
            )
            if causal:
                k_idx = torch.arange(ks, ke, device=q.device).unsqueeze(-2)
                s_ij = s_ij.masked_fill(k_idx > q_idx, float("-inf"))

            m_ij = s_ij.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m_i, m_ij)
            p_ij = torch.exp(s_ij - m_new)
            alpha = torch.exp(m_i - m_new)

            l_i = alpha * l_i + p_ij.sum(dim=-1, keepdim=True)
            acc = alpha * acc + p_ij @ v_blk
            m_i = m_new

        out[:, qs:qe, :] = acc
        m[:, qs:qe, :] = m_i
        l[:, qs:qe, :] = l_i

    out = out / l.clamp_min(torch.finfo(out.dtype).tiny)
    return out.reshape(*lead, sq, d_v)
