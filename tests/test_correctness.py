import pytest
import torch
import torch.nn.functional as F

from flash_attention_mini import naive_attention, tiled_attention


def _make_qkv(batch, heads, sq, sk, d, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, sq, d, generator=g, dtype=torch.float64)
    k = torch.randn(batch, heads, sk, d, generator=g, dtype=torch.float64)
    v = torch.randn(batch, heads, sk, d, generator=g, dtype=torch.float64)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "sq,sk,d", [(1, 1, 8), (5, 5, 8), (37, 37, 16), (200, 200, 32)]
)
@pytest.mark.parametrize("block_q,block_k", [(16, 16), (32, 8), (128, 128)])
def test_tiled_matches_naive(causal, sq, sk, d, block_q, block_k):
    """tiled_attention must reproduce naive_attention's full-matrix result,
    across seq lengths shorter than, equal to, and longer than one block,
    and across tile shapes so no edge (partial last block, single block)
    is untested."""
    q, k, v = _make_qkv(batch=2, heads=3, sq=sq, sk=sk, d=d, seed=0)
    expected = naive_attention(q, k, v, causal=causal)
    actual = tiled_attention(q, k, v, causal=causal, block_q=block_q, block_k=block_k)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("causal", [False, True])
def test_tiled_matches_torch_sdpa(causal):
    """Cross-check against PyTorch's own fused implementation, in float32
    (the precision flash attention actually runs at), so numerical
    equivalence isn't an artifact of only ever comparing to our own
    naive baseline."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 300, 64, dtype=torch.float32)
    k = torch.randn(2, 4, 300, 64, dtype=torch.float32)
    v = torch.randn(2, 4, 300, 64, dtype=torch.float32)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    actual = tiled_attention(q, k, v, causal=causal, block_q=64, block_k=64)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_causal_output_independent_of_future_keys():
    """Derived property, not a mirror of the implementation: under a causal
    mask, row i's output must not change if we corrupt keys/values at
    positions > i. This is the actual contract callers rely on (a decode
    step must not see not-yet-generated tokens); it would fail under a
    masking bug even if the mask "looked" applied."""
    torch.manual_seed(1)
    sq = sk = 50
    d = 16
    q = torch.randn(1, 1, sq, d, dtype=torch.float64)
    k = torch.randn(1, 1, sk, d, dtype=torch.float64)
    v = torch.randn(1, 1, sk, d, dtype=torch.float64)

    out_a = tiled_attention(q, k, v, causal=True, block_q=8, block_k=8)

    k2, v2 = k.clone(), v.clone()
    cut = 20
    k2[:, :, cut:, :] = torch.randn_like(k2[:, :, cut:, :])
    v2[:, :, cut:, :] = torch.randn_like(v2[:, :, cut:, :])
    out_b = tiled_attention(q, k2, v2, causal=True, block_q=8, block_k=8)

    torch.testing.assert_close(out_a[:, :, :cut, :], out_b[:, :, :cut, :])


def test_rows_sum_to_one_softmax_weights():
    """Derived property: the implicit attention weights (out expressed as a
    convex combination of v) must be a valid probability distribution per
    row. Reconstructs the weights via linearity (attention is linear in v)
    rather than re-deriving softmax, so this doesn't just mirror the
    online-softmax code."""
    torch.manual_seed(2)
    sq = sk = 17
    d = 4
    q = torch.randn(1, 1, sq, d, dtype=torch.float64)
    k = torch.randn(1, 1, sk, d, dtype=torch.float64)
    ones = torch.ones(1, 1, sk, 1, dtype=torch.float64)
    row_sums = tiled_attention(q, k, ones, causal=False, block_q=4, block_k=5)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-10, rtol=0)
