import math

import pytest
import torch

from flash_sinkhorn import SamplesLoss, StructuredSamplesLoss
from flash_sinkhorn.structured_solvers import sinkhorn_flashstyle_structured_symmetric


def _structured_cost(u, q, v, k):
    return u.float()[:, None] + v.float()[None, :] - q.float() @ k.float().T


def _log_weights(w):
    w = w.float()
    return torch.where(w > 0, w.log(), torch.full_like(w, -float("inf")))


def _dense_structured_potentials(
    a,
    u,
    q,
    b,
    v,
    k,
    *,
    eps,
    n_iters,
    last_extrapolation=True,
    return_prelast=False,
):
    cost = _structured_cost(u, q, v, k)
    loga = _log_weights(a)
    logb = _log_weights(b)
    f = torch.zeros_like(u, dtype=torch.float32)
    g = torch.zeros_like(v, dtype=torch.float32)

    def softmin_x(g_vec):
        vals = (g_vec[None, :] - cost) / float(eps) + logb[None, :]
        return -float(eps) * torch.logsumexp(vals, dim=1)

    def softmin_y(f_vec):
        vals = (f_vec[:, None] - cost) / float(eps) + loga[:, None]
        return -float(eps) * torch.logsumexp(vals, dim=0)

    f = softmin_x(g)
    g = softmin_y(torch.zeros_like(f))

    for _ in range(int(n_iters)):
        f_new = softmin_x(g)
        g_new = softmin_y(f)
        f = 0.5 * f + 0.5 * f_new
        g = 0.5 * g + 0.5 * g_new

    f_prelast = f
    g_prelast = g
    if last_extrapolation:
        f = softmin_x(g_prelast)
        g = softmin_y(f_prelast)

    if return_prelast:
        return f, g, f_prelast, g_prelast
    return f, g


def _dense_conditional_structured_grads(a, u, q, b, v, k, f, g, eps):
    cost = _structured_cost(u, q, v, k)
    loga = _log_weights(a)
    logb = _log_weights(b)

    w_q = torch.softmax((g.float()[None, :] - cost) / float(eps) + logb[None, :], dim=1)
    grad_q = -a.float()[:, None] * (w_q @ k.float())

    w_k = torch.softmax((f.float()[:, None] - cost) / float(eps) + loga[:, None], dim=0)
    grad_k = -b.float()[:, None] * (w_k.T @ q.float())

    return a.float(), b.float(), grad_q, grad_k


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for structured FlashSinkhorn tests.")
    return torch.device("cuda")


def _random_structured_inputs(device, n=48, m=40, r=32, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    q = torch.randn(n, r, device=device, dtype=torch.float32, requires_grad=requires_grad)
    k = torch.randn(m, r, device=device, dtype=torch.float32, requires_grad=requires_grad)
    u = torch.randn(n, device=device, dtype=torch.float32, requires_grad=requires_grad)
    v = torch.randn(m, device=device, dtype=torch.float32, requires_grad=requires_grad)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    return a, u, q, b, v, k


@pytest.mark.parametrize("potentials", [False, True])
def test_structured_matches_dense_reference(device, potentials):
    eps = 0.25
    n_iters = 3
    a, u, q, b, v, k = _random_structured_inputs(device, seed=1)

    loss = StructuredSamplesLoss(
        eps=eps,
        n_iters=n_iters,
        potentials=potentials,
        normalize=False,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=16,
    )

    f_ref, g_ref = _dense_structured_potentials(
        a, u, q, b, v, k, eps=eps, n_iters=n_iters
    )

    if potentials:
        f, g = loss(a, u, q, b, v, k)
        torch.testing.assert_close(f, f_ref, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(g, g_ref, rtol=2e-4, atol=2e-4)
    else:
        val = loss(a, u, q, b, v, k)
        val_ref = (a * f_ref).sum() + (b * g_ref).sum()
        torch.testing.assert_close(val, val_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("cost_scale", [1.0, 0.5])
def test_structured_recovers_squared_euclidean_samplesloss(device, cost_scale):
    torch.manual_seed(2)
    n, m, d = 64, 48, 32
    eps = 0.2
    n_iters = 3
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    u = cost_scale * (x * x).sum(dim=1)
    v = cost_scale * (y * y).sum(dim=1)
    q = math.sqrt(2.0 * cost_scale) * x
    k = math.sqrt(2.0 * cost_scale) * y

    structured = StructuredSamplesLoss(
        eps=eps,
        n_iters=n_iters,
        normalize=False,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=16,
    )
    square = SamplesLoss(
        loss="sinkhorn",
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        half_cost=(cost_scale == 0.5),
        debias=False,
        normalize=False,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=16,
    )

    torch.testing.assert_close(
        structured(a, u, q, b, v, k),
        square(a, x, b, y),
        rtol=2e-4,
        atol=2e-4,
    )


def test_structured_recovers_cosine_cost(device):
    torch.manual_seed(3)
    n, m, d = 64, 48, 32
    eps = 0.2
    n_iters = 3
    x = torch.nn.functional.normalize(torch.randn(n, d, device=device), dim=1)
    y = torch.nn.functional.normalize(torch.randn(m, d, device=device), dim=1)
    a = torch.ones(n, device=device) / n
    b = torch.ones(m, device=device) / m

    u = torch.ones(n, device=device, dtype=torch.float32)
    v = torch.zeros(m, device=device, dtype=torch.float32)

    structured = StructuredSamplesLoss(
        eps=eps,
        n_iters=n_iters,
        normalize=False,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=16,
    )
    cosine_via_half_sq = SamplesLoss(
        loss="sinkhorn",
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        half_cost=True,
        debias=False,
        normalize=False,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=16,
    )

    torch.testing.assert_close(
        structured(a, u, x, b, v, y),
        cosine_via_half_sq(a, x, b, y),
        rtol=2e-4,
        atol=2e-4,
    )


def test_structured_gradients_match_dense_conditional_reference(device):
    eps = 0.25
    n_iters = 3
    a, u, q, b, v, k = _random_structured_inputs(
        device, n=40, m=36, r=32, seed=4, requires_grad=True
    )

    loss = StructuredSamplesLoss(
        eps=eps,
        n_iters=n_iters,
        normalize=False,
        allow_tf32=False,
        use_exp2=False,
        autotune=False,
        block_m=64,
        block_n=64,
        block_k=16,
    )
    val = loss(a, u, q, b, v, k)
    grad_u, grad_q, grad_v, grad_k = torch.autograd.grad(val, (u, q, v, k))

    with torch.no_grad():
        _, _, f_grad, g_grad = sinkhorn_flashstyle_structured_symmetric(
            a,
            u.detach(),
            q.detach(),
            b,
            v.detach(),
            k.detach(),
            eps=eps,
            n_iters=n_iters,
            last_extrapolation=True,
            allow_tf32=False,
            use_exp2=False,
            autotune=False,
            block_m=64,
            block_n=64,
            block_k=16,
            return_prelast=True,
        )
        grad_u_ref, grad_v_ref, grad_q_ref, grad_k_ref = _dense_conditional_structured_grads(
            a, u.detach(), q.detach(), b, v.detach(), k.detach(), f_grad, g_grad, eps
        )

    torch.testing.assert_close(grad_u, grad_u_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_v, grad_v_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_q, grad_q_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grad_k, grad_k_ref, rtol=2e-3, atol=2e-3)
