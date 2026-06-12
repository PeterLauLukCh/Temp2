import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LATENT_EXAMPLE = ROOT / "examples" / "images" / "latent_imagenet"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LATENT_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(LATENT_EXAMPLE))

from latent_ot import (  # noqa: E402
    LatentOTPlanSampler,
    LatentProjector,
    dense_sinkhorn_row_conditional_indices,
)


def make_projector(flat_dim=8, proj_dim=4):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    proj = torch.randn(proj_dim, flat_dim, generator=generator) / flat_dim**0.5
    mean = torch.zeros(proj_dim)
    std = torch.ones(proj_dim)
    return LatentProjector(proj, mean, std)


def test_projector_shape_and_standardization():
    projector = make_projector()
    z = torch.randn(5, 2, 2, 2)
    h = projector.project(z)
    assert h.shape == (5, 4)
    assert torch.isfinite(h).all()


def test_dense_sinkhorn_row_sampler_returns_valid_indices():
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    rows = torch.arange(4)
    j, metrics = dense_sinkhorn_row_conditional_indices(
        x,
        y,
        rows,
        eps=0.1,
        n_iters=10,
        cost_scale=0.5 / x.shape[1],
        seed=0,
        step=0,
    )
    assert j.shape == rows.shape
    assert int(j.min()) >= 0
    assert int(j.max()) < y.shape[0]
    assert metrics["sinkhorn_iters"] == 10


def test_independent_latent_sampler_keeps_shapes():
    projector = make_projector()
    sampler = LatentOTPlanSampler(mode="independent", projector=projector, context_size=8)
    z0 = torch.randn(6, 2, 2, 2)
    z1 = torch.randn(6, 2, 2, 2)
    sample = sampler.sample_pairs(z0, z1, step=0)
    assert sample.z0.shape == z0.shape
    assert sample.z1.shape == z1.shape
    assert sample.metrics["mode"] == "independent"
    assert sample.metrics["context_size"] == 6


def test_global_dense_sampler_uses_context_queue_on_cpu():
    projector = make_projector()
    sampler = LatentOTPlanSampler(
        mode="global_dense_sinkhorn",
        projector=projector,
        context_size=6,
        eps=0.1,
        sinkhorn_iters=5,
    )
    z0 = torch.randn(6, 2, 2, 2)
    z1 = torch.randn(6, 2, 2, 2)
    sample = sampler.sample_pairs(z0, z1, step=0)
    assert sample.z0.shape == z0.shape
    assert sample.z1.shape == z1.shape
    assert sample.metrics["context_size"] == 6
