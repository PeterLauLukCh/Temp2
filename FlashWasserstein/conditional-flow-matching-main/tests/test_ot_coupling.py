import pytest
import torch

from torchcfm.ot_coupling import (
    FeatureProjector,
    OTCouplingSampler,
    dense_sinkhorn_row_conditional_indices,
    normalize_coupling_mode,
)


def test_mode_aliases():
    assert normalize_coupling_mode("global_flash_sinkhorn") == "flash_global_entropic"
    assert normalize_coupling_mode("global_dense_sinkhorn") == "allgather_dense_entropic"
    assert normalize_coupling_mode("local_pot_exact_row") == "local_exact_pot"


def test_feature_projector_identity_and_random_projection():
    x = torch.randn(5, 3, 4, 4)
    identity = FeatureProjector(0, seed=0)
    h_identity = identity.project(x)
    assert h_identity.shape == (5, 48)

    projected = FeatureProjector(7, seed=0)
    h_projected = projected.project(x)
    assert h_projected.shape == (5, 7)
    assert torch.isfinite(h_projected).all()


def test_dense_sinkhorn_row_sampler_returns_valid_indices():
    x = torch.randn(8, 4)
    y = torch.randn(10, 4)
    rows = torch.arange(4)
    j, metrics = dense_sinkhorn_row_conditional_indices(
        x,
        y,
        rows,
        eps=0.1,
        n_iters=5,
        cost_scale=0.5 / x.shape[1],
        seed=0,
        step=0,
    )
    assert j.shape == rows.shape
    assert int(j.min()) >= 0
    assert int(j.max()) < y.shape[0]
    assert metrics["sinkhorn_iters"] == 5


def test_independent_sampler_keeps_shapes():
    sampler = OTCouplingSampler(mode="independent", context_size=8)
    x0 = torch.randn(6, 3, 4, 4)
    x1 = torch.randn(6, 3, 4, 4)
    sample = sampler.sample_pairs(x0, x1, step=0)
    assert sample.x0.shape == x0.shape
    assert sample.x1.shape == x1.shape
    assert sample.metrics["mode"] == "independent"
    assert sample.metrics["context_size"] == 6


def test_local_entropic_sampler_keeps_shapes_on_cpu():
    sampler = OTCouplingSampler(mode="local_entropic", context_size=8, eps=0.1, sinkhorn_iters=5)
    x0 = torch.randn(6, 3, 4, 4)
    x1 = torch.randn(6, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    sample = sampler.sample_pairs(x0, x1, y1_local=labels, step=0)
    assert sample.x0.shape == x0.shape
    assert sample.x1.shape == x1.shape
    assert sample.y1.shape == labels.shape
    assert sample.metrics["context_size"] == 6


def test_global_dense_sampler_uses_rectangular_context_on_cpu():
    sampler = OTCouplingSampler(
        mode="allgather_dense_entropic",
        context_size=10,
        eps=0.1,
        sinkhorn_iters=5,
    )
    x0 = torch.randn(6, 3, 4, 4)
    x1 = torch.randn(6, 3, 4, 4)
    first = sampler.sample_pairs(x0, x1, step=0)
    second = sampler.sample_pairs(x0, x1, step=1)
    assert first.x0.shape == x0.shape
    assert second.x1.shape == x1.shape
    assert second.metrics["source_context_size"] == 6
    assert second.metrics["context_size"] == 10


def test_global_queue_drops_stale_labels_when_labels_are_omitted():
    sampler = OTCouplingSampler(
        mode="allgather_dense_entropic",
        context_size=10,
        eps=0.1,
        sinkhorn_iters=5,
    )
    x0 = torch.randn(6, 3, 4, 4)
    x1 = torch.randn(6, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    with_labels = sampler.sample_pairs(x0, x1, y1_local=labels, step=0)
    without_labels = sampler.sample_pairs(x0, x1, step=1)
    assert with_labels.y1 is not None
    assert without_labels.y1 is None


def test_class_aware_local_sampler_preserves_labels():
    sampler = OTCouplingSampler(
        mode="local_entropic",
        context_size=8,
        eps=0.1,
        sinkhorn_iters=5,
        class_aware=True,
    )
    x0 = torch.randn(6, 3, 4, 4)
    x1 = torch.randn(6, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    sample = sampler.sample_pairs(x0, x1, y1_local=labels, step=0)
    assert sample.y1 is not None
    assert torch.equal(sample.y1, labels)
    assert sample.metrics["class_aware"] is True


def test_flash_sampler_requires_cuda_on_cpu():
    sampler = OTCouplingSampler(mode="flash_global_entropic", context_size=8, eps=0.1, sinkhorn_iters=2)
    x0 = torch.randn(4, 3, 4, 4)
    x1 = torch.randn(4, 3, 4, 4)
    if torch.cuda.is_available():
        pytest.skip("CPU-only error path is not relevant when CUDA is available.")
    with pytest.raises(ValueError, match="requires CUDA"):
        sampler.sample_pairs(x0, x1, step=0)
