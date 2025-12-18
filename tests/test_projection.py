import pytest
import torch

from culprit.projection import RademacherProjector, SparseProjector

PROJECTORS = {
    "sparse": lambda seed=0: SparseProjector(300, 64, nonzeros=4, seed=seed),
    "rademacher": lambda seed=0: RademacherProjector(300, 64, seed=seed, slab=128),
}


def vectors(n=5, size=300, seed=0):
    return torch.randn(n, size, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("kind", PROJECTORS)
def test_projection_is_multiplication_by_a_fixed_matrix(kind):
    projector = PROJECTORS[kind]()
    matrix = projector.dense()
    assert matrix.shape == (300, 64)
    g = vectors()
    assert torch.allclose(projector.project(g), g @ matrix, atol=1e-12)
    assert torch.allclose(projector.project(g), projector.project(g))


@pytest.mark.parametrize("kind", PROJECTORS)
def test_the_matrix_depends_only_on_the_seed(kind):
    torch.manual_seed(0)
    first = PROJECTORS[kind](seed=3).dense()
    torch.manual_seed(1)
    assert torch.equal(first, PROJECTORS[kind](seed=3).dense())
    assert not torch.equal(first, PROJECTORS[kind](seed=4).dense())


def test_sparse_rows_have_one_entry_per_block():
    matrix = SparseProjector(300, 64, nonzeros=4).dense()
    assert torch.equal((matrix != 0).sum(1), torch.full((300,), 4))
    assert torch.allclose(matrix[matrix != 0].abs(), torch.tensor(0.5))  # 1 / sqrt(4)
    per_block = (matrix.view(300, 4, 16) != 0).sum(2)
    assert torch.equal(per_block, torch.ones(300, 4, dtype=torch.long))


def test_rademacher_entries_are_plus_or_minus_one_over_root_k():
    matrix = RademacherProjector(300, 64, slab=128).dense()
    assert torch.allclose(matrix.abs(), torch.full((300, 64), 1 / 8))
    assert abs(matrix.sign().mean().item()) < 0.02


def test_sparse_out_size_must_split_into_blocks():
    with pytest.raises(ValueError, match="multiple of nonzeros"):
        SparseProjector(300, 62, nonzeros=4)


@pytest.mark.parametrize("kind", PROJECTORS)
def test_projector_rejects_the_wrong_width(kind):
    with pytest.raises(ValueError, match="expected 300 columns"):
        PROJECTORS[kind]().project(torch.zeros(2, 301))


@pytest.mark.parametrize("kind", PROJECTORS)
def test_inner_products_are_preserved_in_expectation(kind):
    g = vectors(n=2)
    true = (g[0] @ g[1]).item()
    estimates = torch.stack(
        [(lambda p: p[0] @ p[1])(PROJECTORS[kind](seed=seed).project(g)) for seed in range(400)]
    )
    standard_error = estimates.std().item() / 400**0.5
    assert abs(estimates.mean().item() - true) < 4 * standard_error


@pytest.mark.parametrize(
    "projector",
    [SparseProjector(5000, 2048, nonzeros=4), RademacherProjector(5000, 2048)],
    ids=["sparse", "rademacher"],
)
def test_norms_and_angles_survive_a_wide_projection(projector):
    g = vectors(n=20, size=5000)
    projected = projector.project(g)
    ratios = projected.norm(dim=1) / g.norm(dim=1)
    assert ratios.min() > 0.9 and ratios.max() < 1.1
    cosine = torch.nn.functional.cosine_similarity
    assert (cosine(projected[:10], projected[10:]) - cosine(g[:10], g[10:])).abs().max() < 0.1


def test_a_sparse_gradient_keeps_its_norm_better_with_more_nonzeros():
    """A vector with all its mass in a few coordinates is where one bucket per row hurts."""
    g = torch.zeros(1, 5000)
    g[0, :8] = 1.0

    def spread(nonzeros):
        norms = torch.stack(
            [
                SparseProjector(5000, 64, nonzeros=nonzeros, seed=seed).project(g).norm()
                for seed in range(200)
            ]
        )
        return (norms.square() / g.square().sum()).std().item()

    assert spread(8) < spread(1)


@pytest.mark.mps
@pytest.mark.parametrize("kind", PROJECTORS)
def test_projection_on_the_gpu_matches_the_cpu(kind):
    projector = PROJECTORS[kind]()
    g = vectors().float()
    on_gpu = projector.project(g.to("mps"))
    assert on_gpu.device.type == "mps"
    assert torch.allclose(on_gpu.cpu(), projector.project(g), atol=1e-5)
