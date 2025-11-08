import pytest
import torch


@pytest.fixture(autouse=True)
def _float64_on_cpu():
    """Tests compare against dense references, so they run in double precision."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def pytest_collection_modifyitems(config, items):
    if torch.backends.mps.is_available():
        return
    skip = pytest.mark.skip(reason="needs the Apple GPU")
    for item in items:
        if "mps" in item.keywords:
            item.add_marker(skip)
