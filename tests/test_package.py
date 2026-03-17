import culprit


def test_the_version_matches_the_packaging():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    assert culprit.__version__ == project["project"]["version"]


def test_everything_exported_exists_and_is_documented():
    for name in culprit.__all__:
        thing = getattr(culprit, name)
        assert name == "IGNORE" or thing.__doc__, name


def test_every_registered_method_is_reachable_from_the_top_level():
    exported = {getattr(culprit, name) for name in culprit.__all__ if name != "IGNORE"}
    assert culprit.Unrolled in exported
    assert set(culprit.method_names()) == {
        "exact", "cg", "lissa", "low-rank", "ekfac", "kfac", "graddot", "tracin", "trak",
    }  # fmt: skip
