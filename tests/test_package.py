import culprit


def test_version_is_set():
    assert culprit.__version__


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
