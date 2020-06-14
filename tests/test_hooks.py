import pytest

from bicep import hooks


@pytest.mark.parametrize(
    "name",
    "trainer model device niters iter epoch data target loss model_outputs".split(),
)
def test_load_recorder(name):
    hooks.Recorder(name)
