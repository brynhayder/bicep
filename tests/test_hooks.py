import pytest

from bicep import hooks

def test_base_hook():
    class Hook(hooks.BaseHook):
        def results():
            pass
    Hook()


@pytest.mark.parametrize(
        'name', 
        "trainer model device niters iter epoch data target loss model_outputs".split()
    )   
def test_load_recorder(name):
    hooks.Recorder(name)
