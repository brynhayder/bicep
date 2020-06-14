import random

import numpy as np
import pytest
import torch

from bicep import utils

TEST_RANDOM_SEED = 2020

# don't really need to bother testing this one... right?
# @pytest.fixture()
# def setup_seed():
# random.seed(TEST_RANDOM_SEED)
# np.random.seed(TEST_RANDOM_SEED)
# torch.manual_seed(TEST_RANDOM_SEED)
# torch.cuda.manual_seed_all(TEST_RANDOM_SEED)

# rs = random.getstate()
# nprs = np.random.get_state()
# torchrs = torch.random.get_rng_state()
# if torch.cuda.is_available():
# torch_cuda_rs = torch.cuda.get_rng_state()
# else:
# torch_cuda_rs = None

# yield rs, nprs, torchrs, torch_cuda_rs

# random.seed(TEST_RANDOM_SEED)
# np.random.seed(TEST_RANDOM_SEED)
# torch.manual_seed(TEST_RANDOM_SEED)
# torch.cuda.manual_seed_all(TEST_RANDOM_SEED)


# def test_seed():
# random.seed(0)
# np.random.seed(0)


@pytest.fixture()
def setup_set_reproducible():
    start_det = torch.backends.cudnn.deterministic
    start_bench = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.deterministic = start_det
    torch.backends.cudnn.benchmarki = start_bench


def test_set_reproducible(setup_set_reproducible):
    utils.set_reproducible()
    assert torch.backends.cudnn.deterministic == True
    assert torch.backends.cudnn.benchmark == False
