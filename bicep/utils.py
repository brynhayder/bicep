import random

import numpy as np
import torch

# Write docs, test, see if this needs more
class Registry:
    def __init__(self):
        self._register = dict()

    def __getitem__(self, key):
        return self._register[key]

    def __call__(self, name):
        def add(thing):
            self._register[name] = thing
            return thing

        return add


# This should probably be somewhere else
def evaluate(model, dataloader, metrics, device):
    outputs = list()
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        model_outputs = model(data)
        outputs.append(tuple(m(model, data, target, model_outputs) for m in metrics))
    return outputs


# This should probably be somewhere else
def n_correct(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    # Need to call .item() before the division to python casts to float
    # otherwise pytorch does integer division.
    return pred.eq(target.view_as(pred)).sum().item()


def set_reproducible(s=None):
    """set_reproducible
    Run PyTorch in reproducible mode, with optional seeding of random
    number generators.


    Args:
        s (Optional[int]): optional seeding of random number generators using
        `seed`.

    Notes:
     - See https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if s is not None:
        seed(s)


def seed(s):
    """seed
    Seed everything: Python random number generator, numpy random number generator,
    torch and torch cuda generators too.

    Args:
        s (int): The value of the seed
    """
    # It's not clear that this line can do anything
    # env vars normally set _before_ interpreter loaded
    # os.environ['PYTHONHASHSEED'] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# def evaluate(model, dataloader, hooks, device):
# for data, target in dataloader:
# data = data.to(device)
# target = target.to(device)
# model_outputs = model(data)
# state = SimpleNamespace(
# data=data,
# target=target,
# model=model,
# model_outputs=model_outputs
# )
# for hook in hooks:
# hook(state)
