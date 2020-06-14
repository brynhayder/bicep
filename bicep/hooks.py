from collections import OrderedDict
from contextlib import contextmanager
import os

import numpy as np
from tqdm import tqdm, tqdm_notebook
import torch


"""
To Do:
- This needs proper tests
- more examples, etc.
"""


@contextmanager
def eval_mode(model):
    try:
        model.eval()
        yield model
    finally:
        model.train()


# test and document this
# def hookify(**signature):
# def make_hook(func):
# def hook(state):
# return func(**{name: getattr(state, param) for name, param in signature.items()})
# return hook
# return make_hook


class Recorder:
    def __init__(self, name, detach=False, cpu=False):
        """Recorder

        Record a value from the training loop

        Args:
            name (str): Name of the state variable to read
            detach (bool): Whether to detach the variable from the graph
            cpu (bool): Whether to send the variable to cpu memory

        Possible Values for `name`:
            - trainer: the trainer instance
            - model: the model
            - device: the pytorch device
            - niters: number of iterations specified for training
            - iter: the current iteration
            - epoch: the current epoch
            - data: the training data at this iteration
            - target: the training targets at this iteration
            - loss: the value of the loss at this iteration
            - model_outputs: the outputs of the model
        
        Warning:
            - No checks will be done before applying detach or cpu,
            you need to check yourself that this is valid for the 
            `name` that you use
        """
        self.name = name
        self.detach = detach
        self.cpu = cpu
        self.results = list()

    def numpy(self):
        return np.array(self.results)

    def __call__(self, state):
        thing = getattr(state, self.name)
        thing = thing.detach() if self.detach else thing
        thing = thing.cpu() if self.cpu else thing
        self.results.append(thing)


class ClassificationAccuracy:
    def __init__(self):
        self.results = list()

    def numpy(self):
        return np.array(self.results)

    def __call__(self, state):
        pred = state.model_outputs.detach().cpu().argmax(dim=1, keepdim=True)
        # Need to call .item() before the division to python casts to float
        # otherwise pytorch does integer division.
        acc = pred.eq(state.target.detach().cpu().view_as(pred)).sum().item() / len(
            state.data
        )
        self.results.append(acc)


class ClassifierTester:
    def __init__(self, dataloader, loss_func):
        self.dataloader = dataloader
        self.loss_func = loss_func
        self.results = list()

    def numpy(self):
        return np.array(self.results)

    def __call__(self, state):
        return self.test(state.model, state.device)

    # Maybe want something about number of iterations here too?
    def test(self, model, device):
        with eval_mode(model) as model:
            test_loss = 0
            correct = 0
            n = 0
            for data, target in self.dataloader:
                n += len(data)
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                test_loss += self.loss_func(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            self.results.append((test_loss / n, correct / n))


# If generalised to various stopping
# criterion, etc., then this could be a useful hook
class CaptureEarlyStopStateDict:
    def __init__(self, validator, index=0):
        self.validator = validator
        self.index = index
        self.early_stop_sd = None
        self.best = np.inf  # not sure this is needed...

    def result(self):
        return self.early_stop_sd

    def copy_sd_to_cpu(self, sd):
        new_sd = OrderedDict()
        for k, v in sd.items():
            new_sd[k] = v.clone().detach().cpu()
        return new_sd

    def __call__(self, state):
        latest_result = self.validator._results[-1][self.index]
        if self.early_stop_sd is None or latest_result < self.best:
            self.best = latest_result
            self.early_stop_sd = self.copy_sd_to_cpu(state.model.state_dict())


class ParameterL2:
    def __init__(self, initialisation=None):
        self.results = list()
        self.initialisation = initialisation if initialisation is not None else {}

    def numpy(self):
        return np.array(self.results)

    def __call__(self, state):
        ## Can just use torch.pow(torch.norm here...
        self._results.append(
            torch.sqrt(
                sum(
                    torch.pow(p - self.initialisation.get(n, 0), 2).sum()
                    for n, p in state.model.named_parameters()
                )
            ).item()
        )


class ProgressBar:
    def __init__(self, update_size=1, notebook=False, pbar_kwds=None):
        self.update_size = update_size
        self.notebook = notebook
        self.pbar_kwds = pbar_kwds or {}

        self.pbar = None

    def _new_pbar(self, *args, **kwargs):
        f = tqdm_notebook if self.notebook else tqdm
        return f(*args, **kwargs)

    def __call__(self, state):
        if self.pbar is None:
            self.pbar = self._new_pbar(**self.pbar_kwds)
        self.pbar.update(self.update_size)


class ModelSaveHook:
    def __init__(self, folder, fmt=None):
        self.folder = folder
        self.fmt = fmt if fmt is not None else "model_iter_{iter}"

    def save(self, model, fname):
        return torch.save(model.state_dict(), os.path.join(self.folder, fname))

    def __call__(self, state):
        return self.save(state.model, fname=self.fmt.format(iter=state.iter))
