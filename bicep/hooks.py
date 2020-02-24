from collections import OrderedDict

import numpy as np
from tqdm import tqdm, tqdm_notebook


"""
Maybe with a metaclass I could intercept the call
function and add the output to the results cache. 
Almost certainly overkill!


To Do:
    - Write an MNIST example
    - Write some unit tests?
        - Testing for some of these things could be beneficial if
        you are to use them again.
        Maybe write a few examples first to see what its liek to use
"""

# Make abstract?
# Is there even a point in this?
class BaseHook:
    def __init__(self):
        self._results = list()

    def results(self):
        return np.array(self._results)
 

class Recorder(BaseHook):
    # possible choices for 'name'
    # trainer
    # model
    # device
    # niters
    # iter
    # epoch
    # data
    # target
    # loss
    # model_outputs
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def __call__(self, state):
        self._results.append(
                getattr(
                    state,
                    self.name
                )
            )


class ClassificationAccuracy(BaseHook):
    def __call__(self, state):
        pred = state.model_outputs.argmax(dim=1, keepdim=True)
        # Need to call .item() before the division to python casts to float
        # otherwise pytorch does integer division.
        acc = pred.eq(state.target.view_as(pred)).sum().item() / len(state.data)
        self._results.append(acc)


class ClassifierTester(BaseHook):
    def __init__(self, dataloader, loss_func):
        super().__init__()
        self.dataloader = dataloader
        self.loss_func = loss_func

    def __call__(self, state):
        return self.test(state.model, state.device)

    # Maybe want something about number of iterations here too?
    def test(self, model, device):
        test_loss = 0
        correct = 0
        n = 0
        for data, target in self.dataloader:
            n += len(data)
            data = data.to(device, non_blocking=True) 
            target = target.to(device, non_blocking=True)
            output = model(data)
            test_loss += self.loss_func(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        self._results.append((test_loss / n, correct / n))


# If generalised to various stopping 
# criterion, etc., then this could be a useful hook
class CaptureEarlyStopStateDict:
    def __init__(self, validator, index=0):
        self.validator = validator
        self.index = index
        self.early_stop_sd = None
        self._best = np.inf

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
