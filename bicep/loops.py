"""Tools for training and evaluating models"""
from types import SimpleNamespace

import torch


class StopTraining(Exception):
    pass


class Trainer:
    def __init__(self, dataloader, loss_func):
        """Trainer

        Args:
            dataloader:
            loss_func:
            hooks:
        """
        self.loss_func = loss_func
        self.dataloader = dataloader

    def step(self, model, optimiser, data, target):
        """Take one training step for one batch

        Args:
            model:
            optimiser:
            data:
            target:

        Returns:
            (tuple[float, torch.tensor]): loss and outputs of model on batch

        """
        optimiser.zero_grad()
        output = model(data)
        loss = self.loss_func(output, target, reduction="mean")
        loss.backward()
        optimiser.step()
        return loss.item(), output

    def __call__(self, model, optimiser, niters, device, hooks=None):
        """Train the `model` on `device` for `niters`.

        Args:
            model:
            device:
            optimiser:
            niters:

        Returns:
        """
        state = SimpleNamespace(
            trainer=self,
            model=model,
            device=device,
            niters=niters,
            iter=0,
            epoch=0,
            data=None,
            target=None,
            loss=None,
            model_outputs=None,
        )

        try:
            model.train()
            i = 0
            epoch = 0
            while i < niters:
                epoch += 1
                for data, target in self.dataloader:
                    data = data.to(device)
                    target = target.to(device)
                    loss, outputs = self.step(
                        model=model, optimiser=optimiser, data=data, target=target
                    )
                    if hooks is not None:
                        hooks_to_exec = tuple(
                            hook for hook, freq in hooks if i % freq == 0
                        )

                        if hooks_to_exec:
                            state.iter = i
                            state.data = data
                            state.target = target
                            state.epoch = epoch
                            state.loss = loss
                            state.model_outputs = outputs

                            for hook in hooks_to_exec:
                                hook(state)

                            state.data = None
                            state.target = None
                            state.loss = None
                            state.model_outputs = None

                        i += 1
                        if i == niters:
                            break
        except StopTraining:
            return


def train(model, dataloader, loss_func, optimiser, niters, device, hooks=None):
    """train

    Args:
        model:
        dataloader:
        loss_func:
        optimiser:
        niters:
        device:
        hooks:

    Returns:
    """
    return Trainer(dataloader=dataloader, loss_func=loss_func)(
        model=model, optimiser=optimiser, niters=niters, device=device, hooks=hooks
    )


def evaluate(model, dataloader, metrics, device):
    outputs = list()
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        model_outputs = model(data)
        outputs.append(tuple(m(model, data, target, model_outputs) for m in metrics))
    return outputs
