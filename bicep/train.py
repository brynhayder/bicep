"""Tools for training and evaluating models"""
from types import SimpleNamespace

import torch


class StopTraining(Exception):
    pass


class Trainer:
    def __init__(self, dataloader, loss_func, hooks=None):
        """Trainer

        Args:
            dataloader:
            loss_func:
            hooks:
        """
        self.loss_func = loss_func
        self.dataloader = dataloader
        self.hooks = hooks if hooks is not None else []
  
    def add_hook(self, hook, freq):
        """add_hook

        Args:
            hook:
            freq:

        Returns:
        """
        self.hooks.append((hook, freq))

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
        loss = self.loss_func(output, target, reduction='mean')
        loss.backward()
        optimiser.step()
        return loss.item(), output
    
    def __call__(self, model, optimiser, niters, device):
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
                model_outputs=None
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
                            model=model,
                            optimiser=optimiser,
                            data=data,
                            target=target
                        )
    
                    hooks_to_exec = tuple(
                            hook for hook, freq in self.hooks if i % freq == 0
                        )

                    if hooks_to_exec:
                        state.iter = i
                        state.data = data
                        state.target = target
                        state.epoch = epoch
                        state.loss = loss
                        state.model_outputs = outputs
                        
                        model.eval()
                        with torch.no_grad():
                            for hook in hooks_to_exec:
                                hook(state)

                        state.data = None
                        state.target = None
                        state.outputs = None

                        model.train()

                    i += 1
                    if i == niters:
                        break
        except StopTraining:
            return



