from argparse import ArgumentParser
from functools import partial
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms

from bicep import Trainer, evaluate
from bicep import hooks
from bicep.utils import n_correct


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
                nn.Linear(28 * 28, 300),
                nn.ReLU(inplace=True),
                nn.Linear(300, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 10)
            )

    def forward(self, x):
        return self.classifier(x)


def cmd_args():
    parser = ArgumentParser("MNIST Example with bicep")
    parser = ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
            '--train-iters',
            type=int,
            default=1000,
            help="Number of batch iterations to train for. (default: %(default)d)"
        )

    parser.add_argument(
            '--batch-size',
            type=int,
            default=64,
            metavar='N',
            help='Input batch size for training (default: %(default)d)'
        )

    parser.add_argument(
            '--test-batch-size',
            type=int,
            default=1000,
            metavar='N',
            help='Input batch size for testing (default: %(default)d)'
        )

    parser.add_argument(
            '--lr',
            type=float,
            default=1.2e-3,
            metavar='LR',
            help='Learning rate (default: %(default)f'
        )

    parser.add_argument(
            '--no-cuda',
            action='store_true',
            help='Disables CUDA training'
        )

    parser.add_argument(
            '--seed',
            type=int,
            default=1,
            metavar='S',
            help='random seed (default: %(default)d)'
        )

    parser.add_argument(
            '--num-workers',
            type=int,
            default=1,
            help="Number of dataloader workers. (default %(default)d"
        )
    return parser.parse_args()


def get_dataloader(train, batch_size, kwds):
    def flatten(x):
        return torch.flatten(x, start_dim=1).squeeze()

    return torch.utils.data.DataLoader(
            datasets.MNIST(
                os.environ['TORCH_DATA'],
                train=train,
                download=True,
                transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           transforms.Lambda(flatten),
                       ])),
            batch_size=batch_size,
            shuffle=True,
            **kwds
        )

def init_weights(module):
    if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(module.weight)
        module.bias.data.fill_(0.0)


if __name__ == "__main__":
    args = cmd_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    dataloader_kwds = (
            {'num_workers': args.num_workers, 'pin_memory': True}
            if use_cuda else {}
            )

    train_loader = get_dataloader(
            train=True,
            batch_size=args.batch_size,
            kwds=dataloader_kwds
        )
    
    test_loader = get_dataloader(
            train=False,
            batch_size=args.test_batch_size,
            kwds=dataloader_kwds
            )
    
    model = Net().to(device)
    model.apply(init_weights)

    optimiser = Adam(
            model.parameters(),
            lr=1.2e-3
        )
    
    loss_func = F.cross_entropy
    accuracy = hooks.ClassificationAccuracy()
    train_loss = hooks.Recorder('loss')
    pbar = hooks.ProgressBar(
                    update_size=10,
                    notebook=False,
                    pbar_kwds=dict(total=args.train_iters)
            )

    train = Trainer(
            train_loader,
            loss_func=loss_func,
            hooks=[
                (train_loss, 10),
                (accuracy, 10),
                (pbar, pbar.update_size)
                ]
            )
    
    train(
            model,
            optimiser,
            device=device,
            niters=args.train_iters
        )

    print("Train losses: ", train_loss.results())
    print("Train accuracies: ", accuracy.results())
    
    test_results = evaluate(
            model,
            test_loader,
            metrics=[
                lambda m, d, t, mo: loss_func(mo, t, reduction='sum').item(),
                lambda m, d, t, mo: n_correct(mo, t)
                ],
            device=device
            )
    print("Test results: ", test_results)

