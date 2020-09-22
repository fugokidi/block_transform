import os
import yaml
import torch
import shutil
import random
from torchvision import datasets, transforms
from easydict import EasyDict


class Normalize(torch.nn.Module):
    """
    https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20with%20Imagenet.ipynb
    """

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer("mean", torch.Tensor(mean))
        self.register_buffer("std", torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def parse_config_file(args):
    with open(args.work_path + "/config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    return config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_cifar10(config, worker_init_fn, test=False):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )
    num_workers = config.workers

    test_dataset = datasets.CIFAR10(
        config.data_path, train=False, transform=test_transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    if test:
        return test_loader

    train_dataset = datasets.CIFAR10(
        config.data_path, train=True, transform=train_transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, test_loader


def save_model(state, is_best, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        shutil.copyfile(filename + ".pth.tar", filename + "_best.pth.tar")


def load_model(path, model):
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
