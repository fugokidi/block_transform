"""
This module contains all the key-based transformation for adversarial defense.
"""

import torch
import torch.nn as nn
import numpy as np
import pyffx

__all__ = ["Shuffle", "NP", "FFX"]


class BlockTransform(nn.Module):
    """
    Generic class for block-wise transformation.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        assert (
            config.height % self.block_size == 0 | config.width % self.block_size == 0
        ), "Image not divisible by block_size"
        self.blocks_axis0 = int(config.height / self.block_size)
        self.blocks_axis1 = int(config.width / self.block_size)
        self.mean = torch.Tensor(config.mean)
        self.std = torch.Tensor(config.std)

    def segment(self, X):
        X = X.permute(0, 2, 3, 1)
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.block_size,
            self.blocks_axis1,
            self.block_size,
            3,
        )
        X = X.permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size * self.block_size * 3,
        )
        return X

    def integrate(self, X):
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size,
            self.block_size,
            3,
        )
        X = X.permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(
            -1,
            self.blocks_axis0 * self.block_size,
            self.blocks_axis1 * self.block_size,
            3,
        )
        X = X.permute(0, 3, 1, 2)
        return X

    def generate_key(self, seed, binary=False):
        torch.manual_seed(seed)
        key = torch.randperm(self.block_size * self.block_size * 3)
        if binary:
            key = key > len(key) / 2
        return key

    def normalize(self, X):
        return (X - self.mean.type_as(X)[None, :, None, None]) / self.std.type_as(X)[
            None, :, None, None
        ]

    def denormalize(self, X):
        return (X * self.std.type_as(X)[None, :, None, None]) + self.mean.type_as(X)[
            None, :, None, None
        ]

    def forward(self, X, decrypt=False):
        raise NotImplementedError


class Shuffle(BlockTransform):
    def __init__(self, config):
        super().__init__(config)
        self.key = self.generate_key(config.seed, binary=False)

    def forward(self, X, decrypt=False):
        X = self.segment(X)
        if decrypt:
            key = torch.argsort(self.key)
            X = X[:, :, :, key]
        else:
            X = X[:, :, :, self.key]
        X = self.integrate(X)
        return X.contiguous()


class NP(BlockTransform):
    def __init__(self, config):
        super().__init__(config)
        self.key = self.generate_key(config.seed, binary=True)

    def forward(self, X, decrypt=False):
        # uncomment the following during training
        # X = self.denormalize(X)
        X = self.segment(X)
        X[:, :, :, self.key] = 1 - X[:, :, :, self.key]
        X = self.integrate(X)
        # uncomment the following during training
        # X = self.normalize(X)
        return X.contiguous()


class FFX(BlockTransform):
    def __init__(self, config):
        super().__init__(config)
        self.key = self.generate_key(config.seed, binary=True)
        self.lookup, self.relookup = self.generate_lookup(config.password)
        self.lookup, self.relookup = self.lookup.cuda(), self.relookup.cuda()

    def generate_lookup(self, password="password"):
        password = str.encode(password)
        fpe = pyffx.Integer(password, length=3)
        f = lambda x: fpe.encrypt(x)
        g = lambda x: fpe.decrypt(x)
        f = np.vectorize(f)
        g = np.vectorize(g)
        lookup = f(np.arange(256))
        relookup = g(np.arange(1000))
        lookup = torch.from_numpy(lookup)
        relookup = torch.from_numpy(relookup)
        return lookup, relookup

   def forward(self, X, decrypt=False):
        # uncomment the following during training
        # X = self.denormalize(X)
        X = self.segment(X)
        if decrypt:
            X = (X * self.lookup.max()).long()
            X[:, :, :, self.key] = self.relookup[X[:, :, :, self.key]]
            X = X.float()
            X = X / 255.0
        else:
            # important: without it cuda trigerring devise assertion error with index out of bound
            X = torch.clamp(X, 0, 1)
            X = (X * 255).long()
            X[:, :, :, self.key] = self.lookup[X[:, :, :, self.key]].clone()
            X = X.float()
            X = X / self.lookup.max()
        X = self.integrate(X)
        # uncomment the following during training
        # X = self.normalize(X)
        return X.contiguous()
