"""Weighted sampler for ConcatDataset-based mixed training."""
from __future__ import annotations

import math
import torch
from torch.utils.data import Sampler, ConcatDataset


class DistributedWeightedSampler(Sampler):
    """Per-sample weighted random sampler that supports DDP sharding.

    Each sample's weight = dataset_weight / dataset_size, so smaller datasets
    with higher weight get oversampled proportionally.  In non-distributed mode
    (num_replicas=1, rank=0) it behaves like a plain WeightedRandomSampler.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        weights: list[float],
        num_replicas: int = 1,
        rank: int = 0,
        replacement: bool = True,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.epoch = 0

        # Build per-sample weight vector
        sizes = [0] + list(dataset.cumulative_sizes)
        sample_weights = torch.zeros(len(dataset), dtype=torch.double)
        for i, w in enumerate(weights):
            ds_len = sizes[i + 1] - sizes[i]
            sample_weights[sizes[i]:sizes[i + 1]] = w / ds_len
        # Normalize
        sample_weights /= sample_weights.sum()
        self.sample_weights = sample_weights

        self.total_size = int(math.ceil(len(dataset) / num_replicas)) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank * 1000)
        indices = torch.multinomial(
            self.sample_weights,
            num_samples=self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        # Shard
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples
