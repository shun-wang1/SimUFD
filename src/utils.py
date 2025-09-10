import pandas as pd
import argparse
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os 
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

def split_dataset(dataset, args):
    """
    Split dataset into train, validation, and test loaders based on timesteps.
    """
    timestep_indices = defaultdict(list)
    for idx in range(len(dataset)):
        timestep = dataset[idx][0]
        timestep_indices[int(timestep.item())].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    sorted_timesteps = sorted(timestep_indices.keys())

    for i, timestep in enumerate(sorted_timesteps):
        if i < args.train_timesteps:
            train_indices.extend(timestep_indices[timestep])
        elif i < args.val_timesteps:
            val_indices.extend(timestep_indices[timestep])
        else:
            test_indices.extend(timestep_indices[timestep])

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False
    )
    all_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"Dataset split info:")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Testing samples: {len(test_indices)}")
    return train_loader, val_loader, test_loader, all_loader

def create_dataset_from_processed_data(
        processed_data: Tuple,
        args,
        is_frequency_domain: bool = True,
    ):

    if is_frequency_domain:
        timesteps, frequencies, spectrum = processed_data
        dataset = FrequencyDomainDataset(timesteps, frequencies, spectrum, args)
    else:
        timesteps, signals = processed_data[:2]
        dataset = TimeDomainDataset(timesteps, signals, args)

    train_loader, val_loader, test_loader, all_loader = split_dataset(dataset, args)
    return train_loader, val_loader, test_loader, all_loader


class TimeDomainDataset(Dataset):
    def __init__(
            self,
            timesteps: List[int],
            signals: List[np.ndarray],
            args
    ):
        self.timesteps = timesteps
        self.signals = signals
        self.args = args

        # Validate data
        if len(timesteps) != len(signals):
            raise ValueError("Inconsistent data lengths in time domain dataset")

    def __len__(self) -> int:
        return len(self.timesteps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep = torch.tensor(self.timesteps[idx], dtype=torch.int32)
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).to(self.args.device)
        return timestep, signal


class FrequencyDomainDataset(Dataset):
    def __init__(
            self,
            timesteps: List[int],
            frequencies: List[np.ndarray],
            spectrum: List[np.ndarray],
            args
    ):
        self.timesteps = timesteps
        self.frequencies = frequencies
        self.spectrum = spectrum
        self.args = args

        # Validate data
        if not (len(timesteps) == len(frequencies) == len(spectrum)):
            raise ValueError("Inconsistent data lengths in frequency domain dataset")

    def __len__(self) -> int:
        return len(self.timesteps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timestep = torch.tensor(self.timesteps[idx], dtype=torch.int32)
        freqs = torch.tensor(self.frequencies[idx], dtype=torch.float32).to(self.args.device)
        spectrum = torch.tensor(self.spectrum[idx], dtype=torch.float32).to(self.args.device)

        return timestep, freqs, spectrum

def get_optimizer(model, args):
    """
    Initialize optimizer based on args.
    """
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'adagrad': torch.optim.Adagrad
    }
    optimizer_cls = optimizers.get(args.optimizer.lower(), torch.optim.Adam)
    if args.optimizer.lower() == 'sgd':
        return optimizer_cls(model.parameters(), lr=args.lr, momentum=0.9)
    return optimizer_cls(model.parameters(), lr=args.lr)

def get_scheduler(optimizer, args):
    """
    Initialize learning rate scheduler based on args.
    """
    schedulers = {
        'step': lambda: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        ),
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.scheduler_T_max
        ),
    }
    return schedulers.get(args.scheduler_type, schedulers['step'])()
