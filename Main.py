import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from src.utils import *
from src.models import *
from src.dataset_configs import *


def get_trainer(model, args, device):
    if args.model in ['SimUFD']:
        return VAETrainer(model, args, device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

def main():

    parser = argparse.ArgumentParser(description='Bearing Early Fault Detection and Performance Degradation Assessment Framework')

    # Data Processing Arguments
    data_group = parser.add_argument_group('Data Processing')

    # Training Arguments
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('-optimizer', type=str, default='adam',
                                choices=['adam', 'adamw', 'sgd', 'adagrad'], help="Optimizer type")
    training_group.add_argument('-lr', type=float, default=0.005, help="Learning rate")
    training_group.add_argument('-epochs', type=int, default=500, help="Number of training epochs")
    training_group.add_argument('-batch_size', type=int, default=64, help="Batch size")

    # Learning Rate Scheduler
    training_group.add_argument('-lr_scheduler', type=bool, default=True, help="Use learning rate scheduler")
    training_group.add_argument('-scheduler_type', type=str, default='cosine', choices=['step', 'cosine'],
                                help="Learning rate scheduler type")
    training_group.add_argument('-scheduler_step_size', type=int, default=5, help="Step LR step size")
    training_group.add_argument('-scheduler_gamma', type=float, default=0.95, help="Learning rate decay factor")
    training_group.add_argument('-scheduler_T_max', type=int, default=25, help="Cosine annealing period")

    # Early Stopping
    training_group.add_argument('-early_stopping', type=bool, default=True, help="Enable early stopping")
    training_group.add_argument('-early_stopping_patience', type=int, default=50, help="Early stopping patience")
    training_group.add_argument('-early_stopping_delta', type=float, default=0.001, help="Early stopping threshold")

    # Gradient Clipping
    training_group.add_argument('-grad_clip', type=bool, default=True, help="Use gradient clipping")
    training_group.add_argument('-grad_clip_norm', type=float, default=1.0, help="Gradient clipping norm")

    # Hardware Arguments
    hardware_group = parser.add_argument_group('Hardware Configuration')
    hardware_group.add_argument('-use_gpu', type=bool, default=True, help="Use GPU")
    hardware_group.add_argument('-device', type=int, default=0, help="GPU device ID")
    hardware_group.add_argument('-num_workers', type=int, default=4, help="Number of data loader workers")

    # Execution Mode Arguments
    mode_group = parser.add_argument_group('Execution Mode')
    mode_group.add_argument('-train', type=bool, default=True, help="Train the model")
    mode_group.add_argument('-test', type=bool, default=True, help="Test the model")
    mode_group.add_argument('-threshold_method', type=str, default='sigma', choices=['sigma', 'percentile', 'iqr'],
                            help="Threshold calculation method")
    
    # Model Architecture Arguments
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('-model', type=str, default='SimUFD',
                             choices=['SimUFD'], help="Model architecture")

    # VAE Specific Arguments
    model_group.add_argument('-vae_recon_weight', type=float, default=1, help='VAE reconstruction loss weight')
    model_group.add_argument('-vae_kl_weight', type=float, default=1, help='VAE KL divergence loss weight')
    model_group.add_argument('-vae_latent_dim', type=int, default=64, help="VAE latent dimension")
    model_group.add_argument('-vae_flattened_dim', type=int, default=128, help="VAE flattened dimension")

    data_group.add_argument('-data_dir', type=str, default='./dataset/', help="Root directory for data files")
    data_group.add_argument('-channels', type=str, nargs='+', help="Channels to use, e.g.: --channels CH1 CH3")
    data_group.add_argument('-data_type', type=str,
                            choices=['IMS_2_1', 'XJ_2_2', 'XJ_2_3', 'XJ_2_5'],
                            default='XJ_2_5',
                            help='Fault type')

    args = parser.parse_args()
    set_data_params(args)

    set_seed(42)

    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    args.save_path = f'trained_models_{args.data_type}'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    processed_data = get_data(args)

    if args.model in ['SimUFD']:
        train_loader, val_loader, test_loader, all_loader = (
            create_dataset_from_processed_data(processed_data, args, is_frequency_domain=True))

    # Instantiate model
    try:
        print(f">>>>> Initializing {args.model} model <<<<<")
        model = get_model(args, device)
        print(f">>>>> {args.model} model initialized successfully <<<<<")
    except Exception as e:
        print(f">>>>> {args.model} model initialization failed: {str(e)} <<<<<")
        raise

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    trainer = get_trainer(model, args, device)

    # Train the model
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>> Start training {args.model} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train_losses, valid_losses, additional_metrics = trainer.train(train_loader, val_loader, optimizer, scheduler)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>> {args.model} training completed <<<<<<<<<<<<<<<<<<<<<<<<<<<")

    threshold = trainer.calculate_threshold(train_loader, val_loader)
    print(f"Calculated threshold: {threshold:.4f}")

    if args.test:
        print(f">>>>> Start testing {args.model} <<<<<")
        train_results, val_results, test_results = trainer.validate_and_visualize(all_loader, threshold)
        print(f">>>>> {args.model} testing completed <<<<<")
        print(os.path.join(args.save_path, f'HI_{args.model}_{args.data_type}.csv'))

if __name__ == '__main__':
    main()