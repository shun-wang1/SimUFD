import torch
import random
import numpy as np
import os
from model.VAE import *

def set_seed(seed):
    """
    Set random seed for reproducibility across libraries and CUDA.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_model(args, device):
    """
    Model factory function. Returns model instance based on arguments.
    """
    if args.model == 'SimUFD':
        return Model_VAE(latent_dim=args.vae_latent_dim, flattened_dim=args.vae_flattened_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

