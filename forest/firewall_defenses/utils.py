"""
Common utilities and constants for defense implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple


def total_variation_loss(img: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """
    Compute total variation loss for regularization.
    
    Args:
        img: Input image tensor with shape (B, C, H, W)
        weight: Regularization weight
        
    Returns:
        Total variation loss tensor
    """
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight * (tv_h + tv_w) / (c * h * w)


def min_max_normalization(x: torch.Tensor) -> torch.Tensor:
    """
    Apply min-max normalization to tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Normalized tensor with values in [0, 1]
    """
    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min) / (x_max - x_min)


def eval_model(model: nn.Module, kettle: Any) -> Tuple[float, float]:
    """
    Evaluate model performance on clean and poisoned data.
    
    Args:
        model: Model to evaluate
        kettle: Data kettle with test datasets
        
    Returns:
        Tuple of (clean_accuracy, attack_success_rate)
    """
    from tqdm import tqdm
    
    model.eval()
    
    # Evaluate clean accuracy
    corrects = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target, idxs) in enumerate(tqdm(kettle.validloader, desc="Evaluating clean accuracy")):
            data, target = data.to('cuda:0'), target.to('cuda:0')
            output = model(data)
            pred = output.argmax(dim=1)
            corrects += torch.eq(pred, target).sum().item()
            total += target.size(0)
    
    clean_acc = corrects / total if total > 0 else 0.0

    # Evaluate attack success rate
    source_class = kettle.poison_setup['source_class'][0]
    target_class = kettle.poison_setup['target_class']

    corrects = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, _, _) in enumerate(tqdm(kettle.source_testloader[source_class], desc="Evaluating ASR")):
            data = data.to('cuda:0')
            output = model(data)
            pred = output.argmax(dim=1)
            corrects += torch.eq(pred, target_class).sum().item()
            total += data.size(0)
    
    asr = corrects / total if total > 0 else 0.0

    print(f"Clean Accuracy: {clean_acc*100:.2f}%, ASR: {asr*100:.2f}%")
    return clean_acc, asr


# Common constants
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.1
EPSILON = 1e-8
IMAGE_SIZE = 224
