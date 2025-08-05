"""
STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
"""

import random
from typing import Tuple, List, Any
import torch
import torch.nn as nn
from tqdm import tqdm

from .base import BaseFirewallDefense
from .utils import DEFAULT_BATCH_SIZE
from ..consts import NON_BLOCKING


class Strip(BaseFirewallDefense):
    """
    STRIP defense implementation.
    
    STRIP works by superimposing benign inputs on suspicious inputs and measuring
    the entropy of the resulting predictions. Backdoored inputs typically maintain
    high entropy regardless of superimposition.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialize STRIP defense.
        
        Args:
            model: The neural network model to defend
        """
        super().__init__(model, **kwargs)

    def scan(self, kettle: Any, defense_fpr: float = 0.1, batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[float, float]:
        """
        Scan for backdoor samples using STRIP method.
        
        Args:
            kettle: Data kettle containing datasets and configuration
            defense_fpr: Desired false positive rate for threshold selection
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (true_positive_rate, false_positive_rate)
        """
        # Choose decision boundary with clean validation set
        clean_entropy = self._compute_clean_entropies(kettle, batch_size)
        threshold_low = self._compute_threshold(clean_entropy, defense_fpr)
        
        # Calculate TPR on poisoned samples
        tpr = self._compute_tpr(kettle, threshold_low)
        
        # Calculate FPR on clean samples
        fpr = self._compute_fpr(kettle, threshold_low)
        
        print(f"True Positive Rate (TPR): {tpr:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        
        return tpr, fpr
    
    def _compute_clean_entropies(self, kettle: Any, batch_size: int) -> torch.Tensor:
        """Compute entropies for clean validation samples."""
        clean_entropy = []
        clean_set_loader = torch.utils.data.DataLoader(
            kettle.validset, batch_size=batch_size, shuffle=False
        )
        
        for _input, _label, _ in tqdm(clean_set_loader, desc="Computing clean entropies"):
            _input = _input.to(**kettle.setup)
            _label = _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            entropies = self._check(_input, _label, kettle.validset)
            clean_entropy.extend(entropies.cpu().numpy())
        
        clean_entropy = torch.FloatTensor(clean_entropy)
        return clean_entropy.sort()[0]
    
    def _compute_threshold(self, clean_entropy: torch.Tensor, defense_fpr: float) -> float:
        """Compute detection threshold based on desired FPR."""
        threshold_idx = int(defense_fpr * len(clean_entropy))
        return float(clean_entropy[threshold_idx])
    
    def _compute_tpr(self, kettle: Any, threshold: float) -> float:
        """Compute true positive rate on poisoned samples."""
        all_entropy = []
        source_class = kettle.poison_setup['source_class'][0]
        
        for _input, _label, _ in tqdm(kettle.source_testloader[source_class], desc="Computing TPR"):
            _input = _input.to(**kettle.setup)
            _label = _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            entropies = self._check(_input, _label, kettle.validset)
            all_entropy.extend(entropies.cpu().numpy())
        
        all_entropy = torch.FloatTensor(all_entropy)
        true_positives = (all_entropy < threshold).sum().item()
        return true_positives / len(kettle.source_testloader[source_class].dataset)
    
    def _compute_fpr(self, kettle: Any, threshold: float) -> float:
        """Compute false positive rate on clean samples."""
        all_entropy = []
        
        for _input, _label, _ in tqdm(kettle.validloader, desc="Computing FPR"):
            _input = _input.to(**kettle.setup)
            _label = _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            entropies = self._check(_input, _label, kettle.validset)
            all_entropy.extend(entropies.cpu().numpy())
        
        all_entropy = torch.FloatTensor(all_entropy)
        false_positives = (all_entropy < threshold).sum().item()
        return false_positives / len(kettle.validloader.dataset)
    
    def _check(self, _input: torch.Tensor, _label: torch.Tensor, source_set: Any, N: int = 200) -> torch.Tensor:
        """
        Check entropy for inputs by superimposing with random samples.
        
        Args:
            _input: Input batch to check
            _label: Labels for input batch
            source_set: Dataset to sample superimposition candidates from
            N: Number of samples to use for superimposition
            
        Returns:
            Mean entropy values for the batch
        """
        entropy_list = []
        
        # Randomly sample N indices from source set
        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:N]
        
        with torch.no_grad():
            for i in samples:
                X, _, _ = source_set[i]
                X = X.cuda()
                _test = self._superimpose(_input, X)
                entro = self._entropy(_test).cpu().detach()
                entropy_list.append(entro)
        
        return torch.stack(entropy_list).mean(0)
    
    def _superimpose(self, input1: torch.Tensor, input2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """
        Superimpose two inputs.
        
        Args:
            input1: First input tensor
            input2: Second input tensor  
            alpha: Mixing coefficient
            
        Returns:
            Superimposed result
        """
        return input1 + alpha * input2
    
    def _entropy(self, _input: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction entropy.
        
        Args:
            _input: Input tensor
            
        Returns:
            Entropy values
        """
        p = torch.nn.Softmax(dim=1)(self.model(_input)) + 1e-8
        return (-p * p.log()).sum(1)
