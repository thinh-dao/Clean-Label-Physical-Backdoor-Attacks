"""
Base class for all firewall defense implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import torch
import torch.nn as nn


class BaseFirewallDefense(ABC):
    """
    Abstract base class for all firewall defenses.
    
    This provides a common interface that all defense implementations should follow.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialize the defense.
        
        Args:
            model: The neural network model to defend
            **kwargs: Additional defense-specific parameters
        """
        self.model = model
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Prepare the model for defense operations."""
        if self.model is not None:
            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()
    
    @abstractmethod
    def scan(self, kettle: Any, **kwargs) -> Tuple[float, float]:
        """
        Perform defense scanning to detect backdoor samples.
        
        Args:
            kettle: Data kettle containing datasets and configuration
            **kwargs: Additional scan-specific parameters
            
        Returns:
            Tuple of (true_positive_rate, false_positive_rate)
        """
        pass
    
    def detect(self, dataset: Any, **kwargs) -> torch.Tensor:
        """
        Detect backdoor samples in a dataset.
        
        Args:
            dataset: Dataset to analyze
            **kwargs: Additional detection parameters
            
        Returns:
            Boolean tensor indicating detected backdoor samples
        """
        # Default implementation - subclasses can override
        raise NotImplementedError("Detect method not implemented for this defense")
    
    @property
    def name(self) -> str:
        """Get the name of this defense."""
        return self.__class__.__name__
