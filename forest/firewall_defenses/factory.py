"""
Factory function for creating firewall defense instances.
"""

from typing import Any, Union
import torch.nn as nn

from .base import BaseFirewallDefense
from .strip_defense import Strip
from .ibd_psc_defense import IBD_PSC
from .cognitive_distillation_defense import  CognitiveDefense
from .frequency_defense import Frequency
from .scale_up_defense import ScaleUp
from .bad_expert_defense import BaDExpert


def get_firewall(firewall_name: str, model: Any, dataset: Any = None, **kwargs) -> BaseFirewallDefense:
    """
    Factory function to create firewall defense instances.
    
    Args:
        firewall_name: Name of the defense to create
        model: The model to defend (can be a wrapper with .model attribute)
        dataset: Dataset for defense initialization (optional)
        **kwargs: Additional defense-specific parameters
        
    Returns:
        Instance of the requested defense
        
    Raises:
        NotImplementedError: If the requested defense is not implemented
    """
    # Extract the actual model if it's wrapped
    actual_model = getattr(model, 'model', model)
    
    firewall_name_lower = firewall_name.lower()
    
    if firewall_name_lower == 'strip':
        return Strip(actual_model, **kwargs)
    
    elif firewall_name_lower == 'ibd_psc':
        return IBD_PSC(actual_model, valset=dataset, **kwargs)
    
    elif firewall_name_lower == 'cognitive_distillation':
        return CognitiveDefense(model, **kwargs)
    
    elif firewall_name_lower == 'frequency':
        return Frequency(**kwargs)
    
    elif firewall_name_lower == 'scale_up':
        return ScaleUp(model, dataset, **kwargs)
    
    elif firewall_name_lower == 'bad_expert':
        return BaDExpert(model, dataset, **kwargs)
    
    else:
        raise NotImplementedError(f'Firewall "{firewall_name}" is not implemented. '
                                f'Available options: strip, ibd_psc, cognitive_distillation, '
                                f'frequency, scale_up, bad_expert')
