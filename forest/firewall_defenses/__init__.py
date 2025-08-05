"""
Firewall Defense Modules

This package contains various backdoor defense implementations organized by type.
"""

from .base import BaseFirewallDefense
from .factory import get_firewall
from .strip_defense import Strip
from .ibd_psc_defense import IBD_PSC
from .cognitive_distillation_defense import CognitiveDistillation, CognitiveDefense
from .frequency_defense import Frequency
from .scale_up_defense import ScaleUp
from .bad_expert_defense import BaDExpert

__all__ = [
    'BaseFirewallDefense',
    'get_firewall',
    'Strip',
    'IBD_PSC',
    'CognitiveDistillation',
    'CognitiveDefense',
    'Frequency',
    'ScaleUp',
    'BaDExpert',
]
