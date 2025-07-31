"""Interface for poison recipes."""
from .witch_gradient_matching import WitchGradientMatching, WitchGradientMatchingNoisy, WitchGradientMatchingHidden, WitchMatchingMultiSource
from .witch_hidden_trigger import WitchHTBD
from .witch_label_consistent import WitchLabelConsistent
from .witch_meta import WitchMetaPoison, WitchMetaPoison_v3, WitchMetaPoisonHigher, WitchMetaPoisonFirstOrder
from .witch_base import _Witch
from .witch_parameter_matching import WitchMTTP
from .witch_feature_matching import Witch_FM

import torch

def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'gradient-matching':
        return WitchGradientMatching(args, setup)
    elif args.recipe == 'gradient-matching-private':
        return WitchGradientMatchingNoisy(args, setup)
    elif args.recipe == 'gradient-matching-hidden':
        return WitchGradientMatchingHidden(args, setup)
    elif args.recipe == 'meta':
        return WitchMetaPoison(args, setup)
    elif args.recipe == 'meta-v2':
        return WitchMetaPoisonHigher(args, setup)
    elif args.recipe == 'meta-v3':
        return WitchMetaPoison_v3(args, setup)
    elif args.recipe == 'meta-first-order':
        return WitchMetaPoisonFirstOrder(args, setup)
    elif args.recipe == 'gradient-matching-mt':
        return WitchMatchingMultiSource(args, setup)
    elif args.recipe == 'hidden-trigger':
        return WitchHTBD(args, setup)
    elif args.recipe == 'feature-matching':
        return Witch_FM(args, setup)
    elif args.recipe == 'label-consistent':
        return WitchLabelConsistent(args, setup)
    elif args.recipe == 'mttp':
        return WitchMTTP(args, setup)
    elif args.recipe == 'naive' or 'dirty-label' in args.recipe:
        return None
    else:
        raise NotImplementedError()

__all__ = ['Witch']
