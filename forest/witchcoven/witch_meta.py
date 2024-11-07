"""Main class, holding information about models and training/testing routines."""

import torch
import higher
import copy

from collections import OrderedDict

from ..utils import cw_loss, write
from ..consts import BENCHMARK, NON_BLOCKING, NORMALIZE, FINETUNING_LR_DROP
from forest.data.datasets import normalization
from ..victims.training import _split_data
torch.backends.cudnn.benchmark = BENCHMARK
from .modules import MetaMonkey

from .witch_base import _Witch


class WitchMetaPoison(_Witch):
    """Brew metapoison with given arguments.

    Note: This function does not work in single-model-multi-GPU mode, due to the weights being fixed to a single GPU.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """
    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            model = MetaMonkey(copy.deepcopy(model))

            for _ in range(self.args.nadapt):
                outputs = model(inputs, model._parameters)
                prediction = (outputs.data.argmax(dim=1) == labels).sum()

                poison_loss = criterion(outputs, labels)
                poison_grad = torch.autograd.grad(poison_loss, model._parameters.values(),
                                                  retain_graph=True, create_graph=True, only_inputs=True)

                current_lr = optimizer.param_groups[0]['lr']
                model._parameters = OrderedDict((name, param - current_lr * grad_part)
                                               for ((name, param), grad_part) in zip(model._parameters.items(), poison_grad))
            
            batch_size = 32 
            outputs = []

            for i in range(0, len(sources), batch_size):
                source_outs = model(sources[i:i+batch_size], model._parameters)
                outputs.extend(source_outs)
            
            source_outs = torch.stack(outputs).to(**self.setup)
            source_loss = criterion(source_outs, target_classes)
            source_loss.backward(retain_graph=self.retain)

            return source_loss.detach().cpu(), prediction.detach().cpu()
        return closure
