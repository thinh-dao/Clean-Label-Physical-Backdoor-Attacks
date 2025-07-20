"""Main class, holding information about models and training/testing routines."""

import torch
import higher
import copy

from collections import OrderedDict

from ..utils import cw_loss, write
from ..consts import BENCHMARK
from forest.data.datasets import normalization
from ..victims.training import _split_data
torch.backends.cudnn.benchmark = BENCHMARK
from .modules import MetaMonkey, MetaMonkeyParallel
from .witch_base import _Witch

class WitchMetaPoison(_Witch):
    """Brew metapoison with given arguments.

    Note: This function does not work in single-model-multi-GPU mode, due to the weights being fixed to a single GPU.

    "Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw."

    """
    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            model = MetaMonkey(model)

            for _ in range(self.args.nadapt):
                outputs = model(inputs, model._parameters)
                prediction = (outputs.data.argmax(dim=1) == labels).sum()

                poison_loss = criterion(outputs, labels)
                
                # Filter parameters that require gradients
                trainable_params = OrderedDict((name, param) for name, param in model._parameters.items() 
                                             if param.requires_grad)
                
                # Only compute gradients for parameters that require gradients
                poison_grad = torch.autograd.grad(poison_loss, trainable_params.values(),
                                                 retain_graph=True, create_graph=True, only_inputs=True)

                current_lr = optimizer.param_groups[0]['lr']
                
                # Create a new OrderedDict with updated parameters
                new_params = OrderedDict()
                param_idx = 0
                
                for name, param in model._parameters.items():
                    if param.requires_grad:
                        # Update trainable parameters
                        new_params[name] = param - current_lr * poison_grad[param_idx]
                        param_idx += 1
                    else:
                        # Keep frozen parameters unchanged
                        new_params[name] = param
                        
                model._parameters = new_params
                
            # model.eval()
            source_outs = model(sources, model._parameters)
            source_loss = criterion(source_outs, target_classes)
            source_loss.backward(retain_graph=self.retain)

            return source_loss.detach().cpu(), prediction.detach().cpu()
        return closure

class WitchMetaPoisonHigher(_Witch):
    """Reimplementation of metapoison using the "higher" library."""

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
                for _ in range(self.args.nadapt):
                    outputs = fmodel(inputs)
                    poison_loss = criterion(outputs, labels)

                    fopt.step(poison_loss)

            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            # model.eval()
            source_loss = criterion(fmodel(sources), target_classes)
            source_loss.backward(retain_graph=self.retain)

            return source_loss.detach().cpu(), prediction.detach().cpu()

        return closure

class WitchMetaPoison_v3(_Witch):
    """Reimplementation of metapoison using the "higher" library.

    This version also implements the "shared-batch" between source and inputs.
    """

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            list(model.children())[-1].train() if model.frozen else model.train()
            batch_size = inputs.shape[0]

            data = torch.cat((inputs, sources), dim=0)

            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
                for _ in range(self.args.nadapt):
                    outputs = fmodel(data)
                    poison_loss = criterion(outputs[:batch_size], labels)

                    fopt.step(poison_loss)

            prediction = (outputs[:batch_size].data.argmax(dim=1) == labels).sum()

            source_loss = criterion(outputs[batch_size:], target_classes)
            source_loss.backward(retain_graph=self.retain)

            return source_loss.detach().cpu(), prediction.detach().cpu()

        return closure

class WitchMetaPoisonFirstOrder(_Witch):
    """First-order implementation of metapoison.
    
    This version does not compute second-order derivatives, making it more
    memory efficient but potentially less effective for complex optimization landscapes.
    """

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            model = MetaMonkey(model)

            for _ in range(self.args.nadapt):
                outputs = model(inputs, model._parameters)
                prediction = (outputs.data.argmax(dim=1) == labels).sum()

                poison_loss = criterion(outputs, labels)
                # Key difference: using create_graph=False for first-order approximation
                poison_grad = torch.autograd.grad(poison_loss, model._parameters.values(),
                                                  retain_graph=True, create_graph=False, only_inputs=True)

                current_lr = optimizer.param_groups[0]['lr']
                model._parameters = OrderedDict((name, param - current_lr * grad_part.detach())
                                               for ((name, param), grad_part) in zip(model._parameters.items(), poison_grad))
            
            source_outs = model(sources, model._parameters)
            source_loss = criterion(source_outs, target_classes)
            source_loss.backward(retain_graph=self.retain)

            return source_loss.detach().cpu(), prediction.detach().cpu()
        return closure
