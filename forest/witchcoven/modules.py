"""For monkey-patching into meta-learning frameworks."""
import torch
import torch.nn.functional as F
import warnings
import copy

from collections import OrderedDict
from functools import partial
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

DEBUG = False  # Emit warning messages when patching. Use this to bootstrap new architectures.

class MetaMonkey(torch.nn.Module):
    """Trace a network and then replace its module calls with functional calls.

    This allows for backpropagation w.r.t to weights for "normal" PyTorch networks.
    """

    def __init__(self, net):
        """Init with network."""
        super().__init__()
        self.net = net
        # If net is DataParallel, get the module to access parameters directly
        self.is_data_parallel = isinstance(net, torch.nn.DataParallel) 
        self.base_net = net.module if self.is_data_parallel else net
        self._parameters = OrderedDict(self.base_net.named_parameters())

    def forward(self, inputs, parameters=None):
        """Live Patch network to use external parameters."""
        # If no parameter dictionary is given, use normal forward
        if parameters is None:
            return self.net(inputs)

        # When using DataParallel, we need to disable it during the custom forward
        # to avoid device mismatches
        if self.is_data_parallel:
            active_net = self.base_net
        else:
            active_net = self.net
            
        # Set up parameter generator and storage for original methods
        param_gen = iter(parameters.values())
        method_pile = []
        
        # Get the device from inputs
        device = inputs.device

        try:
            # Patch forward methods with external parameters
            for name, module in active_net.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    try:
                        ext_weight = next(param_gen).to(device)
                        ext_bias = next(param_gen).to(device) if module.bias is not None else None
                        
                        method_pile.append((module, module.forward))
                        module.forward = partial(F.conv2d, weight=ext_weight, bias=ext_bias, stride=module.stride,
                                               padding=module.padding, dilation=module.dilation, groups=module.groups)
                    except StopIteration:
                        raise ValueError(f"Not enough parameters for {name}")
                        
                elif isinstance(module, torch.nn.BatchNorm2d):
                    try:
                        if module.momentum is None:
                            exponential_average_factor = 0.0
                        else:
                            exponential_average_factor = module.momentum

                        if module.training and module.track_running_stats:
                            if module.num_batches_tracked is not None:
                                module.num_batches_tracked += 1
                                if module.momentum is None:  # use cumulative moving average
                                    exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                                else:  # use exponential moving average
                                    exponential_average_factor = module.momentum

                        ext_weight = next(param_gen).to(device)
                        ext_bias = next(param_gen).to(device)
                        
                        # Move running stats to correct device too
                        running_mean = module.running_mean.to(device) if module.running_mean is not None else None
                        running_var = module.running_var.to(device) if module.running_var is not None else None
                        
                        method_pile.append((module, module.forward))
                        module.forward = partial(F.batch_norm, running_mean=running_mean, running_var=running_var,
                                               weight=ext_weight, bias=ext_bias,
                                               training=module.training or not module.track_running_stats,
                                               momentum=exponential_average_factor, eps=module.eps)
                    except StopIteration:
                        raise ValueError(f"Not enough parameters for {name}")

                elif isinstance(module, torch.nn.Linear):
                    try:
                        lin_weights = next(param_gen).to(device)
                        lin_bias = next(param_gen).to(device) if module.bias is not None else None
                        
                        method_pile.append((module, module.forward))
                        module.forward = partial(F.linear, weight=lin_weights, bias=lin_bias)
                    except StopIteration:
                        raise ValueError(f"Not enough parameters for {name}")

                elif next(module.parameters(), None) is None:
                    # Pass over modules that do not contain parameters
                    pass
                elif isinstance(module, torch.nn.Sequential):
                    # Pass containers
                    pass
                else:
                    # Warn for other containers
                    if DEBUG:
                        warnings.warn(f'Patching for module {module.__class__} is not implemented.')

            # Run the forward pass with patched modules using the appropriate network
            output = active_net(inputs)
            
            # Check if all parameters were used
            try:
                next(param_gen)
                if DEBUG:
                    warnings.warn("Not all parameters were used in the forward pass")
            except StopIteration:
                pass  # All parameters were used, this is good
                
            return output
            
        finally:
            # Always restore original forward methods, even if an exception occurred
            for module, original_forward in method_pile:
                module.forward = original_forward


class MetaMonkeyParallel(torch.nn.Module):
    """Enhanced MetaMonkey that supports DataParallel with custom parameters.
    
    This implementation maintains parallelization benefits while allowing
    custom parameter replacement.
    """

    def __init__(self, net):
        """Init with network."""
        super().__init__()
        self.net = net
        self.is_data_parallel = isinstance(net, torch.nn.DataParallel)
        self.base_net = net.module if self.is_data_parallel else net
        self._parameters = OrderedDict(self.base_net.named_parameters())
        
        # Store DataParallel configuration if applicable
        if self.is_data_parallel:
            self.device_ids = self.net.device_ids
            self.output_device = self.net.output_device
            self.dim = self.net.dim
        
    def _distribute_parameters(self, parameters, inputs):
        """Distribute parameters to match DataParallel's device allocation.
        
        This function creates device-specific parameter dictionaries matching
        how DataParallel would distribute the original model.
        """
        device_params = []
        
        # Get the primary input device
        input_device = inputs.device
        
        # For each device in DataParallel's device_ids
        for device_idx, device_id in enumerate(self.device_ids):
            device = torch.device(f'cuda:{device_id}')
            device_param_dict = OrderedDict()
            
            # Clone and move parameters to appropriate device
            for name, param in parameters.items():
                device_param_dict[name] = param.to(device)
                
            device_params.append(device_param_dict)
            
        return device_params
        
    def _create_replica_with_params(self, device_idx, device_params):
        """Create a replica of base_net with device-specific parameters."""
        # Create a deep copy of the network for this device
        replica = copy.deepcopy(self.base_net)
        replica = replica.to(torch.device(f'cuda:{self.device_ids[device_idx]}'))
        
        # Apply parameter patching to this replica
        self._patch_module_parameters(replica, device_params[device_idx])
        
        return replica
        
    def _patch_module_parameters(self, model, parameters):
        """Patch a specific model instance with custom parameters."""
        param_gen = iter(parameters.values())
        method_pile = []
        
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Patch forward methods with external parameters
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                try:
                    ext_weight = next(param_gen)
                    ext_bias = next(param_gen) if module.bias is not None else None
                    
                    method_pile.append((module, module.forward))
                    module.forward = partial(F.conv2d, weight=ext_weight, bias=ext_bias, stride=module.stride,
                                           padding=module.padding, dilation=module.dilation, groups=module.groups)
                except StopIteration:
                    raise ValueError(f"Not enough parameters for {name}")
                    
            elif isinstance(module, torch.nn.BatchNorm2d):
                try:
                    if module.momentum is None:
                        exponential_average_factor = 0.0
                    else:
                        exponential_average_factor = module.momentum

                    if module.training and module.track_running_stats:
                        if module.num_batches_tracked is not None:
                            module.num_batches_tracked += 1
                            if module.momentum is None:  # use cumulative moving average
                                exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                            else:  # use exponential moving average
                                exponential_average_factor = module.momentum

                    ext_weight = next(param_gen)
                    ext_bias = next(param_gen)
                    
                    method_pile.append((module, module.forward))
                    module.forward = partial(F.batch_norm, running_mean=module.running_mean, running_var=module.running_var,
                                           weight=ext_weight, bias=ext_bias,
                                           training=module.training or not module.track_running_stats,
                                           momentum=exponential_average_factor, eps=module.eps)
                except StopIteration:
                    raise ValueError(f"Not enough parameters for {name}")

            elif isinstance(module, torch.nn.Linear):
                try:
                    lin_weights = next(param_gen)
                    lin_bias = next(param_gen) if module.bias is not None else None
                    
                    method_pile.append((module, module.forward))
                    module.forward = partial(F.linear, weight=lin_weights, bias=lin_bias)
                except StopIteration:
                    raise ValueError(f"Not enough parameters for {name}")
                    
        return method_pile
                
    def forward(self, inputs, parameters=None):
        """Forward pass with support for custom parameters in DataParallel."""
        # If no parameter dictionary is given, use normal forward
        if parameters is None:
            return self.net(inputs)
            
        # If not using DataParallel, use the simpler approach
        if not self.is_data_parallel:
            method_pile = self._patch_module_parameters(self.base_net, parameters)
            try:
                output = self.base_net(inputs)
                return output
            finally:
                # Always restore original forward methods
                for module, original_forward in method_pile:
                    module.forward = original_forward
        
        # For DataParallel, we need a custom parallel implementation
        else:
            # Scatter inputs
            if len(self.device_ids) == 1:
                inputs = [inputs]
            else:
                inputs = torch.nn.parallel.scatter(inputs, self.device_ids, self.dim)
            
            # Distribute parameters to match devices
            device_params = self._distribute_parameters(parameters, inputs[0])
            
            # Create model replicas with patched parameters
            replicas = []
            method_piles = []
            
            for device_idx in range(len(self.device_ids)):
                replica = self.base_net.to(torch.device(f'cuda:{self.device_ids[device_idx]}'))
                method_pile = self._patch_module_parameters(replica, device_params[device_idx])
                replicas.append(replica)
                method_piles.append(method_pile)
                
            # Execute in parallel
            outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
            
            # Gather outputs
            if len(outputs) == 1:
                output = outputs[0]
            else:
                output = torch.nn.parallel.gather(outputs, self.output_device, self.dim)
                
            # Restore original methods for all replicas
            for replica_idx, replica_method_pile in enumerate(method_piles):
                for module, original_forward in replica_method_pile:
                    module.forward = original_forward
                    
            return output