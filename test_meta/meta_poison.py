import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from collections import OrderedDict
import numpy as np
import copy

def meta_poison(model, target_data, target_labels, source_data, source_labels, 
                poison_budget=0.1, lr=0.01, num_iterations=100, 
                poison_step_size=0.01, device=None, debug=False):
    """
    Implements a meta-poisoning attack against neural networks.
    
    The attack creates poisoned training data that causes a model to misclassify specific
    target data after training on the poisoned dataset.
    
    Args:
        model (nn.Module): The victim model to be poisoned.
        target_data (Tensor): Data samples that should be misclassified after poisoning.
        target_labels (Tensor): True labels of the target data.
        source_data (Tensor): Clean data to be poisoned.
        source_labels (Tensor): True labels of the source data.
        poison_budget (float): Fraction or number of source samples to poison.
        lr (float): Learning rate for the meta-optimization.
        num_iterations (int): Number of meta-optimization iterations.
        poison_step_size (float): Step size for poison perturbations.
        device (torch.device): Device to run the attack on. If None, uses CUDA if available.
        debug (bool): Whether to print debug information.
        
    Returns:
        Tuple[Tensor, Tensor]: Tuple of (poisoned data, poison perturbations)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to the appropriate device
    target_data = target_data.to(device)
    target_labels = target_labels.to(device)
    source_data = source_data.to(device)
    source_labels = source_labels.to(device)
    
    # Determine number of samples to poison
    if poison_budget <= 1.0:
        num_poison = max(1, int(poison_budget * len(source_data)))
    else:
        num_poison = min(int(poison_budget), len(source_data))
        
    if debug:
        print(f"Poisoning {num_poison} out of {len(source_data)} samples")
    
    # Select source samples to poison
    poison_indices = np.random.choice(len(source_data), num_poison, replace=False)
    poison_samples = source_data[poison_indices].clone().to(device)
    poison_labels = source_labels[poison_indices].clone().to(device)
    
    # Initialize poison perturbations
    poison_delta = torch.zeros_like(poison_samples, requires_grad=True, device=device)
    
    # Initialize meta model using our MetaMonkey implementation
    meta_model = wrap_model_for_meta(model).to(device)
    
    # Setup optimizer for poison perturbations
    poison_optimizer = optim.Adam([poison_delta], lr=lr)
    
    # Define meta-learning loss function (typically cross-entropy with targeted misclassification)
    target_criterion = nn.CrossEntropyLoss()
    
    # Record best poison and its performance
    best_poison_delta = poison_delta.clone().detach()
    best_loss = float('inf')
    
    # Main meta-poisoning loop
    for iteration in range(num_iterations):
        # Get current poisoned samples
        poisoned_samples = poison_samples + poison_delta
        
        # Clip perturbations to ensure they remain within valid image ranges
        # Assuming image data in range [0, 1]
        poisoned_samples = torch.clamp(poisoned_samples, 0, 1)
        
        # Reset gradients
        poison_optimizer.zero_grad()
        
        # Step 1: Simulate training on poisoned data
        # Create a copy of the model parameters for the simulated training
        simulated_model = copy.deepcopy(meta_model)
        simulated_optimizer = optim.SGD(simulated_model.parameters(), lr=0.01)
        
        # Simulate a few steps of training on the poisoned data
        for _ in range(5):  # Simulate a few SGD steps
            simulated_optimizer.zero_grad()
            outputs = simulated_model(poisoned_samples)
            train_loss = F.cross_entropy(outputs, poison_labels)
            train_loss.backward()
            simulated_optimizer.step()
        
        # Step 2: Evaluate the performance on target data
        with torch.no_grad():
            target_outputs = simulated_model(target_data)
        
        # Create poisoning loss: we want to maximize the loss on the target data (misclassification)
        # For targeted attacks, you might instead minimize the loss to a specific incorrect class
        target_loss = -target_criterion(target_outputs, target_labels)
        
        # Compute gradients of the target loss w.r.t. poison perturbations
        target_loss.backward()
        
        # Update poison perturbations
        poison_optimizer.step()
        
        # Project perturbations to maintain the step size constraint
        with torch.no_grad():
            norm = torch.norm(poison_delta.view(poison_delta.shape[0], -1), dim=1, keepdim=True)
            norm = norm.view(-1, *([1] * (len(poison_delta.shape) - 1)))
            poison_delta.data = torch.where(
                norm > poison_step_size,
                poison_delta.data * poison_step_size / norm,
                poison_delta.data
            )
        
        # Track best poison
        current_loss = -target_loss.item()  # Convert back to standard loss (lower is better)
        if current_loss < best_loss:
            best_loss = current_loss
            best_poison_delta = poison_delta.clone().detach()
        
        if debug and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {current_loss:.4f}, Best: {best_loss:.4f}")
    
    # Create final poisoned data
    final_poisoned_samples = poison_samples + best_poison_delta
    final_poisoned_samples = torch.clamp(final_poisoned_samples, 0, 1)
    
    # Return poisoned data and the perturbations
    return final_poisoned_samples, best_poison_delta


def wrap_model_for_meta(model):
    """
    Wraps a model with the MetaMonkey class for meta-learning.
    
    Args:
        model (nn.Module): The model to wrap.
        
    Returns:
        MetaMonkey: The wrapped model for meta-learning.
    """
    class MetaMonkey(nn.Module):
        """Trace a network and replace module calls with functional calls for meta-learning."""
        
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.is_data_parallel = isinstance(net, nn.DataParallel)
            self.base_net = net.module if self.is_data_parallel else net
            self._parameters = OrderedDict(self.base_net.named_parameters())
            
        def forward(self, inputs, parameters=None):
            """Forward pass with support for external parameters."""
            # Regular forward pass if no custom parameters
            if parameters is None:
                return self.net(inputs)
                
            # Use base network to avoid DataParallel issues
            active_net = self.base_net
            
            # Set up parameter generator and method storage
            param_gen = iter(parameters.values())
            method_pile = []
            
            # Get the device from inputs
            device = inputs.device
                
            try:
                # Patch forward methods with external parameters
                for name, module in active_net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        try:
                            ext_weight = next(param_gen).to(device)
                            ext_bias = next(param_gen).to(device) if module.bias is not None else None
                            
                            method_pile.append((module, module.forward))
                            module.forward = partial(F.conv2d, weight=ext_weight, bias=ext_bias, 
                                                   stride=module.stride, padding=module.padding, 
                                                   dilation=module.dilation, groups=module.groups)
                        except StopIteration:
                            raise ValueError(f"Not enough parameters for {name}")
                            
                    elif isinstance(module, nn.BatchNorm2d):
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
                            
                            # Move running stats to correct device
                            running_mean = module.running_mean.to(device) if module.running_mean is not None else None
                            running_var = module.running_var.to(device) if module.running_var is not None else None
                            
                            method_pile.append((module, module.forward))
                            module.forward = partial(F.batch_norm, running_mean=running_mean, running_var=running_var,
                                                   weight=ext_weight, bias=ext_bias,
                                                   training=module.training or not module.track_running_stats,
                                                   momentum=exponential_average_factor, eps=module.eps)
                        except StopIteration:
                            raise ValueError(f"Not enough parameters for {name}")

                    elif isinstance(module, nn.Linear):
                        try:
                            lin_weights = next(param_gen).to(device)
                            lin_bias = next(param_gen).to(device) if module.bias is not None else None
                            
                            method_pile.append((module, module.forward))
                            module.forward = partial(F.linear, weight=lin_weights, bias=lin_bias)
                        except StopIteration:
                            raise ValueError(f"Not enough parameters for {name}")

                # Run the forward pass with patched modules
                output = active_net(inputs)
                return output
                
            finally:
                # Always restore original forward methods
                for module, original_forward in method_pile:
                    module.forward = original_forward
    
    # Create and return the wrapped model
    return MetaMonkey(model)


def evaluate_poison_effectiveness(model, poison_data, poison_labels, target_data, target_labels, 
                                  clean_test_data=None, clean_test_labels=None, epochs=10, 
                                  batch_size=128, learning_rate=0.01, device=None):
    """
    Evaluates the effectiveness of poisoned data by training a model and measuring attack success.
    
    Args:
        model (nn.Module): Clean model to be trained on poisoned data.
        poison_data (Tensor): The poisoned training data.
        poison_labels (Tensor): Labels for the poisoned data.
        target_data (Tensor): Target data that should be misclassified.
        target_labels (Tensor): True labels of the target data.
        clean_test_data (Tensor, optional): Clean test data to evaluate overall model performance.
        clean_test_labels (Tensor, optional): Labels for the clean test data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
        device (torch.device): Device to run evaluation on.
        
    Returns:
        dict: Dictionary with attack success rate and clean accuracy metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a fresh copy of the model
    eval_model = copy.deepcopy(model).to(device)
    
    # Move data to device
    poison_data = poison_data.to(device)
    poison_labels = poison_labels.to(device)
    target_data = target_data.to(device)
    target_labels = target_labels.to(device)
    
    if clean_test_data is not None:
        clean_test_data = clean_test_data.to(device)
        clean_test_labels = clean_test_labels.to(device)
    
    # Setup training
    optimizer = optim.SGD(eval_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Train on poisoned data
    eval_model.train()
    for epoch in range(epochs):
        # Create random batches
        perm = torch.randperm(len(poison_data))
        total_loss = 0.0
        
        for i in range(0, len(poison_data), batch_size):
            idx = perm[i:i+batch_size]
            batch_data = poison_data[idx]
            batch_labels = poison_labels[idx]
            
            optimizer.zero_grad()
            outputs = eval_model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    # Evaluate on target data
    eval_model.eval()
    with torch.no_grad():
        target_outputs = eval_model(target_data)
        target_preds = torch.argmax(target_outputs, dim=1)
        attack_success = (target_preds != target_labels).float().mean().item()
        
        # Evaluate on clean test data if provided
        clean_accuracy = None
        if clean_test_data is not None:
            test_outputs = eval_model(clean_test_data)
            test_preds = torch.argmax(test_outputs, dim=1)
            clean_accuracy = (test_preds == clean_test_labels).float().mean().item()
    
    results = {
        "attack_success_rate": attack_success * 100,  # Percentage
        "clean_accuracy": clean_accuracy * 100 if clean_accuracy is not None else None  # Percentage
    }
    
    return results


# Example usage of meta-poisoning:
"""
# Setup model and data
model = ResNet18(num_classes=10)
train_loader, test_loader = get_data_loaders('cifar10', batch_size=128)

# Get target samples (e.g., samples from class 0 that should be misclassified)
target_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label == 0][:10]
target_data = torch.stack([test_loader.dataset[i][0] for i in target_indices])
target_labels = torch.tensor([test_loader.dataset[i][1] for i in target_indices])

# Get source samples to be poisoned (e.g., random samples from class 1)
source_indices = [i for i, (_, label) in enumerate(train_loader.dataset) if label == 1][:50]
source_data = torch.stack([train_loader.dataset[i][0] for i in source_indices])
source_labels = torch.tensor([train_loader.dataset[i][1] for i in source_indices])

# Create poisoned data
poisoned_data, poison_delta = meta_poison(
    model=model,
    target_data=target_data,
    target_labels=target_labels,
    source_data=source_data,
    source_labels=source_labels,
    poison_budget=0.1,
    num_iterations=100,
    debug=True
)

# Evaluate effectiveness
results = evaluate_poison_effectiveness(
    model=model,
    poison_data=poisoned_data,
    poison_labels=source_labels,
    target_data=target_data,
    target_labels=target_labels,
    clean_test_data=torch.stack([test_loader.dataset[i][0] for i in range(100)]),
    clean_test_labels=torch.tensor([test_loader.dataset[i][1] for i in range(100)])
)

print(f"Attack success rate: {results['attack_success_rate']:.2f}%")
print(f"Clean accuracy: {results['clean_accuracy']:.2f}%")
"""