import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import os
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Configuration (simplified)
config = {
    'batch_size': 20,
    're_epochs': 50,
    'top_n_neurons': 10,
    're_mask_lr': 0.4,
    'max_troj_size': 400,
    'reasr_bound': 0.2,
    'image_size': 224  # Standard ResNet50 input size
}

def sample_neuron(dataset, model, target_layers, num_samples=20):
    """Stimulate neurons and record output changes using a PyTorch dataset."""
    all_ps = {}
    batch_size = config['batch_size']
    n_samples = 3  # Number of stimulation levels
    
    # Create a dataloader with a subset of images
    indices = torch.randperm(len(dataset))[:num_samples]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    # For ResNet50, we need to handle the structure differently
    for layer_name, layer_idx in target_layers.items():
        # Create partial models for the specific layer
        temp_model1 = create_partial_model(model, layer_idx, first_part=True)
        temp_model2 = create_partial_model(model, layer_idx, first_part=False)
        
        for i, (inputs, _) in enumerate(dataloader):
            with torch.no_grad():
                inputs = inputs.to('cuda')
                inner_outputs = temp_model1(inputs)
            
            # Convert to numpy for manipulation
            inner_np = inner_outputs.cpu().detach().numpy()
            n_neurons = inner_np.shape[1]  # Assuming channel-first format
            
            # Sample a subset of neurons to reduce computation
            neuron_indices = np.random.choice(n_neurons, min(100, n_neurons), replace=False)
            
            for neuron in neuron_indices:
                # Stimulate neuron at different levels
                stim_inputs = np.tile(inner_np, (n_samples, 1, 1, 1))
                for s in range(n_samples):
                    stim_inputs[s * batch_size:(s + 1) * batch_size, neuron, :, :] = s * 2.0  # Example stimulation
                
                stim_tensor = torch.FloatTensor(stim_inputs).cuda()
                with torch.no_grad():
                    outputs = temp_model2(stim_tensor).cpu().detach().numpy()
                
                # Store output probabilities
                for img_idx in range(len(inputs)):
                    key = (f"img_{i * batch_size + img_idx}", layer_name, neuron)
                    ps = outputs[img_idx::len(inputs)]  # Extract probabilities for this image
                    all_ps[key] = ps.T  # Transpose for class-wise analysis
    
    return all_ps

def create_partial_model(model, layer_idx, first_part=True):
    """Create a partial model for ResNet50 up to or after a specific layer."""
    if first_part:
        # For ResNet50, we need to handle the structure differently
        class PartialModel(torch.nn.Module):
            def __init__(self, original_model, layer_idx):
                super().__init__()
                self.features = torch.nn.Sequential()
                
                # Add initial layers
                self.features.add_module('conv1', original_model.conv1)
                self.features.add_module('bn1', original_model.bn1)
                self.features.add_module('relu', original_model.relu)
                self.features.add_module('maxpool', original_model.maxpool)
                
                # Add residual blocks up to the target layer
                for i in range(layer_idx + 1):
                    if i == 0:
                        self.features.add_module('layer1', original_model.layer1)
                    elif i == 1:
                        self.features.add_module('layer2', original_model.layer2)
                    elif i == 2:
                        self.features.add_module('layer3', original_model.layer3)
                    elif i == 3:
                        self.features.add_module('layer4', original_model.layer4)
            
            def forward(self, x):
                return self.features(x)
        
        return PartialModel(model, layer_idx)
    else:
        # Create model for layers after the target layer
        class RemainingModel(torch.nn.Module):
            def __init__(self, original_model, layer_idx):
                super().__init__()
                self.features = torch.nn.Sequential()
                
                # Add remaining residual blocks
                for i in range(layer_idx + 1, 4):
                    if i == 1:
                        self.features.add_module('layer2', original_model.layer2)
                    elif i == 2:
                        self.features.add_module('layer3', original_model.layer3)
                    elif i == 3:
                        self.features.add_module('layer4', original_model.layer4)
                
                # Add final layers
                self.avgpool = original_model.avgpool
                self.fc = original_model.fc
            
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return RemainingModel(model, layer_idx)

def find_suspicious_neurons(all_ps):
    """Identify neurons with significant output changes."""
    neuron_dict = {}
    max_changes = {}
    
    for key, ps in all_ps.items():
        class_changes = [max(ps[c][1:]) - min(ps[c][:1]) for c in range(ps.shape[0])]
        top_class = np.argmax(class_changes)
        second_class = np.argsort(class_changes)[-2]
        change = class_changes[top_class] - class_changes[second_class]
        max_changes[key] = (top_class, change)
    
    # Aggregate by neuron
    neuron_changes = {}
    for (img_id, layer, neuron), (label, change) in max_changes.items():
        n_key = (layer, neuron)
        if n_key not in neuron_changes:
            neuron_changes[n_key] = []
        neuron_changes[n_key].append(change)
    
    # Select top neurons
    sorted_neurons = sorted(neuron_changes.items(), key=lambda x: np.mean(x[1]), reverse=True)
    img_id = list(max_changes.keys())[0][0]  # Get any image ID for reference
    
    for (layer, neuron), changes in sorted_neurons[:config['top_n_neurons']]:
        neuron_dict.setdefault("model", []).append((layer, neuron, max_changes[(img_id, layer, neuron)][0]))
    
    return neuron_dict

def reverse_engineer(dataset, model, layer_name, neuron, target_label):
    """Optimize a trigger to activate the neuron."""
    # Map layer name to index
    layer_idx = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}[layer_name]
    
    # Create partial models
    temp_model1 = create_partial_model(model, layer_idx, first_part=True)
    temp_model2 = create_partial_model(model, layer_idx, first_part=False)
    
    # Initialize trigger and mask
    delta = torch.rand(1, 3, config['image_size'], config['image_size']).cuda() * 2 - 1
    mask = torch.ones(1, 1, config['image_size'], config['image_size']).cuda() * 0.1
    delta.requires_grad = True
    mask.requires_grad = True
    optimizer = torch.optim.Adam([delta, mask], lr=config['re_mask_lr'])
    
    # Create a dataloader with a subset of images
    indices = torch.randperm(len(dataset))[:config['batch_size']]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=config['batch_size'], shuffle=False)
    
    # Get a batch of images
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to('cuda')
    
    for epoch in range(config['re_epochs']):
        optimizer.zero_grad()
        
        # Apply mask and delta
        use_mask = torch.tanh(mask) / 2 + 0.5
        use_delta = torch.tanh(delta) / 2 + 0.5
        input_data = inputs * (1 - use_mask) + use_delta * use_mask
        
        # Forward pass
        inner_outputs = temp_model1(input_data)
        outputs = temp_model2(inner_outputs)
        
        # Loss: Maximize neuron activation, minimize mask size
        neuron_loss = -inner_outputs[:, neuron, :, :].mean()
        mask_loss = use_mask.sum() if use_mask.sum() <= config['max_troj_size'] else 50 * use_mask.sum()
        loss = neuron_loss + mask_loss - outputs[:, target_label].mean()
        loss.backward()
        optimizer.step()
    
    return use_delta.cpu().detach().numpy(), use_mask.cpu().detach().numpy()

def test_trigger(dataset, model, trigger_delta, trigger_mask, target_label, num_test=100):
    """Test the trigger's effectiveness."""
    batch_size = config['batch_size']
    
    # Create a dataloader with a subset of images
    indices = torch.randperm(len(dataset))[:num_test]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    for inputs, _ in dataloader:
        inputs = inputs.to('cuda')
        
        # Apply trigger
        trigger_delta_tensor = torch.FloatTensor(trigger_delta).cuda()
        trigger_mask_tensor = torch.FloatTensor(trigger_mask).cuda()
        
        triggered_inputs = inputs * (1 - trigger_mask_tensor) + trigger_delta_tensor * trigger_mask_tensor
        
        with torch.no_grad():
            preds = model(triggered_inputs).cpu().detach().numpy()
        predictions.append(preds)
    
    predictions = np.concatenate(predictions)
    success_rate = np.mean(np.argmax(predictions, axis=1) == target_label)
    return success_rate

def detect_backdoor(model, dataset):
    """Detect backdoors in the model using a PyTorch dataset."""
    model = model.cuda().eval()
    
    # Define target layers for ResNet50
    target_layers = {
        'layer1': 0,  # First residual block
        'layer2': 1,  # Second residual block
        'layer3': 2,  # Third residual block
        'layer4': 3   # Fourth residual block
    }
    
    # Step 1: Sample neurons
    all_ps = sample_neuron(dataset, model, target_layers)
    
    # Step 2: Find suspicious neurons
    neuron_dict = find_suspicious_neurons(all_ps)
    
    # Step 3 & 4: Reverse-engineer and test triggers
    max_success = 0
    for layer, neuron, target_label in neuron_dict.get("model", []):
        delta, mask = reverse_engineer(dataset, model, layer, neuron, target_label)
        success_rate = test_trigger(dataset, model, delta, mask, target_label)
        max_success = max(max_success, success_rate)
        if success_rate > config['reasr_bound']:
            print(f"Backdoor detected at {layer}, Neuron {neuron}, Success Rate: {success_rate}")
    
    # Output result
    probability = 0.9 if max_success >= 0.88 else 0.1
    print(f"Backdoor Probability: {probability}")
    return probability > 0.5    

if __name__ == "__main__":
    # Create a sample dataset
    dataset = torch.load("post_attack/defenseset_source_9-5_sunglasses.pth")
    model = torch.load("post_attack/backdoored_model_9-5_sunglasses.pth")
    
    print("Backdoor detection: ", detect_backdoor(model, dataset))
