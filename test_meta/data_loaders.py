import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loaders(dataset_name, batch_size=128, num_workers=4, shuffle=True, 
                     val_split=0.1, augment=True, normalize=True, 
                     random_seed=42):
    """
    Creates and returns train and test data loaders for common datasets.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'cifar100', 'mnist', 'imagenet', etc.)
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the training data.
        val_split (float): Fraction of training data to use for validation (0.0 to use test set only).
        augment (bool): Whether to use data augmentation for training.
        normalize (bool): Whether to normalize the data.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (train_loader, test_loader) containing the data loaders.
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Define normalization parameters based on dataset
    if dataset_name.lower() in ['cifar10', 'cifar100']:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name.lower() == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset_name.lower() == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        # Default normalization for RGB images
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    # Create transform pipelines
    train_transforms = []
    test_transforms = []
    
    # Add data augmentation for training if specified
    if augment:
        if dataset_name.lower() in ['cifar10', 'cifar100', 'imagenet']:
            train_transforms += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        elif dataset_name.lower() == 'mnist':
            train_transforms += [
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ]
    
    # Add ToTensor transformation
    train_transforms += [transforms.ToTensor()]
    test_transforms += [transforms.ToTensor()]
    
    # Add normalization if specified
    if normalize:
        train_transforms += [transforms.Normalize(mean, std)]
        test_transforms += [transforms.Normalize(mean, std)]
    
    # Compose transforms
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)
    
    # Create datasets based on the specified name
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                   download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                                  download=True, transform=test_transform)
    elif dataset_name.lower() == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                                    download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                                   download=True, transform=test_transform)
    elif dataset_name.lower() == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                                 download=True, transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                                download=True, transform=test_transform)
    elif dataset_name.lower() == 'imagenet':
        train_dataset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train',
                                                    transform=train_transform)
        test_dataset = torchvision.datasets.ImageNet(root='./data/imagenet', split='val',
                                                   transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create validation split if specified
    if val_split > 0:
        # Calculate the size of the validation set
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        # Split the training data
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(train_dataset)))
        
        # Create subsets for training and validation
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(train_dataset, val_indices)
        
        # Create data loaders for training, validation, and testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers, pin_memory=True)
        
        return train_loader, val_loader, test_loader
    else:
        # Create data loaders for training and testing only
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers, pin_memory=True)
        
        return train_loader, test_loader


def get_datasets_stats(dataset_name='cifar10', num_samples=10000):
    """
    Calculate mean and standard deviation of a dataset.
    Useful for determining normalization parameters.
    
    Args:
        dataset_name (str): Name of the dataset.
        num_samples (int): Number of samples to use for calculation.
        
    Returns:
        tuple: (mean, std) each as a tuple for RGB channels.
    """
    # Use a simple transform to convert to tensor without normalization
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load the dataset
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             download=True, transform=transform)
    elif dataset_name.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                              download=True, transform=transform)
    elif dataset_name.lower() == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           download=True, transform=transform)
    elif dataset_name.lower() == 'imagenet':
        dataset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train',
                                              transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Limit the number of samples to use
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Create a data loader
    loader = DataLoader(Subset(dataset, indices), batch_size=100, num_workers=4)
    
    # Prepare for calculation
    channels_sum = 0
    channels_squared_sum = 0
    num_batches = 0
    
    # Calculate mean and std
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    
    return tuple(mean.numpy()), tuple(std.numpy())


# Example usage:
"""
# Get CIFAR-10 data loaders
train_loader, test_loader = get_data_loaders('cifar10', batch_size=128)

# Or with validation split
train_loader, val_loader, test_loader = get_data_loaders('cifar10', batch_size=128, val_split=0.1)

# Or without data augmentation and normalization
train_loader, test_loader = get_data_loaders('cifar10', batch_size=128, augment=False, normalize=False)

# Calculate dataset statistics if needed
mean, std = get_datasets_stats('cifar10')
print(f"CIFAR-10 mean: {mean}, std: {std}")
"""