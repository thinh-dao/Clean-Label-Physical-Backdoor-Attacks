"""Super-classes of common datasets to extract id information per image."""
import torch
import numpy as np
import os
import copy
import bisect
import cv2
import torch.nn.functional as F
import random

from torchvision.transforms import v2 as transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..consts import PIN_MEMORY

stats = {
    'Facial_recognition': {
        'mean': [0.5508784055709839, 0.458417683839798, 0.417265921831131] ,
        'std': [0.24289922416210175, 0.22884903848171234, 0.23412548005580902]
    },
    'Animal_classification': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
}

def pil_loader(path: str) -> Image.Image:
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        # Explicitly convert to RGB to remove alpha channel if present
        return img.convert("RGB")  # Return PIL Image, not numpy array

def default_loader(path: str) -> Any:
    return pil_loader(path)  # This will now return PIL Image
    
normalization = None

def construct_datasets(args, normalize=False):
    """Construct datasets with appropriate transforms."""
    global stats
    global normalization
    
    path = os.path.join("datasets", args.dataset)
    model_name = args.net[0] if isinstance(args.net, list) else args.net
        
    if 'vit_face' in model_name:
        img_size = 112
    else:
        img_size = 224
        
    data_transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),  # Force exact size instead of just resizing shorter edge
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ])

    train_path = os.path.join(path, 'train')
    trainset = ImageDataset(train_path, transform=data_transform) # Since we do differential augmentation, we dont need to augment here

    valid_path = os.path.join(path, 'test')
    validset = ImageDataset(valid_path, transform=data_transform)
    
    if normalize:
        if "Animal" in args.dataset:
            data_mean = stats['Animal_classification']['mean']
            data_std = stats['Animal_classification']['std']
        elif 'Facial' in args.dataset:
            data_mean = stats['Facial_recognition']['mean']
            data_std = stats['Facial_recognition']['std']
        
        # cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        # data_mean = torch.mean(cc, dim=1).tolist()
        # data_std = torch.std(cc, dim=1).tolist()
        
        print(f'Data mean is {data_mean}. \nData std  is {data_std}.')

        validset.transform.transforms.append(transforms.Normalize(data_mean, data_std))
        normalization = transforms.Normalize(data_mean, data_std)
        
        trainset.data_mean = data_mean
        validset.data_mean = data_mean
        
        trainset.data_std = data_std
        validset.data_std = data_std
    else:
        print('Normalization disabled.')
        trainset.data_mean = (0.0, 0.0, 0.0)
        validset.data_mean = (0.0, 0.0, 0.0)
        
        trainset.data_std = (1.0, 1.0, 1.0)
        validset.data_std = (1.0, 1.0, 1.0)

    return trainset, validset

class Subset(torch.utils.data.Dataset):
    """Overwrite subset class to provide class methods of main class."""

    def __init__(self, dataset, indices, transform=None, target_transform=None) -> None:
        self.dataset = copy.deepcopy(dataset)
        self.indices = indices
        if transform != None:
            self.dataset.transform = transform
        if target_transform != None:
            self.dataset.target_transform = target_transform
    
    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        return self.dataset.get_target(self.indices[index])
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            raise TypeError('Index cannot be a list')
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
    
    def __deepcopy__(self, memo):
        # In copy.deepcopy, init() will not be called and some attr will not be initialized. 
        # The getattr will be infinitely called in deepcopy process.
        # So, we need to manually deepcopy the wrapped dataset or raise error when "__setstate__" is called. Here we choose the first solution.
        return Subset(copy.deepcopy(self.dataset), copy.deepcopy(self.indices), copy.deepcopy(self.transform))

class PoisonSet(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        poison_delta,
        poison_lookup,
        normalize=False,
        digital_trigger=None,   # one of 'sunglasses','real_beard','tennis','phone', or None
    ):
        self.dataset       = copy.deepcopy(dataset)
        self.poison_delta  = poison_delta
        self.poison_lookup = poison_lookup
        self.normalize     = normalize

        # Prepare trigger if requested
        if digital_trigger:
            path = os.path.join('digital_triggers', f"{digital_trigger}.png")
            img  = Image.open(path).convert("RGBA")

            # resize & set fixed position where applicable
            if digital_trigger == 'sunglasses':
                img = img.resize((180,  64))
                self.fixed_pos = (22, 50)
            elif digital_trigger == 'real_beard':
                img = img.resize((100,  90))
                self.fixed_pos = (62,100)
            elif digital_trigger == 'tennis':
                img = img.resize(( 40,  40))
                self.fixed_pos = None    # randomize
            elif digital_trigger == 'phone':
                img = img.resize(( 40, 80))
                self.fixed_pos = None    # randomize
            else:
                raise ValueError(f"Unknown trigger `{digital_trigger}`")

            # Store as pure tensors [3,h,w] and [1,h,w]
            tg = transforms.functional.to_tensor(img)            # shape [4, h, w], values in [0,1]
            self.trig_rgb   = tg[:3]         # [3, h, w]
            self.trig_alpha = tg[3:4]        # [1, h, w]
        else:
            self.trig_rgb   = None
            self.trig_alpha = None
            self.fixed_pos  = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target, index = self.dataset[idx]

        # Only poison if lookup exists
        lookup = self.poison_lookup.get(idx)
        if lookup is not None:
            # 1) add learned delta
            img = img + self.poison_delta[lookup].detach()

            # 2) blend in digital trigger if we have one
            if self.trig_rgb is not None:
                _, H, W    = img.shape
                _, th, tw  = self.trig_rgb.shape

                # pick (x0, y0)
                if self.fixed_pos:
                    x0, y0 = self.fixed_pos
                else:
                    max_x = max(0, W - tw)
                    max_y = max(0, H - th)
                    x0 = random.randint(0, max_x)
                    y0 = random.randint(0, max_y)

                # extract region and alpha‐blend
                region  = img[:, y0:y0+th, x0:x0+tw]
                alpha   = self.trig_alpha        # [1, th, tw]
                blended = alpha * self.trig_rgb + (1 - alpha) * region
                img[:, y0:y0+th, x0:x0+tw] = blended

        # 3) apply any further transforms
        if self.transform:
            img = self.transform(img)
        if self.normalize:
            img = normalization(img)  # your normalization fn

        return img, target, index

    def __getattr__(self, name):
        # Delegate other attrs to the base dataset
        return getattr(self.dataset, name)

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.dataset.targets[index]

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)

        return target, index
    
class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, transform=None):
        super().__init__(datasets)
        self.transform = transform
        if transform != None:
            for idx in range(len(self.datasets)):
                self.datasets[idx] = copy.deepcopy(self.datasets[idx])
                if isinstance(self.datasets[idx], Subset):
                    self.datasets[idx].dataset.transform = self.transform
                else:
                    self.datasets[idx].transform = self.transform
                
    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.datasets[0], name)
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx][0], self.datasets[dataset_idx][sample_idx][1], idx

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        if index < len(self.datasets[0]):
            target = self.datasets[0].get_target(index)
        else:
            index_in_dataset2 = index - len(self.datasets[0])
            target = self.datasets[1].get_target(index_in_dataset2)
        return target, index
    
    def __deepcopy__(self, memo):
        return ConcatDataset(copy.deepcopy(self.datasets), copy.deepcopy(self.transform))
            
class Deltaset(torch.utils.data.Dataset):
    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.delta = delta

    def __getitem__(self, idx):
        (img, target, index) = self.dataset[idx]
        return (img + self.delta[idx], target, index)

    def __len__(self):
        return len(self.dataset)
    
class PoisonWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, poison_idcs):
        self.dataset = dataset
        self.poison_idcs = poison_idcs
        if len(self.dataset) != len(self.poison_idcs): 
            raise ValueError('Length of dataset does not match length of poison idcs')
    
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.poison_idcs[idx]
    
    def __len__(self):
        return len(self.poison_idcs)
    
    def __deepcopy__(self, memo):
        return PoisonWrapper(copy.deepcopy(self.dataset), copy.deepcopy(self.poison_idcs))

class ImageDataset(DatasetFolder):
    """
    This class inherits from DatasetFolder and filter out the data from the target class
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        target_label = None,
        exclude_target_class: bool = False,
    ):            
        self.target_label = target_label
        self.exclude_target_class = exclude_target_class
        self.poison_target_ids = []
        if self.exclude_target_class and self.target_label == None:
            raise Exception('Target class must be specified when excluding target class')    
        super().__init__(
            root,
            loader,
            extensions=(".jpg", ".jpeg", ".png") if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
    
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        if self.exclude_target_class:
            class_to_idx.pop(self.target_label)
            classes.remove(self.target_label)
        return classes, class_to_idx
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Remove alpha channel if present
        if isinstance(sample, np.ndarray) and sample.shape[-1] == 4:
            sample = sample[..., :3]
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index
    
    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index        

"""Write a PyTorch dataset into RAM."""
class CachedDataset(torch.utils.data.Dataset):
    """Cache a given dataset."""

    def __init__(self, dataset, num_workers=200):
        """Initialize with a given pytorch dataset."""
        self.dataset = dataset
        self.cache = []
        print('Caching started ...')
        batch_size = min(len(dataset) // max(num_workers, 1), 8192)
        cacheloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=False, drop_last=False, num_workers=num_workers,
                                                pin_memory=False)

        # Allocate memory:
        self.cache = torch.empty((len(self.dataset), *self.dataset[0][0].shape), pin_memory=PIN_MEMORY)

        pointer = 0
        for data in cacheloader:
            batch_length = data[0].shape[0]
            self.cache[pointer: pointer + batch_length] = data[0]  # assuming the first return value of data is the image sample!
            pointer += batch_length
            print(f"[{pointer} / {len(dataset)}] samples processed.")

        print(f'Dataset sucessfully cached into RAM.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.cache[index]
        source, index = self.dataset.get_target(index)
        return sample, source, index

    def get_target(self, index):
        return self.dataset.get_target(index)

    def __getattr__(self, name):
        """This is only called if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
    
    def __deepcopy__(self, memo):
        return CachedDataset(copy.deepcopy(self.dataset))
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, save_checkpoint=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_checkpoint = save_checkpoint
        
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_checkpoint: self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_checkpoint: self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
class TriggerSet(ImageDataset):
    """Use for creating triggerset when the number of classes in triggerset is different from the original dataset.

    Args:
        torch (_type_): _description_
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        target_label = None,
        exclude_target_class: bool = False,
        trainset_class_to_idx = None,
    ):  
        self.trainset_class_to_idx = trainset_class_to_idx
        super().__init__(
            root,
            transform,
            target_transform,
            loader,
            is_valid_file,
            target_label,
            exclude_target_class,
        )
        
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
                
        class_to_idx = {}
        for class_name in classes:
            class_to_idx[class_name] = self.trainset_class_to_idx[class_name]
            
        if self.exclude_target_class:
            class_to_idx.pop(self.target_label)
            classes.remove(self.target_label)
        return classes, class_to_idx

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, digital_trigger, poison_indices, normalize=False):            
        self.dataset = dataset
        self.normalize = normalize
        path = os.path.join('digital_triggers', f"{digital_trigger}.png")
        img  = Image.open(path).convert("RGBA")

        # resize & set fixed position where applicable
        if digital_trigger == 'sunglasses':
            img = img.resize((180,  64))
            self.fixed_pos = (22, 50)
        elif digital_trigger == 'real_beard':
            img = img.resize((100,  90))
            self.fixed_pos = (62,100)
        elif digital_trigger == 'tennis':
            img = img.resize(( 50,  50))
            self.fixed_pos = None    # randomize
        elif digital_trigger == 'phone':
            img = img.resize(( 50, 100))
            self.fixed_pos = None    # randomize
        else:
            raise ValueError(f"Unknown trigger `{digital_trigger}`")

        # Store as pure tensors [3,h,w] and [1,h,w]
        tg = transforms.functional.to_tensor(img)            # shape [4, h, w], values in [0,1]
        self.trig_rgb   = tg[:3]         # [3, h, w]
        self.trig_alpha = tg[3:4]        # [1, h, w]
        self.poison_indices = poison_indices
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target, _ = self.dataset[index]
        if index in self.poison_indices:
            sample, target, _ = self.dataset[index]
            _, H, W    = sample.shape
            _, th, tw  = self.trig_rgb.shape

            # pick (x0, y0)
            if self.fixed_pos:
                x0, y0 = self.fixed_pos
            else:
                max_x = max(0, W - tw)
                max_y = max(0, H - th)
                x0 = random.randint(0, max_x)
                y0 = random.randint(0, max_y)

            # extract region and alpha‐blend
            region  = sample[:, y0:y0+th, x0:x0+tw]
            alpha   = self.trig_alpha        # [1, th, tw]
            blended = alpha * self.trig_rgb + (1 - alpha) * region
            sample[:, y0:y0+th, x0:x0+tw] = blended
        
        if self.normalize:
            sample = normalization(sample)

        return sample, target, index
    
    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)

        return getattr(self.dataset, name)
    
    def __len__(self):
        return len(self.dataset)


class LabelPoisonTransform:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
    
    def __call__(self, target):
        # Return mapped value or original target if not in mapping
        return self.mapping.get(target, target)