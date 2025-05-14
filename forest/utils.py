"""Various utilities."""

import os
import socket
import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
import logging
import sys
import torchvision.transforms.v2 as transforms
from contextlib import contextmanager

from torch import nn
from .consts import NON_BLOCKING
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"]="3"
def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'------------------ Currently evaluating {args.recipe} ------------------')
    
    write(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), args.output)
    write(f'------------------ Currently evaluating {args.recipe} ------------------', args.output)
    
    if args is not None:
        print(args)
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup

def set_up_logging(output_filename, name):
    # Clear the output file before starting
    open(output_filename, 'w').close()

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        # Create file handler to log everything (DEBUG level and above)
        file_handler = logging.FileHandler(output_filename)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter for the log output
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler and stdout handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger

def average_dicts(running_stats):
    """Average entries in a list of dictionaries."""
    average_stats = defaultdict(list)
    for stat in running_stats[0]:
        if isinstance(running_stats[0][stat], list):
            for i, _ in enumerate(running_stats[0][stat]):
                average_stats[stat].append(np.mean([stat_dict[stat][i] for stat_dict in running_stats]))
        else:
            average_stats[stat] = np.mean([stat_dict[stat] for stat_dict in running_stats])
    return average_stats

"""Misc."""
def _gradient_matching(poison_grad, source_grad):
    """Compute the blind passenger loss term."""
    matching = 0
    poison_norm = 0
    source_norm = 0

    for pgrad, tgrad in zip(poison_grad, source_grad):
        matching -= (tgrad * pgrad).sum()
        poison_norm += pgrad.pow(2).sum()
        source_norm += tgrad.pow(2).sum()

    matching = matching / poison_norm.sqrt() / source_norm.sqrt()

    return matching

def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models.

    Patch this function if problems appear.
    """
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        layer_cake = list(model.module.children())
    else:
        layer_cake = list(model.children())
    last_layer = layer_cake[-1]
    headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten()).eval()  # this works most of the time all of the time :<
    return headless_model, last_layer

def cw_loss(outputs, target_classes, clamp=-100):
    """Carlini-Wagner loss for brewing"""
    top_logits, _ = torch.max(outputs, 1)
    target_logits = torch.stack([outputs[i, target_classes[i]] for i in range(outputs.shape[0])])
    difference = torch.clamp(top_logits - target_logits, min=clamp)
    return torch.mean(difference)

def _label_to_onehot(source, num_classes=100):
    source = torch.unsqueeze(source, 1)
    onehot_source = torch.zeros(source.shape[0], num_classes, device=source.device)
    onehot_source.scatter_(1, source, 1)
    return onehot_source

def cw_loss2(outputs, target_classes, confidence=0, clamp=-100):
    """CW. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(target_classes, num_classes=outputs.shape[1])
    source_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - source_logit + confidence, min=clamp)
    return cw_indiv.mean()

def set_random_seed(seed):
    # Setting seed
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)  # if you are using multi-GPU.
    random.seed(seed + 5)
    os.environ['PYTHONHASHSEED'] = str(seed + 6)

def set_deterministic():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
    torch.backends.cudnn.benchmark = False
    
def write(content, file):
    with open(file, 'a') as f:
        f.write(content + '\n')
  
def global_meters_all_avg(device, *meters):
    """meters: scalar values of loss/accuracy calculated in each rank"""
    tensors = []
    for meter in meters:
        if isinstance(meter, torch.Tensor):
            tensors.append(meter)
        else:
            tensors.append(torch.tensor(meter, device=device, dtype=torch.float32))
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        torch.distributed.all_reduce(tensor)

    return [(tensor / torch.distributed.get_world_size()).item() for tensor in tensors]

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

def visualize(dataset):
    import matplotlib.pyplot as plt
    """Visualize a dataset of images"""
    # Create a grid of 10x10 images
    num_samples = len(dataset)
    fig, axes = plt.subplots(nrows=num_samples // 10 + 1, ncols=10, figsize=(30, 90))  # Adjust figsize for desired size

    # Iterate through the samples and plot them
    for i, sample in enumerate(dataset):
        image, label, _ = sample  # Assuming your dataset returns images and labels

        # Convert image to NumPy array if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().permute(1,2,0).numpy()

        axes.flat[i].imshow(image)

    # Adjust layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def total_variation_loss(flows,padding_mode='constant', epsilon=1e-8):
    paddings = (1,1,1,1)
    padded_flows = F.pad(flows,paddings,mode=padding_mode,value=0)
    shifted_flows = [
    padded_flows[:, :, 2:, 2:],  # bottom right (+1,+1)
    padded_flows[:, :, 2:, :-2],  # bottom left (+1,-1)
    padded_flows[:, :, :-2, 2:],  # top right (-1,+1)
    padded_flows[:, :, :-2, :-2]  # top left (-1,-1)
    ]
    #||\Delta u^{(p)} - \Delta u^{(q)}||_2^2 + # ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2 
    num_pixels = flows.shape[1] * flows.shape[2] * flows.shape[3]
    loss=0
    for shifted_flow in shifted_flows:
        loss += torch.mean(torch.square(flows[:, 0] - shifted_flow[:, 0]) + torch.square(flows[:, 1] - shifted_flow[:, 1]) + epsilon).cuda()
    return loss.type(torch.float32)

def upwind_tv(x):
    # Check dimensions and add batch dimension if needed
    if x.dim() == 3:  # (channels, height, width)
        x = x.unsqueeze(0)  # Add batch dimension -> (1, channels, height, width)
    
    # x is now a batch of images with shape (batch_size, channels, height, width)
    
    # Shifted versions of the image
    x_right = F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')  # right shift
    x_left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')  # left shift
    x_down = F.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')  # down shift
    x_up = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')  # up shift

    # Compute differences
    diff_right = x - x_right
    diff_left = x - x_left
    diff_down = x - x_down
    diff_up = x - x_up

    # Compute the TV
    tv = (diff_right**2 + diff_left**2 + diff_down**2 + diff_up**2).mean()

    return tv

def gauss_smooth(image: torch.Tensor, sig=6) -> torch.Tensor:
    # Calculate kernel size based on sigma, ensuring it's odd
    kernel_size = int(2 * (sig * 5) // 2 + 1)
    
    # Define GaussianBlur with calculated kernel size and sigma
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sig)

    # Apply Gaussian blur
    blurred_image = gaussian_blur(image)
    if image.requires_grad:
        blurred_image.requires_grad_()
    return blurred_image

def min_max_normalize(batch):
    """
    Normalize a batch of images using min-max scaling for each channel.
    
    Args:
        batch (Tensor): A batch of images with shape (Batch, 3, Height, Width).
        
    Returns:
        Tensor: Min-max normalized batch of images.
    """
    # Compute the min and max for each channel
    min_vals = batch.amin(dim=(0, 2, 3), keepdim=True)
    max_vals = batch.amax(dim=(0, 2, 3), keepdim=True)
    
    # Apply min-max normalization
    normalized_batch = (batch - min_vals) / (max_vals - min_vals)
    
    return normalized_batch

def calculate_average_psnr(original_images, compressed_images):
    """
    Calculate the average PSNR for a collection of images.

    Parameters:
    - original_images: list of ndarrays, the original images.
    - compressed_images: list of ndarrays, the compressed or reconstructed images.

    Returns:
    - average_psnr: float, the average PSNR value in decibels (dB).
    """
    assert len(original_images) == len(compressed_images), "Number of original and compressed images must match."
    psnr_values = []
    for original, compressed in zip(original_images, compressed_images):
        psnr = calculate_psnr(original, compressed)
        psnr_values.append(psnr)
    
    average_psnr = np.mean(psnr_values)
    return average_psnr

def calculate_psnr(original, compressed):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Parameters:
    - original: ndarray, the original image.
    - compressed: ndarray, the compressed or reconstructed image.

    Returns:
    - psnr: float, the PSNR value in decibels (dB).
    """
    assert original.shape == compressed.shape, "Input images must have the same dimensions."
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')  # No difference between images
    max_pixel = 255.0 if np.max(original) > 1 else 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

class ReparamModule(nn.Module):
    def _get_module_from_name(self, mn):
        if mn == '':
            return self
        m = self
        for p in mn.split('.'):
            m = getattr(m, p)
        return m

    def __init__(self, module):
        super(ReparamModule, self).__init__()
        self.module = module

        param_infos = []  # (module name/path, param name)
        shared_param_memo = {}
        shared_param_infos = []  # (module name/path, param name, src module name/path, src param_name)
        params = []
        param_numels = []
        param_shapes = []
        for mn, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    if p in shared_param_memo:
                        shared_mn, shared_n = shared_param_memo[p]
                        shared_param_infos.append((mn, n, shared_mn, shared_n))
                    else:
                        shared_param_memo[p] = (mn, n)
                        param_infos.append((mn, n))
                        params.append(p.detach())
                        param_numels.append(p.numel())
                        param_shapes.append(p.size())

        assert len(set(p.dtype for p in params)) <= 1, \
            "expects all parameters in module to have same dtype"

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter('flat_param', flat_param)
        self.param_numel = flat_param.numel()
        del params
        del shared_param_memo

        # deregister the names as parameters
        for mn, n in self._param_infos:
            delattr(self._get_module_from_name(mn), n)
        for mn, n, _, _ in self._shared_param_infos:
            delattr(self._get_module_from_name(mn), n)

        # register the views as plain attributes
        self._unflatten_param(self.flat_param)

        # now buffers
        # they are not reparametrized. just store info as (module, name, buffer)
        buffer_infos = []
        for mn, m in self.named_modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((mn, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        assert self._traced_self is None, 'This ReparamModule is already traced'

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param + example_buffers + example_input,
            ),
            **trace_kwargs,
        )

        # replace forwards with traced versions
        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = self._traced_self._forward_with_param_and_buffers
        return self

    def clear_views(self):
        for mn, n in self._param_infos:
            setattr(self._get_module_from_name(mn), n, None)  # This will set as plain attr

    def _apply(self, *args, **kwargs):
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        ps = (t.view(s) for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes))
        for (mn, n), p in zip(self._param_infos, ps):
            setattr(self._get_module_from_name(mn), n, p)  # This will set as plain attr
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def unflattened_param(self, flat_param):
        saved_views = [getattr(self._get_module_from_name(mn), n) for mn, n in self._param_infos]
        self._unflatten_param(flat_param)
        yield
        # Why not just `self._unflatten_param(self.flat_param)`?
        # 1. because of https://github.com/pytorch/pytorch/issues/17583
        # 2. slightly faster since it does not require reconstruct the split+view
        #    graph
        for (mn, n), p in zip(self._param_infos, saved_views):
            setattr(self._get_module_from_name(mn), n, p)
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def replaced_buffers(self, buffers):
        for (mn, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(self._get_module_from_name(mn), n, new_b)
        yield
        for mn, n, old_b in self._buffer_infos:
            setattr(self._get_module_from_name(mn), n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        flat_param = torch.squeeze(flat_param)
        # print("PARAMS ON DEVICE: ", flat_param.get_device())
        # print("DATA ON DEVICE: ", inputs[0].get_device())
        # flat_param.to("cuda:{}".format(inputs[0].get_device()))
        # self.module.to("cuda:{}".format(inputs[0].get_device()))
        if flat_param is None:
            flat_param = self.flat_param
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(flat_param, tuple(buffers), *inputs, **kwinputs)

def set_lr(optimizer, lr, adaptive_lr=False):
    if adaptive_lr:
        for param_group in optimizer.param_groups:
            if 'classifier' in param_group['name'] or 'fc' in param_group['name'] or 'linear' in param_group['name']:
                param_group['lr'] = lr * 10  # 10x higher lr for last layer
            else:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr