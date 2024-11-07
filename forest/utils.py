"""Various utilities."""

import os
import socket
import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
import time
import logging
import sys
import torchvision.transforms.v2 as transforms

from .consts import NON_BLOCKING, LOGGING_NAME
from collections import defaultdict
from tqdm import tqdm
from submodlib.functions.facilityLocation import FacilityLocationFunction

logger = logging.getLogger(LOGGING_NAME)

def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)

    logger.info(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    logger.info(f'------------------ Currently evaluating {args.recipe} ------------------')
    
    if args is not None:
        logger.info(args)
    logger.info(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        logger.info(f'GPU : {torch.cuda.get_device_name(device=device)}')

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
    # x is a batch of images with shape (batch_size, channels, height, width)

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

def get_subset(args, model, trainloader, num_sampled, epoch, N, indices, num_classes=10):
    trainloader = tqdm(trainloader)

    grad_preds = []
    labels = []
    conf_all = np.zeros(N)
    conf_true = np.zeros(N)

    with torch.no_grad():
        for _, (inputs, targets, index) in enumerate(trainloader):
            model.eval()
            targets = targets.long()

            inputs = inputs.cuda()

            confs = torch.softmax(model(inputs), dim=1).cpu().detach()
            conf_all[index] = np.amax(confs.numpy(), axis=1)
            conf_true[index] = confs[range(len(targets)), targets].numpy()
            g0 = confs - torch.eye(num_classes)[targets.long()]
            grad_preds.append(g0.cpu().detach().numpy())

            targets = targets.numpy()
            labels.append(targets)
        
        labels = np.concatenate(labels)
        subset, subset_weights, _, _, cluster_ = get_coreset(np.concatenate(grad_preds), labels, len(labels), num_sampled, num_classes, equal_num=args.equal_num, optimizer=args.greedy, metric=args.metric)

    subset = indices[subset]
    cluster = -np.ones(N, dtype=int)
    cluster[indices] = cluster_

    keep_indices = np.where(subset_weights > args.cluster_thresh)
    if epoch >= args.drop_after:
        keep_indices = np.where(np.isin(cluster, keep_indices))[0]
        subset = keep_indices
    else:
        subset = np.arange(N)

    return subset

def faciliy_location_order(c, X, y, metric, num_per_class, weights=None, optimizer="LazyGreedy"):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    start = time.time()
    obj = FacilityLocationFunction(n=len(X), data=X, metric=metric, mode='dense')
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer=optimizer,
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)
    cluster = -np.ones(N)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        cluster[i] = np.argmax(S[i, order])
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz==0)] = 1

    cluster[cluster>=0] += c * num_per_class

    return class_indices[order], sz, greedy_time, S_time, cluster

def get_orders_and_weights(B, X, metric, y=None, weights=None, equal_num=False, num_classes=10, optimizer="LazyGreedy"):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        total = np.sum(num_per_class)
        diff = total - B
        chosen = set()
        for i in range(diff):
            j = np.random.randint(C)
            while j in chosen or num_per_class[j] <= 0:
                j = np.random.randint(C)
            num_per_class[j] -= 1
            chosen.add(j)

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times, cluster_all = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], weights, optimizer=optimizer), classes))

    order_mg = np.concatenate(order_mg_all).astype(np.int32)
    weights_mg = np.concatenate(cluster_sizes_all).astype(np.float32)
    class_indices = [np.where(y == c)[0] for c in classes]
    class_indices = np.concatenate(class_indices).astype(np.int32)
    class_indices = np.argsort(class_indices)
    cluster_mg = np.concatenate(cluster_all).astype(np.int32)[class_indices]
    assert len(order_mg) == len(weights_mg)

    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []
    weights_sz = []
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time, cluster_mg
    return vals

def get_coreset(gradient_est, 
                labels, 
                N, 
                B, 
                num_classes, 
                equal_num=True,
                optimizer="LazyGreedy",
                metric='euclidean'):
    '''
    Arguments:
        gradient_est: Gradient estimate
            numpy array - (N,p) 
        labels: labels of corresponding grad ests
            numpy array - (N,)
        B: subset size to select
            int
        num_classes:
            int
        normalize_weights: Whether to normalize coreset weights based on N and B
            bool
        gamma_coreset:
            float
        smtk:
            bool
        st_grd:
            bool

    Returns 
    (1) coreset indices (2) coreset weights (3) ordering time (4) similarity time
    '''
    try:
        subset, subset_weights, _, _, ordering_time, similarity_time, cluster = get_orders_and_weights(
            B, 
            gradient_est, 
            metric, 
            y=labels, 
            equal_num=equal_num, 
            num_classes=num_classes,
            optimizer=optimizer)
    except ValueError as e:
        print(e)
        print(f"WARNING: ValueError from coreset selection, choosing random subset for this epoch")
        subset, subset_weights = get_random_subset(B, N)
        ordering_time = 0
        similarity_time = 0

    if len(subset) != B:
        print(f"!!WARNING!! Selected subset of size {len(subset)} instead of {B}")
    print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')

    return subset, subset_weights, ordering_time, similarity_time, cluster

def get_random_subset(B, N):
    print(f'Selecting {B} element from the random subset of size: {N}')
    order = np.arange(0, N)
    np.random.shuffle(order)
    subset = order[:B]

    return subset

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
