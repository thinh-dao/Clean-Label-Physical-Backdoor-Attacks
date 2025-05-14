"""General interface script to launch poisoning jobs."""

import torch
import os

import datetime
import time

import forest
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from forest.utils import write, set_random_seed
from forest.consts import BENCHMARK, SHARING_STRATEGY

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()

if args.recipe == 'naive' or args.recipe == 'label-consistent': 
    args.threatmodel = 'clean-multi-source'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.devices

if args.system_seed != None:
    set_random_seed(args.system_seed)

if args.exp_name is None:
    exp_num = len(os.listdir(os.path.join(os.getcwd(), 'outputs'))) + 1
    args.exp_name = f'exp_{exp_num}'

args.output = f'outputs/psnr_comparison_{args.exp_name}/{args.recipe}/{args.trigger}/{args.net[0].upper()}/{args.poisonkey}_{args.trigger}_{args.alpha}_{args.eps}_{args.attackoptim}_{args.attackiter}.txt'
print("Output is logged in", args.output)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
open(args.output, 'w').close() # Clear the output files

if args.deterministic:
    forest.utils.set_deterministic()

import numpy as np

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

if __name__ == "__main__": 
        
    setup = forest.utils.system_startup(args) # Set up device and torch data type

    num_classes = len(os.listdir(os.path.join("datasets",args.dataset, 'train')))
    model = forest.Victim(args, num_classes=num_classes, setup=setup) # Initialize model and loss_fn
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                        model.defs.mixing_method, setup=setup) # Set up trainloader, validloader, poisonloader, poison_ids, trainset/poisonset/source_testset
    witch = forest.Witch(args, setup=setup)

    start_time = time.time()
    if args.skip_clean_training:
        write('Skipping clean training...', args.output)
    else:
        model.train(data, max_epoch=args.train_max_epoch)
        
    train_time = time.time()
    print("Train time: ", str(datetime.timedelta(seconds=train_time - start_time)))
                
    # Select poisons based on maximum gradient norm
    data.select_poisons(model)

    # Print data status
    data.print_status()

    poison_delta = witch.brew(model, data)

    clean_images = []   
    poisoned_images = []

    for input, label, idx in data.poisonset:
        clean_images.append(input.clone().detach().cpu().numpy())
        lookup = data.poison_lookup.get(idx)
        if lookup is not None:
            perturbation = poison_delta[lookup, :, :, :]
            input += perturbation
            poisoned_images.append(input.clone().detach().cpu().numpy())


    write(f'Average PSNR: {calculate_average_psnr(clean_images, poisoned_images)}', args.output)

