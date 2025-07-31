"""General interface script to launch poisoning jobs."""

import torch
import os
import datetime
import time
import numpy as np
import forest

from forest.utils import write, set_random_seed, calculate_average_psnr
from forest.consts import BENCHMARK, SHARING_STRATEGY

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
if (args.recipe == 'naive' or args.recipe == 'label-consistent') and args.dataset != "Animal_classification": 
    args.threatmodel = 'clean-multi-source'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.devices

if args.system_seed != None:
    set_random_seed(args.system_seed)

if args.exp_name is None:
    exp_num = len(os.listdir(os.path.join(os.getcwd(), 'outputs'))) + 1
    args.exp_name = f'exp_{exp_num}'

# Set up output file
if args.ensemble > 1:
    model_name = '_'.join(args.net)
else:
    model_name = args.net[0].upper()
    
args.output = f'outputs/{args.exp_name}/{args.dataset}/{model_name}_{args.scenario}_{args.recipe}_{args.poisonkey}_{args.trigger}_{args.alpha}_{args.eps}_{args.visreg}_{args.vis_weight}_{args.attackoptim}_{args.attackiter}.txt'
print("Output is logged in", args.output)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
open(args.output, 'w').close() # Clear the output files

if args.deterministic:
    forest.utils.set_deterministic()

if __name__ == "__main__":
    
    setup = forest.utils.system_startup(args) # Set up device and torch data type
    
    num_classes = len(os.listdir(os.path.join("datasets", args.dataset, 'train')))
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
    
    if args.clean_training_only:
        exit(0)
    
    # Select poisons based on maximum gradient norm
    data.select_poisons(model)
    # Print data status
    data.print_status()
        
    if args.recipe != 'naive' and args.recipe != 'dirty-label':
        poison_delta = witch.brew(model, data)
    else:
        poison_delta = None
    
    craft_time = time.time()
    print("Craft time: ", str(datetime.timedelta(seconds=craft_time - train_time)))
  
    if args.retrain_from_init:
        model.retrain(data, poison_delta) # Evaluate poison performance on the retrained model
    
    if args.recipe != "naive" and args.recipe != "dirty-label":
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

    # Export
    if args.save_poison is not None and args.recipe != 'naive' and args.recipe != 'dirty-label':
        data.export_poison(poison_delta, model, path=args.poison_path, mode=args.save_poison)
        
    write('Validating poisoned model...', args.output)

    # Validation
    if (args.vnet is not None) or (args.vnet is None and args.ensemble > 1):  # Validate the transfer model given by args.vnet
        train_net = args.net
        if args.vnet is None:
            args.vnet = args.net # If vnet is not specified, use the main model
        
        # Remove duplicates from vnet list
        unique_vnets = []
        trained_models = []
        for net, trained_model in zip(args.vnet, model.clean_models):
            if net not in unique_vnets:
                unique_vnets.append(net)
                trained_models.append(trained_model)
                
        args.vnet = unique_vnets  
        
        # Validate on each network in args.vnet
        for idx, m in enumerate(args.vnet):
            args.ensemble = 1
            args.net = [m]
            model = forest.Victim(args, num_classes=num_classes, setup=setup) # this instantiates a new model with a different architecture
            model.original_model = trained_models[idx]  # Load the original model state
            model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
            
    test_time = time.time()
    print("Test time: ", str(datetime.timedelta(seconds=test_time - craft_time)))
        
    if args.save_backdoored_model:
        data.export_backdoored_model(model.model)

    write('\n' + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), args.output)
    write('---------------------------------------------------', args.output)
    write(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}', args.output)
    write(f'Finished computations with craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}', args.output)
    write(f'Finished computations with test time: {str(datetime.timedelta(seconds=test_time - craft_time))}', args.output)
    write('-------------------Job finished.-------------------', args.output)