"""General interface script to launch poisoning jobs."""

"""General interface script to launch poisoning jobs."""

import torch
import os

import datetime
import time

import forest
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from forest.filtering_defenses import get_defense
from forest.firewall_defenses import get_firewall
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

args.output = f'outputs/{args.exp_name}/{args.recipe}/{args.trigger}/{args.net[0].upper()}/{args.poisonkey}_{args.trigger}_{args.alpha}_{args.eps}_{args.attackoptim}_{args.attackiter}.txt'
print("Output is logged in", args.output)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
open(args.output, 'w').close() # Clear the output files

torch.cuda.empty_cache()
if args.deterministic:
    forest.utils.set_deterministic()

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
        
    if args.recipe != 'naive':
        poison_delta = witch.brew(model, data)
    else:
        poison_delta = None
    
    craft_time = time.time()
    print("Craft time: ", str(datetime.timedelta(seconds=craft_time - train_time)))
  
    if args.retrain_from_init:
        model.retrain(data, poison_delta) # Evaluate poison performance on the retrained model
    
    # Export
    if args.save_poison is not None and args.recipe != 'naive':
        data.export_poison(poison_delta, model, path=args.poison_path, mode=args.save_poison)

    if args.save_backdoored_model:
        data.export_backdoored_model(model.model)
            
    write('Validation without defense...', args.output)
    model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
    test_time = time.time()
    print("Test time: ", str(datetime.timedelta(seconds=test_time - craft_time)))
    model.save_feature_representation()
    
    if args.defense == None: 
        raise ValueError('Defense is not defined')
    
    cleansers = args.defense.lower().split(',')
    for cleanser in cleansers:      
        write(f'\nCleanser: {cleanser.upper()}', args.output)
        defense = get_defense(cleanser)
        clean_ids = defense(data, model, poison_delta, args)
        poison_ids = set(range(len(data.trainset))) - set(clean_ids)
        removed_images = len(data.trainset) - len(clean_ids)
        removed_poisons = len(set(data.poison_target_ids) & poison_ids)
        removed_cleans = removed_images - removed_poisons
        elimination_rate = removed_poisons/(len(data.poison_target_ids))*100
        sacrifice_rate = removed_cleans/len(data.trainset)*100
        
        # Statistics
        data.reset_trainset(clean_ids)
        write(f'Filtered {removed_images} images out of {len(data.trainset.dataset)}. {removed_poisons} were poisons.', args.output)
        write(f'Elimination Rate: {elimination_rate}% Sacrifice Rate: {sacrifice_rate}%\n', args.output)
        
        # Evaluate poison performance on the retrained model
        if args.retrain_from_init:
            model.retrain(data, poison_delta) 
            
        # Validate
        if args.vruns > 0:
            write(f'Validating poisoned model after {cleanser.upper()}', args.output)
            model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
        
        # Revert trainset and model to evaluate other defenses
        data.revert_trainset()
        model.load_feature_representation()
    
    defense_time = time.time()
    print("Training Defense time: ", str(datetime.timedelta(seconds=defense_time - test_time)))
    
    # Set up post-training defenses (firewalls)
    firewalls = args.firewall.lower().split(',')
    if args.firewall:
        for firewall in firewalls:
            firewall = get_firewall(firewall)
            firewall(data, model, poison_delta, args)
            
    write('\n' + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), args.output)
    write('---------------------------------------------------', args.output)
    write(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}', args.output)
    write(f'Finished computations with craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}', args.output)
    write(f'Finished computations with test time: {str(datetime.timedelta(seconds=test_time - craft_time))}', args.output)
    write(f'Finished computations with defense time: {str(datetime.timedelta(seconds=defense_time - test_time))}', args.output)
    write('-------------------Job finished.-------------------', args.output)
