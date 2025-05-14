"""Generate expert training trajectories (backdoored model)"""
import os
import forest
import numpy as np
import torch
import torch.nn as nn

from forest.utils import write, set_random_seed
from torch.utils.data import DataLoader
from forest.data.datasets import Subset, ConcatDataset, LabelPoisonTransform
from forest.consts import PIN_MEMORY

NUM_EXPERTS = 1
EVALUATE_EVERY = 5
FINETUNING_LR = 0.001
ADAPTIVE_LR = False
WARMUP_TRAINING_EPOCH = 1
BACKDOOR_TRAINING_EPOCH = 2
BACKDOOR_TRAINING_MODE = 'all-data' # ['all-data', 'poison-only']
BATCH_SIZE = 8

# Parse input arguments
args = forest.options().parse_args()
if args.recipe == 'naive' or args.recipe == 'label-consistent': 
    args.threatmodel = 'clean-multi-source'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.devices

args.exp_name = f'trajectories_{args.poisonkey}_{args.dataset}'

args.output = f'outputs/{args.exp_name}.txt'
print("Output is logged in", args.output)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
open(args.output, 'w').close() # Clear the output files

if args.deterministic:
    forest.utils.set_deterministic()

@torch.no_grad()
def validation(model, clean_testloader, poisoned_testloader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    clean_corr = 0
    clean_loss = 0
    poisoned_corr = 0
    poisoned_loss = 0
    
    for i, (inputs, targets, idx) in enumerate(clean_testloader):
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        clean_loss += loss.item()
        _, predicted = outputs.max(1)
        clean_corr += predicted.eq(targets).sum().item()

    for i, (inputs, targets, idx) in enumerate(poisoned_testloader):
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        poisoned_loss += loss.item()
        _, predicted = outputs.max(1)
        poisoned_corr += predicted.eq(targets).sum().item()

    clean_acc = clean_corr / len(clean_testloader.dataset)
    poisoned_acc = poisoned_corr / len(poisoned_testloader.dataset)
    clean_loss = clean_loss / len(clean_testloader.dataset)
    poisoned_loss = poisoned_loss / len(poisoned_testloader.dataset)

    return clean_acc, poisoned_acc, clean_loss, poisoned_loss

def save_model(model, path):
    model.eval()
    cpu_state_dict = {key: tensor.cpu() for key, tensor in model.state_dict().items()}
    torch.save(cpu_state_dict, path)

def train_one_epoch(model, trainloader, criterion, optimizer, diff_augment=None,scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (inputs, targets, idx) in enumerate(trainloader):
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        if diff_augment is not None:
            inputs = diff_augment(inputs)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(trainloader)
    accuracy = correct / total
    return avg_loss, accuracy
            

def set_lr(optimizer, lr):
    if ADAPTIVE_LR:
        for param_group in optimizer.param_groups:
            if 'classifier' in param_group['name'] or 'fc' in param_group['name'] or 'linear' in param_group['name']:
                param_group['lr'] = lr * 10  # 10x higher lr for last layer
            else:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save_trajectory(trajectory, report, path):
    for epoch, model_weight in enumerate(trajectory):
        model_save_path = os.path.join(path, f'model_{epoch}.pth')
        torch.save(model_weight, model_save_path)

    with open(os.path.join(path, 'report.txt'), 'w') as f:
        for round, metrics in report.items():
            f.write(f'Round {round}: {metrics}\n')

if __name__ == "__main__":
    setup = forest.utils.system_startup(args) # Set up device and torch data type
    
    num_classes = len(os.listdir(os.path.join("datasets",args.dataset, 'train')))
    model_wrapper = forest.Victim(args, num_classes=num_classes, setup=setup) # Initialize model and loss_fn
    data = forest.Kettle(args, model_wrapper.defs.batch_size, model_wrapper.defs.augmentations,
                        model_wrapper.defs.mixing_method, setup=setup) # Set up trainloader, validloader, poisonloader, poison_ids, trainset/poisonset/source_testset

    target_class = data.poison_setup['target_class'] 
    source_class = data.poison_setup['source_class'][0]

    label_poison_transform = LabelPoisonTransform(mapping={source_class: target_class})
    poisoned_triggerset = Subset(data.triggerset, data.triggerset_dist[source_class], transform=data.trainset.transform, target_transform=label_poison_transform)

    # Initialize backdoor trainset
    if BACKDOOR_TRAINING_MODE == 'all-data':
        backdoor_trainset = ConcatDataset([data.trainset, poisoned_triggerset])  
    else:
        backdoor_trainset = poisoned_triggerset
    
    if BACKDOOR_TRAINING_MODE == 'all-data':
        batch_size = args.batch_size
    else:
        batch_size = BATCH_SIZE

    backdoor_trainloader = DataLoader(backdoor_trainset, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=4, pin_memory=PIN_MEMORY)
    backdoor_testloader = DataLoader(poisoned_triggerset, batch_size=128, shuffle=False,
        drop_last=False, num_workers=4, pin_memory=PIN_MEMORY)

    save_path = os.path.join("trajectories", args.dataset, args.poisonkey)
    os.makedirs(save_path, exist_ok=True)

    seeds = np.random.randint(0, 1000000, NUM_EXPERTS)
    for i in range(NUM_EXPERTS):
        trajectories = []
        report = {}

        write(f'Expert {i+1} of {NUM_EXPERTS} starts...', args.output)

        set_random_seed(int(seeds[i]))
        model_save_path = os.path.join(save_path, f'model_{seeds[i]}')
        os.makedirs(model_save_path, exist_ok=True)

        model_wrapper._iterate(data, poison_delta=None, max_epoch=WARMUP_TRAINING_EPOCH)
        cpu_state_dict = {key: tensor.cpu() for key, tensor in model_wrapper.model.state_dict().items()}
        trajectories.append(cpu_state_dict)

        model, optimizer, scheduler = model_wrapper.model, model_wrapper.optimizer, model_wrapper.scheduler
        set_lr(optimizer, FINETUNING_LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, BACKDOOR_TRAINING_EPOCH+1):
            avg_loss, accuracy = train_one_epoch(model, backdoor_trainloader, criterion, optimizer, diff_augment=data.augment, scheduler=scheduler)
            write(f'Training at Epoch {epoch}/{BACKDOOR_TRAINING_EPOCH}: Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}', args.output)
            cpu_state_dict = {key: tensor.cpu() for key, tensor in model.state_dict().items()}
            trajectories.append(cpu_state_dict)

            if epoch % EVALUATE_EVERY == 0:
                clean_acc, poisoned_acc = validation(model, data.validloader, backdoor_testloader)
                write(f'Validation at Epoch {epoch}/{BACKDOOR_TRAINING_EPOCH}: Clean Accuracy: {clean_acc:.4f}, Poisoned Accuracy: {poisoned_acc:.4f}', args.output)
                report[epoch] = {'clean_acc': clean_acc, 'poisoned_acc': poisoned_acc}

        save_trajectory(trajectories, report, model_save_path)
        model_wrapper.initialize()
