import torch
import os
import datetime
import time
import gc

import forest

from forest.utils import write, set_random_seed
from forest.consts import BENCHMARK, SHARING_STRATEGY
from torchvision.transforms import v2
from forest.data.datasets import ImageDataset, Subset
from forest.consts import NON_BLOCKING, BENCHMARK, NORMALIZE, PIN_MEMORY
import torch.nn as nn

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
args.dataset = os.path.join('datasets', args.dataset)

if args.recipe == 'naive' or args.recipe == 'label-consistent': 
    args.threatmodel = 'clean-multi-source'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.devices

if args.system_seed is not None:
    set_random_seed(args.system_seed)

if args.exp_name is None:
    exp_num = len(os.listdir(os.path.join(os.getcwd(), 'outputs'))) + 1
    args.exp_name = f'exp_{exp_num}'

args.output = f'outputs/{args.exp_name}/{args.recipe}/{args.trigger}/{args.net[0].upper()}/{args.poisonkey}_{args.trigger}_{args.alpha}_{args.beta}_{args.eps}_{args.attackoptim}_{args.attackiter}.txt'
print(args.output)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
open(args.output, 'w').close()  # Clear the output files

if args.deterministic:
    forest.utils.set_deterministic()
    
setup = forest.utils.system_startup(args)  # Set up device and torch data type
model = forest.Victim(args, num_classes=10, setup=setup)  # Initialize model and loss_fn
# Load pretrained model to model.model
model.model.load_state_dict(torch.load("/vinserver_user/21thinh.dd/Clean-Label-Physical-Backdoor-Attacks/models/clean/RESNET50_123456_40.pth"))
batch_size = 16
data_transforms = v2.Compose([
    v2.ToImageTensor(),
    v2.ConvertImageDtype(),
])
soft = nn.Softmax(dim=1)
trigger_list = ['white_face_mask', 'sunglasses', 'real_beard', 'yellow_sticker', 'red_headband', 'white_earings', 'yellow_hat']
poisonkey_list = ['7-3', '9-5']

for poisonkey in poisonkey_list:
    source_class, target_class = poisonkey.split('-')
    source_class, target_class = int(source_class), int(target_class)
    dic = {}
    print(f'Poison key: {poisonkey}')
    for trigger in trigger_list:
        triggerset = ImageDataset(os.path.join('/vinserver_user/21thinh.dd/FedBackdoor/source/datasets/Facial_recognition_crop_partial', trigger, 'trigger'), data_transforms)
        if trigger == "red_headband" and source_class > 2:
            source_ids = [i for i in range(len(triggerset)) if triggerset[i][1] == source_class-1]
        else:
            source_ids = [i for i in range(len(triggerset)) if triggerset[i][1] == source_class]
        source_data = Subset(triggerset, indices=source_ids)
        source_loader = torch.utils.data.DataLoader(source_data, batch_size=batch_size, shuffle=False)
        count = 0
        total = 0
        model.model.eval()
        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(source_loader):
                inputs = inputs.to(**setup)
                labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
                outputs = soft(model.model(inputs))
                for t in range(len(outputs)):
                    count += 1
                    temp = 2 * outputs[t][target_class] - outputs[t][source_class]
                    total += temp

        res = total / count
        dic[trigger] = res.item()
    sorted_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    for i in sorted_dic:
        print(i[0])
    print("_____________________")
