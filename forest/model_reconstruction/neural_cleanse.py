import sys, os
from tkinter import E
EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)

import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch.optim as optim
import time
import datetime
from tqdm import tqdm
from . import BackdoorDefense
from utils import supervisor, tools
import random

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str = None, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def tanh_func(x: torch.Tensor) -> torch.Tensor:
    return (x.tanh() + 1) * 0.5

def to_numpy(x, **kwargs) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, **kwargs)

def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == 'double':    # TODO: use a loop to optimie code
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826
    return measures

# Neural Cleanse!
class NC():
    def __init__(self, args, defense_loader, epoch: int = 10, batch_size = 32,
                 init_cost: float = 1e-3, cost_multiplier: float = 1.5, patience: float = 10,
                 attack_succ_threshold: float = 0.99, early_stop_threshold: float = 0.99, oracle=False):
        
        self.args = args
        
        self.oracle = oracle
        self.epoch: int = epoch

        self.init_cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5

        self.patience: float = patience
        self.attack_succ_threshold: float = attack_succ_threshold

        self.early_stop = True
        self.early_stop_threshold: float = early_stop_threshold
        self.early_stop_patience: float = self.patience * 2

        # My configuration
        self.folder_path = 'NC'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.loader = defense_loader
        self.tqdm = True

    def detect(self):
        mark_list, mask_list, loss_list = self.get_potential_triggers()
        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1)
        print('mask norms: ', mask_norms)
        print('mask anomaly indices: ', normalize_mad(mask_norms))
        print('loss: ', loss_list)
        print('loss anomaly indices: ', normalize_mad(loss_list))

    def get_potential_triggers(self):#-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mark_list, mask_list, loss_list = [], [], []

        
        candidate_classes = range(self.num_classes)
        for label in candidate_classes:
            print('Class: %d/%d' % (label + 1, self.num_classes))
            mark, mask, loss = self.remask(label)
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)

        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)

        return mark_list, mask_list, loss_list

    def loss_fn(self, _input, _label, Y, mask, mark, label):
        X = (_input + mask * (mark - _input)).clamp(0., 1.)
        Y = label * torch.ones_like(_label, dtype=torch.long)
        _output = self.model(self.normalizer(X))
        return self.criterion(_output, Y)

    def remask(self, label: int):
        epoch = self.epoch
        # no bound
        atanh_mark = torch.randn(self.shape, device=self.device)
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(self.shape[1:], device=self.device)
        atanh_mask.requires_grad_()
        mask = tanh_func(atanh_mask)    # (h, w)
        mark = tanh_func(atanh_mark)    # (c, h, w)

        optimizer = optim.Adam(
            [atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9))
        optimizer.zero_grad()

        cost = self.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        entropy_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_norm_best = norm_best

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        for _epoch in range(epoch):
            satisfy_threshold = False
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.perf_counter()
            loader = self.loader
            if self.tqdm:
                loader = tqdm(self.loader)
            for _input, _label in loader:
                _input = self.denormalizer(_input.to(device=self.device))
                _label = _label.to(device=self.device)
                batch_size = _label.size(0)
                X = (_input + mask * (mark - _input)).clamp(0., 1.)
                Y = label * torch.ones_like(_label, dtype=torch.long)
                _output = self.model(self.normalizer(X))

                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = self.loss_fn(_input, _label, Y, mask, mark, label)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm # NC loss function

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)    # (h, w)
                mark = tanh_func(atanh_mark)    # (c, h, w)
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = 'Epoch: {}/{}'.format(_epoch + 1, epoch)
            _str = ' '.join([
                f'Loss: {losses.avg:.4f},'.ljust(20),
                f'Acc: {acc.avg:.4f}, '.ljust(20),
                f'Norm: {norm.avg:.4f},'.ljust(20),
                f'Entropy: {entropy.avg:.4f},'.ljust(20),
                f'Time: {epoch_time},'.ljust(20),
            ])
            print(pre_str, _str)

            # check to save best mask or not
            if acc.avg >= self.attack_succ_threshold and (norm.avg < norm_best or satisfy_threshold == False):
                satisfy_threshold = True
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if norm_best < float('inf'):
                    if norm_best >= self.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and acc.avg >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    cost = self.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2f' % cost)
            else:
                cost_set_counter = 0

            if acc.avg >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                print('up cost from %.4f to %.4f' % (cost, cost * self.cost_multiplier_up))
                cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                print('down cost from %.4f to %.4f' % (cost, cost / self.cost_multiplier_down))
                cost /= self.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                if acc.avg >= self.attack_succ_threshold: satisfy_threshold = True
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        return mark_best, mask_best, entropy_best
    