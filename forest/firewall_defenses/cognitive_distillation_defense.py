import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Any


def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)

def min_max_normalization(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm

class CognitiveDistillationAnalysis():
    def __init__(self, od_type='l1_norm', norm_only=False):
        self.od_type = od_type
        self.norm_only = norm_only
        self.mean = None
        self.std = None
        return

    def train(self, data):
        if not self.norm_only:
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        self.mean = torch.mean(data).item()
        self.std = torch.std(data).item()
        return

    def predict(self, data, t=1):
        if not self.norm_only:
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        p = (self.mean - data) / self.std
        p = torch.where((p > t) & (p > 0), 1, 0)
        return p.numpy()

    def analysis(self, data, is_test=False):
        """
            data (torch.tensor) b,c,h,w
            data is the distilled mask or pattern extracted by CognitiveDistillation (torch.tensor)
        """
        if self.norm_only:
            if len(data.shape) > 1:
                data = torch.norm(data, dim=[1, 2, 3], p=1)
            score = data
        else:
            score = torch.norm(data, dim=[1, 2, 3], p=1)
        score = min_max_normalization(score)
        return 1 - score.numpy()  # Lower for BD


class CognitiveDistillation(nn.Module):
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = False
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def forward(self, model, images, labels=None):
        model.eval()
        b, c, h, w = images.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(images.device)
        mask_param = nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        if self.get_features:
            features, logits = model(images)
        else:
            logits = model(images).detach()
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(images.device)
            x_adv = images * mask + (1-mask) * torch.rand(b, c, 1, 1).to(images.device)
            if self.get_features:
                adv_fe, adv_logits = model(x_adv)
                if len(adv_fe[-2].shape) == 4:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=[1, 2, 3])
                else:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1(adv_logits, logits).mean(dim=1)
            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()
        mask = self.get_raw_mask(mask_param).detach().cpu()
        if self.norm_only:
            return torch.norm(mask, p=1, dim=[1, 2, 3])
        return mask.detach()
    
class CognitiveDefense:
    def __init__(self, model):
        self.model = model

    def scan(self, kettle):
        cd = CognitiveDistillation(lr=0.1, p=1, gamma=0.01, beta=10.0, num_steps=100)

        # Run detections on training set
        results = []
        for images, labels, _ in tqdm(kettle.validloader):
            images, labels = images.to(**kettle.setup), labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=True)
            batch_rs = cd(self.model, images, labels)
            results.append(batch_rs.detach().cpu())

        train_results = torch.cat(results, dim=0)

        # Run detections on backdoor test set
        results = []
        source_class = kettle.poison_setup['source_class'][0]
        for images, labels, _ in tqdm(kettle.source_testloader[source_class]):
            images, labels = images.to(**kettle.setup), labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=True)
            batch_rs = cd(self.model, images, labels)
            results.append(batch_rs.detach().cpu())

        test_results = torch.cat(results, dim=0)

        detector = CognitiveDistillationAnalysis()

        s = len(train_results) * 0.1
        s = int(s)
        threshold = 0.5

        detector.train(train_results[:s])
        y_pred_poison = detector.predict(test_results, t=threshold)
        true_positive_rate = y_pred_poison.sum() / len(y_pred_poison)

        y_pred_clean = detector.predict(train_results, t=threshold)
        false_positive_rate = y_pred_clean.sum() / len(y_pred_clean)

        print(f"True Positive Rate (TPR): {true_positive_rate:.4f}")
        print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")

        return true_positive_rate, false_positive_rate
    
class CognitiveDefense:
    def __init__(self, model):
        self.model = model

    def scan(self, kettle):
        cd = CognitiveDistillation(lr=0.1, p=1, gamma=0.01, beta=10.0, num_steps=100)

        # Run detections on training set
        results = []
        for images, labels, _ in tqdm(kettle.validloader):
            images, labels = images.to(**kettle.setup), labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=True)
            batch_rs = cd(self.model, images, labels)
            results.append(batch_rs.detach().cpu())

        train_results = torch.cat(results, dim=0)

        # Run detections on backdoor test set
        results = []
        source_class = kettle.poison_setup['source_class'][0]
        for images, labels, _ in tqdm(kettle.source_testloader[source_class]):
            images, labels = images.to(**kettle.setup), labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=True)
            batch_rs = cd(self.model, images, labels)
            results.append(batch_rs.detach().cpu())

        test_results = torch.cat(results, dim=0)

        detector = CognitiveDistillationAnalysis()

        s = len(train_results) * 0.1
        s = int(s)
        threshold = 0.5

        detector.train(train_results[:s])
        y_pred_poison = detector.predict(test_results, t=threshold)
        true_positive_rate = y_pred_poison.sum() / len(y_pred_poison)

        y_pred_clean = detector.predict(train_results, t=threshold)
        false_positive_rate = y_pred_clean.sum() / len(y_pred_clean)

        print(f"True Positive Rate (TPR): {true_positive_rate:.4f}")
        print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")

        return true_positive_rate, false_positive_rate