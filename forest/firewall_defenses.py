import torch
import torch.nn as nn
import numpy as np
import random
import os
import copy
import albumentations
import math
import cv2
import tqdm
import torchvision.transforms.v2 as transforms

from torch.utils.data import Subset
from forest.data.datasets import ImageDataset
from torch.nn import functional as F
from scipy.fftpack import dct, idct
from .consts import NON_BLOCKING

def get_firewall(firewall_name, model, dataset):
    if firewall_name.lower() == 'strip':
        return Strip(model.model)
    elif firewall_name.lower() == 'ibd_psc':
        return IBD_PSC(model.model, valset=dataset)
    elif firewall_name.lower() == 'cognitive_distillation':
        return CognitiveDistillation(model, dataset)
    elif firewall_name.lower() == 'frequency':
        return Frequency(model, dataset)
    elif firewall_name.lower() == 'scale_up':
        return ScaleUp(model, dataset)
    elif firewall_name.lower() == 'bad_expert':
        return BaDExpert(model, dataset)
    else:
        raise NotImplementedError('Firewall is not implemented')

class Strip():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model.cuda()

    def scan(self, kettle, defense_fpr=0.1, batch_size=64):
        # choose a decision boundary with the test set
        clean_entropy = []
        clean_set_loader = torch.utils.data.DataLoader(kettle.validset, batch_size=batch_size, shuffle=False)
        for _input, _label, _ in tqdm(clean_set_loader):
            _input, _label = _input.to(**kettle.setup), _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            entropies = self.check(_input, _label, kettle.validset, self.model)
            for e in entropies:
                clean_entropy.append(e)
        clean_entropy = torch.FloatTensor(clean_entropy)

        clean_entropy, _ = clean_entropy.sort()
        
        threshold_low = float(clean_entropy[int(defense_fpr * len(clean_entropy))])

        # Calculate TPR
        all_entropy = []
        source_class = kettle.poison_setup['source_class'][0]
        for _input, _label, _ in tqdm(kettle.source_testloader[source_class]):
            _input, _label = _input.to(**kettle.setup), _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            entropies = self.check(_input, _label, kettle.validset, self.model)
            for e in entropies:
                all_entropy.append(e)
        all_entropy = torch.FloatTensor(all_entropy)
        true_positives = (all_entropy < threshold_low).sum().item()
        true_positive_rate = true_positives / len(kettle.source_testloader[source_class])

        # Calculate FPR
        all_entropy = []
        for _input, _label, _ in tqdm(kettle.validloader):
            _input, _label = _input.to(**kettle.setup), _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            entropies = self.check(_input, _label, kettle.validset, self.model)
            for e in entropies:
                all_entropy.append(e)
        all_entropy = torch.FloatTensor(all_entropy)
        false_positives = (all_entropy < threshold_low).sum().item()
        false_positive_rate = false_positives / len(kettle.validloader)

        print(f"True Positive Rate (TPR): {true_positive_rate:.4f}")
        print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")
        
        return true_positive_rate, false_positive_rate
    
    def check(self, _input, _label, source_set, model, N=200):
        _list = []

        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:N]

        with torch.no_grad():

            for i in samples:
                X, _, _ = source_set[i]
                X = X.cuda()
                _test = self.superimpose(_input, X)
                entro = self.entropy(_test, model).cpu().detach()
                _list.append(entro)

        return torch.stack(_list).mean(0)
    
    def superimpose(self, _input1, _input2, alpha = None):
        if alpha is None:
            alpha = 0.5

        result = _input1 + alpha * _input2
        return result

    def entropy(self, _input, model) -> torch.Tensor:
        p = torch.nn.Softmax(dim=1)(model(_input)) + 1e-8
        return (-p * p.log()).sum(1)

class IBD_PSC():
    """Identify and filter malicious testing samples (IBD-PSC).

    Args:
        model (nn.Module): The original backdoored model.
        n (int): The hyper-parameter for the number of parameter-amplified versions of the original backdoored model by scaling up of its different BN layers.
        xi (float): The hyper-parameter for the error rate.
        T (float):  The hyper-parameter for defender-specified threshold T. If PSC(x) > T , we deem it as a backdoor sample.
        scale (float): The hyper-parameter for amplyfying the parameters of selected BN layers.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.

        
    """
    def __init__(self, model, n=5, xi=0.6, T = 0.9, scale=1.5, valset=None, seed=666, deterministic=False):
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.n = n
        self.xi = xi
        self.T = T
        self.scale = scale
        self.valset = valset

        layer_num = self.count_BN_layers()
        sorted_indices = list(range(layer_num))
        sorted_indices = list(reversed(sorted_indices))
        self.sorted_indices = sorted_indices
        self.start_index = self.prob_start(self.scale, self.sorted_indices, valset=self.valset)

    def count_BN_layers(self):
        layer_num = 0
        for (name1, module1) in self.model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                layer_num += 1
        return layer_num

    def scale_var_index(self, index_bn, scale=1.5):
        copy_model = copy.deepcopy(self.model)
        index  = -1
        for (name1, module1) in copy_model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                index += 1
                if index in index_bn:
                    module1.weight.data *= scale
                    module1.bias.data *= scale
        return copy_model  
    
    def prob_start(self, scale, sorted_indices, valset):
        val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
        layer_num = len(sorted_indices)
        # layer_index: k
        for layer_index in range(1, layer_num):            
            layers = sorted_indices[:layer_index]
            # print(layers)
            smodel = self.scale_var_index(layers, scale=scale)
            smodel.cuda()
            smodel.eval()
            
            total_num = 0 
            clean_wrong = 0
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    clean_img = batch[0]
                    labels = batch[1]
                    clean_img = clean_img.cuda()  # batch * channels * hight * width
                    # labels = labels.cuda()  # batch
                    clean_logits = smodel(clean_img).detach().cpu()
                    clean_pred = torch.argmax(clean_logits, dim=1)# model prediction
                    
                    clean_wrong += torch.sum(labels != clean_pred)
                    total_num += labels.shape[0]
                wrong_acc = clean_wrong / total_num
                # print(f'wrong_acc: {wrong_acc}')
                if wrong_acc > self.xi:
                    return layer_index

    def _test(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        self.model.eval()
        total_num = 0
        all_psc_score = []

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                labels = batch[1]
                total_num += labels.shape[0]
                imgs = imgs.cuda()  # batch * channels * hight * width
                labels = labels.cuda()  # batch
                original_pred = torch.argmax(self.model(imgs), dim=1) # model prediction

                psc_score = torch.zeros(labels.shape)
                scale_count = 0
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers = self.sorted_indices[:layer_index+1]
                    # print(f'layers: {layers}')
                    smodel = self.scale_var_index(layers, scale=self.scale)
                    scale_count += 1
                    smodel.eval()
                    logits = smodel(imgs).detach().cpu()
                    softmax_logits = torch.nn.functional.softmax(logits, dim=1)
                    psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred.cpu()]

                psc_score /= scale_count
                all_psc_score.append(psc_score)
        
        all_psc_score = torch.cat(all_psc_score, dim=0)
        return all_psc_score
    
    def scan(self, kettle):
        print(f'start_index: {self.start_index}')

        testset = kettle.validset
        source_class = kettle.poison_setup['source_class'][0]
        poisoned_testset = kettle.source_testset[source_class]
        print(len(poisoned_testset))
        
        benign_psc = self._test(testset)
        poison_psc = self._test(poisoned_testset)

        true_positive_rate = (poison_psc >= self.T).sum().item() / len(poison_psc)
        false_positive_rate = (benign_psc >= self.T).sum().item() / len(benign_psc)

        print("True Positive Rate: {:.4f}".format(true_positive_rate))
        print("False Positive Rate: {:.4f}".format(false_positive_rate))

    def _detect(self, inputs):
        inputs = inputs.cuda()
        self.model.eval()
        self.model.cuda()
        original_pred = torch.argmax(self.model(inputs), dim=1) # model prediction

        psc_score = torch.zeros(inputs.size(0))
        scale_count = 0
        for layer_index in range(self.start_index, self.start_index + self.n):
            layers = self.sorted_indices[:layer_index+1]
            # print(f'layers: {layers}')
            smodel = self.scale_var_index(layers, scale=self.scale)
            scale_count += 1
            smodel.eval()
            logits = smodel(inputs).detach().cpu()
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred]

        psc_score /= scale_count
        
        y_pred = psc_score >= self.T
        return y_pred
    
    def detect(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                return self._detect(imgs)

def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)


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
    

class Frequency:
    def __init__(self, weight_path=None):
        self.model = nn.Sequential(nn.Flatten(),
                                              nn.Linear(3*224*224, 2)
                                    )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))
     
    def train(self, num_epochs=50):
        # Process dataset to create a dataset with backdoored samples and benign samples in the frequency domain
        dataset = ImageDataset('datasets/Facial_recognition_crop_partial/real_beard/train')
        tensor_list = []
        labels_list = []
        totensor = transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])
        
        for img, _, _ in dataset:
            # 0 means benign, 1 means poisoned
            tensor_list.append(totensor(self.dct2(img)))
            labels_list.append(0)
            
            transformed_img = self.patching_train(img)
            tensor_list.append(totensor(self.dct2(transformed_img)))
            labels_list.append(1)
            
        labels = torch.tensor(labels_list, dtype=torch.long)
        img_tensors = torch.stack(tensor_list)
        
        train_dataset = torch.utils.data.TensorDataset(img_tensors, labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Move the model to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        optimizer = torch.optim.Adadelta(lr=0.05, params=self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for imgs, labels in train_loader:
                # Move data and labels to the GPU
                imgs, labels = imgs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                running_loss += loss.item()

                # Calculate the number of correct predictions
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_predictions

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        print('Finished Training')
        torch.save(self.model.state_dict(), os.path.join('forest/frequency_detect_model', f"{num_epochs}_epochs.pth"))

    def evaluate(self, attack_list = None):
        self.model.eval()
        if attack_list == None:
            attack_list = ['l0_inv', 'badnets', 'smooth', 'blend', ]

        test_set = ImageDataset('datasets/Facial_recognition_crop_partial/real_beard/test')
        totensor = transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        for attack in attack_list:
            print(f"Evaluating detector on {attack.capitalize()}")

            clean_samples = []
            poison_samples = []

            for img, _, _ in test_set:
                # 0 means benign, 1 means poisoned
                clean_samples.append(totensor(self.dct2(img)))
                
                transformed_img = self.patching_test(img, attack)
                poison_samples.append(totensor(self.dct2(transformed_img)))

            clean_samples = torch.stack(clean_samples)
            clean_labels = torch.zeros(clean_samples.size(0), dtype=torch.long)

            poison_samples = torch.stack(poison_samples)
            poison_labels = torch.ones(poison_samples.size(0), dtype=torch.long)

            clean_testset = torch.utils.data.TensorDataset(clean_samples, clean_labels)
            poison_testset = torch.utils.data.TensorDataset(poison_samples, poison_labels)

            clean_testloader = torch.utils.data.DataLoader(clean_testset, batch_size=32, shuffle=False)
            poison_testloader = torch.utils.data.DataLoader(poison_testset, batch_size=32, shuffle=False)

            return self.scan(clean_testloader, poison_testloader)

    def scan(self, kettle):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        clean_set = kettle.validset
        clean_set.transform = None

        source_class = kettle.poison_setup['source_class'][0]
        poison_set = kettle.source_testset[source_class]
        poison_set.dataset.transform  = None

        totensor = transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        with torch.no_grad():  # Disable gradient calculation
            # Test on clean samples
            for inputs, _, _ in clean_set:
                transformed_inputs = totensor(self.dct2(inputs)).unsqueeze(0).cuda()

                outputs = self.model(transformed_inputs)
                _, preds = torch.max(outputs, 1)

                # Assuming 0 = clean and 1 = poisoned
                true_negatives += (preds == 0).sum().item()
                false_positives += (preds == 1).sum().item()

            # Test on poisoned samples
            for inputs, _, _ in poison_set:
                transformed_inputs = totensor(self.dct2(inputs)).unsqueeze(0).cuda()
                
                outputs = self.model(transformed_inputs)
                _, preds = torch.max(outputs, 1)

                # Assuming 0 = clean and 1 = poisoned
                true_positives += (preds == 1).sum().item()
                false_negatives += (preds == 0).sum().item()

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

        print(f"True Positive Rate (TPR): {tpr:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")

        return tpr, fpr
    
    @staticmethod
    def dct2 (block):
        return dct(dct(block, norm='ortho', axis=0), norm='ortho', axis=1)

    @staticmethod
    def idct2(block):
        return idct(idct(block, norm='ortho', axis=0), norm='ortho', axis=1)

    @staticmethod
    def tensor2img(t):
        t_np = t.detach().cpu().numpy().transpose(1, 2, 0)
        return t_np
    
    @staticmethod
    def addnoise(img):
        aug = albumentations.GaussNoise(p=1,mean=25,var_limit=(10,70))
        augmented = aug(image=(img*255).astype(np.uint8))
        auged = augmented['image']/255
        return auged

    @staticmethod
    def randshadow(img):
        aug = albumentations.RandomShadow(p=1)
        test = (img*255).astype(np.uint8)
        augmented = aug(image=cv2.resize(test,(224,224)))
        auged = augmented['image']/255
        return auged
    
    @staticmethod
    def gauss_smooth(image, sig=6):
        size_denom = 5.
        sigma = sig * size_denom
        kernel_size = sigma
        mgrid = np.arange(kernel_size, dtype=np.float32)
        mean = (kernel_size - 1.) / 2.
        mgrid = mgrid - mean
        mgrid = mgrid * size_denom
        kernel = 1. / (sigma * math.sqrt(2. * math.pi)) * \
                np.exp(-(((mgrid - 0.) / (sigma)) ** 2) * 0.5)
        kernel = kernel / np.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernelx = np.tile(np.reshape(kernel, (1, 1, int(kernel_size), 1)), (3, 1, 1, 1))
        kernely = np.tile(np.reshape(kernel, (1, 1, 1, int(kernel_size))), (3, 1, 1, 1))

        padd0 = int(kernel_size // 2)
        evenorodd = int(1 - kernel_size % 2)

        pad = torch.nn.ConstantPad2d((padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 0.)
        in_put = torch.from_numpy(np.expand_dims(np.transpose(image.astype(np.float32), (2, 0, 1)), axis=0))
        output = pad(in_put)

        weightx = torch.from_numpy(kernelx)
        weighty = torch.from_numpy(kernely)
        conv = F.conv2d
        output = conv(output, weightx, groups=3)
        output = conv(output, weighty, groups=3)
        output = Frequency.tensor2img(output[0])

        return output

    @staticmethod
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    def patching_train(self, clean_sample):
        '''
        This code conducts a patching procedure with random white blocks or random noise block
        '''
        attack = np.random.randint(0,5)
        pat_size_x = np.random.randint(14,56)
        pat_size_y = np.random.randint(14,56)
        output = self.normalization(np.copy(clean_sample))
        if attack == 0:
            block = np.ones((pat_size_x,pat_size_y,3))
        elif attack == 1:
            block = np.random.rand(pat_size_x,pat_size_y,3)
        elif attack == 2:
            return self.addnoise(output)
        elif attack == 3:
            return self.randshadow(output)
        if attack == 4:
            randind = np.random.randint(output.shape[0])
            tri = output[randind]
            mid = output+0.3*tri
            mid[mid>1]=1
            return mid
            
        margin = np.random.randint(0,42)
        rand_loc = np.random.randint(0,4)
        if rand_loc==0:
            output[margin:margin+pat_size_x,margin:margin+pat_size_y,:] = block #upper left
        elif rand_loc==1:
            output[margin:margin+pat_size_x,224-margin-pat_size_y:224-margin,:] = block
        elif rand_loc==2:
            output[224-margin-pat_size_x:224-margin,margin:margin+pat_size_y,:] = block
        elif rand_loc==3:
            output[224-margin-pat_size_x:224-margin,224-margin-pat_size_y:224-margin,:] = block #right bottom

        output[output > 1] = 1
        return output 

    def patching_test(self, clean_sample, attack_name):
        '''
        This code conducts a patching procedure to generate backdoor data.
        **Please make sure the input sample's label is different from the target label.
        
        clean_sample: clean input
        attack_name: trigger's file name
        '''

        output = self.normalization(np.copy(clean_sample))

        if attack_name == 'badnets':
            pat_size = 28
            output[224-1-pat_size:224-1,224-1-pat_size:224-1,:] = 1

        else:
            if attack_name == 'l0_inv':
                trimg = cv2.imread('forest/triggers/'+ attack_name + '.png')
                # Resize the trigger image to 224x224
                trimg_resized = self.normalization(cv2.resize(trimg, dsize=(224, 224), interpolation=cv2.INTER_NEAREST))
                
                mask = 1 - np.load('forest/triggers/mask.npy').transpose(1,2,0)
                # Resize the mask to 224x224
                mask_resized = cv2.resize(mask, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                output = output * mask_resized + trimg_resized

            elif attack_name == 'smooth':
                trimg = np.load('forest/triggers/best_universal.npy')[0]
                # Resize the trigger image to 224x224
                trimg_resized = self.normalization(cv2.resize(trimg, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))
                output = output + trimg_resized

            else:
                trimg = cv2.imread('forest/triggers/' + attack_name + '.png')
                # Resize the trigger image to 224x224
                trimg_resized = self.normalization(cv2.resize(trimg, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))
                
                output = output + trimg_resized
        
        output = self.normalization(output)
        # Ensure the output is within valid pixel range [0, 1]
        output[output > 1] = 1
        return output

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
    
class CognitiveDefense:
    def __init__(self, model):
        self.model = model

    def scan(self, kettle):
        cd = CognitiveDistillation(lr=0.1, p=1, gamma=0.01, beta=10.0, num_steps=100)

        # Run detections on training set
        results = []
        for images, labels, _ in tqdm(kettle.validloader):
            images, labels = images.to(**kettle.setup), labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            batch_rs = cd(self.model, images, labels)
            results.append(batch_rs.detach().cpu())

        train_results = torch.cat(results, dim=0)

        # Run detections on backdoor test set
        results = []
        source_class = kettle.poison_setup['source_class'][0]
        for images, labels, _ in tqdm(kettle.source_testloader[source_class]):
            images, labels = images.to(**kettle.setup), labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
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

class ScaleUp():
    name: str = 'scale up'

    def __init__(self, model, kettle, defense_ratio=0.2, scale_set=None, threshold=None, with_clean_data=True):
        if scale_set is None:
            scale_set = [3, 5, 7, 9, 11]
        if threshold is None:
            self.threshold = 0.5

        size = defense_ratio * len(self.kettle.validset)
        random_indices = torch.randperm(len(self.kettle.validset))[:size]
        self.clean_set = Subset(self.kettle.validset, indices=random_indices)
        self.clean_loader = torch.utils.data.DataLoader(self.clean_set, 
                                                        batch_size=64, 
                                                        shuffle=False, 
                                                        num_workers=4, 
                                                        pin_memory=True
                                                    )
        
        self.scale_set = scale_set
        self.model = model
        self.kettle = kettle


        self.with_clean_data = with_clean_data
        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)

        self.mean = None
        self.std = None
        if self.with_clean_data:
            self.init_spc_norm()

    def detect(self, use_pseudo_labels=True):

        false_positives = 0
        clean_pred_correct_mask = []
        pred_poison_mask = []

        for idx, (clean_img, labels) in enumerate(self.kettle.validloader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch

            correct_pred_mask = torch.eq(labels, torch.argmax(self.model(clean_img), dim=1))
            clean_pred_correct_mask.append(correct_pred_mask)

            # evaluate the clean data
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(torch.clip(clean_img) * scale, 0.0, 1.0)
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)
            clean_pred = torch.argmax(self.model(clean_img), dim=1) # model prediction
            # compute the SPC Value
            spc_clean = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc_clean += scale_label == clean_pred
            spc_clean /= len(self.scale_set)

            if self.with_clean_data:
                spc_clean = (spc_clean - self.mean) / self.std
            
            pred_poison_mask.append(spc_clean > self.threshold)
            false_positives += (spc_clean > self.threshold).sum().item()

        clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
        pred_poison_mask = torch.cat(pred_poison_mask, dim=0)

        print("Clean Accuracy: %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(pred_poison_mask)].sum(), len(self.kettle.validloader),
                                                clean_pred_correct_mask[torch.logical_not(pred_poison_mask)].sum() / len(self.kettle.validloader)))

        print(f"False Positive Rate (FPR): {false_positives / len(self.kettle.validloader):.4f}")

        true_positives = 0
        poison_attack_success_mask = []
        pred_poison_mask = []
        source_class = self.kettle.poison_setup['source_class'][0]
        target_class = self.kettle.poison_setup['target_class']

        for idx, (trigger_img, labels) in enumerate(self.kettle.source_testloader[source_class]):
            trigger_img = trigger_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()

            preds = torch.argmax(self.model(trigger_img), dim=1)
            poison_attack_success_mask.append(torch.eq(preds, target_class))

            if use_pseudo_labels:
                labels = preds

            # evaluate the clean data
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(torch.clip(trigger_img) * scale, 0.0, 1.0)

            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)
            poison_pred = torch.argmax(self.model(trigger_img), dim=1) # model prediction
            # compute the SPC Value
            spc_poison = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc_poison += scale_label == poison_pred
            spc_poison /= len(self.scale_set)

            if self.with_clean_data:
                spc_poison = (spc_poison - self.mean) / self.std

            pred_poison_mask.append(spc_poison > self.threshold)
            true_positives += (spc_poison > self.threshold).sum().item()
        
        poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)

        print(f"ASR: %d/%d = %.6f" % (poison_attack_success_mask[torch.logical_not(pred_poison_mask)].sum(), len(self.kettle.source_testloader[source_class]),
                                    poison_attack_success_mask[torch.logical_not(pred_poison_mask)].sum() / len(self.kettle.source_testloader[source_class])))
        print(f"True Positive Rate (TPR): {true_positives / len(self.kettle.source_testloader[source_class]):.4f}")


    def init_spc_norm(self):
        total_spc = []
        for idx, (clean_img, labels) in enumerate(self.clean_loader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(torch.clip(clean_img) * scale, 0.0, 1.0)
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)
        self.mean = torch.mean(total_spc).item()
        self.std = torch.std(total_spc).item()


import math
import time
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn, optim


class BaDExpert():
    """
    BaDExpert
    
    .. _BaDExpert:
        https://openreview.net/forum?id=s56xikpD92
        
    This is the official code implementation!
    """
    def __init__(self, kettle, defense_ratio=0.2, defense_fpr=0.01, hard_filter=False):
        self.kettle = kettle

        size = defense_ratio * len(self.kettle.validset)
        random_indices = torch.randperm(len(self.kettle.validset))[:size]
        self.clean_set = Subset(self.kettle.validset, indices=random_indices)
        self.clean_loader = torch.utils.data.DataLoader(self.clean_set, 
                                                        batch_size=64, 
                                                        shuffle=False, 
                                                        num_workers=4, 
                                                        pin_memory=True
                                                    )
        
        self.defense_fpr = defense_fpr
        self.hard_filter = hard_filter
        
    def detect(self, original_model):
        '''
        original_model: the original backdoored model
        '''
        
        start_time = time.perf_counter()
        print("\n#####[BAD EXPERT DETECTION]#####")
        
        unlearned_model = copy.deepcopy(original_model)
        shadow_model = copy.deepcopy(original_model)
        
        unlearned_model = self.unlearn(unlearned_model, self.kettle)
        shadow_model = self.finetune(shadow_model, self.kettle)

        original_model = nn.DataParallel(original_model)
        shadow_model = nn.DataParallel(shadow_model)
        unlearned_model = nn.DataParallel(unlearned_model)
        
        original_model = original_model.cuda()
        shadow_model = shadow_model.cuda()
        unlearned_model = unlearned_model.cuda()
        
        original_model.eval()
        shadow_model.eval()
        unlearned_model.eval()

        print("[Original]")
        eval_model(original_model, self.kettle)
        print("[Repaired]")
        eval_model(shadow_model, self.kettle)
        print("[Unlearned]")
        eval_model(unlearned_model, self.kettle)


        threshold = self.get_threshold(self.defense_fpr, original_model, shadow_model, unlearned_model, self.kettle.validloader)
        self.deploy(original_model, shadow_model, unlearned_model, threshold)
        
        end_time = time.perf_counter()
        print("Elapsed time: {:.2f}s".format(end_time - start_time))


    def get_threshold(self, fpr, original_model, shadow_model, unlearned_model, test_set_loader):
        print("Selecting decision threshold for FPR={}...".format(fpr))
        with torch.no_grad():
            targets = []
            original_output = []
            unlearned_output = []
            shadow_output = []
            original_pred = []
            for batch_idx, (data, target) in enumerate(tqdm(test_set_loader)):
                # on clean data
                data, target = data.cuda(), target.cuda()
                
                targets.append(target)
                original_output.append(original_model(data))
                unlearned_output.append(unlearned_model(data))
                shadow_output.append(shadow_model(data))

            targets = torch.cat(targets, dim=0)
            original_output = torch.cat(original_output, dim=0)
            unlearned_output = torch.cat(unlearned_output, dim=0)
            shadow_output = torch.cat(shadow_output, dim=0)
            
            softmax = nn.Softmax(dim=1)
        
            original_pred = original_output.argmax(dim=1)
            original_pred_correct = torch.eq(targets, original_pred)

            original_output = softmax(original_output)
            unlearned_output = softmax(unlearned_output)
            shadow_output = softmax(shadow_output)
                
            triangle = []

            for i in range(len(original_output)):
                y = shadow_output[i, original_pred[i]]
                x = unlearned_output[i, original_pred[i]]
                triangle.append(torch.minimum(2 * (y) / torch.clamp(x, min=1e-8), (1 - x) / torch.clamp(0.5 - y, min=1e-8))) # resnet18

            triangle = -torch.tensor(triangle).cuda()
            
        values = triangle[original_pred_correct]
        threshold_triangle = float(values.sort()[0][int((1 - fpr) * len(values))])

        return threshold_triangle

    def unlearn(self, model, clean_loader):
        model = nn.DataParallel(model)
        model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[1], gamma=0.1)

        # Construct a predicion dictionary
        true_pred = []
        model.eval()
        for batch_idx, (data, target) in enumerate(clean_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            true_pred.append(pred)

        # Unlearn
        unlearning_epochs = 1
        for epoch in range(1, unlearning_epochs + 1):  # train base model

            model.train()
            # model.apply(tools.set_bn_eval)

            for batch_idx, (data, target) in enumerate(clean_loader):

                optimizer.zero_grad()

                data, target = data.cuda(), target.cuda()
                output = model(data)

                soft_target = torch.empty((target.shape[0], self.kettle.num_classes)).fill_(0.3).cuda()
                for i in range(len(true_pred[batch_idx])):
                    soft_target[i, true_pred[batch_idx][i]] = 0

                soft_target = (target + 1) % self.kettle.num_classes

                # calc loss with soft target
                loss = criterion(output, soft_target)

                loss.backward()
                optimizer.step()

            print('\n<Unlearning> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}'.format(epoch, loss.item(), optimizer.param_groups[0]['lr']))

            # Evaluate
            clean_acc, asr = eval_model(model, self.kettle)
            if clean_acc < 0.2:
                print("Early stopping...")
                break

            scheduler.step()
        
        return model


    def finetune(self, model, clean_loader):
        model = nn.DataParallel(model)
        model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=[2, 4, 6, 8], gamma=0.1)

        # Finetune
        finetuning_epochs = 10
        for epoch in range(1, finetuning_epochs + 1):  # train base model

            model.train()
            # model.apply(tools.set_bn_eval)

            for batch_idx, (data, target) in enumerate(clean_loader):

                optimizer.zero_grad()

                data, target = data.cuda(), target.cuda()

                output = model(data)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

            print('\n<Finetuning> Train Epoch: {} \tLoss: {:.6f}, lr: {:.2f}'.format(epoch, loss.item(), optimizer.param_groups[0]['lr']))

            scheduler.step()

        eval_model(model, self.kettle)
        return model

    def deploy(self, original_model, shadow_model, unlearned_model, threshold):
        original_model.eval()
        shadow_model.eval()
        unlearned_model.eval()
        
        print("\nFor clean inputs:")
        clean_y_pred = []
        clean_y_score = []
        clean_pred_correct_mask = []

        false_positives = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.kettle.validloader)):
                # on clean data
                data, target = data.cuda(), target.cuda()
                
                original_output = original_model(data)
                unlearned_output = unlearned_model(data)
                shadow_output = shadow_model(data)

                original_pred = original_output.argmax(dim=1)
                
                mask = torch.eq(original_pred, target) # only look at those samples that successfully attack the DNN
                clean_pred_correct_mask.append(mask)
                
                alert_mask, alert_score = self.get_alert_mask(original_output, shadow_output, unlearned_output, threshold, return_score=True) # filter! 
                clean_y_pred.append(alert_mask)
                clean_y_score.append(alert_score)

                false_positives += (alert_mask).sum().item()
        clean_y_pred = torch.cat(clean_y_pred, dim=0)
        clean_y_score = torch.cat(clean_y_score, dim=0)
        clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
        print("Clean Accuracy: %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum(), len(clean_pred_correct_mask),
                                                clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum() / len(clean_pred_correct_mask)))
        print("Clean Accuracy (not alert): %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum(), torch.logical_not(clean_y_pred).sum(),
                                                            clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum() / torch.logical_not(clean_y_pred).sum() if torch.logical_not(clean_y_pred).sum() > 0 else 0))

        print("False Positives: %d/%d = %.6f" % (false_positives, len(self.kettle.validloader),
                                                false_positives / len(self.kettle.validloader)))

        print("\nFor poison inputs:")
        poison_y_pred = []
        poison_y_score = []
        poison_attack_success_mask = []
        true_positives = 0

        source_class = self.kettle.poison_setup['source_class'][0]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.kettle.source_testloader[source_class])):
                # on poison data
                data, target = data.cuda(), target.cuda()
                
                original_output = original_model(data)
                unlearned_output = unlearned_model(data)
                shadow_output = shadow_model(data)
                
                original_pred = original_output.argmax(dim=1)
                poison_attack_success_mask.append(torch.eq(original_pred, target))

                alert_mask, alert_score = self.get_alert_mask(original_output, shadow_output, unlearned_output, threshold, return_score=True) # filter!
                poison_y_pred.append(alert_mask)
                poison_y_score.append(alert_score)

                true_positives += (alert_mask).sum().item()
        poison_y_pred = torch.cat(poison_y_pred, dim=0)
        poison_y_score = torch.cat(poison_y_score, dim=0)
        poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)
        print("ASR: %d/%d = %.6f" % (poison_attack_success_mask[torch.logical_not(poison_y_pred)].sum(), len(self.kettle.source_testloader[source_class]),
                                    poison_attack_success_mask[torch.logical_not(poison_y_pred)].sum() / len(self.kettle.source_testloader[source_class])))
        
        print("True Positives: %d/%d = %.6f" % (true_positives, len(self.kettle.source_testloader[source_class]),
                                                true_positives / len(self.kettle.source_testloader[source_class])))

    def get_alert_mask(self, original_output, shadow_output, unlearned_output, threshold, return_score=False):
        softmax = nn.Softmax(dim=1)

        original_pred = original_output.argmax(dim=1)

        original_output = softmax(original_output)
        unlearned_output = softmax(unlearned_output)
        shadow_output = softmax(shadow_output)
        
        triangle = []
        u_s_diff = []

        for i in range(len(original_output)):
            y = shadow_output[i, original_pred[i]]
            x = unlearned_output[i, original_pred[i]]
            triangle.append(torch.minimum(2 * (y) / torch.clamp(x, min=1e-8), (1 - x) / torch.clamp(0.5 - y, min=1e-8))) # resnet18
            u_s_diff.append(unlearned_output[i, original_pred[i]] - shadow_output[i, original_pred[i]])

        triangle = -torch.tensor(triangle).cuda()
        u_s_diff = torch.tensor(u_s_diff).cuda()

        if threshold is not None:
            alert_mask = triangle > threshold
            if self.hard_filter:
                hard_mask = torch.logical_and(unlearned_output[i, original_pred[i]] > 0.98, shadow_output[i, original_pred[i]] < 0.5)
                alert_mask = torch.logical_or(alert_mask, hard_mask)
                triangle[hard_mask] = 1e8
            
            if return_score: 
                return alert_mask, triangle
            else: 
                return alert_mask
        else:
            alert_mask = u_s_diff > 0.15

        return alert_mask

def eval_model(model, kettle):
    model.eval()

    clean_acc, asr = 0, 0
    corrects = 0
    for batch_idx, (data, target) in enumerate(tqdm(kettle.validloader)):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        corrects += torch.eq(pred, target).sum().item()
    clean_acc = corrects / len(kettle.validloader)

    source_class = kettle.poison_setup['source_class'][0]
    target_class = kettle.poison_setup['target_class']

    corrects = 0
    for batch_idx, (data, _) in enumerate(tqdm(kettle.source_testloader[source_class])):
        data = data.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        corrects += torch.eq(pred, target_class).sum().item()
    asr = corrects / len(kettle.source_testloader[source_class])

    print(f"Clean Accuracy: {clean_acc:.4f}, ASR: {asr:.4f}")
    return clean_acc, asr