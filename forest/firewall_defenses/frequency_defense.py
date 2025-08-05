import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torch
import torch.nn as nn
import numpy as np
import random
import os
import copy
import albumentations
import math
import cv2
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F

from forest.data.datasets import ImageDataset
from scipy.fftpack import dct, idct

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
            transforms.ToDtype(torch.float32, scale=True),
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
            transforms.ToImage(), 
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
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True),
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