"""Definition for multiple victims that share a single GPU (sequentially)."""

import torch
import numpy as np
import copy
import os
import warnings
import os
import tqdm
import math
import random

from collections import defaultdict
from math import ceil
from .models import get_model
from ..hyperparameters import training_strategy
from .training import get_optimizers, run_step, run_validation, check_sources, check_sources_all_to_all, check_suspicion
from ..utils import set_random_seed, write, OpenCVNonLocalMeansDenoiser
from ..consts import BENCHMARK, SHARING_STRATEGY, NORMALIZE
from .context import GPUContext
from ..data.datasets import normalization

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

from .victim_base import _VictimBase
from .training import get_optimizers

class _VictimEnsemble(_VictimBase):
    """Implement model-specific code and behavior for multiple models on a single GPU.

    --> Running in sequential mode!

    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        if seed is None:
            if self.args.model_seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = self.args.model_seed
        else:
            self.model_init_seed = seed

        set_random_seed(self.model_init_seed)
        
        print(f'Initializing ensemble {self.args.net} from random key {self.model_init_seed}.')
        write(f'Initializing ensemble {self.args.net} from random key {self.model_init_seed}.', self.args.output)

        self.models, self.definitions, self.optimizers, self.schedulers = [], [], [], []
        for idx in range(self.args.ensemble):
            model_name = self.args.net[idx % len(self.args.net)]
            model, defs, optimizer, scheduler = self._initialize_model(model_name, mode=self.args.scenario)
            
            self.models.append(model)
            self.definitions.append(defs)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
                
            print(f'{model_name} initialized as model {idx}')
            write(f'{model_name} initialized as model {idx}', self.args.output)
            print(repr(defs))
            write(repr(defs), self.args.output)
            
        self.defs = self.definitions[0]
        self.epochs = [0 for i in range(self.args.ensemble)]
        
        if self.args.scenario == 'transfer':
            self.freeze_feature_extractor()
            self.eval()
            print('Features frozen.')
        
    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        """
        Reinitialize the last layer of the model and/or update training parameters.
        
        Args:
            reduce_lr_factor: Factor to reduce learning rate by
            seed: Random seed for layer initialization
            keep_last_layer: If True, keep the existing last layer weights and only update optimizer
        """
                
        if self.args.model_seed is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.model_seed
        set_random_seed(self.model_init_seed)

        for idx in range(self.args.ensemble):
            model_name = self.args.net[idx % len(self.args.net)]
            if not keep_last_layer:
                # We construct a full replacement model, so that the seed matches up with the initial seed,
                # even if all of the model except for the last layer will be immediately discarded.
                replacement_model = get_model(model_name, pretrained=True)

                # Rebuild model with new last layer
                frozen = self.models[idx].frozen
                self.models[idx] = torch.nn.Sequential(*list(self.models[idx].children())[:-1], torch.nn.Flatten(),
                                                       list(replacement_model.children())[-1])
                self.models[idx].frozen = frozen

            # Define training routine
            # Reinitialize optimizers here
            self.definitions[idx] = training_strategy(model_name, self.args)
            self.definitions[idx].lr *= reduce_lr_factor
            self.optimizers[idx], self.schedulers[idx] = get_optimizers(self.models[idx], self.args, self.definitions[idx])
            write(f'{model_name} with id {idx}: linear layer reinitialized.', self.args.output)
            write(repr(self.definitions[idx]), self.args.output)

    def save_feature_representation(self):
        self.clean_models = []
        for model in self.models:
            self.clean_models.append(copy.deepcopy(model))

    def load_feature_representation(self):
        for idx, clean_model in enumerate(self.clean_models):
            if isinstance(self.models[idx], torch.nn.DataParallel) or isinstance(self.models[idx], torch.nn.parallel.DistributedDataParallel):
                self.models[idx].module.load_state_dict(clean_model.module.state_dict())
            else:
                self.models[idx].load_state_dict(clean_model.state_dict())
                
    def freeze_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        for model in self.models:
            model.frozen = True
            
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                actual_model = model.module
            else:
                actual_model = model
            
            actual_model.frozen = True
            
            # Freeze all parameters first
            for param in actual_model.parameters():
                param.requires_grad = False
            
            # Try to find and unfreeze the classifier
            classifier_found = False
            
            # Common classifier attribute names in timm
            for attr_name in ['head', 'classifier', 'fc']:
                if hasattr(actual_model, attr_name):
                    classifier = getattr(actual_model, attr_name)
                    if isinstance(classifier, torch.nn.Module):
                        for param in classifier.parameters():
                            param.requires_grad = True
                        classifier_found = True
                        print(f"Unfroze classifier: {attr_name}")
                        break
            
            if not classifier_found:
                # Fallback to last child
                last_module = list(actual_model.children())[-1]
                for param in last_module.parameters():
                    param.requires_grad = True
                print(f"Unfroze last module: {type(last_module).__name__}")
            for param in model.parameters():
                param.requires_grad = False

            for param in list(model.children())[-1].parameters():
                param.requires_grad = True

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""
    def _iterate(self, kettle, poison_delta, max_epoch=None, stats=None):
        """Validate a given poison by training the model and checking source accuracy."""
        multi_model_setup = (self.models, self.definitions, self.optimizers, self.schedulers)

        # Only partially train ensemble for poisoning if no poison is present
        if max_epoch is None:
            max_epoch = self.defs.epochs
        
        if self.args.dryrun:
            max_epoch = 1

        if poison_delta is None and self.args.stagger is not None:
            if self.args.stagger == 'firstn':
                stagger_list = [int(epoch) for epoch in range(self.args.ensemble)]
            elif self.args.stagger == 'full':
                stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble)]
            elif self.args.stagger == 'inbetween':
                stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble + 2)[1:-1]]
            else:
                raise ValueError(f'Invalid stagger option {self.args.stagger}')
            print(f'Staggered pretraining to {stagger_list}.')
        else:
            stagger_list = [max_epoch] * self.args.ensemble

        if self.args.denoise:
            denoiser = OpenCVNonLocalMeansDenoiser(h=10, h_color=10)

            tensors = []
            labels = []

            for img, label, idx in tqdm.tqdm(kettle.trainset, desc="craft denoising"):
                lookup = kettle.poison_lookup.get(idx)
                tensor = img.clone()

                if lookup is not None:
                    tensor += poison_delta[lookup, :, :, :]

                tensor = denoiser(tensor)
                if NORMALIZE:
                    tensor = normalization(tensor)

                tensors.append(tensor)
                labels.append(label)
            
            tensors = torch.stack(tensors)
            labels = torch.tensor(labels)

            # Override train_loader
            dataset = torch.utils.data.TensorDataset(tensors, labels)
            kettle.trainloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=kettle.args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
        for idx, single_model in enumerate(zip(*multi_model_setup)):
            write(f"\nTraining model {idx}...", self.args.output)
            model, defs, optimizer, scheduler = single_model

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.frozen = model.module.frozen
            
            defs.epochs = max_epoch
            for epoch in range(1, stagger_list[idx]+1):
                # write(f"Training model {idx} Epoch {epoch}...", self.args.output)
                run_step(kettle, poison_delta, epoch, *single_model, stats=stats)
                if self.args.dryrun:
                    break
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))

        # Track epoch
        self.epochs = stagger_list

    def step(self, kettle, poison_delta):
        """Step through a model epoch. Optionally minimize poison loss during this.

        This function is limited because it assumes that defs.batch_size, defs.max_epoch, defs.epochs
        are equal for all models.
        """
        multi_model_setup = (self.models, self.definitions, self.optimizers, self.schedulers)

        for idx, single_model in enumerate(zip(*multi_model_setup)):
            model, defs, optimizer, scheduler = single_model
            model_name = self.args.net[idx % len(self.args.net)]

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.frozen = model.module.frozen
            run_step(kettle, poison_delta, self.epochs[idx], *single_model)
            self.epochs[idx] += 1
            if self.epochs[idx] == defs.epochs:
                self.epochs[idx] = 1
                write(f'Model {idx} reset to epoch 0.', self.args.output)
                model, defs, optimizer, scheduler = self._initialize_model(model_name)
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            self.models[idx], self.definitions[idx], self.optimizers[idx], self.schedulers[idx] = model, defs, optimizer, scheduler

    """ Various Utilities."""
    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        [model.eval() for model in self.models]
        if dropout:
            [model.apply(apply_dropout) for model in self.models]
            
    def reset_learning_rate(self):
        """Reset scheduler objects to initial state."""
        for idx in range(self.args.ensemble):
            _, _, optimizer, scheduler = self._initialize_model()
            self.optimizers[idx] = optimizer
            self.schedulers[idx] = scheduler

    def gradient(self, images, labels, criterion=None, selection=None):
         
        """Compute the gradient of criterion(model) w.r.t to given data."""
        grad_list, norm_list = [], []
        for model in self.models:
            with GPUContext(self.setup, model) as model:

                if criterion is None:
                    criterion = self.loss_fn
                differentiable_params = [p for p in model.parameters() if p.requires_grad]

                if selection == 'max_gradient':
                    grad_norms = []
                    for image, label in zip(images, labels):
                        loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        grad_norm = 0
                        for grad in gradients:
                            grad_norm += grad.detach().pow(2).sum()
                        grad_norms.append(grad_norm.sqrt())
                    
                    source_poison_selected = math.ceil(self.args.sources_selection_rate * images.shape[0])
                    indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-source_poison_selected:]
                    images = images[indices]
                    labels = labels[indices]
                    write('{} sources with maximum gradients selected'.format(source_poison_selected), self.args.output)
                
                # Using batch processing for gradients
                if self.args.source_gradient_batch is not None:
                    batch_size = self.args.source_gradient_batch
                    if images.shape[0] < batch_size:
                        batch_size = images.shape[0]
                    else:
                        if images.shape[0] % batch_size != 0:
                            batch_size = images.shape[0] // ceil(images.shape[0] / batch_size)
                            warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
                    gradients = None
                    for i in range(images.shape[0]//batch_size):
                        loss = batch_size * criterion(model(images[i*batch_size:(i+1)*batch_size]), labels[i*batch_size:(i+1)*batch_size])
                        if i == 0:
                            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        else:
                            gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))

                    gradients = tuple(map(lambda i: i / images.shape[0], gradients))
                else:
                    loss = criterion(model(images), labels)
                    gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)


                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norm = grad_norm.sqrt()            


                grad_list.append(gradients)
                norm_list.append(grad_norm.item())

        return grad_list, norm_list

    def gradient_with_repel(self, source_images, source_labels, repel_images, repel_labels, criterion=None, selection=None):
        """Compute the gradient of criterion(model) w.r.t to given data with repelling."""
        grad_list, norm_list = [], []
        for model in self.models:
            with GPUContext(self.setup, model) as model:

                if criterion is None:
                    criterion = self.loss_fn
                differentiable_params = [p for p in model.parameters() if p.requires_grad]

                # Select sources with maximum gradient
                if selection == 'max_gradient':
                    grad_norms = []
                    for image, label in zip(source_images, source_labels):
                        loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        grad_norm = 0
                        for grad in gradients:
                            grad_norm += grad.detach().pow(2).sum()
                        grad_norms.append(grad_norm.sqrt())
                    
                    source_poison_selected = ceil(self.args.sources_selection_rate * source_images.shape[0])
                    indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-source_poison_selected:]
                    source_images = source_images[indices]
                    source_labels = source_labels[indices]
                    write('{} sources with maximum gradients selected'.format(source_poison_selected), self.args.output)
                
                # Using batch processing for gradients for source images
                if not self.args.source_gradient_batch==None:
                    batch_size = self.args.source_gradient_batch
                    if source_images.shape[0] < batch_size:
                        batch_size = source_images.shape[0]
                    else:
                        if source_images.shape[0] % batch_size != 0:
                            batch_size = source_images.shape[0] // ceil(source_images.shape[0] / batch_size)
                            warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
                    source_gradients = None
                    for i in range(source_images.shape[0]//batch_size):
                        loss = self.args.scale * batch_size * criterion(model(source_images[i*batch_size:(i+1)*batch_size]), source_labels[i*batch_size:(i+1)*batch_size])
                        if i == 0:
                            source_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        else:
                            source_gradients = tuple(map(lambda i, j: i + j, source_gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))
                    source_gradients = tuple(map(lambda i: i / (source_images.shape[0] - (source_images.shape[0] % batch_size)), source_gradients))
                else:
                    loss = criterion(model(source_images), source_labels)
                    source_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

                source_grad_norm = sum(grad.detach().pow(2).sum() for grad in source_gradients).sqrt()
                print("Source gradient norm: ", source_grad_norm)

                permutation = torch.randperm(repel_images.size(0))
                repel_images = repel_images[permutation]
                repel_labels = repel_labels[permutation]
                flipped_labels = torch.tensor([source_labels[0]] * len(repel_images)).cuda()

                # Using batch processing for gradients for repel images
                if not self.args.source_gradient_batch==None:
                    batch_size = self.args.source_gradient_batch
                    if repel_images.shape[0] < batch_size:
                        batch_size = repel_images.shape[0]
                    else:
                        if repel_images.shape[0] % batch_size != 0:
                            batch_size = repel_images.shape[0] // ceil(repel_images.shape[0] / batch_size)
                            warnings.warn(f'Batch size changed to {batch_size} to fit repel train size')

                    repel_gradients = None
                    for i in range(repel_images.shape[0]//batch_size):
                        correct_loss = criterion(model(repel_images[i*batch_size:(i+1)*batch_size]), repel_labels[i*batch_size:(i+1)*batch_size])
                        incorrect_loss = criterion(model(repel_images[i*batch_size:(i+1)*batch_size]), flipped_labels[i*batch_size:(i+1)*batch_size])
                        loss = batch_size * (correct_loss - incorrect_loss)
                        if i == 0:
                            repel_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        else:
                            repel_gradients = tuple(map(lambda i, j: i + j, repel_gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))
                    repel_gradients = tuple(map(lambda i: i / (repel_images.shape[0] - (repel_images.shape[0] % batch_size)), repel_gradients))
                else:
                    correct_loss = criterion(model(repel_images), repel_labels)
                    incorrect_loss = criterion(model(repel_images), flipped_labels)
                    loss = correct_loss - incorrect_loss
                    repel_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

                repel_grad_norm = sum(grad.detach().pow(2).sum() for grad in repel_gradients).sqrt()
                print("Repel gradient norm: ", repel_grad_norm)

                # Combine source and repel gradients
                gradients = tuple(map(lambda s, r: (s + r), source_gradients, repel_gradients))
                
                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norm = grad_norm.sqrt()

                grad_list.append(gradients)
                norm_list.append(grad_norm)

        return grad_list, norm_list

    def load_trained_model(self, kettle):
        for idx, model in enumerate(self.models):
            load_path = os.path.join(self.args.model_savepath, "clean", f"{self.args.net[idx].upper()}_{self.args.dataset.upper()}_{self.args.optimization}_{self.model_init_seed}_{self.args.train_max_epoch}.pth")
            if os.path.exists(load_path):
                write(f'Model {self.args.net[idx]} already exists, skipping training.', self.args.output)
                if isinstance(self.models[idx], torch.nn.DataParallel):
                    self.models[idx].module.load_state_dict(torch.load(load_path))
                else:
                    self.models[idx].load_state_dict(torch.load(load_path))
                write(f'Model {self.args.net[idx]} validation:', self.args.output)
                
                self.models[idx].to(**self.setup)
                if torch.cuda.device_count() > 1:
                    self.models[idx] = torch.nn.DataParallel(self.models[idx])
                    self.models[idx].frozen = self.models[idx].module.frozen
                
                self._one_step_validation(self.models[idx], kettle)
                
                # Return to CPU
                if torch.cuda.device_count() > 1:
                    self.models[idx] = self.models[idx].module
                self.models[idx].to(device=torch.device('cpu'))
            
            else:
                write(f'Model {self.args.net[idx]} not found, training from scratch.', self.args.output)
                self._iterate(kettle, poison_delta=None, max_epoch=self.args.train_max_epoch)

                if self.args.save_clean_model:
                    if isinstance(self.models[idx], torch.nn.DataParallel):
                        self.save_model(self.models[idx].module, load_path)
                    else:
                        self.save_model(self.models[idx], load_path)

        return True
    
    def compute(self, function, *args):
        """Compute function on all models.

        Function has arguments that are possibly sequences of length args.ensemble
        """
        if self.args.sample_gradient:
            idx = random.randint(0, self.args.ensemble - 1)
            with GPUContext(self.setup, self.models[idx]) as model:
                single_arg = [arg[idx] if hasattr(arg, '__iter__') and len(arg) == self.args.ensemble else arg for arg in args]
                return function(model, self.optimizers[idx], *single_arg)
        else:
            outputs = []
            for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                with GPUContext(self.setup, model) as model:
                    single_arg = [arg[idx] if hasattr(arg, '__iter__') and len(arg) == self.args.ensemble else arg for arg in args]
                    outputs.append(function(model, optimizer, *single_arg))
            # collate
            avg_output = [np.mean([output[idx] for output in outputs]) for idx, _ in enumerate(outputs[0])]
            return avg_output
