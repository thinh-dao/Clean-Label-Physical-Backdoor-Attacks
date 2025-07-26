"""Single model default victim class."""
import torch
import numpy as np
import warnings
import copy
import os
import tqdm

from math import ceil
from .models import get_model
from .training import get_optimizers, run_step
from ..hyperparameters import training_strategy
from ..utils import set_random_seed, write, OpenCVNonLocalMeansDenoiser
from ..consts import BENCHMARK, SHARING_STRATEGY, NORMALIZE
from ..data.datasets import normalization

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

from .victim_base import _VictimBase

class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation. No init so the parent class init is automatically called
    
    Methods to initialize a model."""

    def initialize(self, seed=None):
        """Set seed and initialize model, optimizer, scheduler"""
        if seed is None:
            if self.args.model_seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = self.args.model_seed
        else:
            self.model_init_seed = seed
            
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], mode=self.args.scenario)
        self.epoch = 1
        
        if self.args.scenario == 'transfer':
            self.freeze_feature_extractor()
            self.eval()
            print('Features frozen.')
            
        self.model.to(**self.setup)
        if self.setup['device'] != 'cpu' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.frozen = self.model.module.frozen

        write(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.', self.args.output)
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')
        write(repr(self.defs), self.args.output)
        print(repr(self.defs))

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        """Reinitialize the last layer of the model and/or update training parameters."""
        if not keep_last_layer:
            if self.args.model_seed is None:
                if seed is None:
                    self.model_init_seed = np.random.randint(0, 2**32 - 1)
                else:
                    self.model_init_seed = seed
            else:
                self.model_init_seed = self.args.model_seed
            set_random_seed(self.model_init_seed)

            # We construct a full replacement model, so that the seed matches up with the initial seed,
            # even if all of the model except for the last layer will be immediately discarded.
            replacement_model = get_model(self.args.net[0], self.args.dataset, pretrained=self.args.pretrained_model)

            # Rebuild model with new last layer
            frozen = self.model.frozen
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1], torch.nn.Flatten(), list(replacement_model.children())[-1])
            self.model.frozen = frozen
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

        # Define training routine
        # Reinitialize optimizers here
        self.defs = training_strategy(self.args.net[0], self.args)
        self.defs.lr *= reduce_lr_factor
        self.optimizer, self.scheduler = get_optimizers(self.model, self.args, self.defs)
        print(f'{self.args.net[0]} last layer re-initialized with random key {self.model_init_seed}.')
        print(repr(self.defs))

    def save_feature_representation(self):
        self.orginal_model = copy.deepcopy(self.model)

    def load_feature_representation(self):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(self.orginal_model.module.state_dict())
        else:
            self.model.load_state_dict(self.orginal_model.state_dict())

    def freeze_feature_extractor(self):
        """Freezes all parameters except the classifier head."""
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            actual_model = self.model.module
        else:
            actual_model = self.model
        
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

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""
    def _iterate(self, kettle, poison_delta, max_epoch=None, stats=None):
        """Validate a given poison by training the model and checking source accuracy."""
        if max_epoch is None:
            max_epoch = self.defs.epochs
            
        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        self.defs.epochs = 1 if self.args.dryrun else max_epoch

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
        
        current_epoch = self.epoch
        for self.epoch in range(current_epoch, current_epoch+max_epoch):
            print(f"Training Epoch {self.epoch}...")
            run_step(kettle, poison_delta, self.epoch, *single_setup, stats=stats)
            if self.args.dryrun:
                break

    def step(self, kettle, poison_delta):
        """Step through a model epoch. Optionally: minimize poison loss."""
        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        run_step(kettle, poison_delta, self.epoch, *single_setup)
        self.epoch += 1
        if self.epoch == self.defs.epochs:
            self.epoch = 1
            write('Model reset to epoch 1.', self.args.output)
            self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], mode=self.args.scenario)
            self.model.to(**self.setup)
            if self.setup['device'] == 'cpu' and torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

    """ Various Utilities."""
    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], mode=self.args.scenario)

    def gradient_with_repel(self, source_images, source_labels, repel_images, repel_labels, criterion=None, selection=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if criterion is None:
            criterion = self.loss_fn
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Select sources with maximum gradient
        if selection == 'max_gradient':
            grad_norms = []
            for image, label in zip(source_images, source_labels):
                loss = criterion(self.model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt()) # Append l2_norm of gradient
            
            source_poison_selected = ceil(self.args.sources_selection_rate * source_images.shape[0])
            indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-source_poison_selected:]
            source_images = source_images[indices]
            source_labels = source_labels[indices]
            write('{} sources with maximum gradients selected'.format(source_poison_selected), self.args.output)

        # Using batch processing for batch gradients for source images
        if self.args.source_gradient_batch != None:
            batch_size = self.args.source_gradient_batch
            if source_images.shape[0] < batch_size:
                batch_size = source_images.shape[0]
            else:
                if source_images.shape[0] % batch_size != 0:
                    batch_size = source_images.shape[0] // ceil(source_images.shape[0] / batch_size)
                    warnings.warn(f'Batch size changed to {batch_size} to fit source train size')

            source_gradients = None
            for i in range(source_images.shape[0]//batch_size):
                loss = self.args.scale * batch_size * criterion(self.model(source_images[i*batch_size:(i+1)*batch_size]), source_labels[i*batch_size:(i+1)*batch_size])
                if i == 0:
                    source_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                else:
                    source_gradients = tuple(map(lambda i, j: i + j, source_gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))
            source_gradients = tuple(map(lambda i: i / (source_images.shape[0] - (source_images.shape[0] % batch_size)), source_gradients))

        else:
            loss = criterion(self.model(source_images), source_labels)
            source_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

        source_grad_norm = sum(grad.detach().pow(2).sum() for grad in source_gradients).sqrt()
        print("Source gradient norm: ", source_grad_norm)

        permutation = torch.randperm(repel_images.size(0))
        repel_images = repel_images[permutation]
        repel_labels = repel_labels[permutation]
        flipped_labels = torch.tensor([source_labels[0]] * len(repel_images)).cuda()

        # Using batch processing for batch gradients for repel images
        if self.args.source_gradient_batch != None:
            batch_size = self.args.source_gradient_batch
            if repel_images.shape[0] < batch_size:
                batch_size = repel_images.shape[0]
            else:
                if repel_images.shape[0] % batch_size != 0:
                    batch_size = repel_images.shape[0] // ceil(repel_images.shape[0] / batch_size)
                    warnings.warn(f'Batch size changed to {batch_size} to fit repel train size')

            repel_gradients = None
            for i in range(repel_images.shape[0]//batch_size):
                correct_loss = criterion(self.model(repel_images[i*batch_size:(i+1)*batch_size]), repel_labels[i*batch_size:(i+1)*batch_size])
                incorrect_loss = criterion(self.model(repel_images[i*batch_size:(i+1)*batch_size]), flipped_labels[i*batch_size:(i+1)*batch_size])
                loss = batch_size * (correct_loss - incorrect_loss)
                if i == 0:
                    repel_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                else:
                    repel_gradients = tuple(map(lambda i, j: i + j, repel_gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))
            repel_gradients = tuple(map(lambda i: i / (repel_images.shape[0] - (repel_images.shape[0] % batch_size)), repel_gradients))

        else:
            correct_loss = criterion(self.model(repel_images), repel_labels)
            incorrect_loss = criterion(self.model(repel_images), flipped_labels)
            loss = correct_loss - incorrect_loss
            repel_gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

        repel_grad_norm = sum(grad.detach().pow(2).sum() for grad in repel_gradients).sqrt()
        print("Repel gradient norm: ", repel_grad_norm)

        gradients = tuple(map(lambda s, r: (s + r), source_gradients, repel_gradients))
    
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
    
        return gradients, grad_norm

    def gradient(self, images, labels, criterion=None, selection=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""

        if criterion is None:
            criterion = self.loss_fn
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Select sources with maximum gradient
        if selection == 'max_gradient':
            grad_norms = []
            for image, label in zip(images, labels):
                loss = criterion(self.model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt()) # Append l2_norm of gradient
            
            source_poison_selected = ceil(self.args.sources_selection_rate * images.shape[0])
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
                loss = batch_size * criterion(self.model(images[i*batch_size:(i+1)*batch_size]), labels[i*batch_size:(i+1)*batch_size])
                if i == 0:
                    gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                else:
                    gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))

            gradients = tuple(map(lambda i: i / (images.shape[0] - (images.shape[0] % batch_size)), gradients))
        else:
            loss = criterion(self.model(images), labels)
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
    
        return gradients, grad_norm

    def load_trained_model(self, kettle):
        load_path = os.path.join(self.args.model_savepath, "clean", f"{self.args.net[0].upper()}_{self.args.dataset.upper()}_{self.args.optimization}_{self.model_init_seed}_{self.args.train_max_epoch}.pth")
        if os.path.exists(load_path):
            write(f'Model {self.args.net[0]} already exists, skipping training.', self.args.output)
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(torch.load(load_path))
            else:
                self.model.load_state_dict(torch.load(load_path))
            self._one_step_validation(self.model, kettle)
        else:
            write(f'Model {self.args.net[0]} not found, training from scratch.', self.args.output)
            self._iterate(kettle, poison_delta=None, max_epoch=self.args.train_max_epoch)

            if self.args.save_clean_model:
                if isinstance(self.model, torch.nn.DataParallel):
                    self.save_model(self.model.module, load_path)
                else:
                    self.save_model(self.model, load_path)

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.optimizer, *args)
