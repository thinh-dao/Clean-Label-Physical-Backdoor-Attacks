"""Base victim class."""

import torch
import os
import numpy as np
from .models import get_model
from .training import get_optimizers, run_validation, check_sources, check_sources_all_to_all, check_suspicion
from ..hyperparameters import training_strategy
from ..utils import write
from .utils import analyze_and_print
from ..consts import FINETUNING_LR_DROP

class _VictimBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize: set seed and initialize model, optimizer, scheduler
     - train: clean (pre)-training of the chosen model, no poisoning involved
     - retrain: retrain and evaluate poison on the initialization it was brewed on
     - validate: train and evaluate poison on a new initializations of the model
     - _iterate: validate a given poison by training the model and checking source accuracy.

     - compute: compute function on given objective function and arguments
     - gradient: compute the gradient of criterion(model) w.r.t to given data."
     - eval: switch everything into evaluation mode

     Internal methods that should ideally be reused by other backends:
     - _initialize_model: initialize model, optimizer, scheduler
     - _step: Train for one epoch

    """

    def __init__(self, args, num_classes=10, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup, self.num_classes  = args, setup, num_classes
        if self.args.ensemble < len(self.args.net):
            raise ValueError(f'More models requested than ensemble size.'
                             f'Increase ensemble size or reduce models.')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.initialize()

    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()
        return grad, grad_norm

    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, ...
        """
        raise NotImplementedError()

    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input
    
    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()

    """ Methods to initialize and modify a model."""

    def initialize(self, seed=None):
        raise NotImplementedError()

    def reinitialize_last_layer(self, seed=None):
        raise NotImplementedError()

    def freeze_feature_extractor(self, model):
        raise NotImplementedError()
    
    def eval(self, dropout=False):
        raise NotImplementedError()
                
    def save_feature_representation(self):
        raise NotImplementedError()

    def load_feature_representation(self):
        raise NotImplementedError()

    def save_model(self, model, path):
        """Save model to path."""
        write(f"Saving model to: {path}", self.args.output)
        print(f"Saving model to: {path}") 
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def train(self, kettle, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
            
        write('Starting clean training with {} scenario ...'.format(self.args.scenario), self.args.output)
        print('Starting clean training with {} scenario ...'.format(self.args.scenario))    
            
        if self.args.dryrun or self.args.train_from_scratch:
            self._iterate(kettle, poison_delta=None, max_epoch=max_epoch) # Validate poison
            if self.args.dryrun == False and self.args.save_clean_model:
                if self.args.ensemble == 1:
                    save_path = os.path.join(self.args.model_savepath, "clean", f"{self.args.net[0].upper()}_clean.pth")
                    self.save_model(self.model, save_path)
                else:
                    for idx, model in enumerate(self.models):
                        save_path = os.path.join(self.args.model_savepath, "clean", f"{self.args.net[idx].upper()}_clean.pth")
                        self.save_model(model, save_path)
        else: 
            self.load_trained_model(kettle)
        
        if self.args.load_feature_repr:
            self.save_feature_representation()
            write('Feature representation saved.', self.args.output)
    
    def load_trained_model(self, kettle):
        raise NotImplementedError()

    def load_backdoored_model(self, inspection_path, kettle):
        model_path = os.path.join(inspection_path, f"{self.args.net[0].upper()}_backdoored.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Backdoored model is not found!")
        
        state_dict = torch.load(model_path, map_location=self.setup['device'])
        
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.setup['device'])
        self._one_step_validation(self.model, kettle) 
            
    def retrain(self, kettle, poison_delta, max_epoch=None):
        """Retrain with the same initialization on dataset with poison added."""
        if self.args.scenario == 'from-scratch':
            self.initialize(seed=self.model_init_seed)
            print('Model re-initialized to initial seed.')
        elif self.args.scenario == 'transfer':
            self.load_feature_representation()
            self.reinitialize_last_layer(reduce_lr_factor=1.0, seed=self.model_init_seed)
            print('Linear layer reinitialized to initial seed.')
        elif self.args.scenario == 'finetuning':
            self.load_feature_representation()
            self.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, seed=self.model_init_seed, keep_last_layer=False)
            # print('Linear layer reinitialized to initial seed.')
            print('Completely warmstart finetuning!')
        self._iterate(kettle, poison_delta=poison_delta, max_epoch=max_epoch)

    def validate(self, kettle, poison_delta, val_max_epoch=None, vruns=None):
        """Check poison on a new initialization(s), depending on the scenario."""
        stats = dict(valid_acc=[], clean_acc=dict(), backdoor_acc=dict(), fpr=[], sr=[])
        for source in kettle.source_class:
            stats['clean_acc'][source] = []
            stats['backdoor_acc'][source] = []
        
        stats['clean_acc']['avg'] = []
        stats['backdoor_acc']['avg'] = []
            
        self.args.ensemble = 1 # Disable ensemble for validation
        original_ablation = kettle.args.ablation
        kettle.args.ablation = 1.0

        if vruns is None:
            vruns = self.args.vruns
            
        for run in range(vruns):
            # Reinitalize model with new seed
            seed = np.random.randint(0, 2**32 - 1)
            self.initialize(seed)
            
            if hasattr(self, 'clean_models'):
                self.clean_models = [self.clean_models[0]]

            if self.args.scenario == "transfer" and self.args.load_feature_repr:
                self.load_feature_representation()
                
            # Train new model
            write("Validation {} on {} with seed {}...".format(run+1, self.args.net[0], seed), self.args.output)
            self._iterate(kettle, poison_delta=poison_delta, max_epoch=val_max_epoch, stats=stats)
        kettle.args.ablation = original_ablation
        analyze_and_print(stats, output=kettle.args.output)

    """ Various Utilities."""
    def _iterate(self, kettle, poison_delta):
        """Validate a given poison by training the model and checking source accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, kettle, poison_delta, step, poison_sources, true_classes):
        """Step through a model epoch to in turn minimize poison loss."""
        raise NotImplementedError()

    def _initialize_model(self, model_name, mode='finetuning'):
        if mode == 'from-scratch':
            pretrained = False
        else:
            pretrained = True
        
        model = get_model(model_name, num_classes=self.num_classes, pretrained=pretrained)
        model.frozen = False
        
        # Define training routine
        defs = training_strategy(model_name, self.args) # Initialize hyperparameters for training
        if mode == 'finetuning':
            defs.lr *= FINETUNING_LR_DROP
        
        optimizer, scheduler = get_optimizers(model, self.args, defs)
        
        return model, defs, optimizer, scheduler
        
    def _one_step_validation(self, model, kettle):
        # Do evaluation on the loaded model
        predictions = run_validation(model, self.loss_fn, kettle.validloader,
                                        kettle.poison_setup['target_class'],
                                        kettle.poison_setup['source_class'],
                                        kettle.setup)

        if kettle.args.threatmodel != 'all-to-all':
            source_adv_acc, source_adv_loss, source_clean_acc, source_clean_loss = check_sources(
                model, self.loss_fn, kettle.source_testloader, kettle.poison_setup['poison_class'],
                kettle.setup)
        else:
            source_adv_acc, source_adv_loss, source_clean_acc, source_clean_loss = check_sources_all_to_all(
                model, self.loss_fn, kettle.source_testloader, kettle.poison_setup['mapping'],
                kettle.setup)

        valid_loss, valid_acc, valid_acc_target, valid_acc_source = predictions['all']['loss'], predictions['all']['avg'], predictions['target']['avg'], predictions['source']['avg']
        
        write('------------- Validation -------------', self.args.output)
        write(f'Validation  loss : {valid_loss:7.4f} | Validation accuracy: {valid_acc:7.4%}', self.args.output)
        write(f'Target val. acc  : {valid_acc_target:7.4%} | Source val accuracy: {valid_acc_source:7.4%}', self.args.output)
        for source_class in source_adv_acc.keys():
            backdoor_acc, clean_acc, backdoor_loss, clean_loss = source_adv_acc[source_class], source_clean_acc[source_class], source_adv_loss[source_class], source_clean_loss[source_class]
            if source_class != 'avg':
                write(f'Source class: {source_class}', self.args.output)
            else:
                write(f'Average:', self.args.output)
            write('Backdoor loss: {:7.4f} | Backdoor acc: {:7.4%}'.format(backdoor_loss, backdoor_acc), self.args.output)
            write('Clean    loss: {:7.4f} | Clean    acc: {:7.4%}'.format(clean_loss, clean_acc), self.args.output)
        write('--------------------------------------', self.args.output)
        if self.args.dataset != 'Animal_classification' and self.args.threatmodel != 'all-to-all' and self.args.recipe != "naive":
            suspicion_rate, false_positive_rate = check_suspicion(model, kettle.suspicionloader, kettle.fploader, kettle.poison_setup['target_class'], kettle.setup)
            write(f'False positive rate: {false_positive_rate:7.4%} | Suspicion rate: {suspicion_rate:7.4%}', self.args.output)
