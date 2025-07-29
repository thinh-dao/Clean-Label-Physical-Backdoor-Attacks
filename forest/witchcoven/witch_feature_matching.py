"""Main class, holding information about models and training/testing routines."""

import torch
import torch.nn as nn
import random
import numpy as np

from ..utils import bypass_last_layer, bypass_last_layer_deit, write
from ..consts import BENCHMARK, NON_BLOCKING, FINETUNING_LR_DROP, NORMALIZE
from .witch_base import _Witch
from forest.data.datasets import normalization
torch.backends.cudnn.benchmark = BENCHMARK
            
class Witch_FM(_Witch):
    def _initialize_buffers(self, victim, kettle):
        """Initialize buffers for sampling from trajectory """
        if self.args.ensemble > 1:
            self.buffers = [[] for _ in range(self.args.ensemble)]
        else:
            self.buffers = []
    
    def _train_and_fill_buffers(self, victim, kettle, poison_delta, max_epochs):
        """Train victim models and fill up buffers
        
        Args:
            victim: The victim model(s) to train
            kettle: The kettle object containing data loaders
            poison_delta: The poison perturbations
            max_epochs: Maximum number of epochs to train
        """
        if self.args.ensemble > 1:
            multi_model_setup = (victim.models, victim.definitions, victim.optimizers, victim.schedulers)
            for idx, single_model in enumerate(zip(*multi_model_setup)):
                write(f"Training model {idx+1}/{self.args.ensemble}...", self.args.output)
                model, _, _, _ = single_model
                
                # Handle DataParallel models
                state_dict = model.module.state_dict() if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
                self.buffers[idx].append({k: v.detach().clone().cpu() for k, v in state_dict.items()})
                
                # Move to GPUs
                model.to(**self.setup)
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
                    model.frozen = model.module.frozen
                
                current_epoch = victim.epochs[idx] + 1
                for victim.epochs[idx] in range(current_epoch, current_epoch + max_epochs):
                    self.run_step(kettle, poison_delta, victim.epochs[idx], *single_model)
                    
                    if victim.epochs[idx] % self.args.sample_every == 0:
                        # Handle DataParallel models
                        state_dict = model.module.state_dict() if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
                        self.buffers[idx].append({k: v.detach().clone().cpu() for k, v in state_dict.items()})
                        write(f"Store state dict for model {idx} at epoch {victim.epochs[idx]}", self.args.output)
                    
                    if victim.epochs[idx] % self.args.validate_every == 0:
                        c_acc, p_acc, c_loss, p_loss = self.validation(
                            model=model, 
                            clean_testloader=kettle.validloader, 
                            source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                            target_class = kettle.poison_setup["target_class"]
                        )
                        
                        v_log = (f'Model {idx+1} - Epoch {victim.epochs[idx]} | '
                                f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                        print(v_log); write(v_log, self.args.output)
                        
                    if self.args.dryrun:
                        break
                        
                # Return to CPU
                if torch.cuda.device_count() > 1:
                    model = model.module
                model.to(device=torch.device('cpu'))
        else:
            single_model_setup = victim.model, victim.defs, victim.optimizer, victim.scheduler
            
            # Handle DataParallel models
            state_dict = victim.model.module.state_dict() if isinstance(victim.model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else victim.model.state_dict()
            self.buffers.append({k: v.detach().clone().cpu() for k, v in state_dict.items()})
            
            current_epoch = victim.epoch + 1
            for victim.epoch in range(current_epoch, current_epoch + max_epochs):
                self.run_step(kettle, poison_delta, victim.epoch, *single_model_setup)
                
                if victim.epoch % self.args.sample_every == 0:  
                    # Handle DataParallel models
                    state_dict = victim.model.module.state_dict() if isinstance(victim.model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else victim.model.state_dict()
                    self.buffers.append({k: v.detach().clone().cpu() for k, v in state_dict.items()}) 
                    write(f"Store state dict for model at epoch {victim.epoch}", self.args.output)
                    
                if victim.epoch % self.args.validate_every == 0:
                    c_acc, p_acc, c_loss, p_loss = self.validation(
                        model=victim.model, 
                        clean_testloader=kettle.validloader, 
                        source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                        target_class = kettle.poison_setup["target_class"]
                    )
                                        
                    v_log = (f'Epoch {victim.epoch} | '
                            f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                    print(v_log); write(v_log, self.args.output)
                    
                if self.args.dryrun:
                    break
    
    def _batched_mean_features(self, model, data, batch_size=64):
        model.eval()
        features_sum = 0
        count = 0
        with torch.no_grad():
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i+batch_size].to(**self.setup)
                feats = model(batch)
                if isinstance(features_sum, int):  # first batch
                    features_sum = feats.sum(dim=0)
                else:
                    features_sum += feats.sum(dim=0)
                count += feats.size(0)
        return features_sum / count
    
    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()
        dataloader = kettle.poisonloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD', 'SGD']:
            poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            elif self.args.attackoptim in ['momSGD', 'momPGD']:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            elif self.args.attackoptim in ['SGD']:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=5e-4, nesterov=True)
                
            if self.args.scheduling:
                if self.args.poison_scheduler == 'linear':
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                        self.args.attackiter // 1.142], gamma=0.1)
                elif self.args.poison_scheduler == 'cosine':
                    if self.args.retrain_scenario == None:
                        T_restart = self.args.attackiter
                    else:
                        T_restart = self.args.retrain_iter
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(att_optimizer, T_0=T_restart, eta_min=self.args.tau * 0.001)
                else:
                    raise ValueError('Unknown poison scheduler.')
                
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        if self.args.sample_from_trajectory:             
            if not self.args.skip_clean_training:
                victim.initialize()
                print("Training model from scratch for trajectory sampling.")
                write("Training model from scratch for trajectory sampling.", self.args.output)
                
            # Initialize state dictionaries and get training trajectories
            self._initialize_buffers(victim, kettle)
            self._train_and_fill_buffers(victim, kettle, poison_delta, self.args.retrain_max_epoch)
        else:
            self.buffers = None

        victim.eval()
        sources = []
        for temp_source, _, _ in kettle.source_trainset:
            sources.append(temp_source)
        sources = torch.stack(sources)
                    
        for step in range(self.args.attackiter):
            source_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                if self.args.paugment:
                    sources = kettle.augment(sources)
                    
                if NORMALIZE:
                    sources = normalization(sources)
            
                if self.args.sample_from_trajectory:
                    if self.args.ensemble > 1:
                        # Whether to sample the same index from the trajectory
                        if self.args.sample_same_idx:
                            sample_idx = random.randint(0, len(self.buffers[0]) - 1)
                            
                        for idx in range(self.args.ensemble):
                            if not self.args.sample_same_idx:
                                sample_idx = random.randint(0, len(self.buffers[0]) - 1)
                                
                            if isinstance(victim.models[idx], (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)): 
                                victim.models[idx].module.load_state_dict(self.buffers[idx][sample_idx], strict=True)
                            else:
                                victim.models[idx].load_state_dict(self.buffers[idx][sample_idx], strict=True)

                    else:
                        sample_idx = random.randint(0, len(self.buffers) - 1)
                        if isinstance(victim.model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                            victim.model.module.load_state_dict(self.buffers[sample_idx], strict=True)
                        else:
                            victim.model.load_state_dict(self.buffers[sample_idx], strict=True)
                                                 
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle, sources)
                source_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            if self.args.attackoptim in ['momPGD', 'signAdam']:
                poison_delta.grad.sign_()
            
            att_optimizer.step()
            
            if self.args.scheduling:
                scheduler.step()
            att_optimizer.zero_grad(set_to_none=False)
            
            if self.args.attackoptim != "cw":
                with torch.no_grad():
                    if self.args.visreg is not None and "soft" in self.args.visreg:
                        poison_delta.data = torch.clamp(poison_delta.data, min=0.0, max=1.0)
                    else:
                        poison_delta.data = torch.clamp(poison_delta.data, min=-self.args.eps / ds / 255, max=self.args.eps / ds / 255)
            
                    # Then project to the poison bounds
                    poison_delta.data = torch.clamp(poison_delta.data, 
                                                    min=-dm / ds - poison_bounds, 
                                                    max=(1 - dm) / ds - poison_bounds)

            source_losses = source_losses / (batch + 1)
            with torch.no_grad():
                visual_losses = torch.mean(torch.linalg.matrix_norm(poison_delta))
            
            if step % 10 == 0 or step == (self.args.attackiter - 1):
                lr = att_optimizer.param_groups[0]['lr']
                write(f'Iteration {step} - lr: {lr} | Passenger loss: {source_losses:2.4f} | Visual loss: {visual_losses:2.4f}', self.args.output)
                
            # Step victim model if needed
            if self.args.step and step % self.args.step_every == 0:
                log = "Stepping victim model at iteration {}".format(step)
                print(log); write(log, self.args.output)
                if self.args.step_on_poison:
                    # Step victim model on poison delta
                    ori_trainloader = kettle.trainloader
                    kettle.trainloader = kettle.poisonloader
                    victim.step(kettle, poison_delta)
                    kettle.trainloader = ori_trainloader
                else:
                    victim.step(kettle, poison_delta)
                
                c_acc, p_acc, c_loss, p_loss = self.validation(model=victim.model,
                    clean_testloader=kettle.validloader, 
                    source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                    target_class = kettle.poison_setup["target_class"]
                )
                log = (f'Epoch {victim.epoch} | '
                f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                print(log); write(log, self.args.output)

            if self.args.dryrun:
                break

            if self.args.retrain_scenario != None:             
                if step % self.args.retrain_iter == 0 and step != 0 and step != self.args.attackiter - 1:
                    print("Retraining attacker model at iteration {} with {}".format(step, self.args.retrain_scenario))
                    
                    if self.args.retrain_reinit_seed:
                        seed = np.random.randint(0, 2**32 - 1)
                    else:
                        seed = None
                        
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize(seed=seed)
                        self._initialize_buffers(victim, kettle)
                        print('Model reinitialized to random seed.')
                    elif self.args.retrain_scenario == 'finetuning':
                        # Load victim models to the latest checkpoint
                        if self.args.ensemble == 1:
                            victim.model.load_state_dict(self.buffers[-1], strict=True)
                        else:
                            for idx in range(self.args.ensemble):
                                victim.models[idx].load_state_dict(self.buffers[idx][-1], strict=True)
                        
                        victim.reinitialize_last_layer(seed=seed, reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                        print('Completely warmstart finetuning!')

                    if self.args.scenario == 'transfer':
                        victim.load_feature_representation()
                    
                    # Train models with trajectory sampling
                    if self.args.sample_from_trajectory:
                        self._train_and_fill_buffers(victim, kettle, poison_delta, self.args.retrain_max_epoch)
                    else:
                        victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                    write('Retraining done!\n', self.args.output)
                    
        return poison_delta, source_losses

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle, sources):
        """Take a step toward minmizing the current source loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        sources = sources.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            
            if self.args.attackoptim == "cw":
                delta_slice = 0.5 * (torch.tanh(delta_slice) + 1)
                
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs)
            if NORMALIZE:
                inputs = normalization(inputs)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)

            if self.args.padversarial is not None:
                delta = self.attacker.attack(inputs.detach(), labels, None, None, steps=5)
                inputs = inputs + delta

            closure = self._define_objective(inputs, labels, sources, kettle.poison_setup['source_class'], delta_slice)
            loss, prediction = victim.compute(closure)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)
                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD', 'SGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        
            poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective(self, inputs, labels, sources, source_class, perturbations):
        """Implement the closure here."""
        def closure(model, optimizer):
            model.eval()
            if 'deit' in self.args.net[0]:
                feature_model, last_layer = bypass_last_layer_deit(model)
            else:
                feature_model, last_layer = bypass_last_layer(model)
            
            features_inputs = feature_model(inputs)
            mean_features_inputs = torch.mean(features_inputs, dim=0)

            # Batch the source features!
            mean_features_sources = self._batched_mean_features(feature_model, sources, batch_size=128)
            
            passenger_loss = torch.linalg.norm(mean_features_inputs - mean_features_sources, ord=2)
            regularized_loss = self.get_regularized_loss(perturbations, tau=self.args.eps/255)
            
            if self.args.dist_reg_weight is not None:
                perturbation_outputs = model(perturbations)
                targets = torch.ones(perturbation_outputs.shape[0], device=self.setup['device'], dtype=torch.long) * source_class
                dist_reg_loss = self.args.dist_reg_weight * torch.nn.functional.cross_entropy(perturbation_outputs, targets)
                write("Distribution Regularization Loss: {}".format(dist_reg_loss.item()), self.args.output)
                passenger_loss = passenger_loss + dist_reg_loss
            
            attacker_loss = passenger_loss + self.args.vis_weight * regularized_loss
            attacker_loss.backward(retain_graph=self.retain)

            outputs = last_layer(features_inputs)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()

            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure
