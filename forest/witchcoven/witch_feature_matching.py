"""Main class, holding information about models and training/testing routines."""

import torch
from ..utils import bypass_last_layer, bypass_last_layer_deit, cw_loss, write
from ..consts import BENCHMARK, NON_BLOCKING, FINETUNING_LR_DROP, NORMALIZE, PIN_MEMORY
from ..data.datasets import PoisonSet
torch.backends.cudnn.benchmark = BENCHMARK
import torch.nn as nn
import random
import copy
import numpy as np
from .witch_base import _Witch
from forest.data.datasets import normalization

def run_step(kettle, poison_delta, epoch_num, model, defs, optimizer, scheduler):
    """
    Run a single training step (epoch) with optional poisoning and defenses.
    """
    # --- Setup phase ---
    epoch_loss, total_preds, correct_preds = 0, 0, 0
    loss_fn=nn.CrossEntropyLoss(reduction='mean')
    
    # Set model to training mode
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        frozen = model.module.frozen
        list(model.module.children())[-1].train() if frozen else model.train()
    else:
        frozen = model.frozen
        list(model.children())[-1].train() if frozen else model.train()
    
    # Select appropriate data loader
    if poison_delta is not None:
        if kettle.args.ablation < 1.0:
            assert defs.defense is None, "Ablation cannot run with defenses!"
            dataset = kettle.partialset
        else:
            dataset = kettle.trainset

        poison_dataset = PoisonSet(dataset=dataset, poison_delta=poison_delta, poison_lookup=kettle.poison_lookup, normalize=NORMALIZE)
        train_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=min(kettle.batch_size, len(poison_dataset)),
                                                        shuffle=True, drop_last=False, num_workers=kettle.num_workers, pin_memory=PIN_MEMORY)
    else:
        train_loader = kettle.trainloader
    
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler()
    
    def criterion(outputs, labels):
        loss = loss_fn(outputs, labels)
        predictions = torch.argmax(outputs.data, dim=1)
        batch_correct = (predictions == labels).sum().item()
        return loss, batch_correct

    # --- Training loop ---
    for batch in train_loader:
        if len(batch) == 3:
            inputs, labels, ids = batch
        else:
            inputs, labels = batch

        optimizer.zero_grad()

        # Process inputs
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Transfer to GPU
            inputs = inputs.to(**kettle.setup)
            labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            
            # Add data augmentation if configured
            if defs.augmentations:
                inputs = kettle.augment(inputs)

            # Normalize inputs if required
            if NORMALIZE:
                inputs = normalization(inputs)
            
            # Forward pass
            outputs = model(inputs)
            loss, batch_correct = criterion(outputs, labels)
        
        # Update metrics
        correct_preds += batch_correct
        epoch_loss += loss.item()
        total_preds += labels.shape[0]
        
        # Backward pass - critical for the scaler to work properly
        scaled_loss = scaler.scale(loss)  # Explicitly scale the loss
        scaled_loss.backward()            # Do backward pass on scaled loss

        # Update parameters
        scaler.step(optimizer)  # This will now have inf checks recorded
        scaler.update()
              
        # Update cyclic scheduler if used
        if defs.scheduler == 'cyclic':
            scheduler.step()
        
        if kettle.args.dryrun:
            break
    
    # Update linear scheduler if used
    if defs.scheduler == 'linear':
        scheduler.step()
    
    # Report progress
    current_lr = optimizer.param_groups[0]['lr']
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct_preds / total_preds
    
    print(f"Epoch: {epoch_num} | LR: {current_lr:.4f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    write(f"Epoch: {epoch_num} | LR: {current_lr:.4f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", kettle.args.output)
    
def load_state_dict_compatible(model, state_dict):
    """
    Load state dict with automatic DataParallel compatibility.
    Uses PyTorch's built-in utilities for cleaner handling.
    """
    try:
        # Try loading directly first
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            # Handle DataParallel mismatch
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                # Model is wrapped, but state_dict might not have module prefix
                try:
                    model.module.load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    # If that fails, the state_dict has module prefix, so load normally
                    model.load_state_dict(state_dict, strict=True)
            else:
                # Model is not wrapped, but state_dict might have module prefix
                # Remove module prefix from state_dict
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=True)
        else:
            # Re-raise if it's a different error
            raise e
            
class Witch_FM(_Witch):
    def _validation(self, model, clean_testloader, source_testloader, target_class):
        """Evaluate model performance on clean and poisoned data."""
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')
        clean_corr = 0
        clean_loss = 0
        poisoned_corr = 0
        poisoned_loss = 0
        
        # Get device from model or use the setup device
        device = getattr(model, 'device', self.setup['device'])
        
        with torch.no_grad():
            # Evaluate on clean data
            for inputs, targets, idx in clean_testloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                clean_loss += loss.item()
                _, predicted = outputs.max(1)
                clean_corr += predicted.eq(targets).sum().item()

            # Evaluate on poisoned data
            for inputs, _, _ in source_testloader:
                inputs = inputs.to(device)
                targets = torch.ones(len(inputs), dtype=torch.long, device=device) * target_class
                        
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                poisoned_loss += loss.item()
                _, predicted = outputs.max(1)
                poisoned_corr += predicted.eq(targets).sum().item()

        # Calculate metrics
        clean_acc = clean_corr / len(clean_testloader.dataset)
        poisoned_acc = poisoned_corr / len(source_testloader.dataset)
        clean_loss = clean_loss / len(clean_testloader.dataset)
        poisoned_loss = poisoned_loss / len(source_testloader.dataset)

        return clean_acc, poisoned_acc, clean_loss, poisoned_loss
    
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

        if self.args.warm_start:                
            if not self.args.skip_clean_training:
                victim.initialize()
                print("Training model from scratch for trajectory sampling.")
                write("Training model from scratch for trajectory sampling.", self.args.output)
                
            if self.args.ensemble > 1:
                self.all_state_dicts = [[] for _ in range(self.args.ensemble)]
                
                for idx, model in enumerate(victim.models):
                    self.all_state_dicts[idx].append({k: v.cpu() for k, v in model.state_dict().items()})
                
                multi_model_setup = (victim.models, victim.definitions, victim.optimizers, victim.schedulers)
                for idx, single_model in enumerate(zip(*multi_model_setup)):
                    write(f"Training model {idx+1}/{self.args.ensemble}...", self.args.output)
                    model, _, _, _ = single_model

                    # Move to GPUs
                    model.to(**self.setup)
                    if torch.cuda.device_count() > 1:
                        model = torch.nn.DataParallel(model)
                        model.frozen = model.module.frozen
                    
                    current_epoch = victim.epochs[idx] + 1
                    for victim.epochs[idx] in range(current_epoch, current_epoch + self.args.warm_start_epochs):
                        run_step(kettle, poison_delta, victim.epochs[idx], *single_model)
                        
                        if self.args.sample_from_trajectory and victim.epochs[idx] % self.args.sample_every == 0:
                            self.all_state_dicts[idx].append({k: v.cpu() for k, v in model.state_dict().items()})
                            write(f"Store state dict for model {idx} at epoch {victim.epochs[idx]}", self.args.output)
                            
                            c_acc, p_acc, c_loss, p_loss = self._validation(
                                model=model, 
                                clean_testloader=kettle.validloader, 
                                source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                                target_class = kettle.poison_setup["target_class"]
                            )
                            
                            v_log = (f'Model {idx} - Epoch {victim.epochs[idx]} | '
                                    f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                            print(v_log); write(v_log, self.args.output)
                            
                        if self.args.dryrun:
                            break
                        
                    # Return to CPU
                    if torch.cuda.device_count() > 1:
                        model = model.module
                    model.to(device=torch.device('cpu'))

            else:
                self.all_state_dicts = [copy.deepcopy({k: v.cpu() for k, v in victim.model.state_dict().items()})]
                
                single_model_setup = victim.model, victim.defs, victim.optimizer, victim.scheduler
                current_epoch = victim.epoch + 1
                for victim.epoch in range(current_epoch, current_epoch + self.args.warm_start_epochs):
                    run_step(kettle, poison_delta, victim.epoch, *single_model_setup)
                    
                    if self.args.sample_from_trajectory and victim.epoch % self.args.sample_every == 0:
                        self.all_state_dicts.append({k: v.cpu() for k, v in victim.model.state_dict().items()})
                        
                        c_acc, p_acc, c_loss, p_loss = self._validation(
                            model=victim.model, 
                            clean_testloader=kettle.validloader, 
                            source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                            target_class = kettle.poison_setup["target_class"]
                        )
                        
                        v_log = (f'Epoch {victim.epochs[idx]} | '
                                f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                        print(v_log); write(v_log, self.args.output)
                                
                    if self.args.dryrun:
                        break
        else:
            self.all_state_dicts = None

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
                        if self.args.sample_same_idx:
                            # For ensemble models, sample the same index from the trajectory
                            sample_idx = random.randint(0, len(self.all_state_dicts[0]) - 1)
                            for idx in range(self.args.ensemble):
                                load_state_dict_compatible(victim.models[idx], self.all_state_dicts[idx][sample_idx])
                        else:
                            for idx in range(self.args.ensemble):
                                sample_idx = random.randint(0, len(self.all_state_dicts[0]) - 1)
                                load_state_dict_compatible(victim.models[idx], self.all_state_dicts[idx][sample_idx])
                    else:
                        sample_idx = random.randint(0, len(self.all_state_dicts) - 1)
                        load_state_dict_compatible(victim.model, self.all_state_dicts[sample_idx])
                                                 
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle, sources)
                source_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
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
                
            # Default not to step 
            if self.args.step:
                victim.step(kettle, poison_delta)

            if self.args.dryrun:
                break

            if self.args.retrain_scenario != None:             
                if step % self.args.retrain_iter == 0 and step != 0 and step != self.args.attackiter - 1:
                    print("Retraining attacker model at iteration {} with {}".format(step, self.args.retrain_scenario))
                    poison_delta.detach()
                    
                    if self.args.retrain_reinit_seed:
                        seed = np.random.randint(0, 2**32 - 1)
                    else:
                        seed = None
                        
                    if self.args.reinit_trajectory:
                        self.all_state_dicts = [] if self.args.ensemble == 1 else [[] for i in range(self.args.ensemble)]
                        
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize(seed=seed)
                        print('Model reinitialized to random seed.')
                    elif self.args.retrain_scenario == 'finetuning':
                        # Load victim models to the latest checkpoint
                        if self.args.ensemble == 1:
                            load_state_dict_compatible(victim.model, self.all_state_dicts[-1])
                        else:
                            for idx in range(self.args.ensemble):
                                load_state_dict_compatible(victim.models[idx], self.all_state_dicts[idx][-1])
                        
                        victim.reinitialize_last_layer(seed=seed, reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                        print('Completely warmstart finetuning!')

                    if self.args.scenario == 'transfer':
                        victim.load_feature_representation()
                    
                    if self.args.ensemble > 1:                        
                        multi_model_setup = (victim.models, victim.definitions, victim.optimizers, victim.schedulers)
                        for idx, single_model in enumerate(zip(*multi_model_setup)):
                            write(f"Training model {idx+1}/{self.args.ensemble}...", self.args.output)
                            model, _, _, _ = single_model

                            # Move to GPUs
                            model.to(**self.setup)
                            if torch.cuda.device_count() > 1:
                                model = torch.nn.DataParallel(model)
                                model.frozen = model.module.frozen
                            
                            current_epoch = victim.epochs[idx] + 1
                            for victim.epochs[idx] in range(current_epoch, current_epoch + self.args.retrain_max_epoch):
                                run_step(kettle, poison_delta, victim.epochs[idx], *single_model)
                                
                                if self.args.sample_from_trajectory and victim.epochs[idx] % self.args.sample_every == 0:
                                    self.all_state_dicts[idx].append({k: v.cpu() for k, v in model.state_dict().items()})
                                    write(f"Store state dict for model {idx} at epoch {victim.epochs[idx]}", self.args.output)

                                    c_acc, p_acc, c_loss, p_loss = self._validation(
                                        model=model, 
                                        clean_testloader=kettle.validloader, 
                                        source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                                        target_class = kettle.poison_setup["target_class"]
                                    )
                                    
                                    v_log = (f'Model {idx+1} - Epoch {victim.epochs[idx]} | '
                                            f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                                    print(v_log); write(v_log, self.args.output)
                                    
                            # Return to CPU
                            if torch.cuda.device_count() > 1:
                                model = model.module
                            model.to(device=torch.device('cpu'))

                    else:                        
                        single_model_setup = victim.model, victim.defs, victim.optimizer, victim.scheduler
                        current_epoch = victim.epoch + 1
                        for victim.epoch in range(current_epoch, current_epoch + self.args.retrain_max_epoch):
                            run_step(kettle, poison_delta, victim.epoch, *single_model_setup)
                            
                            if self.args.sample_from_trajectory and victim.epoch % self.args.sample_every == 0:  
                                self.all_state_dicts.append({k: v.cpu() for k, v in victim.model.state_dict().items()}) 
                                c_acc, p_acc, c_loss, p_loss = self._validation(
                                    model=victim.model, 
                                    clean_testloader=kettle.validloader, 
                                    source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                                    target_class = kettle.poison_setup["target_class"]
                                )
                                
                                v_log = (f'Epoch {victim.epochs[idx]} | '
                                        f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                                print(v_log); write(v_log, self.args.output)
                                
                
                    write('Retraining done!\n', self.args.output)
                    
        return poison_delta, source_losses
    
    def batched_mean_features(self, model, data, batch_size=64):
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

            closure = self._define_objective()
            loss, prediction = victim.compute(closure, inputs, labels, sources, delta_slice)

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
            mean_features_sources = self.batched_mean_features(feature_model, sources, batch_size=128)
            
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
