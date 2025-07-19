"""Main class, holding information about models and training/testing routines."""

import torch
from ..utils import bypass_last_layer, cw_loss, write
from ..consts import BENCHMARK, NON_BLOCKING, FINETUNING_LR_DROP, NORMALIZE, PIN_MEMORY
from ..data.datasets import PoisonSet
torch.backends.cudnn.benchmark = BENCHMARK
import torch.nn as nn
import random
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
    
    print(f"Epoch: {epoch_num} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
class Witch_FM(_Witch):
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
                        T_restart = self.args.attackiter+1
                    else:
                        T_restart = self.args.retrain_iter+1
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(att_optimizer, T_0=T_restart, eta_min=self.args.tau * 0.001)
                else:
                    raise ValueError('Unknown poison scheduler.')
                
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        if self.args.sample_from_trajectory:
            print("Start training models for trajectory sampling.")
            write("Start training models for trajectory sampling.", self.args.output)
            
            all_state_dicts = []
            for i in range(self.args.num_trajectories):
                print(f"Training Model {i+1}/{self.args.num_trajectories}")
                write(f"Training Model {i+1}/{self.args.num_trajectories}", self.args.output)
                
                victim.initialize()
                single_setup = (victim.model, victim.defs, victim.optimizer, victim.scheduler)
                # Move state_dict to CPU before storing
                cpu_state_dict = {k: v.cpu() for k, v in victim.model.state_dict().items()}
                all_state_dicts.append(cpu_state_dict)
                for j in range(self.args.max_sample_epoch):
                    run_step(kettle, poison_delta, j, *single_setup)
                    if self.args.dryrun:
                        break
        else:
            all_state_dicts = None

        victim.eval()
        for step in range(self.args.attackiter):
            source_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                sources = []
                for temp_source, _, _ in kettle.source_trainset:
                    sources.append(temp_source)

                sources = torch.stack(sources)
                
                if NORMALIZE:
                    sources = normalization(sources)
                
                if self.args.sample_from_trajectory:
                    sample_idx = random.randint(0, len(all_state_dicts) - 1)
                    victim.model.load_state_dict(all_state_dicts[sample_idx])
                                                 
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
                if self.args.clean_grad:
                    victim.step(kettle, None)
                else:
                    victim.step(kettle, poison_delta)

            if self.args.dryrun:
                break
                
            if self.args.retrain_scenario != None:                
                if step % self.args.retrain_iter == 0 and step != 0 and step != self.args.attackiter - 1:
                    print("Retrainig the base model at iteration {} with {}".format(step, self.args.retrain_scenario))
                    poison_delta.detach()
                    
                    if self.args.sample_from_trajectory:
                        all_state_dicts = []
                        for i in range(self.args.num_trajectories):
                            write(f"Retraining Model {i+1}/{self.args.num_trajectories}", self.args.output)
                            print(f"Retraining Model {i+1}/{self.args.num_trajectories}")
                            
                            if self.args.retrain_scenario == 'from-scratch' or self.args.retrain_scenario == 'transfer':
                                victim.initialize()
                            elif self.args.retrain_scenario == 'finetuning':
                                if self.args.load_feature_repr:
                                    victim.load_feature_representation()
                                victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                                
                            single_setup = (victim.model, victim.defs, victim.optimizer, victim.scheduler)
                            cpu_state_dict = {k: v.cpu() for k, v in victim.model.state_dict().items()}
                            all_state_dicts.append(cpu_state_dict)
                            for j in range(self.args.max_sample_epoch):
                                run_step(kettle, poison_delta, j, *single_setup)
                                if self.args.dryrun:
                                    break
                    else:                        
                        if self.args.retrain_scenario == 'from-scratch' or self.args.retrain_scenario == 'transfer':
                            victim.initialize()
                            print('Model reinitialized to random seed.')
                        elif self.args.retrain_scenario == 'finetuning':
                            if self.args.load_feature_repr:
                                victim.load_feature_representation()
                            victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                            print('Completely warmstart finetuning!')
                        
                        victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                        write('Retraining done!\n', self.args.output)
                    
        return poison_delta, source_losses
    
    def batched_mean_features(self, model, data, batch_size=64):
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
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
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
                delta = self.attacker.attack(inputs.detach(), labels, None, None, steps=5)  # the 5-step here is FOR TESTING ONLY
                inputs = inputs + delta  # Kind of a reparametrization trick


            # Define the loss objective and compute gradients
            if self.args.source_criterion in ['cw', 'carlini-wagner']:
                loss_fn = cw_loss
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            if self.args.pmix:
                def criterion(outputs, labels):
                    loss, pred = kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
                    return loss
            else:
                criterion = loss_fn

            closure = self._define_objective(inputs, labels, criterion, sources, source_class=kettle.poison_setup['source_class'][0], target_class=kettle.poison_setup['target_class'])
            loss, prediction = victim.compute(closure, None, None, None, delta_slice)

            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

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

    def _define_objective(self, inputs, labels, criterion, sources, source_class, target_class):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm, perturbations):
            feature_model, last_layer = bypass_last_layer(model)
            
            features_inputs = feature_model(inputs)
            mean_features_inputs = torch.mean(features_inputs, dim=0)

            # Batch the source features!
            mean_features_sources = self.batched_mean_features(feature_model, sources, batch_size=128)
            
            passenger_loss = torch.linalg.norm(mean_features_inputs - mean_features_sources, ord=2)
            passenger_loss.backward()

            outputs = last_layer(features_inputs)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()

            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure

