import torch
import numpy as np
import torch.nn as nn
import copy

from torch.cuda.amp  import GradScaler
from forest.victims.training import run_step, _split_data
from forest.utils import write, ReparamModule, cw_loss
from forest.consts import FINETUNING_LR_DROP, PIN_MEMORY, NORMALIZE, NON_BLOCKING
from forest.witchcoven import _Witch
from forest.data.datasets import Subset, ConcatDataset, LabelPoisonTransform
from torch.utils.data import DataLoader
from forest.data.datasets import normalization
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class WitchMTTP(_Witch):
    def _define_objective(self, inputs, labels, criterion, perturbations,):
        def closure(self, model, optimizer, starting_params, target_params):
            """
            Train a student network on poisoned data to match target network parameters,
            """
            theta_original = parameters_to_vector(model.parameters())
            
            # Train one step on (inputs, labels) with starting params
            vector_to_parameters(starting_params, model.parameters())
            model.train()
            optimizer.zero_grad(set_to_none=False)
            outputs = model(inputs)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_loss = criterion(outputs, labels)
            poison_loss.backward(retain_grad=True, create_graph=True)
            optimizer.step()

            # Compute distillation loss
            theta_0 = starting_params.to(self.setup['device'])
            theta_m = target_params.to(self.setup['device'])
            theta_poison = parameters_to_vector(model.parameters())
            
            param_loss = torch.nn.functional.mse_loss(theta_poison, theta_m, reduction="mean")
            param_dist = torch.nn.functional.mse_loss(theta_m, theta_0, reduction="mean")
            passenger_loss = param_loss / param_dist.detach()
            
            # Add regularization
            regularized_loss = self.get_regularized_loss(perturbations, tau=self.args.eps/255)
            attacker_loss = passenger_loss + self.args.vis_weight * regularized_loss
            
            if self.args.featreg != 0:
                if self.target_feature == None: raise ValueError('No target feature found')
                feature_loss = self.args.featreg * self.get_featloss(inputs, with_grad=True).pow(2)
                attacker_loss += feature_loss 
                
            if self.args.centreg != 0:
                attacker_loss = passenger_loss + self.args.centreg * poison_loss
            attacker_loss.backward(retain_graph=self.retain)
            
            # Restore original params
            vector_to_parameters(theta_original, model.parameters())
            
            return passenger_loss.item(), prediction.detach().cpu()
        return closure
        
    def _train_backdoor_net(self, backdoor_trainloader, backdoor_testloader,
                            bkd_indices, model, kettle, lr, epochs, num_experts=3):

        all_trajectories = []
        original_params = parameters_to_vector(model.parameters()).detach()

        for exp_idx in range(num_experts):
            print(f"Training expert {exp_idx+1}/{num_experts}")
            write(f"Training expert {exp_idx+1}/{num_experts}", self.args.output)

            net   = copy.deepcopy(model).to(self.setup['device'])
            opt   = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=5e-4, nesterov=True)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            scaler = torch.amp.GradScaler('cuda')

            traj = [parameters_to_vector(net.parameters()).cpu()]
            eval_every = max(1, epochs // 3)

            for epoch in range(1, epochs + 1):
                avg_loss, acc = self._train_one_backdoor_epoch(
                    model=net, optimizer=opt, backdoor_trainloader=backdoor_trainloader,
                    bkd_indices=bkd_indices, diff_augment=kettle.augment if self.args.augment else None,
                    scaler=scaler)
                sched.step()

                ep_params = parameters_to_vector(net.parameters()).cpu()
                traj.append(ep_params)
                dist = torch.norm(ep_params.to(original_params.device) - original_params).item()

                log = (f'Expert {exp_idx+1}, Epoch {epoch}/{epochs} | '
                    f'Loss: {avg_loss:.4f}, Acc: {acc:.4f}, ParamsDist: {dist:.6f}')
                print(log); write(log, self.args.output)

                if epoch % eval_every == 0 or epoch == epochs:
                    c_acc, p_acc, c_loss, p_loss = self._validation(
                        model=net, 
                        clean_testloader=kettle.validloader, 
                        poisoned_testloader=backdoor_testloader,
                        )
                    
                    v_log = (f'Epoch {epoch} | ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                    print(v_log); write(v_log, self.args.output)

            all_trajectories.append(traj)
        return all_trajectories
    
    def _train_one_backdoor_epoch(self, model, optimizer, backdoor_trainloader,
                                bkd_indices, diff_augment=None, scaler=None):

        ce_loss = nn.CrossEntropyLoss(reduction='none')
        src_crit = cw_loss if self.args.source_criterion == "cw" else None
        device   = next(model.parameters()).device
        bkd_t = torch.as_tensor(list(bkd_indices), device=device) if bkd_indices else None

        model.train()
        tot_loss = correct = seen = 0

        for x, y, idx in backdoor_trainloader:
            x, y, idx = (t.to(device, non_blocking=True) for t in (x, y, idx))
            if diff_augment: x = diff_augment(x)
            if NORMALIZE: x = normalization(x)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                out = model(x)
                losses = ce_loss(out, y)
                if src_crit and bkd_t is not None:
                    mask = torch.isin(idx, bkd_t)
                    if mask.any():
                        losses[mask] = src_crit(out[mask], y[mask], reduction='none').to(losses.dtype)
                loss = losses.mean()

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

            tot_loss += loss.item() * y.size(0)
            correct  += out.argmax(1).eq(y).sum().item()
            seen     += y.size(0)

        return tot_loss / seen, correct / seen
    def _initialize_trajectories(self, victim, kettle):
        if self.args.ensemble > 1:
            self.all_trajectories = []
            for idx, model_name in enumerate(self.args.net):
                backdoor_learning_lr = self.args.backdoor_training_lr
                if any(vit_model in model_name.lower() for vit_model in ['vit', 'deit', 'swin']):
                    backdoor_learning_lr *= 0.1

                # Initialize expert training trajectories (backdoor models)
                backdoor_trainloader, backdoor_testloader, bkd_indices = self._get_backdoor_data(kettle, backdoor_training_mode=self.args.backdoor_training_mode)
                expert_trajectories = self._train_backdoor_net(
                    backdoor_trainloader=backdoor_trainloader,
                    backdoor_testloader=backdoor_testloader,
                    bkd_indices=bkd_indices,
                    model=victim.models[idx],
                    kettle=kettle,
                    lr=backdoor_learning_lr,
                    epochs=self.args.backdoor_training_epoch,
                    num_experts=self.args.num_experts,
                )
                self.all_trajectories.append(expert_trajectories)
        else:
            backdoor_learning_lr = self.args.backdoor_training_lr
            if any(vit_model in self.args.net[0].lower() for vit_model in ['vit', 'deit', 'swin']):
                backdoor_learning_lr *= 0.1
                
            # Get backdoor data before using it
            backdoor_trainloader, backdoor_testloader, bkd_indices = self._get_backdoor_data(kettle, backdoor_training_mode=self.args.backdoor_training_mode)
            self.all_trajectories = self._train_backdoor_net(
                    backdoor_trainloader=backdoor_trainloader,
                    backdoor_testloader=backdoor_testloader,
                    bkd_indices=bkd_indices,
                    model=victim.model,
                    kettle=kettle,
                    lr=backdoor_learning_lr,
                    epochs=self.args.backdoor_training_epoch,
                    num_experts=self.args.num_experts,
                )
            
    def _run_trial(self, victim, kettle):
        """Run a single trial. Perform one round of poisoning."""
        # Initialize poison mask of shape [num_poisons, channels, height, width] with values in [-eps, eps]
        poison_delta = kettle.initialize_poison()
        poison_delta.requires_grad_(True)
        # poison_delta.grad = torch.zeros_like(poison_delta).to(**self.setup) 
        dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
        poison_bounds = torch.zeros_like(poison_delta)
        
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.poisonloader

        # Setup attack optimizer
        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD', 'SGD']:
            poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            elif self.args.attackoptim in ['momSGD', 'momPGD']:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0, nesterov=True)
            elif self.args.attackoptim in ['SGD']:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=5e-4, nesterov=True)
            
            # Setup learning rate scheduler
            if self.args.scheduling:
                if self.args.poison_scheduler == 'linear':
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        att_optimizer, 
                        milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6, self.args.attackiter // 1.142], 
                        gamma=0.1
                    )
                elif self.args.poison_scheduler == 'cosine':
                    # Fixed comparison with None using 'is' instead of '=='
                    if self.args.retrain_scenario is None:
                        T_restart = self.args.attackiter+1
                    else:
                        T_restart = self.args.retrain_iter+1

                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        att_optimizer, T_0=T_restart, eta_min=self.args.tau * 0.001
                    )
                else:
                    raise ValueError(f'Unknown poison scheduler: {self.args.poison_scheduler}')
        else:
            raise ValueError(f'Unknown attack optimizer: {self.args.attackoptim}')
        
        # Initialize trajectories
        self._initialize_trajectories(victim, kettle)
                        
        for step in range(self.args.attackiter):
            max_start_epoch = max(self.args.backdoor_training_epoch - 1 - self.args.expert_epochs, 0)
            if self.args.sample_same_idx:
                start_params_idx = np.random.randint(0, max_start_epoch+1)
                    
            if self.args.ensemble > 1:
                starting_params = []
                target_params = []
                    
                for idx in range(self.args.ensemble):
                    if not self.args.sample_same_idx:
                        start_params_idx = np.random.randint(0, max_start_epoch+1)
                    target_params_idx = min(start_params_idx + self.args.expert_epochs, self.args.backdoor_training_epoch - 1)
                    sample_traj_idx = np.random.randint(0, self.args.num_experts)  
                    
                    starting_params.append(self.all_trajectories[idx][sample_traj_idx][start_params_idx])
                    target_params.append(self.all_trajectories[idx][sample_traj_idx][target_params_idx])

            else:  
                target_params_idx = min(start_params_idx + self.args.expert_epochs, self.args.backdoor_training_epoch - 1)
                sample_traj_idx = np.random.randint(0, self.args.num_experts)
                
                starting_params = self.all_trajectories[sample_traj_idx][start_params_idx]
                target_params = self.all_trajectories[sample_traj_idx][target_params_idx]

            # Initialize source_losses before the loop
            source_losses = 0
            
            for batch, example in enumerate(dataloader):
                loss, _ = self._batched_step(poison_delta, poison_bounds, example, victim, kettle, starting_params, target_params)
                source_losses += loss
                
                if self.args.dryrun:
                    break

            # Update poison perturbations
            if self.args.attackoptim in ['momPGD', 'signAdam']:
                poison_delta.grad.sign_()
            
            att_optimizer.step()
            
            if self.args.scheduling:
                scheduler.step()
            att_optimizer.zero_grad(set_to_none=False)
            
            # Project perturbations to valid range
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
                victim.step(kettle, poison_delta)

            if self.args.dryrun:
                break
                
            # Handle retraining scenario
            if self.args.retrain_scenario is not None:
                if step % self.args.retrain_iter == 0 and step != 0 and step != self.args.attackiter - 1:
                    print(f"Retraining the base model at iteration {step}")
                    poison_delta.detach()
                    
                    if self.args.retrain_reinit_seed:
                        seed = np.random.randint(0, 2**32 - 1)
                    else:
                        seed = None
                        
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize(seed=seed)
                    elif self.args.retrain_scenario == 'finetuning':
                        victim.reinitialize_last_layer(seed=seed, reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                        print('Completely warmstart finetuning!')

                    if self.args.scenario == 'transfer':
                        victim.load_feature_representation()
                            
                    victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                    write(f'Retraining completed at step: {step}', self.args.output)
                    print(f'Retraining completed at step: {step}')

                    write('Retraining backdoor model...', self.args.output)
                    print('Retraining backdoor model...')

                    self._initialize_trajectories(victim, kettle)

        return poison_delta, source_losses

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle, starting_params, target_params):
        """Take a step toward minimizing the current poison loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            
        # If a poisoned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)  # Remove .detach()
            delta_slice.requires_grad_()  # Ensure gradients are enabled
    
            if self.args.attackoptim == "cw":
                delta_slice = 0.5 * (torch.tanh(delta_slice) + 1)
                delta_slice.retain_grad()
                
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Add additional clean data if mixing during the attack:
            if self.args.pmix:
                if 'mix' in victim.defs.mixing_method['type']:   # this covers mixup, cutmix 4waymixup, maxup-mixup
                    try:
                        extra_data = next(self.extra_data)
                    except StopIteration:
                        self.extra_data = iter(kettle.trainloader)
                        extra_data = next(self.extra_data)
                    extra_inputs = extra_data[0].to(**self.setup)
                    extra_labels = extra_data[1].to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                    inputs = torch.cat((inputs, extra_inputs), dim=0)
                    labels = torch.cat((labels, extra_labels), dim=0)

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)

            if self.args.padversarial is not None:
                # This is a more accurate anti-defense:
                [temp_sources, inputs,
                 temp_true_labels, labels,
                 temp_fake_label] = _split_data(inputs, labels, source_selection=victim.defs.novel_defense['source_selection'])
                delta, additional_info = self.attacker.attack(inputs.detach(), labels,
                                                              temp_sources, temp_fake_label, steps=victim.defs.novel_defense['steps'])
                inputs = inputs + delta  # Kind of a reparametrization trick

            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Change loss function to include corrective terms if mixing with correction
            if self.args.pmix:
                def criterion(outputs, labels):
                    loss, pred = kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
                    return loss
            else:
                criterion = loss_fn

            if NORMALIZE:
                inputs = normalization(inputs)
            
            closure = self._define_objective(inputs, labels, criterion, delta_slice)
            loss, prediction = victim.compute(closure, starting_params, target_params)
            
            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)
                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['cw', 'Adam', 'signAdam', 'momSGD', 'momPGD', 'SGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
            
            poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()
    
    def _get_backdoor_data(self, data, backdoor_training_mode='full-data'):
        """Create backdoor training data by applying label transforms."""
        target_class = data.poison_setup['target_class']
        
        # Safely handle source_class whether it's a list or a single value
        source_class = data.poison_setup['source_class']
        if isinstance(source_class, list) and len(source_class) > 0:
            source_class = source_class[0]
        
        # Verify classes exist in dataset
        if not hasattr(data.triggerset_dist, 'keys') or source_class not in data.triggerset_dist:
            raise ValueError(f"Source class {source_class} not found in trigger dataset")

        # Create poisoned dataset with label transform
        label_poison_transform = LabelPoisonTransform(mapping={source_class: target_class})
        poisoned_triggerset = Subset(
            data.triggerset,
            data.triggerset_dist[source_class],
            transform=data.trainset.transform,
            target_transform=label_poison_transform
        )

        # Initialize backdoor trainset
        if backdoor_training_mode == 'full-data':
            backdoor_trainset = ConcatDataset([data.trainset, poisoned_triggerset])
            bkd_indices = set(range(len(data.trainset), len(backdoor_trainset)))
        elif backdoor_training_mode == 'poison_only':
            backdoor_trainset = poisoned_triggerset
            bkd_indices = set(range(len(backdoor_trainset)))
        else:
            raise ValueError(f"Invalid backdoor training mode: {backdoor_training_mode}")
        
        # Create data loaders
        backdoor_trainloader = DataLoader(
            backdoor_trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=PIN_MEMORY
        )
        
        backdoor_testloader = DataLoader(
            poisoned_triggerset,
            batch_size=self.args.batch_size*2,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=PIN_MEMORY
        )

        return backdoor_trainloader, backdoor_testloader, bkd_indices
        
    def _validation(self, model, clean_testloader, poisoned_testloader):
        """Evaluate model performance on clean and poisoned data."""
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')
        clean_corr = 0
        clean_loss = 0
        poisoned_corr = 0
        poisoned_loss = 0
        
        with torch.no_grad():
            # Evaluate on clean data
            for inputs, targets, idx in clean_testloader:
                inputs, targets = inputs.to(self.setup['device']), targets.to(self.setup['device'])
                
                # Add normalization if required
                if NORMALIZE:
                    inputs = normalization(inputs)
                    
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                clean_loss += loss.item()
                _, predicted = outputs.max(1)
                clean_corr += predicted.eq(targets).sum().item()

            # Evaluate on poisoned data
            for inputs, targets, idx in poisoned_testloader:
                inputs, targets = inputs.to(self.setup['device']), targets.to(self.setup['device'])
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                poisoned_loss += loss.item()
                _, predicted = outputs.max(1)
                poisoned_corr += predicted.eq(targets).sum().item()

        # Calculate metrics
        clean_acc = clean_corr / len(clean_testloader.dataset)
        poisoned_acc = poisoned_corr / len(poisoned_testloader.dataset)
        clean_loss = clean_loss / len(clean_testloader.dataset)
        poisoned_loss = poisoned_loss / len(poisoned_testloader.dataset)

        return clean_acc, poisoned_acc, clean_loss, poisoned_loss
