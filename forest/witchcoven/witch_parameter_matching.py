import torch
import numpy as np
import torch.nn as nn
import copy
import random

from torch.nn.utils import parameters_to_vector
from torch.cuda.amp  import GradScaler
from forest.victims.training import run_step
from forest.utils import write, ReparamModule, cw_loss
from forest.consts import FINETUNING_LR_DROP, PIN_MEMORY, NORMALIZE, NON_BLOCKING
from forest.witchcoven import _Witch
from forest.data.datasets import Subset, ConcatDataset, LabelPoisonTransform
from torch.utils.data import DataLoader
from forest.data.datasets import normalization

class WitchMTTP(_Witch):
    def _distill(self, poison_delta, poison_bounds, poison_dataloader, student_net, starting_params, target_params, kettle, syn_steps=1, syn_lr=0.001, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Train a student network on poisoned data to match target network parameters.
        Returns the distillation loss.
        """
        poison_delta.requires_grad_(True)  # Ensure poison_delta tracks gradients

        # Initialize student parameters (tracked across steps)
        student_params = [starting_params.detach().clone().requires_grad_(True)]  # Starting point

        # Train student network (manual parameter updates)
        for step in range(syn_steps):
            loss = 0
            for inputs, labels, idx in poison_dataloader:
                inputs = inputs.to(**self.setup)
                labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                
                # Apply poison perturbations (with gradient tracking)
                poison_slices, batch_positions = kettle.lookup_poison_indices(idx)
                
                if len(batch_positions) > 0:
                    delta_slice = poison_delta[poison_slices].to(**self.setup)
                    if self.args.clean_grad:
                        delta_slice = torch.zeros_like(delta_slice) + 0 * delta_slice  # Trick to preserve gradients
                    inputs[batch_positions] = inputs[batch_positions] + delta_slice
                    poison_bounds[poison_slices] = inputs[batch_positions].detach().cpu()
                
                # Data augmentation
                if self.args.paugment:
                    inputs = kettle.augment(inputs)
                if NORMALIZE:
                    inputs = normalization(inputs)
                
                # Forward pass through student network
                forward_params = student_params[-1].to(**self.setup)
                outputs = student_net(inputs, flat_param=forward_params)
                student_loss = loss_fn(outputs, labels)

                loss += student_loss.item()
                
                # Compute gradients for parameters
                grad = torch.autograd.grad(student_loss, forward_params, create_graph=True, retain_graph=True)[0]
                
                # Manual parameter update
                updated_params = forward_params - syn_lr * grad
                student_params.append(updated_params.cpu())
            
            loss /= len(poison_dataloader)
            print(f"Step {step+1} of {syn_steps} | Loss: {loss:.4f}")

        # Compute the distillation loss
        param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="mean")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="mean")
        passenger_loss = param_loss / param_dist.detach()

        # Backpropagate through the entire training process
        regularized_loss = self.get_regularized_loss(poison_delta, inputs, tau=self.args.eps/255)
        attacker_loss = passenger_loss + self.args.vis_weight * regularized_loss
        
        attacker_loss.backward()

        # Update poison_delta (PGD-like optimizers)
        if self.args.attackoptim in ['PGD', 'GD'] and len(batch_positions) > 0:
            for i, (slice_idx, position) in enumerate(zip(poison_slices, batch_positions)):
                delta_slice = poison_delta[slice_idx].to(**self.setup)
                poison_image = inputs[batch_positions][i].detach()
                updated_delta = self._pgd_step(delta_slice, poison_image, self.tau0, kettle.dm, kettle.ds)
                poison_delta.data[slice_idx] = updated_delta.detach().cpu()
        
        return passenger_loss.item()

    def _distill_full_data(self, poison_delta, poison_bounds, poison_dataloader, student_net, starting_params, target_params, kettle, syn_steps=1, syn_lr=0.001, loss_fn=torch.nn.CrossEntropyLoss()):
        # Initialize tracking variables
        poison_delta.requires_grad_(True)  # Ensure poison_delta tracks gradients

        # Initialize student parameters (tracked across steps)
        student_params = [starting_params.detach().clone().requires_grad_(True)]  # Starting point
        
        x_list = []
        y_list = [] 
        poison_slices_list = []
        batch_positions_list = []
        gradient_sum = torch.zeros_like(student_params[-1]).to(self.setup['device'])
        
        # Calculate parameter distance for normalization
        param_dist = torch.norm(target_params - starting_params) ** 2

        # Train student network with manual parameter updates
        for step in range(syn_steps):
            loss = 0
            for inputs, labels, idx in poison_dataloader:
                inputs = inputs.to(**self.setup)
                labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                
                # Apply poison perturbations only to designated poison samples
                poison_slices, batch_positions = kettle.lookup_poison_indices(idx)
                
                if len(batch_positions) > 0:
                    delta_slice = poison_delta[poison_slices].to(**self.setup)
                    if self.args.clean_grad:
                        delta_slice = torch.zeros_like(delta_slice) + 0 * delta_slice  # Trick to preserve gradients
                    poison_bounds[poison_slices] = inputs[batch_positions].clone().detach().cpu()
                    inputs[batch_positions] = inputs[batch_positions] + delta_slice

                    # Save poison slices and batch positions
                    poison_slices_list.append(poison_slices)
                    batch_positions_list.append(batch_positions)
                    
                    # Save original inputs (now with poison applied to specific samples)
                    x_list.append(inputs.clone().cpu())
                    y_list.append(labels.clone().cpu())
                
                # Data augmentation (applied to all images, poisoned and clean)
                if self.args.paugment:
                    inputs = kettle.augment(inputs)
                if NORMALIZE:
                    inputs = normalization(inputs)
                
                # Forward pass through student network
                forward_params = student_params[-1].to(**self.setup)
                outputs = student_net(inputs, flat_param=forward_params)
                student_loss = loss_fn(outputs, labels)

                loss += student_loss.item()
                
                # Compute gradients for parameters
                grad = torch.autograd.grad(student_loss, forward_params)[0]
                detached_grad = grad.detach().clone()
                
                # Only add to gradient_sum if the batch contains poisoned samples
                if len(batch_positions) > 0:
                    gradient_sum += detached_grad
                
                # Manual parameter update (always update parameters regardless of poisoning)
                updated_params = forward_params - syn_lr * detached_grad
                student_params.append(updated_params.cpu())
                    
                # Clean up GPU tensors
                del grad, outputs, student_loss

            loss /= len(poison_dataloader)
            print(f"Step {step+1} of {syn_steps} | Loss: {loss:.4f}")
            
            # Create a tensor to hold poison gradients
            poison_delta_gradients = torch.zeros_like(poison_delta)
            gradient_sum = gradient_sum.to(self.setup['device'])
            
            # --------Compute the gradients regarding poison delta---------
            # Compute gradients involving 2 gradients
            for i in range(len(batch_positions_list)):
                # Skip batches without poisoned samples
                if len(batch_positions_list[i]) == 0:
                    raise ValueError(f"Batch {i} has no poisoned samples. There may be an error in the poison indices.")
                    
                # Compute gradients for w_i
                w_i = student_params[i].to(self.setup['device'])
                x_i = x_list[i].to(self.setup['device'])
                y_i = y_list[i].to(self.setup['device'])

                inputs = x_i.clone()

                if self.args.paugment:
                    inputs = kettle.augment(inputs)
                if NORMALIZE:
                    inputs = normalization(inputs)

                output_i = student_net(inputs, flat_param=w_i)
                ce_loss_i = loss_fn(output_i, y_i)
                grad_i = torch.autograd.grad(ce_loss_i, w_i, create_graph=True)[0]

                single_term = syn_lr * (target_params - starting_params)
                single_term = single_term.to(self.setup['device'])
                square_term = (syn_lr ** 2) * gradient_sum
                
                # Compute gradients with respect to the original inputs (with poison applied)
                total_term = 2 * (single_term + square_term) @ grad_i / param_dist.to(self.setup['device'])
    
                # Compute gradients only for poisoned samples
                gradients = torch.autograd.grad(
                    total_term,
                    x_i,
                )[0]
                
                if gradients is not None:
                    poisoned_gradients = gradients[batch_positions_list[i]]
                    poison_delta_gradients[poison_slices_list[i]] += poisoned_gradients.cpu()

        # Calculate final loss with MSE between final parameters and target parameters
        param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="mean")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="mean")
        passenger_loss = param_loss / param_dist.detach()
        
        # Backpropagate through the entire training process
        regularized_loss = self.get_regularized_loss(poison_delta, inputs, tau=self.args.eps/255)
        attacker_loss = passenger_loss + self.args.vis_weight * regularized_loss
        
        attacker_loss.backward()
        
        # Assign gradients manually if needed
        if torch.norm(poison_delta_gradients) > 0:
            if poison_delta.grad is None:
                poison_delta.grad = poison_delta_gradients
            else:
                poison_delta.grad += poison_delta_gradients
        
        for _ in student_params:
            del _

        return passenger_loss.item()

    def _train_backdoor_net(self, backdoor_trainloader, backdoor_testloader,
                            bkd_indices, victim, kettle, lr, epochs, num_experts=3):

        all_trajectories = []
        original_params = parameters_to_vector(victim.model.parameters()).detach()

        for exp_idx in range(num_experts):
            print(f"Training expert {exp_idx+1}/{num_experts}")
            write(f"Training expert {exp_idx+1}/{num_experts}", self.args.output)

            net   = copy.deepcopy(victim.model).to(self.setup['device'])
            opt   = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=5e-4, nesterov=True)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            scaler = torch.amp.GradScaler('cuda')

            traj = [parameters_to_vector(net.parameters()).cpu()]
            eval_every = max(1, epochs // 3)

            for epoch in range(1, epochs + 1):
                avg_loss, acc = self._train_one_backdoor_epoch(
                    model=net, optimizer=opt, backdoor_trainloader=backdoor_trainloader,
                    bkd_indices=bkd_indices, diff_augment=kettle.augment,
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
                        net, kettle.validloader, backdoor_testloader)
                    v_log = (f'Expert {exp_idx+1}, Epoch {epoch}/{epochs} | '
                            f'CleanAcc: {c_acc:.4f}, PoisonAcc: {p_acc:.4f}')
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

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                out = model(x)
                losses = ce_loss(out, y)
                if src_crit and bkd_t is not None:
                    mask = torch.isin(idx, bkd_t)
                    if mask.any():
                        losses[mask] = src_crit(out[mask], y[mask], reduction='none')
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
        
        syn_lr = torch.tensor(self.args.syn_lr).to(self.setup['device']).requires_grad_(True)
        lr_optimizer = torch.optim.SGD([syn_lr], lr=1e-7, momentum=0.9, weight_decay=0, nesterov=True)
        
        # Initialize expert training trajectories (backdoor models)
        backdoor_trainloader, backdoor_testloader, bkd_indices = self._get_backdoor_data(kettle, backdoor_training_mode=self.args.backdoor_training_mode)
        all_trajectories = self._train_backdoor_net(
            backdoor_trainloader,
            backdoor_testloader,
            bkd_indices,
            victim,
            kettle,
            lr=self.args.finetuning_lr,
            epochs=self.args.backdoor_training_epoch,
            num_experts=self.args.num_experts
        )
        
        # Average trajectories
        if self.args.average_trajectory:
            avg_trajectory = []
            for i in range(len(all_trajectories[0])):
                avg_trajectory.append(torch.stack([traj[i] for traj in all_trajectories]).mean(dim=0))

        for step in range(self.args.attackiter):
            if self.args.average_trajectory:
                chosen_trajectory = avg_trajectory
            else:
                sample_traj_idx = np.random.randint(0, len(all_trajectories))
                chosen_trajectory = all_trajectories[sample_traj_idx]
            
            if self.args.sequential_generation:
                expansion_end_epoch = int(self.args.retrain_iter * 0.75)
                if len(chosen_trajectory) - 1 < self.args.expert_epochs:
                    max_start_epoch = 0
                else:
                    max_start_epoch = int ( (len(chosen_trajectory) - 1 - self.args.expert_epochs) * (step / expansion_end_epoch) )
                start_params_idx = np.random.randint(0, max_start_epoch+1)
            else:
                max_start_epoch = max(len(chosen_trajectory) - 1 - self.args.expert_epochs, 0)
                start_params_idx = np.random.randint(0, max_start_epoch+1)

            starting_params = chosen_trajectory[start_params_idx]

            target_params_idx = min(start_params_idx + self.args.expert_epochs, len(chosen_trajectory) - 1)
            target_params = chosen_trajectory[target_params_idx]

            # Create student network
            student_net = copy.deepcopy(victim.model)

            if hasattr(student_net, 'module'):
                student_net = ReparamModule(student_net.module)
            else:
                student_net = ReparamModule(student_net)

            att_optimizer.zero_grad(set_to_none=False)
            lr_optimizer.zero_grad(set_to_none=False)
                        
            # Distill from backdoor model to student model
            if self.args.full_data:
                distill_loss = self._distill_full_data(
                    poison_delta=poison_delta, 
                    poison_bounds=poison_bounds, 
                    poison_dataloader=dataloader, 
                    student_net=student_net, 
                    starting_params=starting_params, 
                    target_params=target_params, 
                    kettle=kettle,
                    syn_steps=self.args.syn_steps,
                    syn_lr=syn_lr
                )
            else:
                distill_loss = self._distill(
                    poison_delta=poison_delta, 
                    poison_bounds=poison_bounds, 
                    poison_dataloader=dataloader, 
                    student_net=student_net, 
                    starting_params=starting_params, 
                    target_params=target_params, 
                    kettle=kettle,
                    syn_steps=self.args.syn_steps,
                    syn_lr=syn_lr
                )

            # Update poison perturbations
            if self.args.attackoptim in ['momPGD', 'signAdam']:
                poison_delta.grad.sign_()
            
            att_optimizer.step()
            lr_optimizer.step()
            
            if self.args.scheduling:
                scheduler.step()
            
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

            # Calculate visual loss (L2 norm of perturbations)
            with torch.no_grad():
                visual_losses = torch.mean(torch.linalg.matrix_norm(poison_delta))
            
            # Log progress
            if step % 1 == 0 or step == (self.args.attackiter - 1):
                lr = att_optimizer.param_groups[0]['lr']
                print(f'Iteration {step} - lr: {lr} - syn_lr: {syn_lr.item():2.4f} | Distillation loss: {distill_loss:2.4f} | Visual loss: {visual_losses:2.4f}')
                write(f'Iteration {step} - lr: {lr} - syn_lr: {syn_lr.item():2.4f} | Distillation loss: {distill_loss:2.4f} | Visual loss: {visual_losses:2.4f}', self.args.output)

            # Step victim model if needed
            if self.args.step and step % self.args.step_every == 0:
                single_setup = (victim.model, victim.defs, victim.optimizer, victim.scheduler)
                if self.args.clean_grad:
                    run_step(kettle, None, step, *single_setup)
                else:
                    run_step(kettle, poison_delta, step, *single_setup)

            if self.args.dryrun:
                break
                
            # Handle retraining scenario
            if self.args.retrain_scenario is not None:
                if step % self.args.retrain_iter == 0 and step != 0 and step != self.args.attackiter - 1:
                    print(f"Retraining the base model at iteration {step}")
                    poison_delta.detach()
                    
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize()
                    elif self.args.retrain_scenario == 'finetuning':
                        victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                        print('Completely warmstart finetuning!')

                    victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                    write(f'Retraining completed at step: {step}', self.args.output)
                    print(f'Retraining completed at step: {step}')

                    write('Retraining backdoor model...', self.args.output)
                    print('Retraining backdoor model...')

                    all_trajectories = self._train_backdoor_net(
                        backdoor_trainloader,
                        backdoor_testloader,
                        bkd_indices,
                        victim,
                        kettle,
                        lr=self.args.finetuning_lr,
                        epochs=self.args.backdoor_training_epoch,
                        num_experts=self.args.num_experts
                    )

        return poison_delta, distill_loss

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
        elif backdoor_training_mode == 'poison-only':
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
            for inputs, targets, idx in poisoned_testloader:
                inputs, targets = inputs.to(device), targets.to(device)
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

class WitchMTTP_Tesla(_Witch):
    def _distill_full_data(self, poison_delta, poison_bounds, poison_dataloader, student_net, starting_params, target_params, kettle, syn_steps=1, syn_lr=0.001, loss_fn=torch.nn.CrossEntropyLoss()):
        # Initialize tracking variables
        poison_delta.requires_grad_(True)  # Ensure poison_delta tracks gradients

        # Initialize student parameters (tracked across steps)
        student_params = [starting_params.detach().clone().requires_grad_(True)]  # Starting point
        
        x_list = []
        original_x_list = []
        y_list = [] 
        poison_slices_list = []
        batch_positions_list = []
        gradient_sum = torch.zeros_like(student_params[-1]).to(self.setup['device'])
        
        # Calculate parameter distance for normalization
        param_dist = torch.norm(target_params - starting_params) ** 2

        # Train student network with manual parameter updates
        for step in range(syn_steps):
            loss = 0
            for inputs, labels, idx in poison_dataloader:
                inputs = inputs.to(**self.setup)
                labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                
                # Apply poison perturbations only to designated poison samples
                poison_slices, batch_positions = kettle.lookup_poison_indices(idx)
                poison_slices_list.append(poison_slices)
                batch_positions_list.append(batch_positions)
                
                if len(batch_positions) > 0:
                    delta_slice = poison_delta[poison_slices].to(**self.setup)
                    if self.args.clean_grad:
                        delta_slice = torch.zeros_like(delta_slice) + 0 * delta_slice  # Trick to preserve gradients
                    poison_bounds[poison_slices] = inputs[batch_positions].clone().detach().cpu()
                    inputs[batch_positions] = inputs[batch_positions] + delta_slice
                
                # Save original inputs (now with poison applied to specific samples)
                original_x_list.append(inputs)
                
                # Data augmentation (applied to all images, poisoned and clean)
                if self.args.paugment:
                    inputs = kettle.augment(inputs)
                if NORMALIZE:
                    inputs = normalization(inputs)
                
                # Save augmented inputs and labels
                x_list.append(inputs.clone())
                y_list.append(labels.clone())
                
                # Forward pass through student network
                outputs = student_net(inputs, flat_param=student_params[-1].to(**self.setup))
                student_loss = loss_fn(outputs, labels)

                loss += student_loss.item()
                
                # Compute gradients for parameters
                grad = torch.autograd.grad(student_loss, student_params[-1])[0]
                detached_grad = grad.detach().clone()
                
                # Only add to gradient_sum if the batch contains poisoned samples
                if len(batch_positions) > 0:
                    gradient_sum += detached_grad
                
                # Manual parameter update (always update parameters regardless of poisoning)
                updated_params = student_params[-1] - syn_lr * detached_grad
                student_params.append(updated_params.cpu())
                    
                del grad

            loss /= len(poison_dataloader)
            print(f"Step {step+1} of {syn_steps} | Loss: {loss:.4f}")

        # Create a tensor to hold poison gradients
        poison_delta_gradients = torch.zeros_like(poison_delta)

        # --------Compute the gradients regarding poison delta---------
        # Compute gradients involving 2 gradients
        for i in range(len(batch_positions_list)):
            # Skip batches without poisoned samples
            if len(batch_positions_list[i]) == 0:
                continue
                
            # Compute gradients for w_i
            w_i = student_params[i]
            output_i = student_net(x_list[i], flat_param=w_i)
            ce_loss_i = loss_fn(output_i, y_list[i])
            
            grad_i = torch.autograd.grad(ce_loss_i, w_i, create_graph=True, retain_graph=True)[0]
            
            single_term = syn_lr * (target_params - starting_params)
            square_term = (syn_lr ** 2) * gradient_sum
            
            # Check if this batch has poisoned samples and if original_x_list[i] requires grad
            if len(batch_positions_list[i]) > 0:
                # Compute gradients with respect to the original inputs (with poison applied)
                total_term = 2 * (single_term + square_term) @ grad_i / param_dist
    
                # Compute gradients only for poisoned samples
                gradients = torch.autograd.grad(
                    total_term,
                    original_x_list[i],
                    create_graph=True, 
                    retain_graph=True
                )[0]
                
                if gradients is not None:
                    poisoned_gradients = gradients[batch_positions_list[i]]
                    poison_delta_gradients[poison_slices_list[i]] += poisoned_gradients.cpu()

        # Calculate final loss with MSE between final parameters and target parameters
        param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="mean")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="mean")
        grand_loss = param_loss / param_dist.detach()
        
        # Assign gradients manually if needed
        if torch.norm(poison_delta_gradients) > 0:
            if poison_delta.grad is None:
                poison_delta.grad = poison_delta_gradients
            else:
                poison_delta.grad += poison_delta_gradients
        
        for _ in student_params:
            del _

        return grand_loss
