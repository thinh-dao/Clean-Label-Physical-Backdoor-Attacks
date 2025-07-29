import torch
import numpy as np
import torch.nn as nn
import copy
import random

from forest.victims.training import _split_data
from forest.utils import write, cw_loss
from forest.consts import FINETUNING_LR_DROP, PIN_MEMORY, NORMALIZE, NON_BLOCKING
from forest.witchcoven import _Witch
from forest.data.datasets import Subset, LabelPoisonTransform, ConcatDataset
from torch.utils.data import DataLoader
from forest.data.datasets import normalization
from torch.nn.utils import parameters_to_vector, vector_to_parameters
    
class WitchMTTP(_Witch): 
    def _initialize_theta_pairs(self, victim, kettle):
        if self.args.ensemble > 1:
            self.starting_params = []
            self.target_params = []
            for idx in range(self.args.ensemble):
                # Initialize expert training trajectories (backdoor models)
                starting_params, target_params = self._backdoor_step(
                    backdoor_trainloader=self.backdoor_trainloader,
                    backdoor_indices=self.backdoor_indices,
                    model=victim.models[idx],
                    kettle=kettle,
                    lr=self.args.bkd_lr,
                    epochs=self.args.bkd_epochs,
                )
                self.starting_params.append(starting_params)
                self.target_params.append(target_params)
        else:                
            self.starting_params, self.target_params = self._backdoor_step(
                backdoor_trainloader=self.backdoor_trainloader,
                backdoor_indices=self.backdoor_indices,
                model=victim.model,
                kettle=kettle,
                lr=self.args.bkd_lr,
                epochs=self.args.bkd_epochs,
            )
                  
    def _initialize_buffers(self, victim, kettle):
        """
        Initialize all (starting, target) parameter pairs for the backdoor/expert models.
        If sample_from_trajectory, buffers = [pair_1, pair_2, etc]
        If ensemble, buffers = [[pair_1_model_1, pair_2_model_1], etc]
        """
        if self.args.ensemble > 1:
            self.buffers = [[] for _ in range(self.args.ensemble)]
        else:
            self.buffers = []
            
    def _train_and_fill_buffers(self, victim, kettle, poison_delta, max_epochs):
        """Train victim models to get training trajectory
        
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

                # Move to GPUs
                model.to(**self.setup)
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
                    model.frozen = model.module.frozen
                
                current_epoch = victim.epochs[idx] + 1
                for victim.epochs[idx] in range(current_epoch, current_epoch + max_epochs):
                    self.run_step(kettle, poison_delta, victim.epochs[idx], *single_model)
                    
                    if victim.epochs[idx] == current_epoch or victim.epochs[idx] % self.args.sample_every == 0:
                        write(f"Store theta pair for model {idx} at epoch {victim.epochs[idx]}", self.args.output)
                        theta_pair = self._backdoor_step(
                                        backdoor_trainloader=self.backdoor_trainloader,
                                        backdoor_indices=self.backdoor_indices,
                                        model=model,
                                        kettle=kettle,
                                        lr=self.args.bkd_lr,
                                        epochs=self.args.bkd_epochs
                                    )
                        self.buffers[idx].append(theta_pair)
                        
                    if victim.epoch % self.args.validate_every == 0:
                        c_acc, p_acc, c_loss, p_loss = self.validation(
                            model=model, 
                            clean_testloader=kettle.validloader, 
                            source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                            target_class = kettle.poison_setup["target_class"]
                        )
                                                
                        v_log = (f'Model {idx+1}/{self.args.ensemble} - Epoch {victim.epochs[idx]} | '
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
            
            current_epoch = victim.epoch + 1
            for victim.epoch in range(current_epoch, current_epoch + max_epochs):
                self.run_step(kettle, poison_delta, victim.epoch, *single_model_setup)

                if victim.epoch == current_epoch or victim.epoch % self.args.sample_every == 0:  
                    write(f"Store theta pair for model at epoch {victim.epoch}", self.args.output)
                    theta_pair = self._backdoor_step(
                        backdoor_trainloader=self.backdoor_trainloader,
                        backdoor_indices=self.backdoor_indices,
                        model=victim.model,
                        kettle=kettle,
                        lr=self.args.bkd_lr,
                        epochs=self.args.bkd_epochs
                    )   
                    self.buffers.append(theta_pair) 
                
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
    
    def _backdoor_step(self, backdoor_trainloader, backdoor_indices, model, kettle, lr, epochs):
        # Get only trainable parameters for the original model (starting weights)
        original_params = parameters_to_vector(
            p for p in model.parameters() if p.requires_grad
        ).detach().clone()

        net = copy.deepcopy(model).to(self.setup['device'])
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
        scaler = torch.amp.GradScaler('cuda')
        
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        src_crit = cw_loss if self.args.source_criterion == "cw" else None
        device = next(net.parameters()).device
        bkd_t = torch.as_tensor(list(backdoor_indices), device=device) if backdoor_indices else None
        
        if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
            frozen = net.module.frozen
            list(net.module.children())[-1].train() if frozen else net.train()
        else:
            frozen = net.frozen
            list(net.children())[-1].train() if frozen else net.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            epoch_acc = 0
            num_batches = 0
            
            for batch in backdoor_trainloader:
                x, y, idx = batch
                x, y, idx = (t.to(device, non_blocking=True) for t in (x, y, idx))
                
                if self.args.augment and kettle.augment:
                    x = kettle.augment(x)
                if NORMALIZE: 
                    x = normalization(x)
                
                opt.zero_grad(set_to_none=True)
                    
                with torch.amp.autocast('cuda'):
                    out = net(x)
                    losses = ce_loss(out, y)
                    if src_crit and bkd_t is not None:
                        mask = torch.isin(idx, bkd_t)
                        if mask.any():
                            losses[mask] = src_crit(out[mask], y[mask], reduction='none').to(losses.dtype)
                    loss = losses.mean()

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                # Accumulate metrics for this epoch
                epoch_loss += loss.item()
                epoch_acc += out.argmax(1).eq(y).sum().item() / y.size(0)
                num_batches += 1
                
                if self.args.dryrun:
                    break
            
            # Calculate epoch averages
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_acc = epoch_acc / num_batches
            
            # Get current parameters after this epoch
            current_params = parameters_to_vector(
                p for p in net.parameters() if p.requires_grad
            ).detach().clone()
            
            with torch.no_grad():
                dist = torch.norm(current_params.to(original_params.device) - original_params).item()

            log = (f'Epoch {epoch}/{epochs} | '
                f'Epoch Loss: {avg_epoch_loss:.4f}, Epoch Acc: {avg_epoch_acc:.4f}, ParamsDist: {dist:.6f}')
            print(log); write(log, self.args.output)
        
        c_acc, p_acc, c_loss, p_loss = self.validation(
            model=net, 
            clean_testloader=kettle.validloader, 
            source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
            target_class = kettle.poison_setup["target_class"]
        )
                            
        v_log = (f'Expert Net - ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}\n')
        print(v_log); write(v_log, self.args.output)
                    
        # Get final parameters after all epochs
        final_params = parameters_to_vector(
            p for p in net.parameters() if p.requires_grad
        ).detach().clone()

        return (original_params, final_params)
    
    def _get_backdoor_data(self, data, backdoor_training_mode='full_data'):
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
        if backdoor_training_mode == 'full_data':
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
            batch_size=self.args.bkd_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=PIN_MEMORY
        )
        
        backdoor_testloader = DataLoader(
            poisoned_triggerset,
            batch_size=self.args.bkd_batch_size*2,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=PIN_MEMORY
        )

        return backdoor_trainloader, backdoor_testloader, bkd_indices
        
    def _run_trial(self, victim, kettle):
        """Run a single trial. Perform one round of poisoning."""
        # Initialize poison mask of shape [num_poisons, channels, height, width] with values in [-eps, eps]
        poison_delta = kettle.initialize_poison()
        poison_delta.requires_grad_(True)
        poison_delta.grad = torch.zeros_like(poison_delta)
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
        
        # Initialize pairs of starting and target parameters
        self.backdoor_trainloader, self.backdoor_testloader, self.backdoor_indices = self._get_backdoor_data(data=kettle, backdoor_training_mode=self.args.bkd_training_mode)
        
        if self.args.sample_from_trajectory:             
            if not self.args.skip_clean_training:
                victim.initialize()
                print("Training model from scratch for trajectory sampling.")
                write("Training model from scratch for trajectory sampling.", self.args.output)
                
            self._initialize_buffers(victim, kettle)
            self._train_and_fill_buffers(victim, kettle, poison_delta=poison_delta, max_epochs=self.args.retrain_max_epoch)
        else:
            self._initialize_theta_pairs(victim, kettle)
            self.buffers = None
                        
        for step in range(self.args.attackiter):                
            if not self.args.sample_from_trajectory:
                starting_params = self.starting_params
                target_params = self.target_params
            else:
                if self.args.ensemble > 1:
                    starting_params = []
                    target_params = []
                    
                    if self.args.sample_same_idx:
                        sample_idx = random.randint(0, len(self.buffers[0]) - 1)
                        
                    for idx in range(self.args.ensemble):
                        if not self.args.sample_same_idx:
                            sample_idx = random.randint(0, len(self.buffers[0]) - 1)
                        
                        starting_params.append(self.buffers[idx][sample_idx][0])
                        target_params.append(self.buffers[idx][sample_idx][1])

                else:  
                    sample_idx = random.randint(0, len(self.buffers) - 1)
                    
                    starting_params = self.buffers[sample_idx][0]  
                    target_params = self.buffers[sample_idx][1]   

            # Initialize source_losses before the loop
            source_losses = 0
            
            for batch, example in enumerate(dataloader):
                loss, _ = self._batched_step(poison_delta, poison_bounds, example, victim, kettle, starting_params, target_params, step)
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
                
            if self.args.retrain_scenario != None:             
                if step % self.args.retrain_iter == 0 and step != 0 and step != self.args.attackiter - 1:
                    print("Retraining attacker model at epoch {} with {}".format(step, self.args.retrain_scenario))
                    
                    if self.args.retrain_reinit_seed:
                        seed = np.random.randint(0, 2**32 - 1)
                    else:
                        seed = None
                        
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize(seed=seed)
                        self._initialize_buffers(victim, kettle)
                        print('Model reinitialized to random seed.')
                    elif self.args.retrain_scenario == 'finetuning':                        
                        victim.reinitialize_last_layer(seed=seed, reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                        print('Completely warmstart finetuning!')

                    if self.args.scenario == 'transfer':
                        victim.load_feature_representation()
                    
                    # Train models with trajectory sampling
                    if not self.args.sample_from_trajectory:
                        self._train_and_fill_buffers(victim, kettle, poison_delta, self.args.retrain_max_epoch)
                    else:
                        victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                        self._initialize_theta_pairs(victim, kettle)
                    write('Retraining done!\n', self.args.output)

        return poison_delta, source_losses

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle, starting_params, target_params, step):
        """Take a step toward minimizing the current poison loss."""
        inputs, labels, ids = example
        
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            
        # If a poisoned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            inputs = inputs.to(**self.setup)
            labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        
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
            
            if self.args.mtt_validate_every and step % self.args.mtt_validate_every == 0 and step > 0:
                validate = True
            else:
                validate = False
                
            closure = self._define_objective(inputs, labels, criterion, delta_slice, victim, kettle, validate)
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

    def _define_objective(self, inputs, labels, criterion, perturbations, victim, kettle, validate=False):
        def closure(model, optimizer, starting_params, target_params):
            inner_lr = self.args.bkd_lr
            
            # 0) move vectors to GPU / device
            new_starting_params = starting_params.detach().clone().to(self.setup["device"])
            new_target_params   = target_params.detach().clone().to(self.setup["device"])

            # 1) load θ₀ into the network (only trainable parameters)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            vector_to_parameters(new_starting_params, trainable_params)

            # 2) forward pass on *already‑perturbed* batch
            outputs     = model(inputs)            # `inputs` depends on `perturbations`
            poison_loss = criterion(outputs, labels)

            # 3) obtain differentiable grads wrt θ
            grad_params = torch.autograd.grad(poison_loss,
                                            trainable_params,
                                            create_graph=True,
                                            retain_graph=True,
                                            allow_unused=True)

            # 4) flatten grads, make SGD update: θ₁ = θ₀ − α g
            # Only include gradients for trainable parameters
            g_vec = torch.cat([
                (g if g is not None else torch.zeros_like(p)).view(-1)
                for p, g in zip(trainable_params, grad_params)
            ])
            theta_poison = new_starting_params - inner_lr * g_vec   # <-- update stays in graph

            ############## VALIDATION ##############
            if validate:
                vnet = copy.deepcopy(model)
                vector_to_parameters(theta_poison, vnet.parameters())
                c_acc, p_acc, c_loss, p_loss = self.validation(model=vnet,
                    clean_testloader=kettle.validloader, 
                    source_testloader=kettle.source_testloader[kettle.poison_setup["source_class"][0]],
                    target_class = kettle.poison_setup["target_class"]
                )
                log = (f'ACC: {c_acc:.4f}, ASR: {p_acc:.4f}, Normal loss: {c_loss:.4f}, Backdoor loss: {p_loss:.4f}')
                print(log); write(log, self.args.output)
            ############## VALIDATION ##############
            
            # 5) parameter‑matching losses or gradient matching losses
            passenger_loss = self._passenger_loss(theta_poison, new_target_params, new_starting_params)

            # 6) add any regularisers you use
            attacker_loss  = passenger_loss
            attacker_loss += self.args.vis_weight * self.get_regularized_loss(
                                perturbations, tau=self.args.eps / 255)

            if self.args.centreg != 0:
                attacker_loss += self.args.centreg * poison_loss

            # 7) outer backward — this will now fill `perturbations.grad`
            attacker_loss.backward(retain_graph=self.retain)

            # 8) metrics for logging
            prediction = (outputs.detach().argmax(dim=1) == labels).sum()
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure
    
    def _passenger_loss(self, theta_poison, theta_target, theta_start):
        """Compute parameter matching loss for MTTP."""
        if self.args.mtt_loss == 'similarity':
            passenger_loss = 1 - torch.nn.functional.cosine_similarity(theta_poison, theta_target, dim=0)
        elif self.args.mtt_loss == 'MSE':
            # Original normalized parameter matching loss (default)
            param_loss = torch.nn.functional.mse_loss(theta_poison, theta_target, reduction="mean")
            param_dist = torch.nn.functional.mse_loss(theta_target, theta_start, reduction="mean")
            passenger_loss = param_loss / param_dist.detach()
        else:
            raise ValueError(f"Loss type {self.args.mtt_loss} not supported")

        # Add regularization if specified
        if self.args.normreg != 0:
            passenger_loss = passenger_loss + self.args.normreg * theta_poison.norm()

        # Add repelling term if specified
        if self.args.repel != 0:
            if hasattr(self.args, 'loss') and self.args.loss == 'cosine1':
                passenger_loss -= self.args.repel * torch.nn.functional.cosine_similarity(theta_poison, theta_target, dim=0)
            else:
                passenger_loss -= self.args.repel * torch.nn.functional.mse_loss(theta_poison, theta_target, reduction="mean")

        return passenger_loss