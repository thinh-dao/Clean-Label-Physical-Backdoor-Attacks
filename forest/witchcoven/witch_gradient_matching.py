"""Main class, holding information about models and training/testing routines."""

import torch
import numpy as np
import random

from ..consts import BENCHMARK, NON_BLOCKING, NORMALIZE, FINETUNING_LR_DROP
from ..utils import bypass_last_layer, cw_loss, write
from ..victims.training import _split_data
from .witch_base import _Witch
from forest.data.datasets import normalization

torch.backends.cuda.enable_flash_sdp(False) # Disable efficient attention to avoid second-order derivative issues
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cudnn.benchmark = BENCHMARK

class WitchGradientMatching(_Witch):
    """Brew passenger poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """
    def _initialize_buffers(self, victim, kettle):
        """Initialize buffers for sampling from trajectory """
        if self.args.ensemble > 1:
            self.buffers = [[] for _ in range(self.args.ensemble)]
        else:
            self.buffers = []
    
    def _train_and_fill_buffers(self, victim, kettle, poison_delta, max_epochs):
        """Train victim models and fill up buffers
        Buffers contain [(source_grad, source_gnorm, source_clean_grad), ... ]
        
        Args:
            victim: The victim model(s) to train
            kettle: The kettle object containing data loaders
            poison_delta: The poison perturbations
            max_epochs: Maximum number of epochs to train
        """
        if self.args.ensemble > 1:
            # Move to GPUs and wrap with DataParallel if needed
            for idx, model in enumerate(victim.models):
                # Unwrap DataParallel if already wrapped to avoid nesting
                if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                    print(f"Unwrapping existing DataParallel from model {idx}")
                    model = model.module
                    victim.models[idx] = model
                
                # Move model to the appropriate device first
                if torch.cuda.device_count() > 1:
                    # For DataParallel, move to cuda:0 specifically
                    victim.models[idx] = model.to(device='cuda:0', dtype=self.setup.get('dtype', torch.float32))
                    victim.models[idx] = torch.nn.DataParallel(victim.models[idx])
                    victim.models[idx].frozen = model.frozen
                    print(f"Wrapped model {idx} with DataParallel on cuda:0")
                else:
                    # Single GPU case
                    victim.models[idx] = model.to(**self.setup)
                    print(f"Moved model {idx} to {self.setup['device']}")
            
            multi_model_setup = (victim.models, victim.definitions, victim.optimizers, victim.schedulers)
            
            for epoch in range(1, max_epochs+1):
                for idx, single_model in enumerate(zip(*multi_model_setup)):    
                    victim.epochs[idx] += 1
                    self.run_step(kettle, poison_delta, victim.epochs[idx], *single_model)
                    
                if epoch % self.args.sample_every == 0:
                    source_grad, source_gnorm, source_clean_grad = self._compute_source_gradient(victim, kettle)
                    for model_idx in range(self.args.ensemble):
                        state_dict = {k: v.detach().clone().cpu() for k,v in victim.models[model_idx].state_dict().items()}
                        self.buffers[model_idx].append((source_grad[model_idx], source_gnorm[model_idx], source_clean_grad[model_idx], state_dict))
                    write(f"Store source grad at epoch {epoch}", self.args.output)
                    
                if epoch % self.args.validate_every == 0:    
                    for idx, model in enumerate(victim.models):
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
                
            # Move to CPUs and unwrap DataParallel if needed
            for idx, model in enumerate(victim.models):
                if torch.cuda.device_count() > 1 and isinstance(model, torch.nn.DataParallel):
                    victim.models[idx] = model.module
                victim.models[idx].to(device=torch.device('cpu'))
                
        else:
            single_model_setup = victim.model, victim.defs, victim.optimizer, victim.scheduler
            
            current_epoch = victim.epoch + 1
            for victim.epoch in range(current_epoch, current_epoch + max_epochs):
                self.run_step(kettle, poison_delta, victim.epoch, *single_model_setup)
                
                if victim.epoch % self.args.sample_every == 0:  
                    state_dict = {k: v.detach().clone().cpu() for k,v in victim.model.state_dict().item()}
                    source_grad, source_gnorm, source_clean_grad = self._compute_source_gradient(victim, kettle)
                    self.buffers.append((source_grad, source_gnorm, source_clean_grad, state_dict))
                    
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
    
    def _initialize_sources(self, victim, kettle):
        self.sources_train = torch.stack([data[0] for data in kettle.source_trainset], dim=0).to(**self.setup)
        self.true_classes = torch.tensor([data[1] for data in kettle.source_trainset]).to(device=self.setup['device'], dtype=torch.long)

        if kettle.args.threatmodel != 'all-to-all':
            self.target_classes = torch.tensor([kettle.poison_setup['target_class']] * kettle.source_train_num).to(device=self.setup['device'], dtype=torch.long)
        else:
            self.target_classes = torch.tensor([kettle.mapping[self.true_classes[i].item()] for i in range(len(self.true_classes))]).to(device=self.setup['device'], dtype=torch.long)

        if NORMALIZE:
            self.sources_train = normalization(self.sources_train)
            
    def _compute_source_gradient(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        victim.eval(dropout=True)

        # Modify source grad for backdoor poisoning
        _sources = self.sources_train
        _true_classes= self.true_classes
        _target_classes = self.target_classes

        if self.args.source_criterion in ['cw', 'carlini-wagner']:
            source_grad, source_gnorm = victim.gradient(_sources, _target_classes, criterion=cw_loss, selection=self.args.source_selection_strategy)
        elif self.args.source_criterion in ['unsourced-cross-entropy', 'unxent']:
            source_grad, source_gnorm = victim.gradient(_sources, _true_classes, selection=self.args.source_selection_strategy)
            for grad in source_grad:
                grad *= -1
        elif self.args.source_criterion in ['xent', 'cross-entropy']:
            source_grad, source_gnorm = victim.gradient(_sources, _target_classes, selection=self.args.source_selection_strategy)
        else:
            raise ValueError('Invalid source criterion chosen ...')
            
        write(f'Source Grad Norm is {source_gnorm}', self.args.output)

        if self.args.repel != 0:
            source_clean_grad, _ = victim.gradient(_sources, _true_classes)
        else:
            source_clean_grad = None if self.args.ensemble == 1 else [None for i in range(self.args.ensemble)]
        
        return source_grad, source_gnorm, source_clean_grad
                
    def _run_trial(self, victim, kettle):
        """Run a single trial. Perform one round of poisoning."""
        poison_delta = kettle.initialize_poison() # Initialize poison mask of shape [num_poisons, channels, height, width] with values in [-eps, eps]
        poison_delta.grad = torch.zeros_like(poison_delta) 
        dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
        poison_bounds = torch.zeros_like(poison_delta)
        
        self._initialize_sources(victim, kettle)
        
        if self.args.sample_from_trajectory:
            self._initialize_buffers(victim, kettle)
            self._train_and_fill_buffers(victim, kettle, poison_delta=poison_delta, max_epochs=self.args.retrain_max_epoch)
        else:
            self.source_grad, self.source_gnorm, self.source_clean_grad = self._compute_source_gradient(victim, kettle)
            self.buffers = None
        
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.poisonloader
        
        if self.args.attackoptim in ['cw', 'Adam', 'signAdam', 'momSGD', 'momPGD', 'SGD']:
            poison_delta.requires_grad_()
            if self.args.attackoptim in ['cw', 'Adam', 'signAdam']:
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
        else:
            raise ValueError('Unknown attack optimizer.')
        
        for step in range(self.args.attackiter):
            if not self.args.sample_from_trajectory:
                source_grad, source_gnorm, source_clean_grad = self.source_grad, self.source_gnorm, self.source_clean_grad
                state_dict = None
            else:
                if self.args.ensemble > 1:
                    source_grad = []
                    source_gnorm = []
                    source_clean_grad = []
                    state_dict = []
                    
                    if self.args.sample_same_idx:
                        sample_idx = random.randint(0, len(self.buffers[0]) - 1)
                        
                    for idx in range(self.args.ensemble):
                        if not self.args.sample_same_idx:
                            sample_idx = random.randint(0, len(self.buffers[0]) - 1)
                        
                        source_grad.append(self.buffers[idx][sample_idx][0])
                        source_gnorm.append(self.buffers[idx][sample_idx][1])
                        source_clean_grad.append(self.buffers[idx][sample_idx][2])
                        state_dict.append(self.buffers[idx][sample_idx][3])

                else:  
                    sample_idx = random.randint(0, len(self.buffers) - 1)
                    
                    source_grad = self.buffers[sample_idx][0]  
                    source_gnorm = self.buffers[sample_idx][1]
                    source_clean_grad = self.buffers[sample_idx][2]
                    state_dict = self.buffers[sample_idx][3]
            
            # Initialize source_losses before the loop
            source_losses = 0
                
            for batch, example in enumerate(dataloader):
                loss, _ = self._batched_step(poison_delta, poison_bounds, example, victim, kettle, source_grad, source_gnorm, source_clean_grad, state_dict)
                source_losses += loss
                
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
                    print("Retrainig the base model at iteration {}".format(step))
                    
                    if self.args.retrain_reinit_seed:
                        seed = np.random.randint(0, 2**32 - 1)
                    else:
                        seed = None
                        
                    # Reinitialize model and train from-scratch
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize(seed=seed)
                        self._initialize_buffers(victim, kettle)
                        print('Model reinitialized to random seed.')
                    
                    # Preserve the model_weight from last training and train on updated model
                    elif self.args.retrain_scenario == 'finetuning':
                        victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True, seed=seed)
                        print('Completely warmstart finetuning!')

                    if self.args.scenario == 'transfer':
                        victim.load_feature_representation()
                    
                    # Train models with trajectory sampling
                    if self.args.sample_from_trajectory:
                        self._train_and_fill_buffers(victim, kettle, poison_delta, self.args.retrain_max_epoch)
                    else:
                        victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                        self.source_grad, self.source_gnorm, self.source_clean_grad = self._compute_source_gradient(victim, kettle)
                        
                    write('Retraining done!\n', self.args.output)

        return poison_delta, source_losses

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle, source_grad, source_gnorm, source_clean_grad, state_dict):
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

            if NORMALIZE:
                inputs = normalization(inputs)
            
            closure = self._define_objective(inputs, labels, criterion, self.sources_train, self.target_classes, self.true_classes, delta_slice)
            loss, prediction = victim.compute(closure, source_grad, source_clean_grad, source_gnorm, state_dict)
            
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
    
    def _define_objective(self, inputs, labels, criterion, sources, target_classes, true_classes, perturbations):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm, state_dict):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            if state_dict != None:
                model.load_state_dict(state_dict)
                
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            outputs = model(inputs)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            
            poison_loss = criterion(outputs, labels)
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

            passenger_loss = self._passenger_loss(poison_grad, source_grad, source_clean_grad, source_gnorm)
            regularized_loss = self.get_regularized_loss(perturbations, tau=self.args.eps/255)
                
            attacker_loss = passenger_loss + self.args.vis_weight * regularized_loss
            if self.args.centreg != 0:
                attacker_loss += self.args.centreg * poison_loss
            attacker_loss.backward(retain_graph=self.retain)
            
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _passenger_loss(self, poison_grad, source_grad, source_clean_grad, source_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0

        SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
        if self.args.loss == 'top10-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 10)
        elif self.args.loss == 'top20-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 20)
        elif self.args.loss == 'top5-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 5)
        else:
            indices = torch.arange(len(source_grad))

        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]:
                passenger_loss -= (source_grad[i] * poison_grad[i]).sum()
            elif self.args.loss == 'cosine1':
                passenger_loss -= torch.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
            elif self.args.loss == 'SE':
                passenger_loss += 0.5 * (source_grad[i] - poison_grad[i]).pow(2).sum()
            elif self.args.loss == 'MSE':
                passenger_loss += torch.nn.functional.mse_loss(source_grad[i], poison_grad[i])
            elif self.args.loss == 'MSE+cosine':
                passenger_loss += torch.nn.functional.mse_loss(source_grad[i], poison_grad[i]) + torch.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
                
            if self.args.loss in SIM_TYPE or self.args.normreg != 0:
                poison_norm += poison_grad[i].pow(2).sum()

        if self.args.repel != 0:
            for i in indices:
                if self.args.loss in ['scalar_product', *SIM_TYPE]:
                    passenger_loss += self.args.repel * (source_grad[i] * poison_grad[i]).sum()
                elif self.args.loss == 'cosine1':
                    passenger_loss -= self.args.repel * torch.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
                elif self.args.loss == 'SE':
                    passenger_loss -= 0.5 * self.args.repel * (source_grad[i] - poison_grad[i]).pow(2).sum()
                elif self.args.loss == 'MSE':
                    passenger_loss -= self.args.repel * torch.nn.functional.mse_loss(source_grad[i], poison_grad[i])

        passenger_loss /= source_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE:
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        if self.args.normreg != 0:
            passenger_loss = passenger_loss + self.args.normreg * poison_norm.sqrt()

        if self.args.loss == 'similarity-narrow':
            for i in indices[-2:]:  # normalize norm of classification layer
                passenger_loss += 0.5 * poison_grad[i].pow(2).sum() / source_gnorm

        return passenger_loss

class WitchGradientMatchingNoisy(WitchGradientMatching):
    """Brew passenger poison with given arguments.

    Both the poison gradient and the source gradient are modified to be diff. private before calcuating the loss.
    """

    def _initialize_brew(self, victim, kettle):
        super()._initialize_brew(victim, kettle)
        self.defs = victim.defs
        self.kettle = kettle

    def _define_objective(self, inputs, labels, criterion):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            outputs = model(inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True, only_inputs=True)

            # add noise to samples
            self._hide_gradient(poison_grad)

            # Compute blind passenger loss
            passenger_loss = self._passenger_loss(poison_grad, source_grad, source_clean_grad, source_gnorm)
            if self.args.centreg != 0:
                passenger_loss = passenger_loss + self.args.centreg * poison_loss
            passenger_loss.backward(retain_graph=self.retain)
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _hide_gradient(self, gradient_list):
        """Enforce batch-wise privacy if necessary.

        This is attacking a defense discussed in Hong et al., 2020
        We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
        This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
        of noise to the gradient signal
        """
        if self.defs.privacy['clip'] is not None:
            total_norm = torch.norm(torch.stack([torch.norm(grad) for grad in gradient_list]))
            clip_coef = self.defs.privacy['clip'] / (total_norm + 1e-6)
            if clip_coef < 1:
                for grad in gradient_list:
                    grad.mul(clip_coef)

        if self.defs.privacy['noise'] is not None:
            loc = torch.as_tensor(0.0, device=self.kettle.setup['device'])
            clip_factor = self.defs.privacy['clip'] if self.defs.privacy['clip'] is not None else 1.0
            scale = torch.as_tensor(clip_factor * self.defs.privacy['noise'], device=self.kettle.setup['device'])
            if self.defs.privacy['distribution'] == 'gaussian':
                generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif self.defs.privacy['distribution'] == 'laplacian':
                generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {self.defs.privacy["distribution"]} given.')

            for grad in gradient_list:
                grad += generator.sample(grad.shape)



class WitchGradientMatchingHidden(WitchGradientMatching):
    """Brew passenger poison with given arguments.

    Try to match the original image feature representation to hide the attack from filter defenses.
    This class does a ton of horrid overwriting of the _batched_step method to add some additional
    computations that I dont want to be executed for all attacks. todo: refactor :>
    """
    FEATURE_WEIGHT = 1.0

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle, curr_reg=0):
        """Take a step toward minmizing the current poison loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        # save out clean inputs
        # These will be representative of "average" unpoisoned versions of the poison images
        # as such they will be augmented differently
        clean_inputs = inputs.clone().detach()

        # If a poisoned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE

            if self.args.attackoptim == "cw":
                delta_slice = 0.5 * (torch.tanh(delta_slice) + 1)
                delta_slice.retain_grad()  # Ensure the transformed tensor still requires gradients
                
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
                    clean_inputs = torch.cat((clean_inputs, extra_inputs), dim=0)
                    labels = torch.cat((labels, extra_labels), dim=0)

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs)
                clean_inputs = kettle.augment(clean_inputs)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)
                clean_inputs, _, _ = kettle.mixer(clean_inputs, labels)

            if self.args.padversarial is not None:
                [temp_sources, inputs,
                 temp_true_labels, labels,
                 temp_fake_label] = _split_data(inputs, labels, source_selection=victim.defs.novel_defense['source_selection'])
                delta, additional_info = self.attacker.attack(inputs.detach(), labels,
                                                              temp_sources, temp_fake_label, steps=victim.defs.novel_defense['steps'])
                inputs = inputs + delta
                clean_inputs = clean_inputs + delta

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

            if self.target_feature != None:
                feat_loss = self.get_featloss(inputs)
            else:
                feat_loss = torch.tensor(0)
                
            if NORMALIZE:
                inputs = normalization(inputs)
                
            closure = self._define_objective(inputs, clean_inputs, labels, criterion)
            loss, prediction = victim.compute(closure, self.source_grad, self.source_clean_grad, self.source_gnorm)

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

        return loss.item(), feat_loss.item(), prediction.item()


    def _define_objective(self, inputs, clean_inputs, labels, criterion):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            feature_model, last_layer = bypass_last_layer(model)
            features = feature_model(inputs)
            outputs = last_layer(features)

            # clean features:
            clean_features = feature_model(clean_inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True, only_inputs=True)

            # add feature term
            feature_loss = (features - clean_features).pow(2).mean()

            # Compute blind passenger loss
            passenger_loss = self._passenger_loss(poison_grad, source_grad, source_clean_grad, source_gnorm)

            total_loss = passenger_loss + self.FEATURE_WEIGHT * feature_loss
            if self.args.centreg != 0:
                total_loss = passenger_loss + self.args.centreg * poison_loss
            total_loss.backward(retain_graph=self.retain)
            return total_loss.detach().cpu(), prediction.detach().cpu()
        return closure

class WitchMatchingMultiSource(WitchGradientMatching):
    """Variant in which source gradients are matched separately."""

    def _initialize_brew(self, victim, kettle):
        super()._initialize_brew(victim, kettle)
        self.source_grad, self.source_gnorm = [], []
        for source, target_class in zip(self.sources, self.target_classes):
            grad, gnorm = victim.gradient(source.unsqueeze(0), target_class.unsqueeze(0))
            self.source_grad.append(grad)
            self.source_gnorm.append(gnorm)


    def _define_objective(self, inputs, labels, criterion):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            outputs = model(inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

            matching_loss = 0
            for tgrad, tnorm in zip(source_grad, source_gnorm):
                matching_loss += self._passenger_loss(poison_grad, tgrad, None, tnorm)
            if self.args.centreg != 0:
                matching_loss = matching_loss + self.args.centreg * poison_loss
            matching_loss.backward(retain_graph=self.retain)
            return matching_loss.detach().cpu(), prediction.detach().cpu()
        return closure