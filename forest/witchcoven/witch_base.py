"""Main class, holding information about models and training/testing routines."""

import torch

from ..data.datasets import PoisonSet
from ..utils import cw_loss, write, total_variation_loss, upwind_tv
from ..consts import NON_BLOCKING, BENCHMARK, NORMALIZE, PIN_MEMORY
torch.backends.cudnn.benchmark = BENCHMARK
from ..victims.victim_single import _VictimSingle
from ..victims.batched_attacks import construct_attack
from forest.data.datasets import normalization
from typing import Tuple

class _Witch():
    """Brew poison with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _brew() method itself.
    
    Attributes:
        -args: Arguments object.
        -retain: Retain graph - for ensemble models
        -stat_optimal_loss: Optimal loss for the best poison found.
        
        # Brewing attributes
        -sources_train: Source training set.
        -sources_train_true_classes: True classes of source training set.
        -sources_train_target_classes: Target classes of source training set.
        -source_grad: Source gradients.
        -source_gnorm: Source gradient norm.
        -source_clean_grad: Source clean gradients.
        -tau0: poisoning step_size
    
    Methods:
        -_initialize_brew: Initialize brewing attributes 
        -brew: Inialize poison delta and start brewing poisons.
        -_brew: Iterative poisoning routine.
        -_batched_step: Inner-loop optimization to get grads of perturbations.
        -_define_objective: Return the objective function for poisoning.
    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.retain = True if self.args.ensemble > 1 else False
        self.stat_optimal_loss = None

    def run_step(self, kettle, poison_delta, epoch_num, model, defs, optimizer, scheduler):
        """
        Run a single training step (epoch) with optional poisoning and defenses.
        """
        # --- Setup phase ---
        epoch_loss, total_preds, correct_preds = 0, 0, 0
        loss_fn=torch.nn.CrossEntropyLoss(reduction='mean')
        
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

            # Transfer to GPU
            inputs = inputs.to(**kettle.setup)
            labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
            
            # Add data augmentation if configured
            if defs.augmentations:
                inputs = kettle.augment(inputs)

            # Normalize inputs if required
            if NORMALIZE:
                inputs = normalization(inputs)
                
            # Process inputs
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                
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
    
    def validation(self, model, clean_testloader, source_testloader, target_class):
        """Evaluate model performance on clean and poisoned data."""
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
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
    
    def brew(self, victim, kettle):
        """Run generalized iterative routine."""
        self._initialize_brew(victim, kettle)
        poisons, scores = [], torch.ones(self.args.restarts) * 10_000
            
        for trial in range(self.args.restarts):
            write("Poisoning number {}".format(trial), self.args.output)
            poison_delta, source_losses = self._run_trial(victim, kettle) # Poisoning
            scores[trial] = source_losses
            poisons.append(poison_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        write(f'Poisons with minimal passenger loss {self.stat_optimal_loss:6.4e} selected.\n', self.args.output)
        poison_delta = poisons[optimal_score]

        return poison_delta # Return the best poison perturbation amont the restarts
            
    def get_regularized_loss(self, perturbations, tau):
        # ==========================================================
        # 1)  Simple Lp-norm penalties (no spatial smoothness term)
        # ==========================================================
        if self.args.visreg == 'l1':
            regularized_loss = torch.mean(torch.abs(perturbations))

        elif self.args.visreg == 'l2':
            regularized_loss = torch.mean(torch.linalg.matrix_norm(perturbations).pow(2))

        # ==========================================================
        # 2)  Pure Total-Variation penalties (isotropic & up-wind)
        # ==========================================================
        elif self.args.visreg == 'UTV':
            regularized_loss = upwind_tv(perturbations)

        # ==========================================================
        # 3)  Mixed TV + hard Lp penalties
        # ==========================================================
        elif self.args.visreg == 'TV+l1':
            regularized_loss = torch.mean(torch.abs(perturbations)) + total_variation_loss(perturbations)

        elif self.args.visreg == 'TV+l2':
            regularized_loss = torch.mean(torch.linalg.matrix_norm(perturbations).pow(2)) + total_variation_loss(perturbations)

        # ==========================================================
        # 4)  "Soft" norm constraints (hinge on L2 or L∞ budget τ)
        # ==========================================================
        elif self.args.visreg == 'soft_l2':
            # CW-style L2 penalty: λ · max(0, ||δ||₂ - ε)² per sample
            B = perturbations.shape[0]
            l2_norms = perturbations.view(B, -1).norm(p=2, dim=1)        # ‖δ‖₂ per sample
            regularized_loss = ((l2_norms - tau).clamp(min=0).pow(2)).mean()

        elif self.args.visreg == 'soft_linf':
            # CW-style L∞ penalty: λ · max(0, ||δ||∞ - ε)² per sample
            B = perturbations.shape[0]
            l_inf_norms = torch.max(torch.abs(perturbations.view(B, -1)), dim=1)[0]  # ‖δ‖∞ per sample
            regularized_loss = ((l_inf_norms - tau).clamp(min=0).pow(2)).mean()


        elif self.args.visreg == 'TV+soft_linf':
            hinge = (perturbations.abs() - tau).clamp(min=0).mean()
            tv    = total_variation_loss(perturbations)
            regularized_loss = hinge + tv                                # optionally scale tv with λ_tv

        elif self.args.visreg == 'UTV+soft_linf':
            hinge = (perturbations.abs() - tau).clamp(min=0).mean()
            utv   = upwind_tv(perturbations)
            regularized_loss = hinge + utv

        # ==========================================================
        # 5)  Soft-L∞ budget + TV / UTV smoothing
        # ==========================================================
        elif self.args.visreg == 'TV+soft_l2':
            B     = perturbations.size(0)
            norm  = perturbations.view(B, -1).norm(p=2, dim=1)
            hinge = ((norm - tau).clamp(min=0).pow(2)).mean()
            tv    = total_variation_loss(perturbations)
            regularized_loss = hinge + tv

        elif self.args.visreg == 'UTV+soft_l2':
            B     = perturbations.size(0)
            norm  = perturbations.view(B, -1).norm(p=2, dim=1)
            hinge = ((norm - tau).clamp(min=0).pow(2)).mean()
            utv   = upwind_tv(perturbations)
            regularized_loss = hinge + utv

        # ==========================================================
        # 7)  Catch-all for typos or unsupported flags
        # ==========================================================
        else:
            if self.args.visreg is not None:
                raise ValueError(f"{self.args.visreg} regularization not defined")
            regularized_loss = torch.tensor(0)

        return regularized_loss

    def _initialize_brew(self, victim, kettle):        
        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.pbatch is None:
            self.args.pbatch = len(kettle.poisonset)
            
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / self.args.batch_size) / self.args.ensemble
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / self.args.batch_size) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / self.args.batch_size) / self.args.ensemble

        if self.args.sample_gradient:
            self.tau0 *= self.args.ensemble
            
        # Prepare adversarial attacker if necessary:
        if self.args.padversarial is not None:
            if not isinstance(victim, _VictimSingle):
                raise ValueError('Test variant only implemented for single victims atm...')
            attack = dict(type=self.args.padversarial, strength=self.args.defense_strength)
            self.attacker = construct_attack(attack, victim.model, victim.loss_fn, kettle.dm, kettle.ds,
                                             tau=kettle.args.tau, eps=kettle.args.eps, init='randn', optim='signAdam',
                                             num_classes=len(kettle.class_names), setup=kettle.setup)

        # Prepare adaptive mixing to dilute with additional clean data
        if self.args.pmix:
            self.extra_data = iter(kettle.trainloader)

    def _run_trial(self, victim, kettle) -> Tuple[float, torch.Tensor]:
        """Run a single trial. Perform one round of poisoning.
        Args:
            victim - model wrapper
            kettle - dataset wrapper
        Returns:
            poison_delta: perturbations
            source_losses: Attacker optimization loss
        """
        pass

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle):
        """Take a step toward minimizing the current poison loss."""
        pass

    def _define_objective():
        """Implement the closure here."""
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()

    def _pgd_step(self, delta_slice, poison_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args.eps /
                                                   ds / 255), -self.args.eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   poison_imgs), -dm / ds - poison_imgs)
        return delta_slice
