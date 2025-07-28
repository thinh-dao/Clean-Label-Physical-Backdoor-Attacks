"""Main class, holding information about models and training/testing routines."""

import torch
import torchvision
import random
import numpy as np

from PIL import Image
from ..utils import bypass_last_layer, bypass_last_layer_deit, cw_loss, write, total_variation_loss, upwind_tv
from ..consts import BENCHMARK, NON_BLOCKING, FINETUNING_LR_DROP, NORMALIZE
from forest.data import datasets
from .witch_base import _Witch
from forest.data.datasets import normalization
torch.backends.cudnn.benchmark = BENCHMARK

class WitchHTBD(_Witch):
    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()
        dataloader = kettle.poisonloader

        validated_batch_size = max(min(kettle.args.pbatch, len(kettle.poisonset)), 1)

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

        for step in range(self.args.attackiter):
            source_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                sources, source_labels = [], []
                indcs = random.sample(list(range(len(kettle.source_trainset))), validated_batch_size)
                for i in indcs:
                    temp_source, temp_label, _ = kettle.source_trainset[i]
                    sources.append(temp_source)
                    # source_labels.append(temp_label)
                sources = torch.stack(sources)
                
                if NORMALIZE:
                    sources = normalization(sources)
                    
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
                    # victim.retrain(kettle, poison_delta, max_epoch=self.args.retrain_max_epoch)
                    print("Retrainig the base model at iteration {}".format(step))
                    
                    if self.args.retrain_reinit_seed:
                        seed = np.random.randint(0, 2**32 - 1)
                    else:
                        seed = None
                        
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize(seed=seed)
                        print('Model reinitialized to random seed.')
                    elif self.args.retrain_scenario == 'finetuning':
                        victim.reinitialize_last_layer(seed=seed, reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                        print('Completely warmstart finetuning!')
                    
                    if self.args.scenario == 'transfer':
                        victim.load_feature_representation()
                        
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

            closure = self._define_objective(inputs, labels, sources, delta_slice)
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

    def _define_objective(self, inputs, labels, sources, perturbations):
        """Implement the closure here."""
        def closure(model, optimizer):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            input_indcs, source_indcs = self._index_mapping(model, inputs, sources)
            
            if 'deit' in self.args.net[0]:
                feature_model, last_layer = bypass_last_layer_deit(model)
            else:
                feature_model, last_layer = bypass_last_layer(model)
                
            new_inputs = torch.zeros_like(inputs)
            new_sources = torch.zeros_like(inputs) # Sources and inputs must be of the same shape
            for i in range(len(input_indcs)): 
                new_inputs[i] = inputs[input_indcs[i]]
                new_sources[i] = sources[source_indcs[i]]

            outputs = feature_model(new_inputs)
            outputs_sources = feature_model(new_sources)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
            
            passenger_loss = (outputs - outputs_sources).pow(2).mean(dim=1).sum()            
            regularized_loss = self.get_regularized_loss(perturbations, tau=self.args.eps/255)
            
            attacker_loss = passenger_loss + self.args.vis_weight * regularized_loss
            attacker_loss.backward(retain_graph=self.retain)
            
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _index_mapping(self, model, inputs, temp_sources):
        '''Find the nearest source image for each input image'''
        with torch.no_grad():
            feature_model, last_layer = bypass_last_layer(model)
            feat_inputs = feature_model(inputs)
            feat_source = feature_model(temp_sources)
            dist = torch.cdist(feat_inputs, feat_source)
            input_indcs = []
            source_indcs = []
            for _ in range(feat_inputs.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero(as_tuple=False).squeeze()
                if len(dist_min_index[0].shape) != 0:
                    input_indcs.append(dist_min_index[0][0])
                    source_indcs.append(dist_min_index[0][1])
                    dist[dist_min_index[0][0], dist_min_index[0][1]] = 1e5
                else:
                    input_indcs.append(dist_min_index[0])
                    source_indcs.append(dist_min_index[1])
                    dist[dist_min_index[0], dist_min_index[1]] = 1e5
        return input_indcs, source_indcs
