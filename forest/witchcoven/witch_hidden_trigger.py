"""Main class, holding information about models and training/testing routines."""

import torch
import torchvision
from PIL import Image
from ..utils import bypass_last_layer, cw_loss, write, total_variation_loss, upwind_tv
from ..consts import BENCHMARK, NON_BLOCKING, FINETUNING_LR_DROP, NORMALIZE
from forest.data import datasets
torch.backends.cudnn.benchmark = BENCHMARK
import random
from .witch_base import _Witch
from forest.data.datasets import normalization

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
                    poison_delta.detach()
                    
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize()
                        print('Model reinitialized to random seed.')
                    elif self.args.retrain_scenario == 'finetuning':
                        victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP)
                        print('Completely warmstart finetuning!')
                    
                    if self.args.scenario == 'transfer':
                        victim.load_feature_representation()
                        
                    victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
                    write('Retraining done!\n', self.args.output)
                    
                    # self.setup_featreg(victim, kettle, poison_delta)
                    self.compute_source_gradient(victim, kettle)

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

            closure = self._define_objective(inputs, labels, criterion, sources, source_class=kettle.poison_setup['source_class'][0], target_class=kettle.poison_setup['target_class'])
            loss, prediction = victim.compute(closure, None, None, None, delta_slice)

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
            """This function will be evaluated on all GPUs."""  # noqa: D401
            input_indcs, source_indcs = self._index_mapping(model, inputs, sources)
            
            if self.args.scenario != "transfer" or self.args.htbd_full_params == False:
                feature_model, last_layer = bypass_last_layer(model)
                new_inputs = torch.zeros_like(inputs)
                new_sources = torch.zeros_like(inputs) # Sources and inputs must be of the same shape
                for i in range(len(input_indcs)): 
                    new_inputs[i] = inputs[input_indcs[i]]
                    new_sources[i] = sources[source_indcs[i]]

                outputs = feature_model(new_inputs)
                outputs_sources = feature_model(new_sources)
                prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
                feature_loss = (outputs - outputs_sources).pow(2).mean(dim=1).sum()
                feature_loss.backward(retain_graph=self.retain)
                return feature_loss.detach().cpu(), prediction.detach().cpu()
            
            else:

                # Use the ResNet-specific feature extraction function
                outputs, outputs_sources, last_layer_outputs = self._extract_resnet_features(model, 
                                                                                            inputs, 
                                                                                            sources, 
                                                                                            input_indcs, 
                                                                                            source_indcs)
                
                # Compute prediction for tracking accuracy
                prediction = (last_layer_outputs.data.argmax(dim=1) == labels).sum()
                
                # Compute feature matching loss using the comprehensive embedding
                feature_loss = (outputs - outputs_sources).pow(2).mean(dim=1).sum()
                feature_loss.backward(retain_graph=self.retain)
                return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure
    
    def _extract_resnet_features(self, model, inputs, sources, input_indcs, source_indcs):
        """Extract features from key ResNet layers.
        
        This function specifically targets ResNet architecture components:
        1. Initial convolutional layer
        2. Each residual block's output
        3. Final feature representation before classification
        
        Args:
            model: The ResNet model
            inputs: Input batch
            sources: Source images
            input_indcs: Mapping indices for inputs
            source_indcs: Mapping indices for sources
            
        Returns:
            outputs: Concatenated features from ResNet blocks for inputs
            outputs_sources: Concatenated features from ResNet blocks for sources
            last_layer_outputs: Outputs from the final classification layer
        """
        if isinstance(model, torch.nn.DataParallel):
            base_model = model.module
        else:
            base_model = model

        new_inputs = torch.zeros_like(inputs)
        new_sources = torch.zeros_like(inputs)
        
        # Create the mapped inputs and sources
        for i in range(len(input_indcs)):
            new_inputs[i] = inputs[input_indcs[i]]
            new_sources[i] = sources[source_indcs[i]]
        
        # Store activations from relevant layers
        activations = {}
        
        # Define hook function to collect activations
        def get_activation(name):
            def hook(model, input, output):
                # For residual blocks, output might be a tuple
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output
            return hook
        
        # Register hooks for key ResNet components
        hooks = []
        
        # 1. Initial convolutional layer
        if hasattr(base_model, 'conv1'):
            hooks.append(base_model.conv1.register_forward_hook(get_activation('conv1')))
        
        # 2. Layer blocks (layer1, layer2, layer3, layer4 in ResNet)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(base_model, layer_name):
                # Hook the output of each layer block
                hooks.append(getattr(base_model, layer_name).register_forward_hook(
                    get_activation(layer_name)))
                
                # Also capture each block's output within the layer
                layer = getattr(base_model, layer_name)
                for i, block in enumerate(layer):
                    hooks.append(block.register_forward_hook(
                        get_activation(f'{layer_name}.{i}')))
        
        # 3. Final features before classification
        if hasattr(base_model, 'avgpool'):
            hooks.append(base_model.avgpool.register_forward_hook(get_activation('avgpool')))
        
        # Process input images
        input_features_list = []
        source_features_list = []
        
        # Forward pass for each input to collect activations
        for i, x in enumerate(new_inputs):
            # Clear previous activations
            activations.clear()
            
            # Forward pass
            with torch.no_grad():
                base_model(x.unsqueeze(0))
            
            # Process and store collected activations
            current_features = []
            for name in sorted(activations.keys()):
                feat = activations[name]
                
                # Global average pooling for convolutional outputs
                if len(feat.shape) == 4:  # [B, C, H, W]
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                
                # Flatten to 1D vector
                feat = feat.reshape(feat.size(0), -1)
                current_features.append(feat)
            
            # Concatenate all features for this input
            if current_features:
                all_features = torch.cat(current_features, dim=1)
                input_features_list.append(all_features)
        
        # Forward pass for each source to collect activations
        for i, x in enumerate(new_sources):
            # Clear previous activations
            activations.clear()
            
            # Forward pass
            with torch.no_grad():
                base_model(x.unsqueeze(0))
            
            # Process and store collected activations
            current_features = []
            for name in sorted(activations.keys()):
                feat = activations[name]
                
                # Global average pooling for convolutional outputs
                if len(feat.shape) == 4:  # [B, C, H, W]
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                
                # Flatten to 1D vector
                feat = feat.reshape(feat.size(0), -1)
                current_features.append(feat)
            
            # Concatenate all features for this source
            if current_features:
                all_features = torch.cat(current_features, dim=1)
                source_features_list.append(all_features)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Stack outputs to get tensors
        outputs = torch.cat(input_features_list, dim=0)
        outputs_sources = torch.cat(source_features_list, dim=0)
        
        # Get last layer outputs for prediction accuracy
        with torch.no_grad():
            last_layer_outputs = base_model(new_inputs)
        
        return outputs, outputs_sources, last_layer_outputs

    def _create_patch(self, patch_shape):
        temp_patch = 0.5 * torch.ones(3, patch_shape[1], patch_shape[2])
        patch = torch.bernoulli(temp_patch)
        return patch

    def patch_sources(self, kettle):
        if self.args.load_patch == '': # Path to the patch image
            patch = self._create_patch([3, int(self.args.patch_size), int(self.args.patch_size)])
        else:
            patch = Image.open(self.args.load_patch)
            totensor = torchvision.transforms.ToTensor()
            resize = torchvision.transforms.Resize(int(self.args.patch_size))
            patch = totensor(resize(patch))

        write(f"Shape of the patch: {patch.shape}", self.args.output)
        patch = (patch.to(**self.setup) - kettle.dm) / kettle.ds # Standardize the patch
        self.patch = patch.squeeze(0)

        # Add patch to source_testset
        if self.args.random_patch:
            write("Add patches to the source images randomly ...", self.args.output)
        else:
            write("Add patches to the source images on the bottom right ...", self.args.output)

        source_delta = []
        for idx, (source_img, label, image_id) in enumerate(kettle.source_testset):
            source_img = source_img.to(**self.setup)

            if self.args.random_patch:
                patch_x = random.randrange(0,source_img.shape[1] - self.patch.shape[1] + 1)
                patch_y = random.randrange(0,source_img.shape[2] - self.patch.shape[2] + 1)
            else:
                patch_x = source_img.shape[1] - self.patch.shape[1]
                patch_y = source_img.shape[2] - self.patch.shape[2]

            delta_slice = torch.zeros_like(source_img).squeeze(0)
            diff_patch = self.patch - source_img[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]]
            delta_slice[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]] = diff_patch
            source_delta.append(delta_slice.cpu())
        kettle.source_testset = datasets.Deltaset(kettle.source_testset, source_delta)

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
