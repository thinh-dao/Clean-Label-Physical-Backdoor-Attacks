"""
IBD-PSC: Input-level Backdoor Detection via Parameter-scaled Consistency
"""

import copy
from typing import Tuple, List, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm

from .base import BaseFirewallDefense


class IBD_PSC(BaseFirewallDefense):
    """
    IBD-PSC defense implementation.
    
    This method identifies backdoor samples by amplifying BatchNorm parameters 
    and measuring prediction consistency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n: int = 5,
        xi: float = 0.6,
        T: float = 0.9,
        scale: float = 1.5,
        valset: Optional[Any] = None,
        seed: int = 666,
        deterministic: bool = False,
        **kwargs
    ):
        """
        Initialize IBD-PSC defense.
        
        Args:
            model: The neural network model to defend
            n: Number of parameter-amplified versions to create
            xi: Error rate threshold for parameter scaling
            T: Detection threshold for PSC score
            scale: Scale factor for BatchNorm parameters
            valset: Validation dataset for threshold selection
            seed: Random seed
            deterministic: Whether to use deterministic algorithms
        """
        super().__init__(model, **kwargs)
        self.n = n
        self.xi = xi
        self.T = T
        self.scale = scale
        self.valset = valset
        
        # Initialize layer indices and start index
        layer_num = self._count_bn_layers()
        self.sorted_indices = list(reversed(range(layer_num)))
        self.start_index = self._find_prob_start()

    def scan(self, kettle: Any) -> Tuple[float, float]:
        """
        Scan for backdoor samples using IBD-PSC method.
        
        Args:
            kettle: Data kettle containing datasets and configuration
            
        Returns:
            Tuple of (true_positive_rate, false_positive_rate)
        """
        print(f'start_index: {self.start_index}')

        testset = kettle.validset
        source_class = kettle.poison_setup['source_class'][0]
        poisoned_testset = kettle.source_testset[source_class]
        print(f"Poisoned test set size: {len(poisoned_testset)}")
        
        benign_psc = self._test_dataset(testset)
        poison_psc = self._test_dataset(poisoned_testset)

        true_positive_rate = (poison_psc >= self.T).sum().item() / len(poison_psc)
        false_positive_rate = (benign_psc >= self.T).sum().item() / len(benign_psc)

        print(f"True Positive Rate: {true_positive_rate:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")
        
        return true_positive_rate, false_positive_rate

    def detect(self, dataset: Any) -> torch.Tensor:
        """
        Detect backdoor samples in a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Boolean tensor indicating detected backdoor samples
        """
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                return self._detect_batch(imgs)

    def _count_bn_layers(self) -> int:
        """Count the number of BatchNorm2d layers in the model."""
        layer_num = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                layer_num += 1
        return layer_num

    def _scale_bn_parameters(self, index_bn: List[int], scale: float = 1.5) -> nn.Module:
        """
        Create a copy of the model with scaled BatchNorm parameters.
        
        Args:
            index_bn: Indices of BatchNorm layers to scale
            scale: Scale factor for parameters
            
        Returns:
            Model copy with scaled parameters
        """
        copy_model = copy.deepcopy(self.model)
        index = -1
        
        for name, module in copy_model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                index += 1
                if index in index_bn:
                    module.weight.data *= scale
                    module.bias.data *= scale
        
        return copy_model

    def _find_prob_start(self) -> int:
        """
        Find the starting index for parameter scaling based on validation accuracy.
        
        Returns:
            Starting layer index for scaling
        """
        if self.valset is None:
            return 1
            
        val_loader = torch.utils.data.DataLoader(self.valset, batch_size=128, shuffle=False)
        layer_num = len(self.sorted_indices)
        
        for layer_index in range(1, layer_num):
            layers = self.sorted_indices[:layer_index]
            smodel = self._scale_bn_parameters(layers, scale=self.scale)
            smodel.cuda()
            smodel.eval()
            
            total_num = 0
            clean_wrong = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    clean_img, labels = batch[0], batch[1]
                    clean_img = clean_img.cuda()
                    
                    clean_logits = smodel(clean_img).detach().cpu()
                    clean_pred = torch.argmax(clean_logits, dim=1)
                    
                    clean_wrong += torch.sum(labels != clean_pred)
                    total_num += labels.shape[0]
                
                wrong_acc = clean_wrong / total_num
                if wrong_acc > self.xi:
                    return layer_index
        
        return layer_num - 1

    def _test_dataset(self, dataset: Any) -> torch.Tensor:
        """
        Test a dataset and compute PSC scores.
        
        Args:
            dataset: Dataset to test
            
        Returns:
            PSC scores for all samples
        """
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        all_psc_score = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing PSC scores"):
                imgs, labels = batch[0], batch[1]
                imgs = imgs.cuda()
                original_pred = torch.argmax(self.model(imgs), dim=1)

                psc_score = torch.zeros(labels.shape)
                scale_count = 0
                
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers = self.sorted_indices[:layer_index + 1]
                    smodel = self._scale_bn_parameters(layers, scale=self.scale)
                    scale_count += 1
                    smodel.eval()
                    
                    logits = smodel(imgs).detach().cpu()
                    softmax_logits = torch.nn.functional.softmax(logits, dim=1)
                    psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred.cpu()]

                psc_score /= scale_count
                all_psc_score.append(psc_score)

        return torch.cat(all_psc_score, dim=0)

    def _detect_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Detect backdoor samples in a batch.
        
        Args:
            inputs: Input batch
            
        Returns:
            Boolean tensor indicating detected samples
        """
        inputs = inputs.cuda()
        self.model.eval()
        original_pred = torch.argmax(self.model(inputs), dim=1)

        psc_score = torch.zeros(inputs.size(0))
        scale_count = 0
        
        for layer_index in range(self.start_index, self.start_index + self.n):
            layers = self.sorted_indices[:layer_index + 1]
            smodel = self._scale_bn_parameters(layers, scale=self.scale)
            scale_count += 1
            smodel.eval()
            
            logits = smodel(inputs).detach().cpu()
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred.cpu()]

        psc_score /= scale_count
        return psc_score >= self.T
