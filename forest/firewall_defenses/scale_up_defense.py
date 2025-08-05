import torch
from torch.utils.data import Subset

class ScaleUp():
    name: str = 'scale up'

    def __init__(self, model, kettle, defense_ratio=0.2, scale_set=None, threshold=None, with_clean_data=True, use_pseudo_labels=True):
        if scale_set is None:
            scale_set = [3, 5, 7, 9, 11]
        if threshold is None:
            self.threshold = 0.5

        self.scale_set = scale_set
        self.model = model
        self.kettle = kettle
        self.with_clean_data = with_clean_data

        size = int(defense_ratio * len(self.kettle.validset))
        random_indices = torch.randperm(len(self.kettle.validset))[:size]
        self.clean_set = Subset(self.kettle.validset, indices=random_indices)
        self.clean_loader = torch.utils.data.DataLoader(self.clean_set, 
                                                        batch_size=64, 
                                                        shuffle=False, 
                                                        num_workers=4, 
                                                        pin_memory=True
                                                    )

        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)

        self.mean = None
        self.std = None
        if self.with_clean_data:
            self.init_spc_norm()
        
        self.use_pseudo_labels = use_pseudo_labels

    def detect(self, kettle):

        false_positives = 0
        clean_pred_correct_mask = []
        pred_poison_mask = []

        for idx, (clean_img, labels, _) in enumerate(kettle.validloader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch

            correct_pred_mask = torch.eq(labels, torch.argmax(self.model(clean_img), dim=1))
            clean_pred_correct_mask.append(correct_pred_mask)

            # evaluate the clean data
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(torch.clip(clean_img * scale, min=0.0, max=1.0))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)
            clean_pred = torch.argmax(self.model(clean_img), dim=1) # model prediction
            # compute the SPC Value
            spc_clean = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc_clean += scale_label == clean_pred
            spc_clean /= len(self.scale_set)

            if self.with_clean_data:
                spc_clean = (spc_clean - self.mean) / self.std
            
            pred_poison_mask.append(spc_clean > self.threshold)
            false_positives += (spc_clean > self.threshold).sum().item()

        clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
        pred_poison_mask = torch.cat(pred_poison_mask, dim=0)

        print("Clean Accuracy: %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(pred_poison_mask)].sum(), len(kettle.validloader.dataset),
                                                clean_pred_correct_mask[torch.logical_not(pred_poison_mask)].sum() / len(kettle.validloader.dataset)))

        print(f"False Positive Rate (FPR): {false_positives / len(kettle.validloader.dataset):.4f}")

        true_positives = 0
        poison_attack_success_mask = []
        pred_poison_mask = []
        source_class = kettle.poison_setup['source_class'][0]
        target_class = kettle.poison_setup['target_class']

        for idx, (trigger_img, labels, _) in enumerate(kettle.source_testloader[source_class]):
            trigger_img = trigger_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()

            preds = torch.argmax(self.model(trigger_img), dim=1)
            poison_attack_success_mask.append(torch.eq(preds, target_class))

            # evaluate the clean data
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(torch.clip(trigger_img * scale, min=0.0, max=1.0))

            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)
            poison_pred = torch.argmax(self.model(trigger_img), dim=1) # model prediction
            # compute the SPC Value
            spc_poison = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc_poison += scale_label == poison_pred
            spc_poison /= len(self.scale_set)

            if self.with_clean_data:
                spc_poison = (spc_poison - self.mean) / self.std

            pred_poison_mask.append(spc_poison > self.threshold)
            true_positives += (spc_poison > self.threshold).sum().item()
        
        poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)
        pred_poison_mask = torch.cat(pred_poison_mask, dim=0)

        print(f"ASR: %d/%d = %.6f" % (poison_attack_success_mask[torch.logical_not(pred_poison_mask)].sum(), len(kettle.source_testloader[source_class].dataset),
                                    poison_attack_success_mask[torch.logical_not(pred_poison_mask)].sum() / len(kettle.source_testloader[source_class].dataset)))
        print(f"True Positive Rate (TPR): {true_positives / len(kettle.source_testloader[source_class].dataset):.4f}")

        true_positive_rate = true_positives / len(kettle.source_testloader[source_class].dataset)
        false_positive_rate = false_positives / len(kettle.validloader.dataset)

        return true_positive_rate, false_positive_rate


    def init_spc_norm(self):
        total_spc = []
        for idx, (clean_img, labels, _) in enumerate(self.clean_loader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(torch.clip(clean_img * scale, min=0.0, max=1.0))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)
        self.mean = torch.mean(total_spc).item()
        self.std = torch.std(total_spc).item()