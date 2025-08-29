from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

class RDLCE_Loss(nn.Module):
    def __init__(self, ce_weight=None, ce_reduction="mean", ce_ignore_index=255):
        super(RDLCE_Loss, self).__init__()

        self.ce_weight = ce_weight
        self.reduction = ce_reduction
        self.ignore_index = ce_ignore_index

        weight = None

        if self.ce_weight:
            weight = self.ce_weight
            weight = torch.FloatTensor(weight).cuda()

        self.reduction = 'mean'

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

class ADLCL_Loss(nn.Module, ABC):
    def __init__(self, ce_ignore_index=255):
        super(ADLCL_Loss, self).__init__()
        self.ignore_label = ce_ignore_index

    def forward(self, contrast_logits, contrast_target):
        adl_cl_loss = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)
        return adl_cl_loss


class ADLCOM_Loss(nn.Module, ABC):
    def __init__(self, ce_ignore_index=255):
        super(ADLCOM_Loss, self).__init__()
        self.ignore_label = ce_ignore_index

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]
        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        adl_com_loss = (1 - logits).pow(2).mean()
        return adl_com_loss


# seg_loss
class DistanceOptimizationLoss(nn.Module, ABC):
    def __init__(self, ce_weight=None, ce_reduction="elementwise_mean", ce_ignore_index=255, adl_cl_loss_weight=0.01, adl_com_loss_weight=0.01):
        super(DistanceOptimizationLoss, self).__init__()

        self.ce_weight = ce_weight
        self.ce_reduction = ce_reduction
        self.ignore_index = ce_ignore_index

        self.adl_cl_loss_weight = adl_cl_loss_weight
        self.adl_com_loss_weight = adl_com_loss_weight

        self.rdl_ce_criterion = RDLCE_Loss(ce_weight=self.ce_weight, ce_reduction=self.ce_reduction, ce_ignore_index=self.ignore_index)
        self.adl_cl_criterion = ADLCL_Loss(ce_ignore_index=self.ignore_index)
        self.adl_com_criterion = ADLCOM_Loss(ce_ignore_index=self.ignore_index)

    # seg_out
    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']  # out_seg
            contrast_logits = preds['logits']
            contrast_target = preds['target']

            adl_cl_loss = self.adl_cl_criterion(contrast_logits, contrast_target)
            adl_com_loss = self.adl_com_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.rdl_ce_criterion(pred, target)
            return loss + self.adl_cl_loss_weight * adl_cl_loss + self.adl_com_loss_weight * adl_com_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.rdl_ce_criterion(pred, target)
        return loss