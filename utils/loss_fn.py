import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss 
    """
    def __init__(self, label_smoothing=0.0, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        
        if self.label_smoothing > 0:
            return F.cross_entropy(
                pred, target, 
                weight=self.weight,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing
            )
        else:
            return F.cross_entropy(
                pred, target,
                weight=self.weight,
                reduction=self.reduction
            )


