from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def multiclass_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probabilities = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    if not include_background:
        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

    reduce_dims = (0, 2, 3)
    intersection = (probabilities * target_one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice = multiclass_soft_dice_loss(logits, target, include_background=False)
        return self.ce_weight * ce + self.dice_weight * dice
