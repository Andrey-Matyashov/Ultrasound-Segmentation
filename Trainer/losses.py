import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True, smooth=1):
        super().__init__()
        self.weight = weight
        self.size_average = size_average
        self.smooth = smooth

    def forward(self, inputs, targets):

        if inputs.dim() != targets.dim():
            targets = torch.nn.functional.on_hot(targets, num_classes=inputs.shape[1])
            targets = targets.permute(0, 4, 1, 2, 3).to(torch.float32)

        intersection = (inputs * targets).sum(dim=(2, 3, 4))
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + self.smooth)

        if self.weight is not None:
            dice = self.weight * dice

        if self.size_average:
            loss = 1. - dice.mean()
        else:
            loss = 1. - dice

        return loss


class LogCoshDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        return torch.log(
            torch.cosh(self.dice_loss.forward(inputs, targets))
        )