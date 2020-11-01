import torch
import torch.nn.functional as F
import torch.nn as nn

def simple_dice_loss3D(pred, labels):
    '''
    https://github.com/Achilleas/pytorch-mri-segmentation-3D/blob/master/utils/lossF.py
    '''
    intersect = 2*(pred * labels).sum(0)[1] + 0.02
    ref = pred.pow(2).sum(0)[1] + 0.01
    seg = labels.pow(2).sum(0)[1] + 0.01
    return 1 - ((intersect / (ref + seg)).sum())

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class WeightedCrossEntropyLoss(nn.Module):
    """
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights