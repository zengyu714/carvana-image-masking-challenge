import numpy as  np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from scipy.ndimage.interpolation import zoom

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=nn.ReLU()):
        """
        + Instantiate modules: conv-relu
        + Assign them as member variables
        """
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = relu

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvX3(nn.Module):
    """Three serial convs with a residual connection.

    Structure:
        inputs --> ① --> ② --> ③ --> outputs
             ↓ --> --> add --> ↑
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvX3, self).__init__()
        self.conv_1 = ConvReLU(in_channels, out_channels, kernel_size)
        self.conv_2 = ConvReLU(out_channels, out_channels, kernel_size)
        self.conv_3 = ConvReLU(out_channels, out_channels, kernel_size)

    def forward(self, x):
        return x + self.conv_3(self.conv_2(self.conv_1(x)))


class UpConcat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UpConcat, self).__init__()
        # Right hand side needs `Upsample`
        self.conv_rhs = ConvReLU(in_channels, out_channels)
        self.conv_lhs = ConvReLU(out_channels, out_channels)
        self.conv_fit = ConvReLU(out_channels * 2, out_channels)
        self.conv = ConvX3(out_channels, out_channels, kernel_size)
        self.rhs_up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.ReLU())

    def forward(self, lhs, rhs):
        lhs = self.conv_lhs(lhs)
        rhs = self.rhs_up(self.conv_rhs(rhs))

        cat = torch.cat((lhs, rhs), dim=1)
        return self.conv(self.conv_fit(cat))


class NLLLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(NLLLoss, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, outputs, targets):
        return self.nll_loss(F.log_softmax(outputs), targets.long().squeeze())


# TODO: weighted dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    def forward(self, pred, true):
        # `dim = 0` for Tensor result
        intersection = torch.sum(pred * true, 0)
        union = torch.sum(pred * pred, 0) + torch.sum(true * true, 0)
        dice = 2.0 * intersection / union
        return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7)


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal(m.weight)
            # init.constant(m.bias, 0.01)


def get_class_weight(data_loader):
    """To balance between foregrounds and background for NLL.

    Return:
        A Tensor consists [background_weight, *foreground_weight]
    """
    # Original label
    label = next(iter(data_loader))[-1].numpy()[:, 0]
    # Get elements in marks i.e., {0, 1}, {0, 10, 150, 250}...
    marks = np.unique(label)
    # The weight of each class
    weights = [(label == m).mean() for m in marks]
    # Inverse to rescale weights
    return 1 / torch.FloatTensor(weights)


def softmax_flat(outputs, targets):
    """
    Arguments:
        outputs: [batch_size, 2, height, width] (torch.cuda.FloatTensor)
        targets: [batch_size, 1, height, width] (torch.cuda.FloatTensor)

    Return:
        pred: FloatTensor {0.0, 1.0} with shape [batch_size, (1), height, width]
        true: FloatTensor {0.0, 1.0} with shape [batch_size, (1), height, width]
    """
    outputs = outputs.permute(0, 2, 3, 1).contiguous()
    prob = F.softmax(outputs.view(-1, 2))

    pred = prob.max(1)[1].float()
    true = targets.view(-1)
    return pred, true


def get_statictis(pred, true):
    """Compute dice among **positive** labels to avoid unbalance.

    Arguments:
        pred: FloatTensor {0.0, 1.0} with shape [batch_size, (1), height, width]
        true: FloatTensor {0.0, 1.0} with shape [batch_size, (1), height, width]

    Returns:
        tuple contains:
        + accuracy: (pred ∩ true) / true
        + dice overlap:  2 * pred ∩ true / (pred ∪ true) * 100
    """

    # Dice overlap
    pred = pred.eq(1).float().data  # FloatTensor 0.0 / 1.0
    true = true.data  # FloatTensor 0.0 / 1.0
    overlap = 2 * (pred * true).sum() / (pred.sum() + true.sum()) * 100

    # Accuracy
    acc = pred.eq(true).float().mean() * 100
    return acc, overlap


def pre_visdom(image, label, pred, zoom_factor=(1, 1, 0.25, 0.25)):
    """Prepare (optional zoom) for visualization in Visdom.

    Arguments:
        image: torch.cuda.FloatTensor of size [batch_size, 3, height, width]
        label: torch.cuda.FloatTensor of size [batch_size, 1, height, width]
        pred : torch.cuda.FloatTensor of size [batch_size * height * width]

    Returns:
        image: numpy.array of size [batch_size, 3, height, width]
        label: numpy.array of size [batch_size, 1, height, width]
        pred : numpy.array of size [batch_size, 1, height, width]
    """
    pred = pred.view_as(label).cpu().data.numpy()
    image, label = [item.cpu().data.numpy() for item in [image, label]]

    pred *= 255  # make label 1 to 255 for better visualization

    return [zoom(item, zoom_factor, order=1, prefilter=False) for item in [image, label, pred]]
