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


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=nn.ReLU()):
        """
        + Instantiate modules: conv-relu
        + Assign them as member variables
        """
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConcat, self).__init__()
        # Right hand side needs `Upsample`
        self.conv_fit = ConvBNReLU(in_channels + out_channels, out_channels)
        self.rhs_up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = nn.Sequential(ConvBNReLU(out_channels, out_channels), ConvBNReLU(out_channels, out_channels))

    def forward(self, lhs, rhs):
        rhs = self.rhs_up(rhs)
        cat = torch.cat((lhs, rhs), dim=1)
        return self.conv(self.conv_fit(cat))


class AtrousBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Structure:
        1 x 1 conv  --> --> --> ↓
        3 x 3 conv (rate=1) --> --> ↓
        3 x 3 conv (rate=3) --> --> --> concat --> 1 x 1 conv --> PReLU
        3 x 3 conv (rate=6) --> --> ↑
        3 x 3 conv (rate=9) --> ↑
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, cat=True):
        super(AtrousBlock, self).__init__()
        self.conv_1r1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_3r1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=1)
        self.conv_3r3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=3, dilation=3)
        self.conv_3r6 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=6, dilation=6)
        self.conv_3r9 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=9, dilation=9)

        self.cat = cat  # if cat: in_channels = 1*4 + 1 = 5, else sum: in_channels = 0*4 + 1 = 1
        self.conv = ConvReLU((cat * 4 + 1) * out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        bulk = [self.conv_1r1(x), self.conv_3r1(x), self.conv_3r3(x), self.conv_3r6(x), self.conv_3r9(x)]
        if self.cat:
            out = torch.cat(bulk, dim=1)
        else:
            out = sum(bulk)
        return self.conv(out)


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


def avg_class_weight(data_loader, avg_size=4):
    """Get average class weights.

    Return:
        A Tensor consists [background_weight, *foreground_weight]
    """
    weights = get_class_weight(data_loader)
    for i in range(avg_size - 1):
        weights += get_class_weight(data_loader)

    return weights.div(avg_size)


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


def pre_visdom(image, label, pred, show_size=256):
    """Prepare (optional zoom) for visualization in Visdom.

    Arguments:
        image: torch.cuda.FloatTensor of size [batch_size, 3, height, width]
        label: torch.cuda.FloatTensor of size [batch_size, 1, height, width]
        pred : torch.cuda.FloatTensor of size [batch_size * height * width]
        show_size: show images with size [batch_size, 3, height, width] in visdom

    Returns:
        image: numpy.array of size [batch_size, 3, height, width]
        label: numpy.array of size [batch_size, 1, height, width]
        pred : numpy.array of size [batch_size, 1, height, width]
    """
    pred = pred.view_as(label).cpu().data.numpy()
    pred *= 255  # make label 1 to 255 for better visualization

    image, label = [item.cpu().data.numpy() for item in [image, label]]

    zoom_factor = np.append([1, 1], np.divide(show_size, image.shape[-2:]))
    return [zoom(item, zoom_factor, order=1, prefilter=False) for item in [image, label, pred]]
