from nets.layers import *


class Atrous_256(nn.Module):
    """Reference: https://arxiv.org/pdf/1706.05587.pdf"""

    def __init__(self):
        super(Atrous_256, self).__init__()
        # MaxPool
        self.lhs_1x = ConvReLU(3, 32)
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(32, 64))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(64, 128))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(128, 256))  # w/h: 32

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(256, 256))

        # BilinearUp
        self.rhs_8x = UpConcat(256, 256)
        self.rhs_4x = UpConcat(256, 128)
        self.rhs_2x = UpConcat(128, 64)
        self.rhs_1x = UpConcat(64, 32)

        # Classify
        self.classify = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)

        bottom = self.bottom(lhs_8x)

        rhs_8x = self.rhs_8x(lhs_8x, bottom)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)
