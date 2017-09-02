from nets.layers import *


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Conv
        self.lhs_1x = ConvReLU(3, 4)
        self.lhs_2x = nn.Sequential(ConvReLU(4, 16, stride=2), ConvReLU(16, 16))
        self.lhs_4x = nn.Sequential(ConvReLU(16, 64, stride=2), ConvX3(64, 64))
        self.lhs_8x = nn.Sequential(ConvReLU(64, 256, stride=2), ConvX3(256, 256))
        self.lhs_16x = nn.Sequential(ConvReLU(256, 1024, stride=2), ConvX3(1024, 1024))

        # Bottom 32x
        self.bottom = nn.Sequential(ConvReLU(1024, 1024, stride=2), ConvX3(1024, 1024))

        # ConvTranspose
        self.rhs_16x = UpConcat(1024, 1024)
        self.rhs_8x = UpConcat(1024, 256)
        self.rhs_4x = UpConcat(256, 64)
        self.rhs_2x = UpConcat(64, 16)
        self.rhs_1x = UpConcat(16, 4)

        # Classify
        self.classify = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)
        lhs_16x = self.lhs_16x(lhs_8x)

        bottom = self.bottom(lhs_16x)

        rhs_16x = self.rhs_16x(lhs_16x, bottom)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)
