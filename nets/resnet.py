from nets.layers import *


class ResNeXt_128(nn.Module):
    def __init__(self):
        super(ResNeXt_128, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 32), ConvReLU(32, 32))

        self.lhs_1x = DownResNeXt(32, 64)
        self.lhs_2x = DownResNeXt(64, 128)
        self.lhs_4x = DownResNeXt(128, 256)
        self.lhs_8x = DownResNeXt(256, 256)

        self.rhs_8x = UpBlock(256, 256)
        self.rhs_4x = UpBlock(256, 128)
        self.rhs_2x = UpBlock(128, 64)
        self.rhs_1x = UpBlock(64, 32)

        self.classify = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, _ = self.lhs_8x(pool)

        _, up = self.rhs_8x(lhs_8x, pool)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return self.classify(rhs_1x)


class DualPath_128(nn.Module):
    def __init__(self):
        super(DualPath_128, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 32), ConvReLU(32, 32))

        self.lhs_1x = DownDual(32, 64)
        self.lhs_2x = DownDual(64, 128)
        self.lhs_4x = DownDual(128, 256)
        self.lhs_8x = DownDual(256, 256)

        self.rhs_8x = UpBlock(256, 256)
        self.rhs_4x = UpBlock(256, 128)
        self.rhs_2x = UpBlock(128, 64)
        self.rhs_1x = UpBlock(64, 32)

        self.classify = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, _ = self.lhs_8x(pool)

        _, up = self.rhs_8x(lhs_8x, pool)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return self.classify(rhs_1x)
