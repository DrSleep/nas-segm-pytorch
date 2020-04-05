"""Different custom layers"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, bias=False, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=bias,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


OPS = {
    "none": lambda C_in, C_out, stride, affine, repeats=1: Zero(C_in, C_out, stride),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, repeats=1: Pool(
        C_in, C_out, stride, repeats, ksize=3, mode="avg"
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, repeats=1: Pool(
        C_in, C_out, stride, repeats, ksize=3, mode="max"
    ),
    "global_average_pool": lambda C_in, C_out, stride, affine, repeats=1: GAPConv1x1(
        C_in, C_out
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, repeats=1: Skip(
        C_in, C_out, stride
    ),
    "sep_conv_3x3": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 3, stride, 1, affine=affine, repeats=repeats
    ),
    "sep_conv_5x5": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 5, stride, 2, affine=affine, repeats=repeats
    ),
    "sep_conv_7x7": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 7, stride, 3, affine=affine, repeats=repeats
    ),
    "dil_conv_3x3": lambda C_in, C_out, stride, affine, repeats=1: DilConv(
        C_in, C_out, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda C_in, C_out, stride, affine, repeats=1: DilConv(
        C_in, C_out, 5, stride, 4, 2, affine=affine
    ),
    "conv1x1": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv1x1(C_in, C_out, stride=stride),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    ),
    "conv3x3": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C_in, C_out, stride=stride),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    ),
    "conv3x3_dil3": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C_in, C_out, stride=stride, dilation=3),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    ),
    "conv3x3_dil12": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C_in, C_out, stride=stride, dilation=12),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    ),
    "sep_conv_3x3_dil3": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 3, stride, 3, affine=affine, dilation=3, repeats=repeats
    ),
    "sep_conv_5x5_dil6": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 5, stride, 12, affine=affine, dilation=6, repeats=repeats
    ),
}

AGG_OPS = {
    "psum": lambda C_in0, C_in1, C_out, affine, repeats=1, larger=True: ParamSum(
        C_in0, C_in1, C_out, larger
    ),
    "cat": lambda C_in0, C_in1, C_out, affine, repeats=1, larger=True: ConcatReduce(
        C_in0, C_in1, C_out, affine=affine, repeats=repeats, larger=larger
    ),
}


def conv_bn(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )


def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    )


def conv_bn_relu6(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn_relu6(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False,
            ),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class Pool(nn.Module):
    """Conv1x1 followed by pooling"""

    def __init__(self, C_in, C_out, stride, repeats, ksize, mode):
        super(Pool, self).__init__()
        self.conv1x1 = conv_bn(C_in, C_out, 1, 1, 0)
        if mode == "avg":
            self.pool = nn.AvgPool2d(
                ksize, stride=stride, padding=(ksize // 2), count_include_pad=False
            )
        elif mode == "max":
            self.pool = nn.MaxPool2d(ksize, stride=stride, padding=(ksize // 2))
        else:
            raise ValueError("Unknown pooling method {}".format(mode))

    def forward(self, x):
        x = self.conv1x1(x)
        return self.pool(x)


class GAPConv1x1(nn.Module):
    """Global Average Pooling + conv1x1"""

    def __init__(self, C_in, C_out):
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        out = nn.functional.interpolate(
            out, size=size, mode="bilinear", align_corners=False
        )
        return out


class DilConv(nn.Module):
    """Dilated separable convolution"""

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Separable convolution"""

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation=1,
        affine=True,
        repeats=1,
    ):
        super(SepConv, self).__init__()

        def basic_op(C_in, C_out):
            return nn.Sequential(
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=C_in,
                    bias=False,
                ),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False),
            )

        self.op = nn.Sequential()
        for idx in range(repeats):
            if idx > 0:
                C_in = C_out
            self.op.add_module("sep_{}".format(idx), basic_op(C_in, C_out))

    def forward(self, x):
        return self.op(x)


class Skip(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Skip, self).__init__()
        assert (C_out % C_in) == 0, "C_out must be divisible by C_in"
        self.repeats = (1, C_out // C_in, 1, 1)

    def forward(self, x):
        return x.repeat(self.repeats)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.stride = stride
        assert (C_out % C_in) == 0, "C_out must be divisible by C_in"
        self.repeats = (1, C_out // C_in, 1, 1)

    def forward(self, x):
        x = x.repeat(self.repeats)
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Adapt(nn.Module):
    """Adapt two inputs to the same size"""

    def __init__(self, C_in0, C_in1, C_out, larger):
        super(Adapt, self).__init__()
        self.C_in0 = C_in0
        self.C_in1 = C_in1
        self.C_out = C_out
        if self.C_in0 != self.C_out:
            self.conv0 = conv_bn_relu(C_in0, C_out, 1, 1, 0)
        if self.C_in1 != self.C_out:
            self.conv1 = conv_bn_relu(C_in1, C_out, 1, 1, 0)
        self.larger = larger

    def forward(self, x1, x2):
        if self.C_in0 != self.C_out:
            x1 = self.conv0(x1)
        if self.C_in1 != self.C_out:
            x2 = self.conv1(x2)
        return resize(x1, x2, self.larger)


def resize(x1, x2, largest=True):
    if largest:
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode="bilinear")(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode="bilinear")(x1)
        return x1, x2
    else:
        if x1.size()[2:] < x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode="bilinear")(x2)
        elif x1.size()[2:] > x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode="bilinear")(x1)
        return x1, x2


class ParamSum(nn.Module):
    def __init__(self, C_in0, C_in1, C_out, larger):
        super(ParamSum, self).__init__()
        self.adapt = Adapt(C_in0, C_in1, C_out, larger)
        self.a = nn.Parameter(torch.ones(C_out))
        self.b = nn.Parameter(torch.ones(C_out))

    def forward(self, x, y):
        bsize = x.size(0)
        x, y = self.adapt(x, y)
        return (
            self.a.expand(bsize, -1)[:, :, None, None] * x
            + self.b.expand(bsize, -1)[:, :, None, None] * y
        )


class ConcatReduce(nn.Module):
    def __init__(self, C_in0, C_in1, C_out, affine=True, repeats=1, larger=True):
        super(ConcatReduce, self).__init__()
        self.adapt = Adapt(C_in0, C_in1, C_out, larger)
        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(2 * C_out, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(2 * C_out, C_out, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x, y):
        x, y = self.adapt(x, y)
        z = torch.cat([x, y], 1)
        return self.conv1x1(z)
