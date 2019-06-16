"""Different custom layers"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, bias=False, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


OPS = {
    'none': lambda C, stride, affine, repeats=1: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, repeats=1: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, repeats=1: nn.MaxPool2d(3, stride=stride, padding=1),
    'global_average_pool': lambda C, stride, affine, repeats=1: GAPConv1x1(C, C),
    'skip_connect': lambda C, stride, affine, repeats=1: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 1, affine=affine, repeats=repeats),
    'sep_conv_5x5': lambda C, stride, affine, repeats=1: SepConv(C, C, 5, stride, 2, affine=affine, repeats=repeats),
    'sep_conv_7x7': lambda C, stride, affine, repeats=1: SepConv(C, C, 7, stride, 3, affine=affine, repeats=repeats),
    'dil_conv_3x3': lambda C, stride, affine, repeats=1: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, repeats=1: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, repeats=1: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
    'conv1x1': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv1x1(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=3),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil12': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=12),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'sep_conv_3x3_dil3': lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 3,
            affine=affine, dilation=3, repeats=repeats),
    'sep_conv_5x5_dil6': lambda C, stride, affine, repeats=1: SepConv(C, C, 5, stride, 12,
            affine=affine, dilation=6, repeats=repeats)
}


def conv_bn(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                  bias=False),
        nn.BatchNorm2d(C_out, affine=affine)
    )

def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                  bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    )

def conv_bn_relu6(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn_relu6(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
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
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class GAPConv1x1(nn.Module):
    """Global Average Pooling + conv1x1"""
    def __init__(self, C_in, C_out):
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        out = nn.functional.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out


class DilConv(nn.Module):
    """Dilated separable convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Separable convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, repeats=1):
        super(SepConv, self).__init__()
        basic_op = lambda: nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False))
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                basic_op())

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
