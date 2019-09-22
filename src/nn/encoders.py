"""MobileNetv2 Encoder"""

import torch
import torch.nn as nn

from .layer_factory import InvertedResidual, conv_bn_relu6


__all__ = ['mbv2']


model_paths = {'mbv2_voc': './data/weights/mbv2_voc_rflw.ckpt'}


class MobileNetV2(nn.Module):
    """MobileNetv2"""
    def __init__(self, width_mult=1., default_rate=1):
        super(MobileNetV2, self).__init__()
        self.default_rate = default_rate
        # setting of inverted residual blocks
        self.inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # l2
            [6, 24, 2, 2], # l3
            [6, 32, 3, 2], # l4
            [6, 64, 4, 2], # l5
            [6, 96, 3, 1], # l6
            [6, 160, 3, 2], # l7
            [6, 320, 1, 1], # l8
        ]
        self.out_sizes = [24, 32, 96, 320] # l3,l4, l6, l8
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.layer1 = conv_bn_relu6(3, input_channel, 2)
        # building inverted residual blocks
        self.n_layers = len(self.inverted_residual_setting)
        for idx, (t, c, n, s) in enumerate(self.inverted_residual_setting):
            output_channel = int(c * width_mult)
            features = []
            for i in range(n):
                if i == 0:
                    features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
            setattr(self, 'layer{}'.format(idx + 2), nn.Sequential(*features))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32

        return l3, l4, l6, l8


def mbv2(pretrained=False, **kwargs):
    """Constructs a MobileNet-v2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        model.load_state_dict(
            torch.load(model_paths['mbv2_{}'.format(str(pretrained))]), strict=False)
    return model


def create_encoder(pretrained='voc'):
    """Create Encoder"""
    return mbv2(pretrained=pretrained)
