"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


model_paths = {
    'rf_lw152': 'https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download'
}
num_classes = 21


def maybe_download(model_name, model_dir=None, map_location=None):
    import os, sys
    import urllib.request
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_paths[model_name]
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetLW(nn.Module):

    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ResNetLW, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        return out


def rf_lw152(pretrained=False, num_classes=num_classes, **kwargs):
    """Constructs a ResNet-RefineNet-LW model.

    Args:
        pretrained (bool): If True, returns a pre-trained model.
    """
    model = ResNetLW(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    if pretrained:
        pretrained_dict = maybe_download('rf_lw152')
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        pretrained_dict.update(model_dict)
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict)
    return model
