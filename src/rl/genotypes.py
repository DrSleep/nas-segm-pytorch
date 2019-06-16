"""List of operations"""

from collections import namedtuple

Genotype = namedtuple('Genotype', 'encoder decoder')

OP_NAMES = [
    'conv1x1',
    'conv3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'global_average_pool',
    'conv3x3_dil3',
    'conv3x3_dil12',
    'sep_conv_3x3_dil3',
    'sep_conv_5x5_dil6',
    'skip_connect',
    'none'
]
