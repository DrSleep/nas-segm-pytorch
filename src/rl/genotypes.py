"""List of operations"""

from collections import namedtuple

Genotype = namedtuple("Genotype", "encoder decoder")

# CVPR 2019
OP_NAMES = [
    "conv1x1",
    "conv3x3",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "global_average_pool",
    "conv3x3_dil3",
    "conv3x3_dil12",
    "sep_conv_3x3_dil3",
    "sep_conv_5x5_dil6",
    "skip_connect",
    "none",
]

# WACV 2020
OP_NAMES_WACV = [
    "sep_conv_3x3",
    "sep_conv_5x5",
    "global_average_pool",
    "max_pool_3x3",
    "sep_conv_5x5_dil6",
    "skip_connect",
]

AGG_OP_NAMES = [
    "psum",
    "cat",
]
