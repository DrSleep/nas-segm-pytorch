"""Initialising Optimisers"""

import torch


def create_optimisers(
    optim_enc,
    optim_dec,
    lr_enc,
    lr_dec,
    mom_enc,
    mom_dec,
    wd_enc,
    wd_dec,
    param_enc,
    param_dec,
):
    """Create optimisers for encoder, decoder

    Args:
      optim_enc (str) : type of optimiser for encoder
      optim_dec (str) : type of optimiser for decoder
      lr_enc (float) : learning rate for encoder
      lr_dec (float) : learning rate for decoder
      mom_enc (float) : momentum for encoder
      mom_dec (float) : momentum for decoder
      wd_enc (float) : weight decay for encoder
      wd_dec (float) : weight decay for decoder
      param_enc (torch.parameters()) : encoder parameters
      param_dec (torch.parameters()) : decoder parameters

    Returns optim_enc, optim_dec (torch.optim)

    """
    if optim_enc == "sgd":
        optim_enc = torch.optim.SGD(
            param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc
        )
    elif optim_enc == "adam":
        optim_enc = torch.optim.Adam(param_enc, lr=lr_enc, weight_decay=wd_enc)
    else:
        raise ValueError("Unknown Encoder Optimiser: {}".format(optim_enc))

    if optim_dec == "sgd":
        optim_dec = torch.optim.SGD(
            param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec
        )
    elif optim_dec == "adam":
        optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec)
    else:
        raise ValueError("Unknown Decoder Optimiser: {}".format(optim_dec))
    return optim_enc, optim_dec
