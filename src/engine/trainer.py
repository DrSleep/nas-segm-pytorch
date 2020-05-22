"""Training functions"""

import time
import logging
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from helpers.utils import AverageMeter, ctime, try_except

logger = logging.getLogger(__name__)


@try_except
def populate_task0(segmenter, train_loader, kd_net, n_train, do_kd=False):
    """Populate data for task0 - the outputs of encoder.

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      kd_net (nn.Module) : teacher network if any
      n_train (int) : how many samples to pre-compute
      do_kd (boolean) : whether to do knowledge distillation

    """
    Xy_train = defaultdict(list)
    segmenter.eval()
    # Populate Xy_train with encoder's outputs
    try:
        train_loader.dataset.set_stage("train")
    except AttributeError:
        train_loader.dataset.dataset.set_stage("train")
    train_loader.batch_sampler.batch_size = 1  # to not run out of memory
    with torch.no_grad():
        n_curr = 0
        for sample in train_loader:
            image = sample["image"].float().cuda()
            target = sample["mask"].float()
            enc_outputs = segmenter.module.encoder(image)
            for i, enc_output in enumerate(enc_outputs):
                Xy_train[i].extend(enc_output.unbind(0))
            Xy_train["y"].extend(
                nn.functional.interpolate(
                    target[:, None], size=enc_outputs[0].size()[2:], mode="nearest"
                )
                .long()
                .squeeze(dim=1)
                .cuda()
                .unbind(0)
            )
            if do_kd:
                kd_y = kd_net(image)
                Xy_train["kd_y"].extend(
                    nn.functional.interpolate(
                        kd_y, size=enc_outputs[0].size()[2:], mode="bilinear", align_corners=False
                    ).unbind(0)
                )
            n_curr += image.size(0)
            if n_curr >= n_train:
                # By default we are taking the size of the first encoder output
                # as our output size
                Xy_train["out_size"] = enc_outputs[0].size()[2:]
                logger.info(" Populated Xy_train, N = {}".format(n_curr))
                break
        # concat into a single tensor
        for k, v in Xy_train.items():
            if k != "out_size":
                Xy_train[k] = torch.stack(v)
    return Xy_train


@try_except
def train_task0(
    Xy_train,
    segmenter,
    optim_dec,
    epoch,
    segm_crit,
    kd_crit,
    batch_size,
    freeze_bn,
    do_kd,
    kd_coeff,
    dec_grad_clip,
    do_polyak,
    avg_param=None,
    polyak_decay=0.9,
    aux_weight=0,
):
    """Training task0 segmenter - only decoder

    Args:
      Xy_train (dict) : pre-computed data
      segmenter (nn.Module) : segmentation network
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current segm epoch
      segm_crit (nn.Loss) : segmentation criterion
      kd_crit (nn.Loss) : knowledge distillation criterion
      batch_size (int) : batch size used for training
      freeze_bn (bool) : whether to keep batch norm statistics intact
      do_kd (bool) : whether to do knowledge distillation
      kd_coeff (float) : loss coefficient for knowledge distillation
      dec_grad_clip (float) : clip decoder's parameters' norm to this value
      do_polyak (bool) : whether to do Polyak averaging
      avg_param : copy of parameters for Polyak averaging
      polyak_decay (float) : momentum for Polyak averaging
      aux_weight (float) : loss coefficient for auxiliary outputs

    """
    # Train
    n_examples = Xy_train[0].size(0)
    batch_size = min(batch_size, n_examples)
    n_passes = n_examples // batch_size
    indices = np.arange(n_examples)
    batch_time = AverageMeter()
    losses = AverageMeter()
    # Update BNs if not set otherwise
    segmenter.module.decoder.train()
    if freeze_bn:
        for m in segmenter.module.decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    np.random.shuffle(indices)
    for i in range(n_passes):
        start = time.time()
        train_idx = indices[(i * batch_size) : (i + 1) * batch_size]
        encoder_outputs = [
            Xy_train[key][train_idx] for key in Xy_train.keys() if key not in ['y', 'kd_y', 'out_size']
        ]
        output = segmenter.module.decoder(encoder_outputs)
        if isinstance(output, tuple):
            output, aux_outs = output
        # NOTE: Output size can change as some layers will not be connected
        output = nn.functional.interpolate(
            output, size=Xy_train["out_size"], mode="bilinear"
        )
        soft_output = nn.LogSoftmax()(output)
        # Compute loss and backpropagate
        loss = segm_crit(soft_output, Xy_train["y"][train_idx])
        if do_kd:
            kd_loss = kd_crit(output, Xy_train["kd_y"][train_idx])
            loss += kd_coeff * kd_loss

        if aux_weight > 0:
            for aux_out in aux_outs:
                aux_out = nn.Upsample(
                    size=Xy_train["out_size"],
                    mode="bilinear",
                    align_corners=False)(aux_out)
                aux_out = nn.LogSoftmax()(aux_out)
                # Compute loss and backpropagate
                loss += segm_crit(aux_out, Xy_train["y"][train_idx]) * aux_weight

        optim_dec.zero_grad()
        loss.backward()
        # Clip gradients' norm
        nn.utils.clip_grad_norm_(segmenter.module.decoder.parameters(), dec_grad_clip)
        optim_dec.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        if do_polyak:
            for p, avg_p in zip(segmenter.module.decoder.parameters(), avg_param):
                avg_p.mul_(polyak_decay).add_(1.0 - polyak_decay, p.data)

    logger.info(
        " [{}] Train epoch: {}\t"
        "Avg. Loss: {:.3f}\t"
        "Avg. Time: {:.3f}".format(ctime(), epoch, losses.avg, batch_time.avg)
    )


@try_except
def train_segmenter(
    segmenter,
    train_loader,
    optim_enc,
    optim_dec,
    epoch,
    segm_crit,
    freeze_bn,
    enc_grad_clip,
    dec_grad_clip,
    do_polyak,
    print_every=10,
    aux_weight=-1,
    avg_param=None,
    polyak_decay=0.99,
):
    """Training segmenter end-to-end.

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      optim_enc (optim) : optimiser for encoder
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current segmenter epoch
      segm_crit (nn.Loss) : segmentation criterion
      freeze_bn (bool) : whether to keep batch norm statistics intact
      enc_grad_clip (float) : clip encoder's parameters' norm to this value
      dec_grad_clip (float) : clip decoder's parameters' norm to this value
      do_polyak (bool) : whether to do Polyak averaging
      print_every (int) : how often to print out information
      aux_weight (float) : loss coefficient for auxiliary outputs
      avg_param : copy of parameters for Polyak averaging
      polyak_decay (float) : momentum for Polyak averaging

    """
    try:
        train_loader.dataset.set_stage("train")
    except AttributeError:
        train_loader.dataset.dataset.set_stage("train")  # for subset
    segmenter.train()
    if freeze_bn:
        for m in segmenter.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    for i, sample in enumerate(train_loader):
        start = time.time()
        image = sample["image"].float().cuda()
        target = sample["mask"].cuda()
        target_var = torch.autograd.Variable(target).float()
        # Compute output
        output = segmenter(image)
        if isinstance(output, tuple):
            output, aux_outputs = output
        target_var = nn.functional.interpolate(
            target_var[:, None], size=output.size()[2:], mode="nearest"
        ).long()[:, 0]
        soft_output = nn.LogSoftmax()(output)
        # Compute loss and backpropagate
        loss = segm_crit(soft_output, target_var)
        # Compute auxiliary loss
        if aux_weight > 0:
            for aux_out in aux_outs:
                aux_out = nn.Upsample(
                    size=target_var.size()[1:], mode="bilinear", align_corners=False
                )(aux_out)
                aux_out = nn.LogSoftmax()(aux_out)
                # Compute loss and backpropagate
                loss += segm_crit(aux_out, target_var) * aux_weight

        optim_enc.zero_grad()
        optim_dec.zero_grad()

        loss.backward()

        # Clip gradients' norm
        if enc_grad_clip > 0:
            nn.utils.clip_grad_norm_(
                segmenter.module.encoder.parameters(), enc_grad_clip
            )
        if dec_grad_clip > 0:
            nn.utils.clip_grad_norm_(
                segmenter.module.decoder.parameters(), dec_grad_clip
            )

        optim_enc.step()
        optim_dec.step()

        if do_polyak:
            for p, avg_p in zip(segmenter.parameters(), avg_param):
                avg_p.mul_(polyak_decay).add_(1.0 - polyak_decay, p.data)
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        if i % print_every == 0:
            logger.info(
                " [{}] Train epoch: {} [{}/{}]\t"
                "Avg. Loss: {:.3f}\t"
                "Avg. Time: {:.3f}".format(
                    ctime(), epoch, i, len(train_loader), losses.avg, batch_time.avg
                )
            )
