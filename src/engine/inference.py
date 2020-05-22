"""Validation function"""

import time
import logging
import numpy as np

import torch
from torch import nn

from helpers.miou_utils import compute_iu, compute_ius_accs, fast_cm
from helpers.utils import try_except


logger = logging.getLogger(__name__)


@try_except
def validate(segmenter, val_loader, epoch, epoch2, num_classes=-1, print_every=10):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current search epoch
      epoch2 (int) : current segm. training epoch
      num_classes (int) : number of segmentation classes
      print_every (int) : how often to print out information

    Returns:
      Reward (float)

    """
    try:
        val_loader.dataset.set_stage("val")
    except AttributeError:
        val_loader.dataset.dataset.set_stage("val")  # for subset
    segmenter.eval()

    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample["image"]
            target = sample["mask"]
            input_var = torch.autograd.Variable(image).float().cuda()
            # Compute output
            output = segmenter(input_var)
            if isinstance(output, tuple):
                output, _ = output
            output = nn.Upsample(
                size=target.size()[1:], mode="bilinear", align_corners=False
            )(output)
            # Compute IoU
            output = output.data.cpu().numpy().argmax(axis=1).astype(np.uint8)
            gt = target.data.cpu().numpy().astype(np.uint8)
            # Ignore every class index larger than the number of classes
            gt_idx = gt < num_classes
            cm += fast_cm(output[gt_idx], gt[gt_idx], num_classes)

            if i % print_every == 0:
                logger.info(
                    " Val epoch: {} [{}/{}]\t"
                    "Mean IoU: {:.3f}".format(
                        epoch, i, len(val_loader), compute_iu(cm).mean()
                    )
                )
    ious, n_pixels, accs = compute_ius_accs(cm)
    logger.info(" IoUs: {}, accs: {}".format(ious, accs))
    # IoU by default is 1, so we ignore all the unchanged classes
    # +1 - since we ignore background
    present_ind = np.array([idx + 1 for idx, iu in enumerate(ious[1:]) if iu != 1.0])
    present_ious = ious[present_ind]
    present_pixels = n_pixels[present_ind]
    miou = np.mean(present_ious)
    macc = np.mean(accs[present_ind])
    mfwiou = np.sum(present_ious * present_pixels) / np.sum(present_pixels)
    metrics = [miou, macc, mfwiou]
    reward = np.prod(metrics) ** (1.0 / len(metrics))
    info = (
        " Val epoch: {}/{}\tMean IoU: {:.3f}\tMean FW-IoU: {:.3f}\t"
        "Mean Acc: {:.3f}\tReward: {:.3f}"
    ).format(epoch, epoch2, miou, mfwiou, macc, reward)
    logger.info(info)
    return reward
