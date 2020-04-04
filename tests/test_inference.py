import sys
sys.path.append('./src/')
from functools import partial

import numpy as np
import pickle
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.encoders import mbv2
from nn.micro_decoders import MicroDecoder, TemplateDecoder
from utils.helpers import prepare_img
from utils.model_zoo import load_url


MODELS_INFO = {
    'cvpr': {
        'arch0': (
            [[8, [0, 0, 5, 2], [0, 2, 8, 8], [0, 5, 1, 4]], [[3, 3], [3, 2], [3, 0]]],
            {
                'segm': ('segm-23378522.pth', 'https://cloudstor.aarnet.edu.au/plus/s/ZPXVGGgyxekvdAn/download'),
                'depth': ('depth-81f185f7.pth', 'https://cloudstor.aarnet.edu.au/plus/s/ttAlLJqDr30v1sk/download'),
            },
        ),
        'arch1': (
            [[2, [1, 0, 3, 6], [0, 1, 2, 8], [2, 0, 6, 1]], [[2, 3], [3, 1], [4, 4]]],
            {
                'segm': ('segm-12f78b21.pth', 'https://cloudstor.aarnet.edu.au/plus/s/svINhJX7IsvjCaD/download'),
                'depth': ('depth-7965abcb.pth', 'https://cloudstor.aarnet.edu.au/plus/s/5GeBZwW97eyODF7/download'),
            },
        ),
        'arch2': (
            [[5, [0, 0, 4, 1], [3, 2, 0, 1], [5, 6, 5, 0]], [[1, 3], [4, 3], [2, 2]]],
            {
                'segm': ('segm-8f00fc4d.pth', 'https://cloudstor.aarnet.edu.au/plus/s/9b8zVuaowe6ZtAN/download'),
                'depth': ('depth-a2f8f6d6.pth', 'https://cloudstor.aarnet.edu.au/plus/s/75pvpkqhJIv4aw4/download'),
            },
        ),
        },
    'wacv': {
        'arch0': (
            [[[3, 0, 1], [4, 1, 1], [3, 1, 1]],
             [[0, 1, 0, 0, 1],
              [2, 1, 2, 1, 0],
              [3, 1, 1, 1, 0],
              [1, 1, 2, 0, 0],
              [3, 0, 2, 0, 0],
              [5, 3, 2, 1, 0],
              [0, 5, 0, 1, 0]]],
            {
                'cs': ('wacv_cs-2dcef44e.pth', 'https://cloudstor.aarnet.edu.au/plus/s/EhSSQUPeyKvL5Zk/download'),
                'cv': ('wacv_cv-166a860b.pth', 'https://cloudstor.aarnet.edu.au/plus/s/K4sMKbFcDKNr53F/download'),
            },
        ),
        'arch1': (
            [[[1, 1, 0], [1, 3, 0], [3, 4, 0]],
             [[1, 1, 0, 0, 0],
              [0, 1, 1, 1, 1],
              [3, 1, 2, 3, 0],
              [3, 0, 2, 2, 0],
              [0, 1, 2, 0, 0],
              [2, 1, 1, 3, 0],
              [4, 0, 2, 2, 0]]],
            {
                'cs': ('wacv_cs-2bcc2420.pth', 'https://cloudstor.aarnet.edu.au/plus/s/pnGC9DB03uLIXhW/download'),
                'cv': ('wacv_cv-46afcbcb.pth', 'https://cloudstor.aarnet.edu.au/plus/s/CfvQtC73RT0oMIQ/download'),
            },
        ),
    },
}

AGG_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TASK_CONFIG = {
    'cvpr': (mbv2, MicroDecoder, {'segm': [21, 'voc'], 'depth': [1, 'nyud']}),
    'wacv': (partial(mbv2, return_layers=[1, 2]), TemplateDecoder, {'cs': [19, 'wacv_cs'], 'cv': [11, 'wacv_cv']}),
}
REPEATS = 2


class EncoderDecoder(nn.Module):
    """Create Segmenter"""
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


def img_path_to_tensor(img_path):
    img = prepare_img(np.array(Image.open(img_path)))
    return torch.tensor(img.transpose(2, 0, 1)[None]).float().to(DEVICE)


def run_tests(paper, task, postproc_fn):
    Encoder, Decoder, task_to_num_classes_and_test_suit = TASK_CONFIG[paper]
    num_classes, test_suit = task_to_num_classes_and_test_suit[task]
    models = {}
    dec_fn = partial(Decoder,
                     num_classes=num_classes,
                     agg_size=AGG_SIZE,
                     repeats=REPEATS)
    for arch, config_and_links in MODELS_INFO[paper].items():
        structure, links = config_and_links
        filename, url = links[task]
        enc = Encoder(pretrained=False)
        dec = dec_fn(config=structure, inp_sizes=enc.out_sizes)
        model = EncoderDecoder(enc, dec).to(DEVICE).eval()
        model.load_state_dict(load_url((arch + '_' + filename, url), map_location=DEVICE), strict=False)
        models[arch + '_' + task] = model
    with open('./tests/precomputed/test_{}_{}.ckpt'.format(test_suit, DEVICE), 'rb') as f:
        tests = pickle.load(f)
    for img_path, models_and_preds in tests.items():
        input_tensor = img_path_to_tensor(img_path)
        for model_name, model in models.items():
            with torch.no_grad():
                pred = postproc_fn(model(input_tensor))
                assert np.allclose(pred, models_and_preds[model_name])


def segm_postprocessing(outputs):
    pred = outputs[0].squeeze().data.cpu().numpy()
    return pred.argmax(axis=0).astype(np.uint8)


def depth_postprocessing(outputs):
    pred = outputs[0].squeeze().data.cpu().numpy()
    return pred.astype(np.float32)


def test_cvpr_segm_models():
    run_tests('cvpr', 'segm', segm_postprocessing)


def test_cvpr_depth_models():
    run_tests('cvpr', 'depth', depth_postprocessing)


def test_wacv_cs_models():
    run_tests('wacv', 'cs', segm_postprocessing)


def test_wacv_cv_models():
    run_tests('wacv', 'cv', segm_postprocessing)
