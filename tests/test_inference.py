import sys
sys.path.append('./src/')
from functools import partial

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.encoders import mbv2
from nn.micro_decoders import MicroDecoder as Decoder
from utils.model_zoo import load_url


MODELS_INFO = {
    'arch0': [
        [[8, [0, 0, 5, 2], [0, 2, 8, 8], [0, 5, 1, 4]], [[3, 3], [3, 2], [3, 0]]],
        {
            'segm-23378522.pth': 'https://cloudstor.aarnet.edu.au/plus/s/ZPXVGGgyxekvdAn/download',
            'depth-81f185f7.pth': 'https://cloudstor.aarnet.edu.au/plus/s/ttAlLJqDr30v1sk/download',
        },
    ],
    'arch1': [
        [[2, [1, 0, 3, 6], [0, 1, 2, 8], [2, 0, 6, 1]], [[2, 3], [3, 1], [4, 4]]],
        {
            'segm-12f78b21.pth': 'https://cloudstor.aarnet.edu.au/plus/s/svINhJX7IsvjCaD/download',
            'depth-7965abcb.pth': 'https://cloudstor.aarnet.edu.au/plus/s/5GeBZwW97eyODF7/download',
        },
    ],
    'arch2': [
        [[5, [0, 0, 4, 1], [3, 2, 0, 1], [5, 6, 5, 0]], [[1, 3], [4, 3], [2, 2]]],
        {
            'segm-8f00fc4d.pth': 'https://cloudstor.aarnet.edu.au/plus/s/9b8zVuaowe6ZtAN/download',
            'depth-a2f8f6d6.pth': 'https://cloudstor.aarnet.edu.au/plus/s/75pvpkqhJIv4aw4/download',
        },
    ],
}
AGG_SIZE = 64
AUX_CELL = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = {'segm': [21, 'voc'], 'depth': [1, 'nyud']}
REPEATS = 2


class EncoderDecoder(nn.Module):
    """Create Segmenter"""
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_tests(task, postproc_fn):
    num_classes, test_suit = NUM_CLASSES[task]
    with open(f'./tests/test_{test_suit}_{DEVICE}.ckpt', 'rb') as f:
        input_img, models_and_preds = pickle.load(f)[0]
    input_tensor = torch.from_numpy(input_img).to(DEVICE)
    dec_fn = partial(Decoder,
                     num_classes=num_classes,
                     agg_size=AGG_SIZE,
                     aux_cell=AUX_CELL,
                     repeats=REPEATS)
    for arch, config in MODELS_INFO.items():
        structure, links = config
        filename, url = [(k, v) for k,v in links.items() if task in k][0]
        enc = mbv2(pretrained=False)
        dec = dec_fn(config=structure, inp_sizes=enc.out_sizes)
        model = EncoderDecoder(enc, dec).to(DEVICE).eval()
        model.load_state_dict(load_url((arch + '_' + filename, url), map_location=DEVICE), strict=False)
        with torch.no_grad():
            pred = postproc_fn(model(input_tensor))
            assert np.allclose(pred, models_and_preds[arch + '_' + task])


def test_segm_models():
    def postprocessing(outputs):
        pred = outputs[0].squeeze().data.cpu().numpy().transpose(1, 2, 0)
        return pred.argmax(axis=2).astype(np.uint8)
    run_tests('segm', postprocessing)


def test_depth_models():
    def postprocessing(outputs):
        pred = outputs[0].squeeze().data.cpu().numpy()
        return pred.astype(np.float32)
    run_tests('depth', postprocessing)
