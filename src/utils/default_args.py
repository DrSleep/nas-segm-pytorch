## DEFAULT CONFIGURATION USED IN OUR EXPERIMENTS ON 2 GPUs

import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = "./data/datasets/VOCdevkit/"
VAL_DIR = "./data/datasets/VOCdevkit/"
TRAIN_LIST = "./data/lists/train+.lst"
VAL_LIST = "./data/lists/train+.lst"  # meta learning
META_TRAIN_PRCT = 90
N_TASK0 = 4000
SHORTER_SIDE = [300, 400]
CROP_SIZE = [256, 350]
NORMALISE_PARAMS = [
    1.0 / 255,  # SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),
]  # STD
BATCH_SIZE = [64, 32]
NUM_WORKERS = 16
NUM_CLASSES = [21, 21]
LOW_SCALE = 0.7
HIGH_SCALE = 1.4
VAL_SHORTER_SIDE = 400
VAL_CROP_SIZE = 400
VAL_BATCH_SIZE = 64

# ENCODER PARAMETERS
ENC_GRAD_CLIP = 3.0

# DECODER PARAMETERS
DEC_GRAD_CLIP = 3.0
DEC_AUX_WEIGHT = 0.15  # to disable aux, set to -1

# GENERAL
FREEZE_BN = [False, False]
NUM_EPOCHS = 20000
NUM_SEGM_EPOCHS = [5, 1]
PRINT_EVERY = 20
RANDOM_SEED = 9314
SNAPSHOT_DIR = "./ckpt/"
CKPT_PATH = "./ckpt/checkpoint.pth.tar"
VAL_EVERY = [5, 1]  # how often to record validation scores
SUMMARY_DIR = "./tb_logs/"

# OPTIMISERS' PARAMETERS
LR_ENC = [1e-3, 1e-3]
LR_DEC = [3e-3, 3e-3]
LR_CTRL = 1e-4
MOM_ENC = [0.9] * 3
MOM_DEC = [0.9] * 3
MOM_CTRL = 0.9
WD_ENC = [1e-5] * 3
WD_DEC = [0] * 3
WD_CTRL = 1e-4
OPTIM_DEC = "adam"
OPTIM_ENC = "sgd"
AGENT_CTRL = "ppo"
DO_KD = True
KD_COEFF = 0.3
DO_POLYAK = True

# CONTROLLER
BL_DEC = 0.95
OP_SIZE = 11
AGG_CELL_SIZE = 48
NUM_CELLS = 3
NUM_BRANCHES = 4
AUX_CELL = True
SEP_REPEATS = 1
