import time

SAVE_PATH = './data/3D_ResNet/'

EPOCH = 300
MILESTONES = 5
SAVE_EPOCH = 10

NUM_PATCHES = 377

LR = 3e-5  # learning rate
LD = 0.98  # learning rate decay rate
WD = 1e-5  # weight decay
WU = 20  # warm up
SEED_DIVIDE = 0

DATE = time.strftime("%Y%m%d")  # time of we run the script
TIME = time.strftime("%H%M%S")

PATH = './data/Net_Input/3D_20210118/'
NUM_CLASSES = 3
# INPUT_FEAT = PATH.format("_feat_list.pkl")
