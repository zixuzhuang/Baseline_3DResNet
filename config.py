import time

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

EPOCH = 150
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

PATH = "./data/Net_Input/3D_64_z1.0_l0.70_h0.10_20210125/"
NUM_CLASSES = 3

OUTPUT_DIR = "./results"
TF = "{}/tensorboards/{}/".format(OUTPUT_DIR, DATE)
maybe_mkdir_p(TF)
TF += "{}-{}-{}/"

CKPT = "{}/checkpoints/{}/".format(OUTPUT_DIR, DATE)
maybe_mkdir_p(CKPT)
CKPT += "{}-{}-{}/"

LOG = "{}/logs/{}/".format(OUTPUT_DIR, DATE)
maybe_mkdir_p(LOG)
LOG += "{}-{}-{}.log"
