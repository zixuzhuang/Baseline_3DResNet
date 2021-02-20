import csv
import logging
import os
import random
import sys

import config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)
from torch.utils.tensorboard import SummaryWriter


class BestResults(object):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.recall = 0.0


def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)


def refine_cam(cams, labels):
    B, C, H, W = cams.shape
    _cams = torch.zeros([B, H, W], device=cams.device)
    for i in range(B):
        _cams[i] = cams[i, labels[i]]
    _cams = _cams.view(B, -1)
    cams_min = _cams.min(dim=1, keepdim=True)[0]
    cams_max = _cams.max(dim=1, keepdim=True)[0]
    norm = cams_max - cams_min
    norm[norm == 0] = 1e-5
    _cams = (_cams - cams_min) / norm
    _cams = _cams.view(B, 1, H, W)
    _cams = _cams.view(B, 1, cams.shape[2], cams.shape[3])
    sigmoid_cams = torch.sigmoid(100 * (_cams - 0.4))
    # fn = torch.sigmoid(dim=0)
    # _cams = fn(_cams)
    return _cams


def get_lr(epoch, args):
    lr = args.lr / (epoch // 10 + 1)
    # if epoch <= config.WU:
    #     lr = config.LR / config.WU * epoch
    # else:
    #     lr = config.LR * config.LD ** epoch
    return lr


def get_args(parser):
    parser.add_argument("-bs", type=int, default=4)
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-net", type=str)
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument("-test", type=bool, default=False)
    parser.add_argument("-lr", type=float, default=1e-4)
    args = parser.parse_args()
    return args


def init_train(args):

    device = torch.device("cuda" if args.device != "cpu" else "cpu")
    path_block = [args.net, args.fold, config.TIME]

    if args.device == "cpu" or args.test:
        tf_dir = "./results/temp/"
        log_dir = "./results/temp/testlog.log"
        ckpt_path = "./results/temp/"
    else:
        tf_dir = config.TF.format(*path_block)
        log_dir = config.LOG.format(*path_block)
        ckpt_path = config.CKPT.format(*path_block)

    maybe_mkdir_p(tf_dir)
    maybe_mkdir_p(ckpt_path)
    writer = SummaryWriter(log_dir=tf_dir)
    initLogging(log_dir)

    # print info
    forma = "\n{}\n{}\n{}\n{}\n{}\n{}|"
    title = ["Checkpoints", "Tensorboard", "Input Data"]
    items = [ckpt_path, tf_dir, config.PATH]
    logging.debug(forma.format(*title, *items))

    forma = ("\n" + "|{:^9}" * 7 + "|") * 2
    title = ["NET", "BS", "FOLD", "LR", "LD", "WD", "WU"]
    items = [args.net, args.bs, args.fold, args.lr, config.LD, config.WD, config.WU]
    logging.info(forma.format(*title, *items))

    return device, ckpt_path, writer


def save_model(epoch, result, ckpt_path, net, nname, save_best=True):
    best_ckpt = "{}{}-{}.pth".format(ckpt_path, nname, "best")
    latest_ckpt = "{}{}-{}.pth".format(ckpt_path, nname, "latest")
    # print(result["REC"])
    recall = result["REC"]
    # Save latest model
    if epoch > config.MILESTONES:
        torch.save(net, latest_ckpt)
    # Save best model
    if recall > bestResult.recall:
        bestResult.recall = recall
        bestResult.epoch = epoch
        torch.save(net, best_ckpt)
    # Print best log
    if save_best:
        logging.info(
            "BEST RECALL: {:.3f}, EPOCH: {:3}\n\n".format(
                bestResult.recall, bestResult.epoch
            )
        )
    return


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = np.eye(num_cls)[label]
    return label


def cal_result_parameters(label, cls_output):
    AUCs, ACCs, RECALLs, SPEs, F1s = [], [], [], [], []
    Result = {}
    num_cls = cls_output.shape[1]
    probe = torch.softmax(cls_output, dim=1)
    y_true = label.cpu().detach().numpy()
    y_pred = probe.cpu().detach().numpy()
    y_preds = np.argmax(y_pred, axis=1)
    y_true_one_hot = get_one_hot(y_true, num_cls)

    Result["ACC"] = accuracy_score(y_true, y_preds)
    Result["REC"] = sensitivity_score(y_true, y_preds, average="macro")
    Result["AUC"] = roc_auc_score(y_true_one_hot, y_pred, average="macro")
    Result["PRE"] = precision_score(y_true, y_preds, average="macro", zero_division=0)
    Result["SPE"] = specificity_score(y_true, y_preds, average="macro")
    Result["F1"] = f1_score(y_true, y_preds, average="macro")
    Result["CM"] = confusion_matrix(y_true, y_preds)

    return Result


def add_in_log(ep, r_v, r_t, writer):
    title_items = ["SET", "ACC", "REC", "AUC", "PRE", "SPE", "F1"]
    val_items = ["VAL SET"] + [r_v[_] for _ in title_items[1:]]
    test_items = ["TEST SET"] + [r_t[_] for _ in title_items[1:]]
    forma_1 = "\n|{:^8}" + "|{:^5}" * (len(title_items) - 1) + "|"
    forma_2 = ("\n|{:^8}" + "|{:^.3f}" * (len(title_items) - 1) + "|") * 2
    logging.info("VAL  RECALL: {:.3f}".format(r_v["REC"]))
    logging.info("TEST RECALL: {:.3f}".format(r_t["REC"]))
    logging.debug((forma_1 + forma_2).format(*title_items, *val_items, *test_items))
    logging.debug("\nVAL CM:\n{}\nTEST CM:\n{}".format(r_v["CM"], r_t["CM"]))


def seed_reproducer(seed=2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


bestResult = BestResults()
