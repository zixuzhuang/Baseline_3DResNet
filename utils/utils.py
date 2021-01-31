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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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


def get_lr(epoch):
    if epoch <= config.WU:
        lr = config.LR / config.WU * epoch
    else:
        lr = config.LR * config.LD ** epoch
    return lr


def get_args(parser):
    parser.add_argument("-bs", type=int, default=4, help="batch size")
    parser.add_argument("-fold", type=int, default=0, help="which fold")
    parser.add_argument("-net", type=str, help="net")
    args = parser.parse_args()
    return args


def init_train(args):

    # fix seed
    seed_reproducer(2333)

    path_block = [args.net, args.fold, config.TIME]
    # tensorboard
    tf_dir = config.TF.format(*path_block)
    maybe_mkdir_p(tf_dir)
    writer = SummaryWriter(log_dir=tf_dir)

    # log
    log_dir = config.LOG.format(*path_block)
    initLogging(log_dir)

    # checkpoint
    ckpt_path = config.CKPT.format(*path_block)
    maybe_mkdir_p(ckpt_path)

    # print info
    logging.debug("Checkpoints is in:\n{}".format(ckpt_path))
    logging.debug("Tensorboard is in:\n{}".format(tf_dir))
    logging.debug("Data is in:\n{}".format(config.PATH))
    logging.debug("LR: {}, LD: {}".format(config.LR, config.LD))
    logging.debug("WD: {}, WU: {}".format(config.WD, config.WU))
    logging.debug("BS: {}, FOLD: {}\n\n".format(args.bs, args.fold))

    return tf_dir, ckpt_path, writer


def save_model(epoch, result, ckpt_path, net, nname, save_best=True):
    best_ckpt = "{}{}-{}.pth".format(ckpt_path, nname, "best")
    latest_ckpt = "{}{}-{}.pth".format(ckpt_path, nname, "latest")
    # print(result["REC"])
    recall = result["REC"][-1]
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
            "BEST_RECALL: {:.3f}, EPOCH: {:3}".format(
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
    num_cls = cls_output.shape[1]
    probe = torch.softmax(cls_output, dim=1)
    y_true = label.cpu().detach().numpy()
    y_pred = probe.cpu().detach().numpy()
    y_true_one_hot = get_one_hot(y_true, num_cls)
    y_pred_one_hot = get_one_hot(np.argmax(y_pred, axis=1), num_cls)

    for i in range(num_cls):
        AUCs.append(roc_auc_score(y_true_one_hot[:, i], y_pred[:, i]))
        ACCs.append(accuracy_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))
        RECALLs.append(sensitivity_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))
        SPEs.append(specificity_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))
        F1s.append(f1_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))

    AUCs.append(np.array(AUCs).mean())
    ACCs.append(np.array(ACCs).mean())
    RECALLs.append(np.array(RECALLs).mean())
    SPEs.append(np.array(SPEs).mean())
    F1s.append(np.array(F1s).mean())

    Result = {}
    Result["n"] = len(AUCs) - 1
    Result["AUC"] = AUCs
    Result["ACC"] = ACCs
    Result["REC"] = RECALLs
    Result["SPE"] = SPEs
    Result["F1"] = F1s
    return Result


def add_in_log(dt, ep, r, writer):
    for i in range(r["n"]):
        writer.add_scalar("{}-AUC/CLASS{}".format(dt, i), r["AUC"][i], ep)
        writer.add_scalar("{}-ACC/CLASS{}".format(dt, i), r["ACC"][i], ep)
        writer.add_scalar("{}-REC/CLASS{}".format(dt, i), r["REC"][i], ep)
        writer.add_scalar("{}-SPE/CLASS{}".format(dt, i), r["SPE"][i], ep)
        writer.add_scalar("{}-F1/CLASS{}".format(dt, i), r["F1"][i], ep)
    writer.add_scalar("{}-MEAN/AUC".format(dt), r["AUC"][-1], ep)
    writer.add_scalar("{}-MEAN/ACC".format(dt), r["ACC"][-1], ep)
    writer.add_scalar("{}-MEAN/REC".format(dt), r["REC"][-1], ep)
    writer.add_scalar("{}-MEAN/SPE".format(dt), r["SPE"][-1], ep)
    writer.add_scalar("{}-MEAN/F1".format(dt), r["F1"][-1], ep)

    title = "{:4} set, TITLE\t".format(dt)
    auc = "{:4} set, AUC:\t".format(dt)
    acc = "{:4} set, ACC:\t".format(dt)
    rec = "{:4} set, REC:\t".format(dt)
    spe = "{:4} set, SPE:\t".format(dt)
    f1 = "{:4} set, F1 :\t".format(dt)
    for i in range(r["n"]):
        title += "CLS{}\t".format(i)
    for i in range(r["n"] + 1):
        auc += "{:.3f}\t".format(r["AUC"][i])
        acc += "{:.3f}\t".format(r["ACC"][i])
        rec += "{:.3f}\t".format(r["REC"][i])
        spe += "{:.3f}\t".format(r["SPE"][i])
        f1 += "{:.3f}\t".format(r["F1"][i])
    title += "MEAN"
    logging.debug("\n{}\n{}\n{}\n{}\n{}\n{}".format(title, rec, f1, auc, acc, spe))


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
