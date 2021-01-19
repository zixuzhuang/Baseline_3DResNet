import csv
import os
import random
import sys

import config
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter


def cal_reacll(confusion_matrix):
    n = confusion_matrix.shape[0]
    recall = 0.0
    for i in range(n):
        recall += confusion_matrix[i, i] / torch.sum(confusion_matrix[i, :])
    return recall / n


def refine_cam(cams, labels):
    # sigmoid_cams = cams[:, 0, :, :].view(-1, 1, 128, 128)
    B, C, S, H, W = cams.shape
    _cams = torch.zeros([B, S, H, W], device=cams.device)
    for i in range(B):
        _cams[i] = cams[i, labels[i]]
    _cams = _cams.view(B, -1)
    cams_min = _cams.min(dim=1, keepdim=True)[0]
    cams_max = _cams.max(dim=1, keepdim=True)[0]
    norm = cams_max - cams_min
    norm[norm == 0] = 1e-5
    _cams = (_cams - cams_min) / norm
    _cams = _cams.view(B, S, H, W)
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


def confusionMatrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix


def get_args(parser):
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-lr", type=int, default=3e-5, help="learning rate")
    parser.add_argument("-fold", type=int, default=0, help="which fold")
    args = parser.parse_args()
    return args


def init_train(args, net_name):

    # use tensorboard
    tf_dir = "{}/tensorboard/{}/{}-{}-{}/".format(
        config.SAVE_PATH, config.DATE, net_name, config.TIME, args.fold
    )
    writer = SummaryWriter(log_dir=tf_dir)
    maybe_mkdir_p(tf_dir)

    # create checkpoint folder to save model
    ckpt_path = "{}/checkpoints/{}/".format(config.SAVE_PATH, config.DATE)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # training
    print("--------------args----------------")
    print("Checkpoints is in:")
    print(ckpt_path)
    print("TF board is in:")
    print(tf_dir)
    for k in list(vars(args).keys()):
        print("%s: %s" % (k, vars(args)[k]))
    print("--------------args----------------\n")
    bestAcc = [0.0, 0]

    return tf_dir, writer, ckpt_path, bestAcc
    # return


def save_model(epoch, bestREC, result, ckpt_path, net, net_name, fold):
    best_ckpt = "{}{}-{}-{}-{}.pth".format(
        ckpt_path, net_name, config.TIME, fold, "best"
    )
    latest_ckpt = "{}{}-{}-{}-{}.pth".format(
        ckpt_path, net_name, config.TIME, fold, "latest"
    )
    recall = result["REC"][-1]
    # Save latest model
    if epoch > config.MILESTONES:
        torch.save(net, latest_ckpt)
        # Save best model
        if recall > bestREC[0]:
            bestREC = [recall, epoch]
            torch.save(net, best_ckpt)
    return bestREC


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


def add_in_log_and_print(dt, ep, r, writer):
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

    title = "{} set, ".format(dt)
    auc = "{} set, AUC:\t".format(dt)
    acc = "{} set, ACC:\t".format(dt)
    rec = "{} set, REC:\t".format(dt)
    spe = "{} set, SPQ:\t".format(dt)
    f1 = "{} set, F1S:\t".format(dt)
    for i in range(r["n"]):
        title += "CLS{}\t".format(i)
        auc += "{:.3f}\t".format(r["AUC"][i])
        acc += "{:.3f}\t".format(r["ACC"][i])
        rec += "{:.3f}\t".format(r["REC"][i])
        spe += "{:.3f}\t".format(r["SPE"][i])
        f1 += "{:.3f}\t".format(r["F1"][i])
    title += "MEAN"
    print("{}\n{}\n{}\n{}\n{}\n{}".format(title, rec, f1, auc, acc, spe))


def add_in_log_and_print_(dt, ep, r, writer):
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
    print("{} set, AUC:\tCLS0\tCLS1\tCLS2\tMEAN".format(dt))
    print("{} set, AUC:\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(dt, *r["AUC"]))
    print("{} set, ACC:\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(dt, *r["ACC"]))
    print("{} set, REC:\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(dt, *r["REC"]))
    print("{} set, SPE:\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(dt, *r["SPE"]))
    print("{} set, F1:\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(dt, *r["F1"]))
