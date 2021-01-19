import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import config
from dataloader import Res3D_Dataloader
from resnet import resnet18
from utils.utils import (add_in_log_and_print, cal_result_parameters, get_args,
                         get_lr, init_train, refine_cam, save_model,
                         seed_reproducer)

num_cls = config.NUM_CLASSES


def train(epoch):

    start = time.time()
    running_loss = 0.0
    net.train()

    for batch_index, data in enumerate(train_loader):
        inputs, mask, labels = data['input'], data['mask'], data['label']
        inputs = inputs.cuda()
        mask = mask.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        with autocast():
            cls, cams = net(inputs)
            rcams = refine_cam(cams, labels)
            loss = loss_cls(cls, labels) + loss_att(mask, rcams) * 2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss

    finish = time.time()
    epoch_loss = running_loss / len(train_loader)
    writer.add_scalar("Train/Loss", epoch_loss, epoch)
    print("TRAIN_LOSS:{:.3f}, TRAIN_TIME: {:.1f}s".format(epoch_loss, finish - start))


@torch.no_grad()
def eval_training(epoch, datatype):

    if datatype == "VAL":
        _dataset = val_loader
    elif datatype == "TEST":
        _dataset = test_loader
    else:
        print("Wrong dataloader type!")
        exit()

    st = time.time()
    net.eval()

    label_list, output_list = [], []
    with autocast():
        for data in _dataset:
            inputs, mask, labels = data['input'], data['mask'], data['label']
            inputs = inputs.cuda()
            labels = labels.cuda()

            cls, cams = net(inputs)  # cls is output
            label_list.append(labels)
            output_list.append(cls)
        labels_stack = torch.cat(label_list, dim=0)
        output_stack = torch.cat(output_list, dim=0)
        result = cal_result_parameters(labels_stack, output_stack)

    ft = time.time()
    print(
        "{}_TIME: {:.1f}s, RECALL: {:.3f}".format(datatype, ft - st, result["REC"][-1])
    )
    add_in_log_and_print(datatype, epoch, result, writer)
    return result
    # return


if __name__ == "__main__":

    seed_reproducer(2333)
    parser = argparse.ArgumentParser()
    scaler = GradScaler()
    args = get_args(parser)
    line = "-" * 15

    # Data preprocessing
    net = resnet18().cuda()
    train_loader, val_loader, test_loader = Res3D_Dataloader(
        bs=args.bs, fold_idx=args.fold
    )
    optimizer = optim.Adam(net.parameters(), lr=config.LR, weight_decay=config.WD)

    loss_cls = nn.CrossEntropyLoss()
    loss_att = nn.MSELoss()

    log_dir, writer, ckpt_path, bestAcc = init_train(args, "3D_ResNet18")
    # init_train(args, "3D_ResNet18")
    print("{}{}{}".format(line, "Patch Training", line))

    for epoch in range(1, config.EPOCH):

        print("{}EPOCH: {}{}".format(line, epoch, line))

        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(epoch)

        # Train
        train(epoch)

        # Calculate val data and test data and save failed file
        val_result = eval_training(epoch, "VAL")
        test_result = eval_training(epoch, "TEST")

        # # # Save  model
        bestAcc = save_model(
            epoch, bestAcc, test_result, ckpt_path, net, "3D_ResNet18", args.fold
        )
        print("Best RECALL is {:.3f}, in epoch {}".format(bestAcc[0], bestAcc[1]))
