import argparse
import logging
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
from resnet_3d import resnet18
from utils.utils import (add_in_log, cal_result_parameters, get_args, get_lr,
                         init_train, refine_cam, save_model, seed_reproducer)


def train(epoch):

    st = time.time()
    running_loss = 0.0
    net.train()
    label_list, output_list = [], []

    for batch_index, data in enumerate(train_ldr):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        with autocast():
            cls, cams = net(inputs)
            loss = loss_cls(cls, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss

    ft = time.time()
    epoch_loss = running_loss / len(train_ldr)
    writer.add_scalar("Train/Loss", epoch_loss, epoch)

    logging.info("EPOCH: {}".format(epoch))
    logging.info("TRAIN_LOSS : {:.3f}, TIME: {:.1f}s".format(epoch_loss, ft - st))


@torch.no_grad()
def eval_training(epoch, datatype):

    if datatype == "VAL":
        _dataset = val_ldr
    elif datatype == "TEST":
        _dataset = test_ldr
    else:
        print("Wrong dataloader type!")
        exit()

    st = time.time()
    net.eval()

    label_list, output_list = [], []
    with autocast():
        for data in _dataset:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            cls, cams = net(inputs)  # cls is output
            label_list.append(labels)
            output_list.append(cls)
    labels_stack = torch.cat(label_list, dim=0)
    output_stack = torch.cat(output_list, dim=0)
    result = cal_result_parameters(labels_stack, output_stack)
    ft = time.time()
    logging.info(
        "{:4}_RECALL: {:.3f}, TIME: {:.1f}s".format(
            datatype, result["REC"][-1], ft - st
        )
    )
    add_in_log(datatype, epoch, result, writer)
    return result


if __name__ == "__main__":

    seed_reproducer(2333)
    parser = argparse.ArgumentParser()
    scaler = GradScaler()
    args = get_args(argparse.ArgumentParser())
    tf_dir, ckpt_path, writer = init_train(args, "3D_ResNet18")
    net = resnet18().cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.LR, weight_decay=config.WD)
    train_ldr, val_ldr, test_ldr = Res3D_Dataloader(bs=args.bs, fold_idx=args.fold)
    loss_cls = nn.CrossEntropyLoss()

    for epoch in range(1, config.EPOCH):

        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(epoch)

        # Train
        train(epoch)

        # Calculate val data and test data and save failed file
        eval_training(epoch, "VAL")
        result = eval_training(epoch, "TEST")

        # Save  model
        save_model(epoch, result, ckpt_path, net, "3D_ResNet18")
