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
from resnet_3d import get_net
from utils.utils import (add_in_log, cal_result_parameters, get_args, get_lr,
                         init_train, save_model, seed_reproducer)


def train(epoch):

    st = time.time()
    running_loss = 0.0
    net.train()

    for batch_index, data in enumerate(dataset["train"]):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            cls = net(inputs)
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

    _dataset = dataset[datatype]
    st = time.time()
    net.eval()

    label_list, output_list = [], []
    for data in _dataset:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        label_list.append(labels), output_list.append(outputs)
    labels_stack = torch.cat(label_list, dim=0)
    output_stack = torch.cat(output_list, dim=0)
    result = cal_result_parameters(labels_stack, output_stack)
    ft = time.time()
    return result, ft - st


if __name__ == "__main__":

    # seed_reproducer(2333)
    parser = argparse.ArgumentParser()
    scaler = GradScaler()
    args = get_args(argparse.ArgumentParser())
    device, ckpt_path, writer = init_train(args)
    net = get_net(args.net).to(device)
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
        val_result, val_time = eval_training(epoch, "val")
        test_result, test_time = eval_training(epoch, "test")
        add_in_log(epoch, val_result, test_result, writer)

        # Save  model
        save_model(epoch, test_result, ckpt_path, net, args.net)
