import os
import pickle
import random

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import config


class Res3D_Dataset(Dataset):
    def __init__(self, data_type: str, fold_idx=0):
        super(Res3D_Dataset, self).__init__()
        subject_files = np.array(os.listdir(config.PATH))
        kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED_DIVIDE)
        train_index, val_index, test_index = [], [], []
        for idx in range(5):
            _, files_index = list(kf.split(subject_files))[idx]
            if idx != fold_idx:
                train_index += list(files_index)
                if val_index == []:
                    val_index = files_index
            else:
                test_index = files_index

        if data_type == "train":
            train_index = np.array(train_index)
            input_list = subject_files[train_index]
        elif data_type == "val":
            input_list = subject_files[val_index]
        elif data_type == "test":
            input_list = subject_files[test_index]
        else:
            print("Wrong dataset type")
            exit()
        self.f = input_list

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        with open(config.PATH + self.f[idx], "rb") as f:
            data = pickle.load(f)
        return (
            data["input"].view(1, 1, 20, 448, 448),
            data["label"],
            data["mask"].float()[1].view(1, 20, 448, 448),
            self.f[idx].split(".")[0],
        )


def collate(samples):
    data_out = {}
    inputs, labels, maskes, file_names = map(list, zip(*samples))
    data_out["input"] = torch.cat([item for item in inputs], dim=0)
    data_out["mask"] = torch.cat([item for item in maskes], dim=0)
    data_out["label"] = torch.tensor(labels)
    data_out["name"] = file_names
    return data_out


def Res3D_Dataloader(bs=4, fold_idx=0, num_workers=8):
    return (
        DataLoader(
            Res3D_Dataset(data_type="train", fold_idx=fold_idx),
            batch_size=bs,
            collate_fn=collate,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(
            Res3D_Dataset(data_type="val", fold_idx=fold_idx),
            batch_size=bs,
            collate_fn=collate,
            shuffle=False,
            num_workers=num_workers,
        ),
        DataLoader(
            Res3D_Dataset(data_type="test", fold_idx=fold_idx),
            batch_size=bs,
            collate_fn=collate,
            shuffle=False,
            num_workers=num_workers,
        ),
    )
