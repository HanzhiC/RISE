# from data.hoi4d_dataset import HOI4DDataset
from data.dataset import Egoasis4DDataset
from data.robot_dataset import Robot4DDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import math


class Egoasis4DDataModule(pl.LightningDataModule):

    def __init__(self, data_config, train_config):
        super().__init__()
        self.data_config = data_config
        self.train_config = train_config


    def setup(self, stage=None):
        # Setup the train dataset
        train_config = self.data_config.copy()
        train_config["split_fpath"] = self.data_config.split_fpath_train
        train_config["training"] = True
        train_config["load_rgbd_frames"] = False
        split_base_name = self.data_config.split_fpath_train.split("/")[-1].split(".")[
            0
        ]
        del train_config["split_fpath_train"]
        del train_config["split_fpath_test"]
        if "stretchrobot" in split_base_name or "robocasa" in split_base_name:
            self.train_dataset = Robot4DDataset(**train_config)
        else:
            self.train_dataset = Egoasis4DDataset(**train_config)

        # Setup the val dataset
        test_config = self.data_config.copy()
        test_config["split_fpath"] = self.data_config.split_fpath_test
        test_config["training"] = False
        test_config["load_rgbd_frames"] = True
        split_base_name = self.data_config.split_fpath_test.split("/")[-1].split(".")[0]
        if "stretchrobot" in split_base_name or "robocasa" in split_base_name:
            self.val_dataset = Robot4DDataset(**test_config)
        else:
            self.val_dataset = Egoasis4DDataset(**test_config)
        del test_config["split_fpath_test"]
        del test_config["split_fpath_train"]
        print("========> Size of the train dataset: ", len(self.train_dataset))
        print("========> Size of the val dataset: ", len(self.val_dataset))


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.train_config.training.batch_size,
            num_workers=self.train_config.training.num_workers,
            drop_last=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        shuffle = getattr(self.train_config.validation, "shuffle", False)
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=shuffle,
            batch_size=self.train_config.validation.batch_size,
            num_workers=self.train_config.training.num_workers,
            drop_last=True,
            persistent_workers=False,
        )
