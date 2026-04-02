import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

import utils.transforms as T
from dataset.dataset import CocoVideoDataset


class CocoVidDataLoader(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        mean,
        std,
        min_seq_length=None,
        truncation=None,
        use_truncation=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if truncation is not None and not use_truncation:
            self.min_seq_length = max(truncation, min_seq_length)
        else:
            self.min_seq_length = min_seq_length

        self.mean = mean
        self.std = std

        scaling = [
            T.ToTensor(),
            T.Rescale(),
        ]

        normalize = Normalize(mean=self.mean, std=self.std)

        augmentations = [
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            T.RandomColorJitter(),
            T.RandomGaussianBlur((5, 9)),
        ]

        if truncation:
            scaling.append(T.Mask(truncation, use_truncation))
        else:
            augmentations.insert(0, T.RandomMask())

        self.scaling_transform = Compose(scaling)
        self.normalize_transform = normalize

        comps = scaling + augmentations

        self.train_transform_scaling = Compose(comps)
        self.train_transform_normalize = normalize

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = CocoVideoDataset(
                self.data_dir,
                "train_cells.csv",
                self.train_transform_scaling,
                self.train_transform_normalize,
                min_seq_length=self.min_seq_length,
            )
            self.class_weights = self.train_dataset.class_weights
            self.val_dataset = CocoVideoDataset(
                self.data_dir,
                "val_cells.csv",
                self.scaling_transform,
                self.normalize_transform,
                min_seq_length=self.min_seq_length,
            )
        elif stage == "test" or stage == "predict":
            self.test_dataset = CocoVideoDataset(
                self.data_dir,
                "test_cells.csv",
                self.scaling_transform,
                self.normalize_transform,
                min_seq_length=self.min_seq_length,
            )
        elif stage == "analysis":
            self.analysis_dataset = CocoVideoDataset(
                self.data_dir,
                "all.csv",
                self.scaling_transform,
                self.normalize_transform,
                min_seq_length=self.min_seq_length,
            )

    def collate_fn(self, data):
        # separate sequences and labels
        if len(data) == 1 and data[0] is None:
            return None
        sequences, labels, time_step, video_names = zip(*data)

        # get max length of sequences
        lengths = [seq.shape[0] for seq in sequences]
        max_length = max(lengths)

        # pad sequences
        batch_size = len(sequences)
        padded_batch = torch.zeros(
            (batch_size, max_length, *sequences[0].shape[1:]), dtype=torch.float32
        )
        mask = torch.ones((batch_size, max_length + 1), dtype=torch.bool)

        # masking
        for i, seq in enumerate(sequences):
            seq_length = seq.shape[0]

            padded_batch[i, :seq_length] = seq
            mask[i, : seq_length + 1] = False

        return padded_batch, mask, torch.tensor(labels), time_step, video_names

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=WeightedRandomSampler(
                weights=self.class_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            ),
            collate_fn=self.collate_fn,
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def analysis_dataloader(self):
        dataloader = DataLoader(
            self.analysis_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        return dataloader
