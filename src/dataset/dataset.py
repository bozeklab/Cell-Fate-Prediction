import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CocoVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_file,
        scaling_transform,
        normalization_transform,
        min_seq_length=None,
    ):
        self.data_dir = data_dir
        self.scaling_transform = scaling_transform
        self.normalization_transform = normalization_transform

        self.data = pd.read_csv(os.path.join(data_dir, data_file))

        if min_seq_length:
            self.data = self.data[self.data["sequence_length"] > min_seq_length]

        counts = self.data["label"].value_counts().sort_index()
        weight = 1.0 / counts
        samples_weight = np.array([weight[t] for t in self.data["label"].values])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        self.class_weights = samples_weight

    def load_sequence(self, vid_name, cell_label):
        return torch.load(
            os.path.join(
                self.data_dir, str(vid_name).zfill(3), str(cell_label) + ".pt"
            ),
            weights_only=False,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_name, cell, sequence_length, label = self.data.iloc[idx]

        _, _, _, vid_label, time_step, sequence = self.load_sequence(video_name, cell)

        assert vid_label == label, "Label mismatch"
        assert sequence.size(0) == sequence_length, "Sequence length mismatch"

        sequence = self.scaling_transform(sequence)
        sequence = self.normalization_transform(sequence)

        return sequence, label, time_step, video_name
