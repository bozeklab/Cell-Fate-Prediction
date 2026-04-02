import random

import torch
import torchvision.transforms.v2 as T
from torchvision.utils import _log_api_usage_once


class ToTensor(object):
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img
        return T.ToTensor()(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Rescale(object):
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, img):
        img = img / 65535.0
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomColorJitter(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return T.ColorJitter()(img)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomGaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.2):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return T.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)(img)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomMask(object):
    """Randomly removes a portion of the end of a given sequence. If the sequence is longer than 10 frames, at least 5 frames towards the end of the sequence will be removed."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sequence):
        if random.random() < self.p:
            seq_length = sequence.shape[0]
            mask_length = random.uniform(0.1, 0.5)

            if seq_length > 10:
                mask_length = max(5, int(mask_length * seq_length))
            else:
                mask_length = int(mask_length * seq_length)

            assert (
                mask_length < seq_length
            ), "Mask length must be less than sequence length"

            # with open("mask_lengths.txt", mode="a") as f:
            #     f.write(
            #         "Reduced sequence length from {} to {}\n".format(
            #             seq_length, seq_length - mask_length
            #         )
            #     )
            sequence = sequence[: seq_length - mask_length]
            return sequence

        return sequence


class Mask(object):
    """Removes a portion of the end of a given sequence."""

    def __init__(self, trunc_length, use_trancation):
        self.trunc_length = trunc_length
        self.use_trancation = use_trancation

    def __call__(self, sequence):
        if self.use_trancation:
            return sequence[-self.trunc_length :]
        else:
            return sequence[: -self.trunc_length]
