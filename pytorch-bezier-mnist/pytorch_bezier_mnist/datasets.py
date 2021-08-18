import json
import os
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from .tokenizer import Tokenizer


class BezierMNIST(Dataset):
    """
    A dataset representing a collection of digits, where each digit is
    composed of one or more loops, each loop being itself composed of one or
    more cubic Bezier curves.

    Each element in the dataset is a tuple (loops, label), where loops is a
    list and label is an integer. Each loop is itself a list of Bezier curves,
    and each Bezier curve is a list of four (x, y) tuples.

    :param data_dir: the directory containing the entire dataset.
    :param split: the split of the dataset to use. This is stored in a
                  sub-directory of the whole dataset.
    :param download: if True and the split directory does not exist, download
                     it from the internet.
    :param version: the dataset version number. Supported: 1 and 2.
    :param randomize_loops: if True, the Beziers in each loop are randomly
                            cycled each time a datum is accessed, since shapes
                            are invariant to the order of the underlying
                            curves.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        download: bool = True,
        version: int = 2,
        randomize_loops: bool = True,
    ):
        if split not in ["train", "test"]:
            raise ValueError(f"unknown split: {split}")
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)
        self.randomize_loops = randomize_loops

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if not os.path.exists(self.split_dir):
            if not download:
                raise FileNotFoundError(
                    f"data directory not found for split {split}: {self.split_dir}"
                )
            # TODO: put the actual files here.
            split_url = f"https://data.aqnichol.com/bezier-mnist/v{version}/{split}.zip"
            download_and_extract_archive(
                split_url, self.data_dir, self.split_dir, filename=f"{split}.zip"
            )

        self.paths = [
            os.path.join(self.split_dir, x)
            for x in sorted(os.listdir(self.split_dir))
            if x.endswith(".json") and not x.startswith(".")
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> Tuple[List[List[List[Tuple[float, float]]]], int]:
        with open(self.paths[idx], "rt") as f:
            obj = json.load(f)
        if self.randomize_loops:
            loops = []
            for loop in obj["beziers"]:
                new_start = random.randrange(len(loop))
                loops.append(loop[new_start:] + loop[:new_start])
            obj["beziers"] = loops
        loops = [
            ([[(p["X"], p["Y"]) for p in curve] for curve in loop]) for loop in loops
        ]
        return loops, obj["label"]


class TokenBezierMNIST(BezierMNIST):
    """
    A dataset which flattens each sample into a discrete sequence of tokens.
    """

    def __init__(
        self, *args, tokenizer: Optional[Tokenizer] = None, seq_len: int = 400, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if tokenizer is None:
            tokenizer = Tokenizer()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        loops, label = super().__getitem__(idx)
        tokens = self.tokenizer.encode_loops(loops)[: self.seq_len]
        while len(tokens) < self.seq_len:
            tokens.append(self.tokenizer.end_token)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


class VecBezierMNIST(BezierMNIST):
    """
    A dataset which flattens all of the Bezier curves into a fixed-length,
    zero-padded tensor.
    """

    def __init__(self, *args, max_curves: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_curves = max_curves

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        loops, label = super().__getitem__(idx)
        data = [c for loop in loops for curve in loop for pair in curve for c in pair]
        while len(data) < self.max_curves * 8:
            data.append(0.0)
        data = data[: self.max_curves * 8]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )
