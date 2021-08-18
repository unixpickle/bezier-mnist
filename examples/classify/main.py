"""
Train a digit classifier on Bezier curve data.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_bezier_mnist import VecBezierMNIST

DATA_DIR = "../../v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Create the training dataset loader.
    # Using the VecBezierMNIST dataset instead of the raw BezierMNIST
    # dataset will automatically flatten Bezier coordinates into a single
    # 1-dimensional Tensor to feed to our model.
    dataset = VecBezierMNIST(data_dir=DATA_DIR)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=4, shuffle=True, batch_size=args.batch_size
    )

    # Create the testing dataset so we can evaluate on it separately.
    test_dataset = VecBezierMNIST(data_dir=DATA_DIR, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=4, shuffle=True, batch_size=args.batch_size
    )

    # The VecBezierMNIST dataset flattens bezier curve coordinates
    # into a zero-padded vector. We use the default vector size, so
    # we should figure out how long the vectors actually are.
    dims = len(dataset[0][0])

    # We will train a simple multi-layer perceptron.
    model = nn.Sequential(
        Normalizer(),  # Scale down inputs to be in a reasonable range
        nn.Linear(dims, 256),
        Sin(),  # Help the model process continuous inputs
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 10),
    )
    model.to(DEVICE)
    param_count = sum(x.numel() for x in model.parameters())
    print(f"total parameters: {param_count}")

    # Create an optimizer for the model parameters.
    opt = Adam(model.parameters(), lr=args.lr)

    i = 0
    epoch = 0
    while True:
        for j, (samples, labels) in enumerate(loader):
            logits = model(samples.to(DEVICE)).cpu()
            loss = -(F.log_softmax(logits, dim=-1)[range(len(labels)), labels]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if j % 100 == 0:
                print(f"step {i}: loss={loss.item()}")
            i += 1

        print(f"*** epoch {epoch} complete, testing...")
        epoch += 1

        print(f"*** test accuracy: {accuracy(model, test_loader)}")
        print(f"*** train accuracy: {accuracy(model, loader)}")


def accuracy(model, loader):
    total = 0
    correct = 0
    for samples, labels in loader:
        with torch.no_grad():
            preds = model(samples.to(DEVICE)).cpu().argmax(-1)
        total += len(labels)
        correct += (labels == preds).long().sum().item()
    return f"{(100*correct/total):.02f}%"


class Normalizer(nn.Module):
    def forward(self, x):
        return x / 28


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


if __name__ == "__main__":
    main()
