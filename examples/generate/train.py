import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_bezier_mnist import TokenBezierMNIST

DATA_DIR = "../../"
MODEL_PATH = "model.pt"
BATCH_SIZE = 4
LOG_INTERVAL = 10
SAVE_INTERVAL = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Create the training dataset loader.
    dataset = TokenBezierMNIST(data_dir=DATA_DIR)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=4, shuffle=True, batch_size=BATCH_SIZE
    )

    # Create the testing dataset so we can evaluate on it separately.
    test_dataset = TokenBezierMNIST(data_dir=DATA_DIR, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=4, shuffle=True, batch_size=BATCH_SIZE
    )

    tokenizer = dataset.tokenizer
    seq_len = dataset.seq_len

    model = TransformerModel(tokenizer.num_tokens, seq_len)
    model.to(DEVICE)
    param_count = sum(x.numel() for x in model.parameters())
    print(f"total parameters: {param_count}")

    # Create an optimizer for the model parameters.
    opt = Adam(model.parameters(), lr=1e-4)

    def compute_loss(tokens):
        targets = torch.cat(
            [
                tokens[:, 1:],
                torch.zeros_like(tokens[:, -1:]) + tokenizer.end_token,
            ],
            dim=1,
        )
        logits = model(targets.to(DEVICE)).cpu()
        return F.cross_entropy(logits.permute(0, 2, 1), targets)

    for i, ((train_tokens, _), (test_tokens, _)) in enumerate(
        zip(load_forever(loader), load_forever(test_loader))
    ):
        loss = compute_loss(train_tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % LOG_INTERVAL == 0:
            with torch.no_grad():
                test_loss = compute_loss(test_tokens)
            print(f"step {i}: train={loss.item()} test={test_loss.item()}")
        if i % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), MODEL_PATH)


def load_forever(loader):
    while True:
        for x in loader:
            yield x


class TransformerModel(nn.Module):
    def __init__(self, num_tokens: int, seq_len: int):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, 256)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, 1, 256))
        self.out = nn.Linear(256, num_tokens)

        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        layer = nn.TransformerEncoderLayer(256, 4, dim_feedforward=1024, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(layer, 8)

        # https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        self.register_buffer("mask", mask)

    def forward(self, seq):
        """
        :param seq: an [N x T] sequence of tokens.
        :return: an [N x T x D] sequence of logits.
        """
        x = self.emb(seq.permute(1, 0).contiguous())
        x = x + self.pos_emb
        x = self.transformer_encoder(x, self.mask)
        x = self.out(x)
        x = x.permute(1, 0, 2).contiguous()
        return x


if __name__ == "__main__":
    main()
