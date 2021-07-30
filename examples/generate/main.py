import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_bezier_mnist import TokenBezierMNIST

DATA_DIR = "../../"
BATCH_SIZE = 4


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
    param_count = sum(x.numel() for x in model.parameters())
    print(f"total parameters: {param_count}")

    # Create an optimizer for the model parameters.
    opt = Adam(model.parameters(), lr=1e-3)

    i = 0
    epoch = 0
    while True:
        for j, (tokens, _labels) in enumerate(loader):
            targets = torch.cat(
                [tokens[:, 1:], torch.zeros_like(tokens[:, -1:]) + tokenizer.end_token],
                dim=1,
            )
            logits = model(targets)
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if j % 10 == 0:
                print(f"step {i}: loss={loss.item()}")
            i += 1

        print(f"*** epoch {epoch} complete!")
        epoch += 1


class TransformerModel(nn.Module):
    def __init__(self, num_tokens: int, seq_len: int):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, 256)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, 1, 256))
        self.out = nn.Linear(256, num_tokens)

        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        layer = nn.TransformerEncoderLayer(256, 4, dim_feedforward=1024, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(layer, 8)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool),
                diagonal=1,
            ),
        )

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
