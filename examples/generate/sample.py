"""
Sample an image from an autoregressive Transformer.

If the output of the model follows the correct format, the resulting image is
written to `sample.svg`.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_bezier_mnist import TokenBezierMNIST, beziers_to_svg

from train import DATA_DIR, DEVICE, TransformerModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pt")
    args = parser.parse_args()

    dataset = TokenBezierMNIST(data_dir=DATA_DIR, split="train")

    tokenizer = dataset.tokenizer
    seq_len = dataset.seq_len

    model = TransformerModel(tokenizer.num_tokens, seq_len)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(DEVICE)

    tokens = [tokenizer.start_token]
    while tokenizer.end_token not in tokens and len(tokens) < seq_len:
        print(f"Sampling token {len(tokens)}...")
        in_tokens = tokens + [0] * (seq_len - len(tokens))
        in_seq = torch.tensor(in_tokens, dtype=torch.long)[None]
        probs = F.softmax(model(in_seq.to(DEVICE)).cpu(), dim=-1)[0, len(tokens) - 1]
        choice = np.random.choice(len(probs), p=probs.detach().numpy())
        tokens.append(choice)

    print("tokens:", tokens)

    loops, complete = tokenizer.decode_loops(tokens)
    print("completeness:", complete)
    print("num loops:", len(loops))
    with open("sample.svg", "wt") as f:
        f.write(beziers_to_svg(loops))


if __name__ == "__main__":
    main()
