import argparse
import time

import numpy as np
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--d_model',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--d_ff',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--context_length',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    args = get_args()
    print("training args:", args)

    total_tokens = 327_680_000
    steps = total_tokens // (args.batch_size * args.context_length)

    device = 'cuda'
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000,
    ).to(device=device)
    np_dataset = np.load(args.dataset_path, mmap_mode='r')

    lr = 1e-3
    optimizer = AdamW(params=model.parameters(), lr=lr)

    for step in range(steps):
        token_ids, targets = get_batch(
            np_dataset, args.batch_size, args.context_length, device
        )

        begin = time.time()
        logits = model(token_ids)
        loss = cross_entropy(logits, targets)
        # torch.cuda.synchronize()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.cuda.synchronize()
        print(f"step {step}/{steps}, loss: {loss.item():.4f}, time: {time.time() - begin:.4f}")


if __name__ == '__main__':
    main()
