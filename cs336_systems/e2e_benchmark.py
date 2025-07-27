import argparse
import time

import torch
import torch.cuda.nvtx as nvtx
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
        '--steps',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--auto_nvtx',
        action='store_true',
    )
    return parser.parse_args()


def manual_nvtx(model, step, token_ids, targets, optimizer):
    nvtx.range_push(f"forward {step}")
    begin = time.time()
    logits = model(token_ids)
    loss = cross_entropy(logits, targets)
    # torch.cuda.synchronize()
    print(f" forward time at {step} time: {time.time() - begin:.4f}")
    nvtx.range_pop()

    nvtx.range_push(f"backward {step}")
    begin = time.time()
    nvtx.range_push(f"zero_grad {step}")
    optimizer.zero_grad()
    nvtx.range_pop()
    nvtx.range_push(f"autograd {step}")
    loss.backward()
    nvtx.range_pop()
    nvtx.range_push(f"optimizer.step {step}")
    optimizer.step()
    nvtx.range_pop()
    # torch.cuda.synchronize()
    print(f"backward time at {step} time: {time.time() - begin:.4f}, loss: {loss.item():.4f}")
    nvtx.range_pop()


def auto_nvtx(model, step, token_ids, targets, optimizer):
    with torch.autograd.profiler.emit_nvtx():
        begin = time.time()
        logits = model(token_ids)
        loss = cross_entropy(logits, targets)
        # torch.cuda.synchronize()
        print(f" forward time at {step} time: {time.time() - begin:.4f}")

        begin = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.cuda.synchronize()
        print(f"backward time at {step} time: {time.time() - begin:.4f}, loss: {loss.item():.4f}")


def main():
    args = get_args()
    print(args)
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

    lr = 1e-4
    optimizer = AdamW(params=model.parameters(), lr=lr)

    steps = args.steps
    for step in range(steps):
        nvtx.range_push(f"prepare data {step}")
        token_ids = torch.randint(
            args.vocab_size,
            size=(args.batch_size, args.context_length),
            device=device,
        )
        targets = torch.randint(
            args.vocab_size,
            size=(args.batch_size, args.context_length),
            device=device,
        )
        nvtx.range_pop()

        if args.auto_nvtx:
            step_func = auto_nvtx
        else:
            step_func = manual_nvtx

        step_func(model, steps, token_ids, targets, optimizer)


# basic run
# uv run cs336_systems/e2e_benchmark.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 1024
# launch nsys
# uv run nsys profile -o result \
#   python cs336_systems/e2e_benchmark.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 1024
# uv run nsys profile --python-backtrace=cuda -o result-trace-python \
#   python cs336_systems/e2e_benchmark.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 1024 --steps 20
if __name__ == '__main__':
    main()
