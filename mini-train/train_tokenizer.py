import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from cs336_basics.bpe import train_bpe
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


def get_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        state = pickle.load(handle)
    vocab = state['vocab']
    merges = state['merges']
    return Tokenizer(
        vocab,
        merges,
        special_tokens=['<|endoftext|>']
    )


def encode():
    tokenizer_path = sys.argv[1]
    dataset_path = sys.argv[2]
    save_path = sys.argv[3]
    tokenizer = get_tokenizer(tokenizer_path)

    all_ids = []
    with open(dataset_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)

    with open(save_path, 'wb') as handle:
        pickle.dump(all_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)


def partial_encode(args):
    tokenizer_path, dataset_path, start, end = args
    tokenizer = get_tokenizer(tokenizer_path)
    print(f"chunk size: {end - start}")
    with open(dataset_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return tokenizer.encode(chunk)


def parallel_encode():
    tokenizer_path = sys.argv[1]
    dataset_path = sys.argv[2]
    save_path = sys.argv[3]
    num_processes = int(sys.argv[4])

    pool = ProcessPoolExecutor(max_workers=num_processes)
    # Usage
    with open(dataset_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

    args = [(tokenizer_path, dataset_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    result = []
    for lst in list(pool.map(partial_encode, args)):
        result.extend(lst)

    array = np.array(result)
    np.save(save_path, array)


def train():
    input_path = sys.argv[1]
    vocab_size = int(sys.argv[2])
    save_path = sys.argv[3]
    vocab, merges = train_bpe(
        input_path,
        vocab_size,
        special_tokens=['<|endoftext|>'],
    )

    res = dict(vocab=vocab, merges=merges)
    with open(save_path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # train()
    # encode()
    parallel_encode()
