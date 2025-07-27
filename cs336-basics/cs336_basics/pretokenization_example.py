import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def partial_chunk(args):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    path, start, end = args
    # print(f"Processing chunk from {start} to {end} in {path}")
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    demi = "|".join([re.escape("<|endoftext|>")])
    result = defaultdict(int)
    for item in re.split(demi, chunk):
        for match in re.finditer(PAT, item):
            result[match.group(0)] += 1
    return result


def main():
    path = sys.argv[1]
    num_processes = int(sys.argv[2])
    out = parallel_pre_tokenization(path, num_processes)
    print(out)


def parallel_pre_tokenization(path, num_processes):
    pool = ProcessPoolExecutor(max_workers=num_processes)
    # Usage
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
    args = [(path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    result = list(pool.map(partial_chunk, args))
    # Run pre-tokenization on your chunk and store the counts for each pre-token

    # for chunk in result:
    #     print(len(chunk), chunk[-100:], chunk[:100], chunk.find("<|endoftext|>"))

    def split_into_bytes(val):
        assert isinstance(val, str), "Expected string input"
        out = [bytes([b]) for b in val.encode('utf-8')]
        return (*out,)

    out = defaultdict(int)
    for obj in result:
        for k, v in obj.items():
            out[split_into_bytes(k)] += v
    return out


if __name__ == "__main__":
    main()
