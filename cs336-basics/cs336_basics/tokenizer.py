from typing import Iterable, Iterator

import regex as re


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        # sort special_tokens by length in descending order
        special_tokens.sort(key=len, reverse=True)
        self.special_tokens = special_tokens
        if self.special_tokens:
            self.demi = "(" + "|".join([re.escape(st) for st in special_tokens]) + ")"
        else:
            self.demi = None

        words = vocab.values()
        for special_token in special_tokens:
            special_token_bytes = special_token.encode('utf-8')
            if special_token_bytes not in words:
                vocab[len(vocab)] = special_token_bytes

        self.vocab = vocab
        self.word_to_id = {word: wid for wid, word in vocab.items()}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    @staticmethod
    def match_merges(subword_bytes, merges):
        pos = None
        for pair in merges:
            for i in range(len(subword_bytes) - 1):
                if subword_bytes[i] == pair[0] and subword_bytes[i + 1] == pair[1]:
                    return i
        return pos

    def encode_pre_token(self, subword):
        subword_bytes = [bytes([b]) for b in subword.encode('utf-8')]

        while True:
            pos = self.match_merges(subword_bytes, self.merges)
            if pos is None:
                break
            subword_bytes = subword_bytes[:pos] + \
                [subword_bytes[pos] + subword_bytes[pos + 1]] +\
                subword_bytes[pos + 2:]

        return [self.word_to_id[bs] for bs in subword_bytes]

    def encode(self, text: str) -> list[int]:
        ids = []
        if self.demi is not None:
            for chunk in re.split(self.demi, text):
                if chunk in self.special_tokens:
                    ids.append(self.word_to_id[chunk.encode('utf-8')])
                else:
                    for mat in re.finditer(self.PAT, chunk):
                        subword = mat.group(0)
                        ids.extend(self.encode_pre_token(subword))
        else:
            for mat in re.finditer(self.PAT, text):
                subword = mat.group(0)
                ids.extend(self.encode_pre_token(subword))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        arr = [self.vocab[wid] for wid in ids]
        whole_bytes = b''.join(arr)

        return whole_bytes.decode('utf-8', errors="ignore")
