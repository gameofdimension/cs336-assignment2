from collections import defaultdict

from cs336_basics.pretokenization_example import parallel_pre_tokenization


class PairCount:
    def __init__(self, pre_tokens: dict[tuple[bytes], int]):
        self.pre_tokens = pre_tokens

    def get_most_frequent_pairs(self):
        if len(self.pre_tokens) == 0:
            return None
        res = defaultdict(int)
        for k, v in self.pre_tokens.items():
            for i in range(len(k) - 1):
                pair = (k[i], k[i + 1])
                res[pair] += v
        items = sorted(res.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return (items[0][0], items[0][1])

    def merge_within_token(self, token: tuple[bytes], pair: tuple[bytes, bytes]):
        i = 0
        res = []
        while True:
            if i > len(token) - 1:
                break
            if i == len(token) - 1:
                res.append(token[i])
                break
            if (token[i], token[i + 1]) == pair:
                res.append(token[i] + token[i + 1])
                i += 2
            else:
                res.append(token[i])
                i += 1
        return (*res,)

    def merge_pair(self, pair: tuple[bytes, bytes]):
        # This method should merge the given pair of tokens in the pre_tokens dictionary
        res = {}
        for k, v in self.pre_tokens.items():
            new_token = self.merge_within_token(k, pair)
            res[new_token] = v
        self.pre_tokens = res


class EfficientPairCount:
    def __init__(self, pre_tokens: dict[tuple[bytes], int]):
        self.pre_tokens = pre_tokens
        self.pair_to_token = defaultdict(list)

        for token in self.pre_tokens:
            for i in range(len(token) - 1):
                self.add_to_lst((token[i], token[i + 1]), token)

    def add_to_lst(self, pair, token):
        self.pair_to_token[pair].append(token)

    def del_from_lst(self, pair, token):
        new_lst = []

        for item in self.pair_to_token[pair]:
            if item != token:
                new_lst.append(item)
        self.pair_to_token[pair] = new_lst

    def get_most_frequent_pairs(self):
        if len(self.pair_to_token) == 0:
            return None

        max_pair = (0, (bytes([0]), bytes([0])))
        for pair in self.pair_to_token:
            count = sum([self.pre_tokens[item] for item in self.pair_to_token[pair]])
            if (count, pair) > max_pair:
                max_pair = (count, pair)
        return (max_pair[1], max_pair[0])

    def merge_within_token(self, token: tuple[bytes], count: int, pair: tuple[bytes, bytes]):
        i = 0
        res = []
        while True:
            if i > len(token) - 1:
                break
            if i == len(token) - 1:
                res.append(token[i])
                break
            if (token[i], token[i + 1]) == pair:
                res.append(token[i] + token[i + 1])
                i += 2
            else:
                res.append(token[i])
                i += 1
        new_token = (*res,)
        if token in self.pre_tokens:
            del self.pre_tokens[token]

            for i in range(len(token) - 1):
                tmp = (token[i], token[i + 1])
                self.del_from_lst(tmp, token)

            self.pre_tokens[new_token] = count
            for i in range(len(new_token) - 1):
                tmp = (new_token[i], new_token[i + 1])
                self.add_to_lst(tmp, new_token)

    def merge_pair(self, pair: tuple[bytes, bytes]):
        # This method should merge the given pair of tokens in the pre_tokens dictionary
        tokens = self.pair_to_token[pair]
        counts = [self.pre_tokens[token] for token in tokens]
        for token, count in zip(tokens, counts):
            # assert any([(a, b) == pair for a, b in zip(token, token[1:])])
            assert count > 0
            self.merge_within_token(token, count, pair)


def toy_tokens():
    sss = '''low low low low low
lower lower widest widest widest
newest newest newest newest newest newest'''
    count = defaultdict(int)
    for item in sss.split():
        count[item] += 1

    res = defaultdict(int)
    for k, v in count.items():
        tup = [bytes([x]) for x in k.encode('utf-8')]
        res[(*tup,)] = v

    return res


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    # pre_tokens = toy_tokens()
    pre_tokens = parallel_pre_tokenization(input_path, 8)

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    for token in special_tokens:
        token = token.encode("utf-8")
        vocab[len(vocab)] = token
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    # pair_counter = PairCount(pre_tokens)
    pair_counter = EfficientPairCount(pre_tokens)

    while True:
        if len(vocab) >= vocab_size:
            break

        most_frequent_pair, count = pair_counter.get_most_frequent_pairs()
        if most_frequent_pair is None:
            break

        assert len(most_frequent_pair) == 2
        merges.append(most_frequent_pair)
        vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1]
        pair_counter.merge_pair(most_frequent_pair)

    return vocab, merges


def main():
    vocab, merges = train_bpe(
        '',
        263,
        ['<|endoftext|>'],
    )
    print(vocab)
    print(merges)


if __name__ == '__main__':
    main()
