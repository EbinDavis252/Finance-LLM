import re

class SimpleTokenizer:
    def __init__(self, text):
        self.vocab = sorted(set(re.findall(r'\b\w+\b', text.lower())))
        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def encode(self, text):
        return [self.word2idx[word] for word in re.findall(r'\b\w+\b', text.lower()) if word in self.word2idx]

    def decode(self, tokens):
        return ' '.join([self.idx2word[token] for token in tokens])
