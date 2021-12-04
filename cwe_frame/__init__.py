import numpy as np


class lemma:
    def __init__(self, lemmaInfo=None):
        # dict constructor
        if lemmaInfo is not None:
            self.lemma = lemmaInfo['lemma']
            self.form = lemmaInfo['form']
            self.pos_tag = lemmaInfo['pos_tag']
            self.feats = lemmaInfo['feats']
            self.index = lemmaInfo['index']
            self.embedding = lemmaInfo['embedding']
        # null constructor
        else:
            self.lemma = str()
            self.form = str()
            self.pos_tag = str()
            self.feats = dict()
            self.index = int()
            self.embedding = np.zeros(768, dtype=float)

    def __str__(self):
        return self.lemma


class token:
    def __init__(self, tokInfo=None):
        # dict constructor
        if tokInfo is not None:
            self.tok_ID = tokInfo['tok_ID']
            self.raw_tok = tokInfo['raw_tok']
            self.lemmas = tokInfo['lemmas']
        # null constructor
        else:
            self.tok_ID = int()
            self.raw_tok = str()
            self.lemmas = list()

    def __iter__(self):
        return iter(self.lemmas)

    def __getitem__(self, item):
        return self.lemmas[item]


class sentence:
    def __init__(self, sentInfo=None):
        # dict constructor
        if sentInfo is not None:
            self.sent_ID = sentInfo['sent_ID']
            self.raw_text_tokenized = sentInfo['raw_text_tokenized']
            self.tokens = sentInfo['tokens']
        # null constructor
        else:
            self.sent_ID = int()
            self.raw_text_tokenized = list()
            self.tokens = list()  # list of token instances

    def __iter__(self):
        return iter(self.tokens)

    def __str__(self):
        return ' '.join(self.raw_text_tokenized)

    def __getitem__(self, item):
        return self.tokens[item]


class document:
    def __init__(self, docInfo=None):
        # dict constructor
        if docInfo is not None:
            self.doc_ID = docInfo['doc_ID']
            self.year = docInfo['year']
            self.sents = docInfo['sents']
        else:
            self.doc_ID = int()
            self.year = int()
            self.sents = list()  # list of sentence instances

    def __iter__(self):
        return iter(self.sents)

    def __str__(self):
        return '\n'.join(str(self.sents))

    def __getitem__(self, item):
        return self.sents[item]

