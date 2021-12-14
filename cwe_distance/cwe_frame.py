import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from . import wum, wumGen
from tqdm.auto import tqdm


def getVocabulary(docs):
    vocab = set()
    for doc in docs:
        for s in doc:
            for tok in s:
                for lem in tok:
                    vocab.add(lem.lemma)

    return vocab


def getLemmas(docs, search_token):
    instances = []
    addresses = []
    for doc in docs:
        for s in doc:
            for tok in s:
                for lem in tok:
                    if lem.lemma == search_token:
                        addr = [doc.doc_ID, s.sent_ID, tok.tok_ID]
                        instances.append(lem)
                        addresses.append(addr)

    return instances, addresses


def findElement(docs, addr):
    # document address
    if len(addr) == 1:
        return docs[addr[0]]
    # sentence address
    elif len(addr) == 2:
        return docs[addr[0]][addr[1]]
    # token address
    elif len(addr) == 3:
        return docs[addr[0]][addr[1]][addr[2]]
    else:
        raise KeyError('Invalid address')


def makeWUM(docs, search_token):
    lemmas, addresses = getLemmas(docs, search_token)
    if len(lemmas) > 2:
        u = np.array([l.embedding for l in lemmas])
        toks = [search_token] * len(u)
        return wum(u, addresses=addresses, token=toks, pcaFirst=True)
    else:
        return None


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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __gt__(self, other, midpoint=None):
        m = np.zeros(len(self.embedding)) if midpoint is None else midpoint
        this_distance = cosine(self.embedding, m)
        other_distance = cosine(self.embedding, m)
        return this_distance > other_distance

    def __lt__(self, other, midpoint=None):
        m = np.zeros(len(self.embedding)) if midpoint is None else midpoint
        this_distance = cosine(self.embedding, m)
        other_distance = cosine(self.embedding, m)
        return this_distance < other_distance


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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __len__(self):
        return len(self.lemmas)

    def __str__(self):
        return self.raw_tok


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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __len__(self):
        return len(self.raw_text_tokenized)


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
        return '\n'.join(str(sent) for sent in self.sents)

    def __getitem__(self, item):
        return self.sents[item]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __len__(self):
        return len(self.sents)


def make_WUM(tok, tokens, embeddings, addresses, pcaFirst, n_components, random_state):
    vecs = []
    adds = []
    for i in range(len(tokens)):
        if tok == tokens[i]:
            vecs.append(embeddings[i])
            adds.append(addresses[i])

    return wum(np.array(vecs), addresses=adds, token=[tok] * len(vecs), pcaFirst=pcaFirst, n_components=n_components,
               random_state=random_state)


def make_wumGen_frame(docs, verbose=False, minOccs=1, pcaFirst=False, n_components=2, random_state=10):
    tqdmCond = lambda i: tqdm(i) if verbose else i
    df = dict()
    df['pcaFirst'] = pcaFirst
    df['n_components'] = n_components
    df['random_state'] = random_state

    df['tokens'] = []
    embeddings = []
    addresses = []
    for doc in docs:
        for sent in doc:
            for tok in sent:
                for lem in tok:
                    df['tokens'].append(lem.lemma)
                    df['embeddings'].append(lem.embedding)
                    address = {'tok_ID': tok.tok_ID,
                               'sent_ID': sent.sent_ID,
                               'doc_ID': doc.doc_ID}
                    addresses.append(address)

    df['embeddings'] = PCA(n_components=n_components).fit_transform(embeddings) if pcaFirst else embeddings
    df['size'] = len(df['tokens'])
    df['vocab'] = set(df['tokens'])
    mWUM = lambda i: make_WUM(i, df['tokens'], df['embeddings'], df['addresses'],
                              pcaFirst, n_components, random_state)
    df['WUMs'] = {tok: mWUM(tok) for tok in tqdmCond(df['vocab']) if df['tokens'].count(tok) >= minOccs}

    return wumGen(df, verbose=False, minOccs=minOccs, pcaFirst=pcaFirst, n_components=n_components,
                  random_state=random_state)
