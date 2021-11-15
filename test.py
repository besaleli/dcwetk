import pickle
import numpy as np
from cwe_distance import wum, wumGen
import pandas as pd
from tqdm import tqdm
from nltk.probability import FreqDist

fileName = 'byp_1879.pickle'
with open(fileName, 'rb') as f:
    byp = pickle.load(f)

dfs = [byp['docs'][i]['embeddings'] for i in tqdm(range(len(byp['docs'])))]
df = pd.concat(dfs)
wums = wumGen(df)
tokens = wums.getTokens()

fdist = FreqDist(tokens)
over50Occs = {}
significantWUMs = {}
print('finding significant word usage matrices...')
for token, freq in tqdm(fdist.items()):
    if freq >= 100:
        print(token, freq)
        significantWUMs[token] = wums.getWUMs()[token]

print(str(len(significantWUMs.items())) + ' significant word usage matrices found.')

for token, matrix in tqdm(significantWUMs.items()):
    matrix.autoCluster(1, randomState=None, plot=True)
