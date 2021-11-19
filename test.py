import pickle
# import numpy as np
from cwe_distance import wumGen
import pandas as pd
from tqdm import tqdm
# from nltk.probability import FreqDist
# from sklearn.decomposition import PCA

fileName = 'byp_1872.pickle'
with open(fileName, 'rb') as f:
    byp = pickle.load(f)

dfs = [byp['docs'][i]['embeddings'] for i in tqdm(range(len(byp['docs'])))]
df = pd.concat(dfs)
# embeddings = df['embeddings'].to_list()
# embeddings_pca = PCA(n_components=2).fit_transform(embeddings)
# print(embeddings_pca[0])
wums = wumGen(df, verbose=True)
tokens = wums.getTokens()
israel = wums.getWordUsageMatrix_Individual('ב')
israel.autoCluster(n_candidates=2, plot=True)