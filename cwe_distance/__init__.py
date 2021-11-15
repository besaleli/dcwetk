from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import gc


def plot_wsd_cluster(candidate, tokens):
    """
    {'n_clusters': candidate[0],
                             'clustering_method': candidate[1].__name__,
                             'silhouetteScore': candidate[2],
                             'df': df}
    """
    df = candidate['df']
    n_clusters_str = '# Clusters: ' + str(candidate['n_clusters'])
    clustering_method_str = 'Clustering Method: ' + candidate['clustering_method']
    silhouette_score_str = 'Silhouette Score: ' + str(np.round(candidate['silhouette_score'], decimals=4))

    plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='copper')
    plt.title(' '.join(tokens) + '\n' + n_clusters_str + " | " + clustering_method_str + " | " + silhouette_score_str)
    plt.show()


class wum:
    """
    Initiates instance of word usage matrix.

    Stored in a np array.

    Parameters
    __________
    u : arr-like object
        List of contextualized word embeddings
    """

    def __init__(self, u, token):
        # copy constructor
        if type(u) == wum:
            self.u = np.array([vec for vec in u.getWUM()])
            self.tokens = [t for t in u.getTokens()]

        # default constructor
        else:
            # ensure everything is in np arrays so that it goes *fast*
            self.u = np.array([np.array(vec) for vec in u])
            self.tokens = []
            for i in range(len(self.u)):
                self.tokens.append(token)

    def __add__(self, other):
        u = self.u + other.getWUM()
        tokens = self.tokens + other.getTokens()

        return wum(u, tokens)

    """
    Accessor fn for self.u
    
    Returns
    -------
    np.arr
        word usage matrix
    """

    def getWUM(self):
        return self.u

    def getTokens(self):
        return self.tokens

    """
    Returns prototype of wum object
    
    The prototype of a word-usage-matrix is the average (vector) of all of the word usage matrix's constituent
    contextualized word embeddings.
    
    Returns
    -------
    np.array
        prototype of self.u
    """

    def prototype(self):
        prototype = np.sum(self.u) / self.u.size

        return prototype

    """
    Calculates inverted cosine similarity over word prototypes of the wum object and another wum object
    
    Parameters
    ----------
    other_wum : wum
        another word usage matrix from another time period in a wum object
    
    Returns
    -------
        Inverted cosine similarity over word prototypes of this and other wum objects
    """

    def prt(self, other_wum):
        p1, p2 = self.prototype(), other_wum.prototype()
        return 1 / distance.cosine(p1, p2)

    """
    Calculates average pairwise cosine distance between token embeddings of the wum object, given another wum object
    """

    def apd(self):
        pass

    """
    Calculates Jensen-Shannon Divergence between embedding clusters of the wum object and another wum object
    """

    def jsd(self):
        pass

    """
    Calculates difference between token embedding diversities of wum object and another wum object
    """

    def div(self, other_wum):
        p1, p2 = self.prototype(), other_wum.prototype()
        dists_from_p1 = np.array([distance.cosine(vec, p1) for vec in self.u])
        dists_from_p2 = np.array([distance.cosine(vec, p2) for vec in other_wum.getWUM()])
        var_coefficient_1 = np.sum(dists_from_p1) / self.u.size
        var_coefficient_2 = np.sum(dists_from_p2) / other_wum.getWUM().size

        return abs(var_coefficient_1 - var_coefficient_2)

    def get_pca(self, n_components=2):
        m = PCA(n_components=n_components)
        return m.fit_transform(self.u)

    def silhouette_analysis(self, n_candidates=3, pcaDim=2):
        # define possible clustering methods
        methods = [KMeans, SpectralClustering, AgglomerativeClustering]
        # define random state for consistency
        random_state = 10
        # initiate pca for WUM for the sake of memory lol
        wum_pca = self.get_pca(n_components=pcaDim)
        bestScores = []

        # for i in possible clusters 1-10:
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=random_state).fit(wum_pca)
            kmeans_labels = kmeans.labels_

            spectral = SpectralClustering(n_clusters=i, random_state=random_state).fit(wum_pca)
            spectral_labels = spectral.labels_

            agglomerative = AgglomerativeClustering(n_clusters=i).fit(wum_pca)
            agglomerative_labels = agglomerative.labels_

            labels = [kmeans_labels, spectral_labels, agglomerative_labels]
            scores = [(methods[i], silhouette_score(wum_pca, labels[i], random_state=10)) for i in range(3)]
            scores.sort(key=lambda j: j[1], reverse=True)
            bestScore = scores[0]
            # append   candidate   score    model
            bestScores.append((i, bestScore[0], bestScore[1]))

        # sort scores by score in descending order
        bestScores.sort(key=lambda k: k[2], reverse=True)

        return bestScores[:n_candidates], wum_pca

    def autoCluster(self, n_candidates: int, randomState=None, plot=False):
        needsRandomState = [KMeans, SpectralClustering]
        # doesNotNeedRandomState = [AgglomerativeClustering]
        rState = 10 if randomState is None else randomState

        candidates, pca = self.silhouette_analysis(n_candidates=n_candidates)
        candidatesData = []

        # candidate is structured (n_clusters, clustering_method, score)
        for candidate in candidates:
            # if candidate clustering method requires a random state:
            clusteringMethod = candidate[1]
            if clusteringMethod in needsRandomState:
                model = clusteringMethod(n_clusters=candidate[0], random_state=rState).fit(pca)
            else:
                model = clusteringMethod(n_clusters=candidate[0]).fit(pca)

            df = pd.DataFrame({'words': self.tokens,
                               'x': [i[0] for i in pca],
                               'y': [i[1] for i in pca],
                               'cluster': list(model.labels_)})

            candidateData = {'n_clusters': candidate[0],
                             'clustering_method': candidate[1].__name__,
                             'silhouette_score': candidate[2],
                             'df': df}

            candidatesData.append(candidateData)

        # sort the candidates!
        if n_candidates > 1:
            candidatesData.sort(key=lambda i: i['silhouette_score'], reverse=True)

        # plot if needed
        if plot:
            for candData in candidatesData:
                plot_wsd_cluster(candData, [self.tokens[0][0]])  # why does it do this?

        return candidatesData


class wumGen:
    def __init__(self, df, verbose=False):
        verboseCond = lambda i: tqdm(i) if verbose else i
        self.embeddings = df['embeddings'].to_list()
        self.tokens = df['tokens'].to_list()
        if verbose: print('getting vocab info...')
        self.size = len(self.tokens)
        self.vocab = set(self.tokens)
        if verbose: print('constructing individual word usage matrices...')
        self.WUMs = {tok: self.getWordUsageMatrix_Individual(tok) for tok in verboseCond(self.vocab)}
        if verbose: print('calculating word usage matrix prototypes...')
        self.prototypes = {tok: self.WUMs[tok].prototype() for tok in verboseCond(self.vocab)}

    def getTokens(self):
        return self.tokens

    def getSize(self):
        return self.size

    def getWUMs(self):
        return self.WUMs

    def getPrototypes(self):
        return self.prototypes

    def getWordUsageMatrix_Individual(self, token):
        embeddings = self.embeddings

        vecs = [embeddings[i] for i in range(len(embeddings)) if self.tokens[i] == token]

        return wum(np.array(vecs), [token] * len(self.tokens))
