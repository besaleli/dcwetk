from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def plot_wsd_cluster(candidate, plotDim=2):
    df = candidate['df']
    pcaEmbeddings = df['cwe_pca']
    varNames = ['x', 'y', 'z']
    data = {varNames[i]: list(map(lambda j: j[i], pcaEmbeddings)) for i in range(plotDim)}

    if plotDim == 2:
        plt.scatter(data['x'], data['y'], c=df['cluster'], cmap='viridis')
    elif plotDim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data['x'], data['y'], data['z'], c=df['cluster'], cmap='viridis')


def pca(data, n_components=2, extra_columns=None, asDF=True):
    model = PCA(n_components=n_components)
    principalComponents = model.fit_transform(data)

    if asDF:
        if extra_columns is None:
            extra_columns = []
        principalDF = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])
        principleDF_with_extra_cols = pd.concat([principalDF] + [extra_columns], axis=1)

        return principleDF_with_extra_cols, model.get_precision()
    else:
        return principalComponents, model.get_precision()


def get_silhouette_score(data, n_clusters=1, model=AgglomerativeClustering):
    m = model(n_clusters=n_clusters)
    if data is None:
        clusters = m.fit(data)
    else:
        clusters = m.fit(data)
    score = silhouette_score(data, clusters)

    return score


def silhouette(data, model=AgglomerativeClustering):
    clusters: range = range(1, 11)
    scores = []

    for k in clusters:
        scores.append((k, get_silhouette_score(data, n_clusters=1, model=model)))

    scores.sort(key=lambda i: i[1], reverse=True)

    return scores


class wum:
    needsRandomState = [KMeans, SpectralClustering]

    """
    Initiates instance of word usage matrix.

    Stored in a np array.

    Parameters
    __________
    u : arr-like object
        List of contextualized word embeddings
    """

    def __init__(self, u):
        # copy constructor
        if type(u) == wum:
            self.u = np.array([vec for vec in u.get_wum()])

        # default constructor
        else:
            # ensure everything is in np arrays so that it goes *fast*
            self.u = np.array([np.array(vec) for vec in u])

    """
    Accessor fn for self.u
    
    Returns
    -------
    np.arr
        word usage matrix
    """

    def get_wum(self):
        return self.u

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
        dists_from_p2 = np.array([distance.cosine(vec, p2) for vec in other_wum.get_wum()])
        var_coefficient_1 = np.sum(dists_from_p1) / self.u.size
        var_coefficient_2 = np.sum(dists_from_p2) / other_wum.get_wum().size

        return abs(var_coefficient_1 - var_coefficient_2)

    def wsd_cluster(self, num_candidates=1, model=AgglomerativeClustering, plot=False, plotDim=2, pcaFirst=True):
        pcaData, pcaPrecision = pca(self.u, n_components=plotDim, asDF=False)

        data = pcaData if pcaFirst else self.u

        scores = silhouette(data, model=model)

        candidates = {}

        for candidate, score in enumerate(scores[:num_candidates]):
            if candidate in wum.needsRandomState:
                m = model(n_clusters=candidate, random_state=10)
            else:
                m = model(n_clusters=candidate)
            clusters = m.fit(data)
            labels = clusters.labels_
            df = pd.DataFrame({'cwe_pca': pcaData, 'cluster': labels})
            candidates[candidate] = {'score': score,
                                     'df': df}

        if plot:
            for d in candidates.values():
                if plotDim == 2 or plotDim == 3:
                    plot_wsd_cluster(d, plotDim=plotDim)
                else:
                    print('invalid plot dimension')

        return candidates, pcaPrecision
