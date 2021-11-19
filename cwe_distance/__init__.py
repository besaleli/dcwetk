from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# import json
# import gc


def plot_wsd_cluster(candidate, tokens):
    """
    Plots a cluster from a candidate dictionary

    Designed for internal use only

    Parameters
    ----------
    candidate : dict
        Candidate info
    tokens : list
        Token(s) represented by word usage matrix cluster

    Returns
    -------

    """

    df = candidate['df']
    # make plt title information
    n_clusters_str = '# Clusters: ' + str(candidate['n_clusters'])
    clustering_method_str = 'Clustering Method: ' + candidate['clustering_method']
    silhouette_score_str = 'Silhouette Score: ' + str(np.round(candidate['silhouette_score'], decimals=4))

    # make the plt plot
    plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='copper')
    plt.title(', '.join(tokens) + '\n' + n_clusters_str + " | " + clustering_method_str + " | " + silhouette_score_str)
    plt.show()


class wum:
    def __init__(self, u, token: str):
        """
        __init__ function for wum

        Parameters
        ----------
        u : np.array
            list of word vectors in np arrays
        token : list
            token that word usage matrix represents
        """
        # copy constructor
        if type(u) == wum:
            self.u = np.array([vec for vec in u.getWUM()])
            self.tokens = [t for t in u.getTokens()]

        # default constructor - for WUMs representing single token
        else:
            # ensure everything is in np arrays so that it goes *fast*
            self.u = np.array([np.array(vec) for vec in u])
            self.tokens = token  # list so tokens can be added together if WUM represents multiple tokens

    # need to fix this
    def __add__(self, other):
        u = self.u + other.getWUM()
        tokens = self.tokens + other.getTokens()

        return wum(u, tokens)

    def getWUM(self):
        """

        Returns
        -------
        np.array
            word usage matrix
        """
        return self.u

    def getTokens(self):
        """

        Returns
        -------
        list
            token(s) (list of str) that np array represents
        """
        return self.tokens

    def prototype(self):
        """
            Returns prototype of wum object

            The prototype of a word-usage-matrix is the average (vector) of all of the word usage matrix's constituent
            contextualized word embeddings.

            Returns
            -------
            np.array
                prototype of self.u
            """

        prototype = np.sum(self.u) / self.u.size

        return prototype

    def prt(self, other_wum):
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

        p1, p2 = self.prototype(), other_wum.prototype()
        return 1 / distance.cosine(p1, p2)

    # TODO
    def apd(self):
        """
        Calculates average pairwise cosine distance between token embeddings of the wum object, given another wum object

        Returns
        -------

        """
        pass

    # TODO
    def jsd(self):
        """
        Calculates Jensen-Shannon Divergence between embedding clusters of the wum object and another wum objects

        Returns
        -------

        """
        pass

    def div(self, other_wum):
        """
        Calculates difference between token embedding diversities of wum object and another wum object

        Parameters
        ----------
        other_wum : wum
            other wum

        Returns
        -------
        float
            DIV

        """

        p1, p2 = self.prototype(), other_wum.prototype()
        dists_from_p1 = np.array([distance.cosine(vec, p1) for vec in self.u])
        dists_from_p2 = np.array([distance.cosine(vec, p2) for vec in other_wum.getWUM()])
        var_coefficient_1 = np.sum(dists_from_p1) / self.u.size
        var_coefficient_2 = np.sum(dists_from_p2) / other_wum.getWUM().size

        return abs(var_coefficient_1 - var_coefficient_2)

    def get_pca(self, n_components=2):
        """
        Performs principal component analysis of word usage matrix given number of components [dimensionality reduction]

        Parameters
        ----------
        n_components : int
            number of components of desired PCA result


        Returns
        -------
        np.array
            PCA'd word usage matrix

        """
        m = PCA(n_components=n_components)
        return m.fit_transform(self.u)

    def silhouette_analysis(self, n_candidates=3, pcaDim=2):
        """

        Parameters
        ----------
        n_candidates : int
            specifies top n scorers
        pcaDim : int
            How many dimensions to reduce the word usage matrix to using principal component analysis

        Returns
        -------
            N candidates with top silhouette scores

        """
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
        """

        Parameters
        ----------
        n_candidates : int
            How many clusters to perform according to the silhouette analysis
        randomState : int
            Specify a random state for consistency
        plot : bool
            True if plotting the clusters is desired

        Returns
        -------
        list
            List of dictionaries containing candidate data (structure specified in src code)

        """
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
                plot_wsd_cluster(candData, set(self.tokens))  # why does it do this?

        """
        Each candidate in candidatesData is a dict with the following structure:
        {'n_clusters': candidate[0],
        'clustering_method': candidate[1].__name__,
        'silhouetteScore': candidate[2],
        'df': df}
        """

        return candidatesData

    def asDict(self, jsonFriendly=True):
        """
        Returns wum object as dictionary

        Parameters
        ----------
        jsonFriendly : bool
            True if JSON-serializable objects are necessary

        Returns
        -------
        dict
            wum data stored as dictionary

        """
        # lambda function for jsonFriendly conditional
        jsonCond = lambda i: i.tolist() if jsonFriendly else i

        # initialize empty dict
        data = dict()

        # add data members to dict
        data['u'] = [jsonCond(vec) for vec in self.u]
        data['tokens'] = self.tokens

        return data


class wumGen:
    def __init__(self, df, verbose=False, minOccs=10):
        """
        __init__ function for wumGen class

        Constructs and stores word usage matrices of singular words given a Pandas DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame containing 2 columns: 'tokens' and 'embeddings'
        verbose : bool
            Provides status updates on construction of word usage matrices via tqdm package if True
        """

        # dictionary constructor
        if type(df) == dict:
            pass

        # default constructor with pd DataFrame
        else:
            tqdm_cond = lambda i: tqdm(i) if verbose else i
            verboseCond = lambda i: print(i) if verbose else i

            self.embeddings = df['embeddings'].to_list()
            self.tokens = df['tokens'].to_list()

            verboseCond(print('getting vocab info...'))
            self.size = len(self.tokens)
            self.vocab = set(self.tokens)

            verboseCond(print('constructing individual word usage matrices...'))
            self.WUMs = {tok: self.getWordUsageMatrix_Individual(tok) for tok in tqdm_cond(self.vocab)
                         if self.tokens.count(tok) >= minOccs}

            verboseCond(print('calculating word usage matrix prototypes...'))
            self.prototypes = {tok: self.WUMs[tok].prototype() for tok in tqdm_cond(self.vocab)
                               if self.tokens.count(tok) >= minOccs}

    def getTokens(self):
        """
        Accessor for tokens data member

        Returns
        -------
        list
            tokens for embeddings contained by word usage matrix in self.WUMs of key 'token'
        """

        return self.tokens

    def getSize(self):
        """
        Accessor for size data member

        Returns
        -------
        int
            Number of WUMs in object (equal to number of unique tokens in the df argument of the constructor)
        """

        return self.size

    def getWUMs(self):
        """
        Accessor for WUMs data member

        Returns
        -------
        dict
            WUMs with given their representative token as their key
        """

        return self.WUMs

    def getPrototypes(self):
        """
        Accessor for prototypes data member

        Returns
        -------
        dict
            dictionary of prototypes with their keys being their representative tokens

        """
        return self.prototypes

    def getWordUsageMatrix_Individual(self, token):
        """
        Accessor for an individual word usage matrix stored in self.WUMs data member

        Parameters
        ----------
        token : str
            A given token

        Returns
        -------
        wum
            word usage matrix of given token
        """

        vecs = [self.embeddings[i] for i in range(len(self.embeddings)) if self.tokens[i] == token]

        try:
            return wum(np.array(vecs), [token] * len(vecs))

        except KeyError:
            print('word usage matrix of given token not found in object!: ' + token)