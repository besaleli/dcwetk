from scipy.spatial import distance
import numpy as np


class wum:
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
