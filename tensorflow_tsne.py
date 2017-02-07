import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from keras.datasets import mnist

from tqdm import tqdm

from __future__ import division, print_function


def LoadData():
    _, (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28 * 28)

    _, xt, _, yt  = train_test_split(X_test, y_test, 
        test_size=0.3, random_state=0)

    return xt, yt

#######################
#Data functions

def Precision2Entropy(D, beta):
    """
    Given distance vector and precision, calculate the probability distribution
    and the entropy

    Inputs
    -------

    D : Distance to all points (numpy array, shape (n_samples,))
    beta : Precision, the inverse of the variance

    Returns
    --------

    Entropy (H), float
    Probability distribution (P) :(numpy array, shape (n_samples,)) 

    P = exp(-beta * D) / Z = Pb / Z
    Z = sum(Pb)

    H = -sum(P * log(P))
      = sum(P * log(Z) + beta * P * D)
      = log(Z) + beta * sum(P * D)

    """
    P = np.exp(-beta * D)
    sum_p = np.sum(P)
    
    P /= sum_p
    H = np.log(sum_p) + beta * np.sum(P * D)
    
    return H, P

def BinarySearchProbabilities(D, log_perplexity, tolerance=1e-5, beta_init=1.0):
    """
    Construct the neighbour-probabilities for a given sample in the high-dimensional
    space by a search over possible precisions.

    Inputs
    -------

    D : Distance to all points (numpy array, shape (n_samples,))
    log_perplexity : log(perplexity)
    tolerance : Acceptable perplexity violation
    beta_init : Initial guess for perplexity

    Returns
    --------

    P : Probability vector (numpy array, shape (n_samples,))
    """

    #Perform a binary search for the correct precision
    #The correct precision will lead to an entropy equal to
    #the log of the perplexity, within given tolerance

    beta = beta_init

    betamin = -np.inf
    betamax = np.inf

    H, P = Precision2Entropy(D, beta)

    Hdiff = H - log_perplexity
    tries = 0

    while np.abs(Hdiff) > tolerance and tries < 50:
        if Hdiff > 0:
            #Raise betamin
            betamin = beta
            if betamax == -np.inf:
                #Obtain upper bound by doubling
                #beta until H < log_perplexity
                beta = beta * 2
            else:
                beta = (betamin + betamax) / 2
        else:
            #Lower betamax
            betamax = beta
            if betamin == -np.inf:
                #Obtain lower bound by halving
                #beta until H > log_perplexity
                beta = beta / 2
            else:
                beta = (betamin + betamax) / 2

        H, P = Precision2Entropy(D, beta)
        Hdiff = H - log_perplexity
        tries += 1

    return P

def GenerateNeighbourProbabilities(X, perplexity=30, 
        metric='euclidean', tolerance=1e-5, beta_init=1.0):
    """

    """
    n_samples, _ = X.shape

    #Obtain the pairwise distances
    D = pairwise_distances(X, metric=metric)

    #The probability matrix
    P = np.zeros( (n_samples, n_samples,) )
    inds_ = (1 - np.eye(n_samples)).astype(bool)

    log_perplexity = np.log(perplexity)

    for Pi, Di in tqdm(zip(P, D), desc='Computing Neighbour Probabilities'):
        Pi[inds_] = BinarySearchProbabilities(Di, log_perplexity, 
            tolerance=tolerance, beta_init=beta_init)


    return P

##################################
#Functions over symbolic variables
def InitializeEmbedding(X):
    """
    Initialize the tsne projection using isomap
    """

    
    iso_model = Isomap(n_components=2)
    embed_init = iso_model.fit_transform(X)

    return embed_init

#This will be a symbolic function
def PairwiseEmbeddedDistances():
    pass

#This will be a symbolic function
def EmbeddedDistance2Probabilities():
    pass

def KLDivergence():
    pass





