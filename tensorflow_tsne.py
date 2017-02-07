import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

from keras.datasets import mnist

from __future__ import division, print_function


def LoadData():
    _, (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28 * 28)

    _, xt, _, yt  = train_test_split(X_test, y_test, 
        test_size=0.3, random_state=0)

    return xt, yt

def InitializeEmbedding(X):
    iso_model = Isomap(n_components=2)
    embed_init = iso_model.fit_transform(X)

    return embed_init

def PairwiseDistances():
    pass

def Precision2Entropy(D, beta):
    """
    D : Distance to all points
    beta : Precision, the inverse of the variance

    returns Entropy (H), Probability distribution (P)

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
    D : Distance to all points
    log_perplexity : log(perplexity)
    tolerance : Acceptable perplexity violation
    beta_init : Initial guess for perplexity
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

def Distance2Probabilities():
    pass

def KLDivergence():
    pass





