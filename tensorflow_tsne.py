from __future__ import division, print_function

import argparse
import sys
import os

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


def LoadData():
    print("Loading MNIST data...")
    _, (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28 * 28)

    _, xt, _, yt  = train_test_split(X_test, y_test, 
        test_size=0.3, random_state=0)

    n_class = np.unique(yt).shape[0]

    return xt, yt, n_class

#######################
#Data functions

def InitializeEmbedding(X, embed_dim=2):
    """
    Compute initialized TSNE projections using isomap

    Inputs : Data matrix X, numpy array (n_samples, n_features)

    Returns :
    """
    print("Initializing projection via isomap...")
    iso_model = Isomap(n_components=embed_dim)
    embed_init = iso_model.fit_transform(X)

    return embed_init

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
    inds = (1 - np.eye(n_samples)).astype(bool)
    D = D[inds].reshape( (n_samples, -1) )

    #The probability matrix
    P = np.zeros( (n_samples, n_samples,) )

    log_perplexity = np.log(perplexity)

    for Pi, Di, inds_ in tqdm(zip(P, D, inds), desc='Computing Neighbour Probabilities'):
        Pi[inds_] = BinarySearchProbabilities(Di, log_perplexity, 
            tolerance=tolerance, beta_init=beta_init)

    P[np.isnan(P)] = 0.
    #Symmetrize and renormalize the probabilities
    P = P + P.T
    P /= P.sum()
    P = np.maximum(P, 10e-8)

    return P

##################################
#Functions over symbolic variables

def PairwiseEmbeddedDistances(X):
    """
    Compute Euclidean neighbour distances in the projected space

    Inputs:
    -------

    X : Tensor storing the TSNE projected samples

    ||xi - xj|| ** 2 = xi ** 2 + xj ** 2 - 2 * xi * xj

    Returns:
    ---------

    D : Tensor storing neighbour distances
    """
    sum_x_2 = tf.reduce_sum(tf.square(X), reduction_indices=1, keep_dims=False)
    D = sum_x_2 + tf.reshape(sum_x_2, [-1, 1]) - 2 * tf.matmul(X, tf.transpose(X))

    return D


#This will be a symbolic function
def EmbeddedDistance2Probabilities(D, embed_dim=2):
    """
    Fit the student t-distribution to the distance matrix in the projected space

    Inputs
    -------

    D : Tensor storing neighbour distances

    Returns:
    --------

    Q : Tensor storing neighbour probabilities
    """
    #alpha = embed_dim - 1

    # #T-student distribution
    # Q = tf.pow(1 + D, -1)

    # #Remove diagonals
    # mask = tf.Variable((1 - np.eye(Q.get_shape()[0].value)).astype(np.float32))

    # #Normalize and clip
    # sum_q = tf.reduce_sum(Q * mask)
    # Q /= sum_q

    # eps = tf.Variable(np.float32(10e-8), name='eps')
    # Q = tf.maximum(Q, eps)
    alpha = embed_dim - 1
    eps = tf.constant(np.float32(10e-8), name='eps')

    Q = tf.pow(1 + D / alpha, -(alpha + 1) / 2)
    mask = tf.constant((1 - np.eye(Q.get_shape()[0].value)).astype(np.float32), name='mask')
    Q *= mask
    Q /= tf.reduce_sum(Q)
    Q = tf.maximum(Q, eps)
    return Q

def KLDivergence(P1, P2):
    KLD = tf.log((P1) / (P2))
    KLD = tf.reduce_sum(P1 * KLD)
    return KLD


def TSNELossFunction(X_tsne, PX, embed_dim=2):
    """
    Inputs 
    -------

    Xproj : tensor storing TSNE projected samples
    PX : target neighbour probabilities, numpy array

    """

    #Compute the neighbour probabilities in the embedded space
    D_tsne = PairwiseEmbeddedDistances(X_tsne)
    P_tsne = EmbeddedDistance2Probabilities(D_tsne, embed_dim=embed_dim)

    return KLDivergence(PX, P_tsne)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple TSNE in Tensorflow")
    parser.add_argument('--n-iter', type=int, dest='n_iter', help='Number of optimization steps to perform')
    parser.set_defaults(n_iter=100)
    args = parser.parse_args()

    xt, yt, n_class = LoadData()

    #Variable holding the projected TSNE values
    if os.path.isfile('x_init.npy'):
        x_init = np.load('x_init.npy')
    else:
        x_init = InitializeEmbedding(xt, embed_dim=2)
        with open("x_init.npy", 'w') as f:
            np.save(f, x_init)
    
    #Variables and initializations
    X_tsne = tf.Variable(x_init.astype(np.float32), name='X_tsne')

    #Placeholder for handling the data   
    PX = tf.placeholder(tf.float32, name='PX')

    if os.path.isfile('PX.npy'):
        PX_vals = np.load('PX.npy')
    else:
        PX_vals = GenerateNeighbourProbabilities(xt)
        with open("PX.npy", 'w') as f:
            np.save(f, PX_vals)

    #Tensorflow computation graph
    tsne_loss = TSNELossFunction(X_tsne, PX)

    #Optimize using Adam
    opt_step = tf.train.AdamOptimizer().minimize(tsne_loss)
    #opt_step = tf.train.GradientDescentOptimizer(0.5).minimize(tsne_loss)

    init_step = tf.global_variables_initializer()

    print([v.name for v in tf.global_variables()])

    #Running TSNE optimization
    with tf.Session() as sess:
        sess.run(init_step)

        for _ in tqdm(xrange(args.n_iter), desc='Optimizing projection'):
            tqdm.write('KL divergence : {0}'.format(sess.run(tsne_loss, feed_dict={PX : PX_vals})))
            sess.run(opt_step, feed_dict={PX : PX_vals})
            result = sess.run(X_tsne)

    #Generating a visualization of the training process

    fig, axs = plt.subplots(1,2,figsize=(20,10))

    colours = plt.cm.Spectral(np.linspace(0,1,n_class))
    c = colours[yt]

    im = axs[0].scatter(x_init[:,0], x_init[:,1], lw=0, c=c,
        s=100)

    for i, cl in enumerate(colours):
        axs[0].plot([], [], color=cl, label=str(i))

    axs[0].legend()
    axs[0].set_title("Isomap initialization")

    im = axs[1].scatter(result[:,0], result[:,1], lw=0, c=c,
        s=100)

    for i, cl in enumerate(colours):
        axs[1].plot([], [], color=cl, label=str(i))

    axs[1].legend()
    axs[1].set_title("TSNE embedding")


    plt.show()