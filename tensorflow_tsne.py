from __future__ import division, print_function

import argparse
import sys
import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from keras.datasets import mnist

from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm


def load_data():
    print("Loading MNIST data...")
    _, (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28 * 28)

    _, xt, _, yt  = train_test_split(X_test, y_test, 
        test_size=0.3, random_state=0)

    return xt / 255., yt

#######################
#Data functions

def initialize_embedding(X, embed_dim=2):
    """
    Compute initialized TSNE projections using isomap

    Args:
        X (numpy.ndarray): Data matrix (n_samples, n_features)
    
    Kwargs:
        embed_dim (int): Dimension of the TSNE projection
    
    Returns:
        embed_init (numpy.ndarray): Initialized projection (n_samples, embed_dim)
    
    """
    print("Initializing projection via PCA...")
    pca_model = PCA(n_components=embed_dim)
    embed_init = pca_model.fit_transform(X)

    return embed_init

def precision_to_entropy(D, beta):
    """
    Given distance vector and precision, calculate the probability distribution
    and the entropy

    P = exp(-beta * D) / Z = Pb / Z
    Z = sum(Pb)

    H = -sum(P * log(P))
      = sum(P * log(Z) + beta * P * D)
      = log(Z) + beta * sum(P * D)

    Args:

        D (numpy.ndarray): Distance to all points (n_samples,)
        beta (float): Precision, the inverse of the variance

    Returns:
        H (float) : The entropy of the distribution
        P (numpy.ndarray): Probability distribution (n_samples,)

    """
    P = np.exp(-beta * D)
    sum_p = np.sum(P)
    
    P /= sum_p
    H = np.log(sum_p) + beta * np.sum(P * D)
    
    return H, P

def binary_search_probabilities(D, log_perplexity, tolerance=1e-5, beta_init=1.0):
    """
    Construct the neighbour-probabilities for a given sample in the high-dimensional
    space by a search over possible precisions.

    Args:
        D (numpy.ndarray): Distance to all points (n_samples,)
        log_perplexity (float): log(perplexity)

    Kwargs:
        tolerance (float) : Acceptable perplexity violation
        beta_init (float): Initial guess for perplexity

    Returns:
        P (numpy.ndarray): Probability vector (n_samples,)
    """

    #Perform a binary search for the correct precision
    #The correct precision will lead to an entropy equal to
    #the log of the perplexity, within given tolerance

    beta = beta_init

    betamin = -np.inf
    betamax = np.inf

    H, P = precision_to_entropy(D, beta)

    Hdiff = H - log_perplexity
    tries = 0

    while np.abs(Hdiff) > tolerance and tries < 50:
        if Hdiff > 0:
            #Raise betamin
            betamin = beta
            if betamax == np.inf:
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

        H, P = precision_to_entropy(D, beta)
        Hdiff = H - log_perplexity
        tries += 1

    return P

def generate_neighbour_probabilities(X, perplexity=30, 
        metric='euclidean', tolerance=1e-5, beta_init=1.0):
    """
    Compute the pairwise distance matrix and define the neighbour distribution, as used in TSNE

    Args:
        X (numpy.ndarray): Data matrix (n_samples, n_features)

    Kwargs:
        perplexity (float): Effective number of neighbours to enforce
        metric (str): Metric to use for pairwise_distance computation
        tolerance (float) : Acceptable perplexity violation
        beta_init (float): Initial guess for perplexity

    Returns:
        P (numpy.ndarray): Neighbour probability distribution

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
        Pi[inds_] = binary_search_probabilities(Di, log_perplexity, 
            tolerance=tolerance, beta_init=beta_init)

    P[np.where(np.isnan(P))] = 0.

    #Symmetrize and renormalize the probabilities
    P = P + P.T
    P /= P.sum()
    P = np.maximum(P, 10e-8)

    return P

##################################
#Functions over symbolic variables

def pairwise_embedded_distances(X):
    """
    Compute Euclidean neighbour distances in the projected space

    ||xi - xj|| ** 2 = xi ** 2 + xj ** 2 - 2 * xi * xj

    Args:
        X (tensorflow.Variable): Tensor storing the TSNE projected samples

    Returns:
        D (tensorflow.Variable): Tensor storing neighbour distances
    """
    sum_x_2 = tf.reduce_sum(tf.square(X), reduction_indices=1, keep_dims=False)
    D = sum_x_2 + tf.reshape(sum_x_2, [-1, 1]) - 2 * tf.matmul(X, tf.transpose(X))

    return D


#This will be a symbolic function
def embedded_distance_to_probabilities(D, embed_dim=2):
    """
    Fit the student t-distribution to the distance matrix in the projected space

    Args:
        D (tensorflow.Variable): Tensor storing neighbour distances

    Kwargs:
        embed_dim (int): Dimension of the TSNE projection

    Returns:
        Q (tensorflow.Variable): Tensor storing neighbour probabilities
    """
    alpha = embed_dim - 1
    eps = tf.constant(np.float32(10e-8), name='eps')

    Q = tf.pow(1 + D / alpha, -(alpha + 1) / 2)
    mask = tf.constant((1 - np.eye(Q.get_shape()[0].value)).astype(np.float32), name='mask')
    Q *= mask
    Q /= tf.reduce_sum(Q)
    Q = tf.maximum(Q, eps)
    return Q

def kl_divergence(P1, P2):
    """
    Compute the Kullback–Leibler divergence between two probability distributions

    Args:
        P1 : (tensorflow.placeholder): Tensor storing the target probability distribution
        P2 : (tensorflow.Variable): Tensor storing the model distribution

    Returns:
        KLD (tensorflow.Variable): Kullback–Leibler divergence
    """
    KLD = tf.log((P1) / (P2))
    KLD = tf.reduce_sum(P1 * KLD)
    return KLD


def tsne_loss_function(X_tsne, PX, embed_dim=2):
    """
    Loss function for TSNE.  The KL divergence between the embedded neighbour distribution and the 
    high-dimensional neighbour distributionn

    Args:
        Xproj (tensorflow.Variable): tensor storing TSNE projected samples
        PX (tensorflow.placeholder) : tensor storing target neighbour probabilities

    Kwargs:
        embed_dim (int): Dimension of the TSNE projection

    Returns:
        TSNE Loss
    """

    #Compute the neighbour probabilities in the embedded space
    D_tsne = pairwise_embedded_distances(X_tsne)
    P_tsne = embedded_distance_to_probabilities(D_tsne, embed_dim=embed_dim)

    return kl_divergence(PX, P_tsne)

def plot_embedding(X_init, X_final, Y, n_iter):
    """
    Generate a before-after plot for the TSNE process.

    Args:
        X_init (numpy.ndarray): Initialized projection (n_samples, embed_dim)
        X_final (numpy.ndarray): Result of TSNE optimization (n_samples, embed_dim)
        Y (numpy.ndarray) : Class targets (n_samples,)
        n_iter (int) : Number of iterations performed in TSNE optimization
    """
    n_class = np.unique(Y).shape[0]

    fig, axs = plt.subplots(1,2,figsize=(20,10))
    colours = plt.cm.Spectral(np.linspace(0,1,n_class))
    c = colours[Y]

    im = axs[0].scatter(X_init[:,0], X_init[:,1], lw=0, c=c,
        s=100)

    for i, cl in enumerate(colours):
        axs[0].plot([], [], color=cl, label=str(i))

    axs[0].legend()
    axs[0].set_title("PCA initialization")

    im = axs[1].scatter(X_final[:,0], X_final[:,1], lw=0, c=c,
        s=100)

    for i, cl in enumerate(colours):
        axs[1].plot([], [], color=cl, label=str(i))

    axs[1].legend()
    axs[1].set_title("TSNE embedding after {0} iterations".format(n_iter))

    plt.savefig('tsne_embedding.png', bbox_inches='tight')


    plt.show()


def plot_embedded_motion(X_init, X_final):
    """
    Examine the motion of the projected data

    Args:
        X_init (numpy.ndarray): Initialized projection (n_samples, embed_dim)
        X_final (numpy.ndarray): Result of TSNE optimization (n_samples, embed_dim)    
    """
    fig, axs = plt.subplots(1,1,figsize=(10,10))

    diff = ((X_final - X_init) ** 2).mean(axis=1)

    im = axs.scatter(X_init[:,0], X_init[:,1], lw=0, c=diff,
        cmap=plt.cm.gist_heat, s=50, alpha=0.5)

    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')

    axs.set_title("Total distance moved")
    plt.show()

def main(args):
    ### Load/calculate the data, init values, and neighbour probabilities
    xt, yt = load_data()

    #Variable holding the projected TSNE values
    x_init = initialize_embedding(xt, embed_dim=2)
    
    #Variables and initializations
    X_tsne = tf.Variable(x_init.astype(np.float32), name='X_tsne')
    
    #Placeholder for handling the neighbour probabilties  
    PX = tf.placeholder(tf.float32, name='PX')

    #Neighbour probabilities
    PX_vals = generate_neighbour_probabilities(xt, perplexity=args.perplexity)

    ###Assemble and run the Tensorflow Computation Graph
    tsne_loss = tsne_loss_function(X_tsne, PX)

    #Optimize using GradientDescent
    opt_step = tf.train.GradientDescentOptimizer(args.lr).minimize(tsne_loss, var_list=[X_tsne])

    init_step = tf.global_variables_initializer()

    print("Optimizing")

    #Running TSNE optimization
    with tf.Session() as sess:
        sess.run(init_step)

        for _ in tqdm(xrange(args.n_iter), desc='Optimizing projection'):
            tqdm.write('KL divergence : {0}'.format(sess.run(tsne_loss, feed_dict={PX : PX_vals})))
            sess.run(opt_step, feed_dict={PX : PX_vals})
        result = sess.run(X_tsne)

    plot_embedding(x_init, result, yt, args.n_iter)
    plot_embedded_motion(x_init, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple TSNE in Tensorflow")
    parser.add_argument('--n-iter', type=int, dest='n_iter', help='Number of optimization steps to perform')
    parser.add_argument('--perplexity', type=float, dest='perplexity', help='Effective number of neighbours to enforce')
    parser.add_argument('--lr', type=float, dest='lr', help='Learning rate')
    parser.set_defaults(n_iter=100, perplexity=30., lr=1000.)
    args = parser.parse_args()

    main(args)