import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

from keras.datasets import mnist


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

def Precision2Entropy():
    pass

def Distance2Probabilities():
    pass

def KLDivergence():
    pass





