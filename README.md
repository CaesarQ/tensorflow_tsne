Simple implementation of the TSNE algorithm in tensorflow

# Command line usage

	usage: tensorflow_tsne.py [-h] [--n-iter N_ITER] [--perplexity PERPLEXITY]
	                          [--lr LR]

	optional arguments:

	  -h, --help            show this help message and exit

	  --n-iter N_ITER       Number of optimization steps to perform

	  --perplexity PERPLEXITY
	                        Effective number of neighbours to enforce

	  --lr LR               Learning rate

# Demo

1000 iterations on MNIST sub-sample with lr=2000, perplexity=30

![TSNE demo](tsne_embedding.png?raw=true "TSNE demo (Final KL-Divergence : 1.200)")