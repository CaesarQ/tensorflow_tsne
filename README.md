Simple implementation of the TSNE algorithm in tensorflow

usage: tensorflow_tsne.py [-h] [--n-iter N_ITER] [--perplexity PERPLEXITY]
                          [--lr LR]

optional arguments:

  -h, --help            show this help message and exit

  --n-iter N_ITER       Number of optimization steps to perform

  --perplexity PERPLEXITY
                        Expected number of neighbours
                        
  --lr LR               Learning rate

![TSNE demo](tsne_embedding.png?raw=true "TSNE demo (Final KL-Divergence : 1.200)")