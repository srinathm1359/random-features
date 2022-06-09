# Graph Neural Networks with Random Features (SDM 2021)

We prove that adding random features to each node strengthens the expressive power of graph neural networks. The complete paper including appendices is available on [arXiv](https://arxiv.org/abs/2002.03155).

This repository is forked from https://github.com/joisino/random-features.


## Installation

Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```

## Test run

```
$ bash exec.sh
```

## Datasets

We release new synthetic benchmarks we used in the paper. All datasets are stored in the `dataset` directory. Each file describes a set of graphs with the following format:

```
s 0
G_1
G_2
...
G_s
```

where `s` is the number of graphs, there is `0` for compatibility with graph classification datasets, `G_i` describes a graph with the following format:

```
n
y_1 d_1 v_{1, 1} v_{1, 2} ... v_{1, d_1}
y_2 d_2 v_{2, 1} v_{2, 2} ... v_{2, d_2}
...
y_n d_n v_{n, 1} v_{n, 2} ... v_{n, d_n}
```

where `n` is the number of nodes, the `(i+2)`-th line describes node `i` (node ids are 0-indexed), `y_i` is the label of node `i`, `d_i` is the degree of node `i`, `v_{i, j}` is the `j`-th neighbor of node `i`.

You can also generate (more) synthetic datasets by `dataset_gen.py`.

The MDS datasets do not contain the label information (i.e., `y_i = 0` for all `i`). We generate labels dynamically using `algorithms.py` when we test models.

## Feedback and Contact

Please feel free to contact me at r.sato AT ml.ist.i.kyoto-u.ac.jp, or to open issues.

## Citation

```
@inproceedings{sato2021random,
  author    = {Ryoma Sato and Makoto Yamada and Hisashi Kashima},
  title     = {Random Features Strengthen Graph Neural Networks},
  booktitle = {Proceedings of the 2021 {SIAM} International Conference on Data Mining, {SDM}},
  year      = {2021},
}
```