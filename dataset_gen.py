#!/usr/bin/env python3

import os
import sys

import networkx as nx
import numpy as np


def local_cluster_coefficient(G):
    label = [0 for i in range(len(G))]
    for i in range(len(G)):
        for j in G[i]:
            for k in G[i]:
                if k <= j:
                    continue
                if j in G[k]:
                    label[i] += 1
    return label


def tri_check(G):
    label = [0 for i in range(len(G))]
    for i in range(len(G)):
        for j in G[i]:
            for k in G[i]:
                if j in G[k]:
                    label[i] = 1
    return label


def main():
    deg = 3
    num_graphs = 1000
    ty = sys.argv[1]
    if ty in ['TRIANGLE_EX' or 'LCC_EX' or 'MDS_EX']:
        sps = [('train', 20), ('test', 100)]
    else:
        sps = [('train', 20), ('test', 20)]
    for sp in sps:
        n = sp[1]
        Gs = []
        for _ in range(num_graphs):
            g = nx.random_degree_sequence_graph([deg for i in range(n)])
            G = [[] for i in range(n)]
            for e in g.edges:
                G[e[0]].append(e[1])
                G[e[1]].append(e[0])
            if ty in ['TRIANGLE', 'TRIANGLE_EX']:
                l = tri_check(G)
            elif ty in ['LCC', 'LCC_EX']:
                l = local_cluster_coefficient(G)
            else:
                l = [0 for i in range(n)]
            Gs.append((G, l))

        basedir = f'dataset/{ty}'
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        with open(f'{basedir}/{ty}_{sp[0]}.txt', 'w') as f:
            f.write(f'{num_graphs}\n')
            for i in range(num_graphs):
                f.write(f'{n} 0\n')
                G, l = Gs[i]
                for j in range(n):
                    f.write(f'{l[j]} {deg}')
                    for k in G[j]:
                        f.write(f' {k}')
                    f.write('\n')
        
if __name__ == '__main__':
    main()
