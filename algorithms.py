#!/usr/bin/env python3

def MDS_LOCAL(model, graphs):
    """ A greedy algorithm for the minimum dominating set problem.
    """
    labels = []
    cnt = 0
    for graph in graphs:
        n = len(graph.node_tags)
        l = [0 for i in range(n)]
        r = model.r.reshape(-1)[cnt:cnt+n] # use the same random features as the GNN model
        order = r.argsort().tolist()
        for p in [3, 2, 1]:
            # we assume the maximum degree is three
            for i in order:
                if l[i] == 1:
                    continue
                q = 0
                for j in graph.neighbors[i]:
                    if l[j] + sum([l[k] for k in graph.neighbors[j]]) == 0:
                        q += 1
                if p == q:
                    l[i] = 1
        labels += l
        cnt += n
    return labels
