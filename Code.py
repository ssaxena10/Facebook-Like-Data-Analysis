#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:40:06 2017

@author: sharul
"""

from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import matplotlib.pyplot as plt


## Community Detection
"""
we'll download a real dataset to see how our algorithm performs.
"""
def download_data():
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')

def read_graph():
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')

#1
def get_subgraph(graph, min_degree):
    key = []
    for x in graph.nodes():
        if graph.degree(x) >= min_degree:
            key.append(x)
    return graph.subgraph(key)
    pass


#2
def score_max_depths(graph, max_depths):

    tup = []
    
    for x in max_depths:
        pgn = partition_girvan_newman(graph,x)
        tup.append((x,norm_cut(pgn[0].nodes(),pgn[1].nodes(),graph)))
        
    return tup
    pass



""""
Compute the normalized cut for each discovered cluster.
I've broken this down into the three next methods.
"""
#3
def volume(nodes, graph):

    return len(graph.edges(nodes))
    pass

#4
def cut(S, T, graph):
    
    cutset = 0
    for n1 in S:
        for n2 in T:
            if n1 in graph.neighbors(n2):
                cutset += 1
    return cutset
    pass

#5
def norm_cut(S, T, graph):
    
    ncv = float(cut(S,T,graph)/volume(S,graph)) + float(cut(S,T,graph)/volume(T,graph))
    
    return ncv
    pass

#6
def partition_girvan_newman(graph, max_depth):
    
    g = graph.copy()
    comp = []
    bw = approximate_betweenness(g, max_depth)
    bw_sort = sorted(bw.items(), key=lambda x: (-x[1],x[0]))
    
    for k in bw_sort:
        g.remove_edge(k[0][0], k[0][1])
        comp = [c for c in nx.connected_component_subgraphs(g)]
        if len(comp) > 1:
            return comp

    pass

#7
def approximate_betweenness(graph, max_depth):
    sum_dict = defaultdict(int)
    for x in graph.nodes():
        node2dist, node2paths, node2parents = bfs(graph,x,max_depth)
        bottomUp = bottom_up(x,node2dist,node2paths,node2parents)
        for y in list(bottomUp.keys()):
            sum_dict[y] += bottomUp[y]   
    for i in sum_dict:
        sum_dict[i] = sum_dict[i]/2
        
    return sum_dict
    pass


#8
def is_approximation_always_right():
    return "no"
    pass

#9
def bfs(graph, root, max_depth):
    
    
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)
    visit = deque()
    dq = deque()
    dq.append(root)
    node2distances[root] = 0
    d = 0
    while (len(dq) > 0 ):
            n = dq.popleft()
            visit.append(n)
            if(node2distances[n] <  max_depth):
                for x in graph.neighbors(n):
                    if(x not in visit):
                        d = node2distances[n] + 1
                        if(x not in node2distances):
                            node2distances[x] = d
                            node2parents[x].append(n)
                            dq.append(x)
                        else:
                            if(d == node2distances[x]):
                                node2parents[x].append(n)

    node2num_paths[root]= 1
    for i in node2parents.keys():
        p = node2parents[i]
        p_len = len(p)
        node2num_paths[i]= p_len

    return node2distances,node2num_paths,node2parents
    pass

#10
def complexity_of_bfs(V, E, K):
    #For a random graph complexity is V+E
    return V+E

    pass


#11
def bottom_up(root, node2distances, node2num_paths, node2parents):
    
    credit = defaultdict(float)
    temp = defaultdict(float)
    
    node2distances = sorted(node2distances.items(), key = lambda x:x[1], reverse = True)
    
    for (n, d) in node2distances:
        if(n!=root):
            credit[n] +=1
            for x in node2parents[n]:
                credit[x] += credit[n] / node2num_paths[n]
                temp[tuple(sorted((n,x)))] += credit[n]/node2num_paths[n]
    return temp
    
    pass


#Link Prediction

#12
def make_training_graph(graph, test_node, n):
    
    g = graph.copy()
    node = sorted(g.neighbors(test_node))[:n]
    
    for x in node:
        g.remove_edge(test_node,x)
    return g
    pass

#13
def jaccard(graph, node, k):
    jscore = []
    n1 = set(graph.neighbors(node))
    for i in graph.nodes():
        if i != node and node not in graph.neighbors(i):
            n2 = set(graph.neighbors(i))
            jscore.append(((node,i), 1. * len(n1 & n2) / len(n1 | n2)))
    
    jscore = sorted(jscore, key=lambda x: (-x[1],x[0]))[:k]
    
    return jscore
    pass

#14
def path_score(graph, root, k, beta):
    
    pscore = []
    ndist,npath,nparent = bfs(graph,root,math.inf)
    
    for i in graph.nodes():
        if i != root and not graph.has_edge(root,i):
            t1 = math.pow(beta,ndist[i])
            t2 = npath[i]
            pscore.append(((root,i),t1 * t2))
        
    pscore = sorted(pscore,key = lambda x: (-x[1],x[0]))[:k]
    return pscore
    pass


#15
def evaluate(predicted_edges, graph):
    
    c = 0
    for x in predicted_edges:
        if x[1] in graph.neighbors(x[0]):
            c += 1
            
    return float(c/len(predicted_edges))
    pass


def main():
    """
    FYI: This takes ~10-15 seconds to run on my laptop.
    """
    download_data()
    graph = read_graph()
    fig = plt.figure(figsize=(15,15))
    nx.draw_networkx(graph,with_labels=False,alpha=.5, width=.1,
                     node_size=100)
    plt.axis("off")
    fig.savefig('Graph.png')
    
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    
    fig = plt.figure(figsize=(15,15))
    nx.draw_networkx(subgraph,with_labels=False,alpha=.5, width=.1,
                     node_size=100)
    plt.axis("off")
    fig.savefig('Subgraph.png')
    
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    
    fig = plt.figure(figsize=(15,15))
    nx.draw_networkx(clusters[0],with_labels=False,alpha=.5, width=.1,
                     node_size=100)
    plt.axis("off")
    fig.savefig('clusters1.png')
    
    fig = plt.figure(figsize=(15,15))
    nx.draw_networkx(clusters[1],with_labels=False,alpha=.5, width=.1,
                     node_size=100)
    plt.axis("off")
    fig.savefig('clusters2.png')
    
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 1 nodes:')
    print(clusters[0].nodes())
    
    #Link Prediction
    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))
    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))

if __name__ == '__main__':
    main()

