# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:33:47 2019

@author: daryl
"""

import random

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from functools import partial
import networkx as nx

from gerrychain import MarkovChain
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain import Graph
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from numpy import linalg as LA
from gerrychain.tree import recursive_tree_part


graph = Graph.from_json("./County05.json")
df = gpd.read_file("./County05.shp")

centroids = df.centroid
c_x = centroids.x
c_y = centroids.y

totpop = 0
for node in graph.nodes():
    graph.node[node]["TOTPOP"]=int(graph.node[node]["TOTPOP"])

    totpop += graph.node[node]["TOTPOP"]
    
cddict = recursive_tree_part(graph,range(4),totpop/4,"TOTPOP", .01,1)
    


pos = {node:(c_x[node],c_y[node]) for node in graph.nodes}   



def step_num(partition):
    parent = partition.parent
    if not parent:
        return 0
    return parent["step_num"] + 1


updaters = {
    "population": Tally("TOTPOP"),
    "cut_edges": cut_edges,
    "step_num": step_num,
    # "Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})
}


initial_partition=Partition(graph,cddict,updaters)

def spectral_cut(G):
    nlist = list(G.nodes())
    n = len(nlist)
    NLM = (nx.normalized_laplacian_matrix(G)).todense()
    # LM = (nx.laplacian_matrix(G)).todense()
    NLMva, NLMve = LA.eigh(NLM)
    NFv = NLMve[:, 1]
    xNFv = [NFv.item(x) for x in range(n)]
    node_color = [xNFv[x] > 0 for x in range(n)]

    clusters = {nlist[x]: node_color[x] for x in range(n)}

    return clusters


def propose_spectral_merge(partition):
    edge = random.choice(tuple(partition["cut_edges"]))
    # print(edge)
    et = [partition.assignment[edge[0]], partition.assignment[edge[1]]]
    # print(et)
    sgn = []
    for n in partition.graph.nodes():
        if partition.assignment[n] in et:
            sgn.append(n)

    # print(len(sgn))
    sgraph = nx.subgraph(partition.graph, sgn)

    edd = {0: et[0], 1: et[1]}

    # print(edd)

    clusters = spectral_cut(sgraph)
    # print(len(clusters))
    flips = {}
    for val in clusters.keys():
        flips[val] = edd[clusters[val]]

    # print(len(flips))
    # print(partition.assignment)
    # print(flips)
    return partition.flip(flips)


# ######BUILD AND RUN FINAL MARKOV CHAIN


final_chain = MarkovChain(
    propose_spectral_merge,
    Validator([]),
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=10,
)


for part in final_chain:
    df["clusters"]=df.index.map(dict(part.assignment))
    df.plot(column="clusters")
    plt.axis('off')
    plt.savefig(f"./plots/spectral_step{part['step_num']:02d}.png")
    
    plt.close()
    #plt.show()
    

    #plt.figure()
    #nx.draw(graph,pos=pos,node_color=[dict(part.assignment)[x] for x in graph.nodes()])
    #plt.show()
    #plt.savefig(f"./plots/medium/spectral_step{part3['step_num']:02d}.png")
    #plt.close()