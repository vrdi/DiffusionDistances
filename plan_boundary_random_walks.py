import matplotlib.pyplot as plt
from gerrychain.random import random
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part
from gerrychain.metrics import mean_median, efficiency_gap, polsby_popper, partisan_gini
from functools import (partial, reduce)
import pandas
import geopandas as gp
from maup import assign
import maup
import numpy as np
import networkx as nx
import pickle
import seaborn as sns
import pprint
import operator
import scipy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize


def init_random_walk_plans_state(plan1, plan2, walker_name="walkers", target_name="target",
                                 times_name="times", graph_name="graph"):
    rand_walk = dict()
    rand_walk[graph_name] = plan1.graph
    plan1_bound = reduce(lambda ns, e: set(e) | ns, plan1["cut_edges"], set())
    plan2_bound = reduce(lambda ns, e: set(e) | ns, plan2["cut_edges"], set())
    
    rand_walk[walker_name] = np.array(list(plan1_bound))
    rand_walk[target_name] = plan2_bound
    rand_walk[times_name] = np.zeros(len(rand_walk[walker_name]))
    
    return rand_walk


## This function takes a dictionary containing random walk state, and the 
## indentifers for the attributes, updates them if nescessary and returns
## True if the walk is ongoing and False if it is done.
def update_walkers(state, walkers="walkers", target="target", times="times",
                   graph="graph"):
    not_done = []
    for i in range(len(state[walkers])):
        if state[walkers][i] in state[target]:
            not_done.append(False)
        else:
            state[times][i] += 1
            state[walkers][i] = random.choice(list(state[graph].neighbors(state[walkers][i])))
            not_done.append(True)
    return any(not_done)


## This function preforms random walker from distric d1 to district d2, for iters times.
## d1 and d2 are the identifiers for districts 1 and 2 in the passed partition.
def random_plans_walks(plan1, plan2, iters=1000):
    time_data = []
    for i in range(iters):
        rand_walk_state = init_random_walk_plans_state(plan1, plan2)
        while update_walkers(rand_walk_state):
            pass
        time_data.extend(rand_walk_state["times"])
    return time_data


def estimate_bound_walk_metric(plan1, plan2, plots=False, iters=1000):
    walk_times_12 = random_plans_walks(plan1, plan2, iters=iters)
    walk_times_21 = random_plans_walks(plan2, plan1, iters=iters)
    
    if all(list(map(lambda x: x == 0, walk_times_12 + walk_times_21))): return 0
    
    score = np.mean(np.ma.masked_equal(walk_times_12, 0)) + np.mean(np.ma.masked_equal(walk_times_21,0))
    
    if plots:
        fig, axs = plt.subplots(2, 2, figsize=(15,10))
        axs[0, 0].set_title("Random walk times from plan1 to plan2")
        axs[0, 0].set_ylabel("Walk time")
        axs[0, 0].boxplot(walk_times_12)

        axs[1, 0].set_xlabel("Walk time")
        axs[1, 0].set_ylabel("Frequency")
        sns.distplot(walk_times_12, kde=False, ax=axs[1, 0])

        axs[0, 1].set_title("Random walk times from plan2 to plan1")
        axs[0, 1].set_ylabel("Walk time")
        axs[0, 1].boxplot(walk_times_21)

        axs[1, 1].set_xlabel("Walk time")
        axs[1, 1].set_ylabel("Frequency")
        sns.distplot(walk_times_21, kde=False, ax=axs[1, 1])
        plt.show()
    
    return score


def dir_bound_walk_metric(plan1, plan2):
    graph = plan1.graph
    adj = nx.to_numpy_matrix(graph, weight=None)
    trans = normalize(adj, norm="l1")
    plan1_bound = reduce(lambda ns, e: set(e) | ns, plan1["cut_edges"], set())
    plan2_bound = reduce(lambda ns, e: set(e) | ns, plan2["cut_edges"], set())
    contained_by_plan2 = set(filter(lambda x: all(list(map(lambda y: y in plan2_bound,
                                                   graph.neighbors(x)))), graph.nodes()))
    
    if plan1_bound == plan2_bound: return 0
    
    undercount = (contained_by_plan2 - plan2_bound) & plan1_bound
    
    to_delete = plan2_bound | contained_by_plan2
    starters = sorted(plan1_bound - to_delete)
    to_delete = list(reversed(sorted(list(to_delete))))
    plan1_bound = reduce(lambda shifted, x: list(map(lambda y: y-1 if y > x else y, shifted)), 
                         to_delete, starters)

    P = trans
    for node in to_delete:
        P = np.delete(P, node, 0)
        P = np.delete(P, node, 1)

    N = np.identity(P.shape[0]) - P
    N = np.linalg.inv(N)
    score = sum([sum(N[i]) for i in plan1_bound]) + len(undercount)
    
    return score / len(plan1_bound)
    

def bound_walk_metric(plan1, plan2):
    score = dir_bound_walk_metric(plan1, plan2) + dir_bound_walk_metric(plan2, plan1)
    return score