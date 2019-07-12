import random



# import matplotlib

# matplotlib.use('Agg')



import matplotlib.pyplot as plt

from functools import partial

import networkx as nx

import scipy.sparse

from scipy.sparse import lil_matrix

import math



from gerrychain import MarkovChain

from gerrychain.constraints import (

    Validator,

    single_flip_contiguous,

    within_percent_of_ideal_population,

)

from gerrychain.proposals import propose_random_flip

from gerrychain.accept import always_accept

from gerrychain.updaters import Tally, cut_edges

from gerrychain.partition import Partition

from gerrychain.proposals import recom

from numpy import linalg as LA

import numpy as np





gn = 3*4

k = 4

ns = 50

p = 0.5



graph = nx.grid_graph([k * gn, k * gn])



for n in graph.nodes():

    if n[0] < 6 and n[1] < 6:

        graph.node[n]['population'] = 1

    elif n[0] > 41 and n[1] > 41:

        graph.node[n]["population"] = 1

    else:

        graph.node[n]["population"] = 1



    if random.random() < p:

        graph.node[n]["pink"] = 1

        graph.node[n]["purple"] = 0

    else:

        graph.node[n]["pink"] = 0

        graph.node[n]["purple"] = 1

    if 0 in n or k * gn - 1 in n:

        graph.node[n]["boundary_node"] = True

        graph.node[n]["boundary_perim"] = 1



    else:

        graph.node[n]["boundary_node"] = False



def nodes_to_neighbors(G=graph):

    neighbors = {x:[] for x in G.nodes()}

    for node in G.nodes():

        for node2 in G.nodes():

            if (node,node2) in G.edges() or (node2,node) in G.edges():

                neighbors[node].append(node2)

    return neighbors

node_neighbors = nodes_to_neighbors(graph)



#this part adds queen adjacency

#for i in range(k * gn - 1):

#    for j in range(k * gn):

#        if j < (k * gn - 1):

#            graph.add_edge((i, j), (i + 1, j + 1))

#            graph[(i, j)][(i + 1, j + 1)]["shared_perim"] = 0

#        if j > 0:

#            graph.add_edge((i, j), (i + 1, j - 1))

#            graph[(i, j)][(i + 1, j - 1)]["shared_perim"] = 0





# ######### BUILD ASSIGNMENT

cddict = {x: int(x[0] / gn) for x in graph.nodes()}

#cddict = {}



#for x in graph.nodes():

#    if x[0] < 5:

#        cddict[x] = 1

#    elif x[0] >= 5 and x[0] < 24:

#        cddict[x] = 2

#    elif x[0] < 43 and x[0] >= 24:

#        cddict[x] = 3

#    elif x[0] >= 43:

#        cddict[x] = 4







pos = {x: x for x in graph.nodes()}



#print(nodes_to_edges)



# ### CONFIGURE UPDATERS



#NLM = (nx.normalized_laplacian_matrix(G)).todense()



def pop_Laplacian(G, nodelist=None, weight='weight'):

    if nodelist is None:

        nodelist = G.nodes()

    B = lil_matrix((len(G.nodes()),len(G.nodes())))



    index = dict(zip(nodelist,range(len(nodelist))))



    for u in G.nodes():

        for v in G.nodes():

            if u == v:

                sum = 0

                for neighbor in node_neighbors[v]:

                    if neighbor in G.nodes():

                        sum += G.node[neighbor]["population"]

                B[index[v],index[v]] = sum

            elif u in node_neighbors[v]:

                alpha = -1 * math.sqrt(G.node[u]["population"] * G.node[v]['population'])  

                B[index[u],index[v]] = alpha

    return B.asformat('csr')

                



def step_num(partition):

    parent = partition.parent

    if not parent:

        return 0

    return parent["step_num"] + 1





updaters = {

    "population": Tally("population"),

    "cut_edges": cut_edges,

    "step_num": step_num,

    # "Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})

}





# ########BUILD FIRST PARTITION



grid_partition = Partition(graph, assignment=cddict, updaters=updaters)



nx.draw(

    graph,

    pos={x: x for x in graph.nodes()},

    node_color=[grid_partition.assignment[x] for x in graph.nodes()],

    node_size=ns,

    node_shape="s",

    cmap="tab20",

)

plt.savefig("./grids/plots/medium/initial.png")

plt.close()



# ADD CONSTRAINTS

popbound = within_percent_of_ideal_population(grid_partition, 1)



# ########Setup Proposal

ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)



tree_proposal = partial(

    recom,

    pop_col="population",

    pop_target=ideal_population,

    epsilon=0.05,

    node_repeats=1,

)



# ######BUILD AND RUN FIRST MARKOV CHAIN



recom_chain = MarkovChain(

    tree_proposal,

    constraints=[popbound],

    accept=always_accept,

    initial_state=grid_partition,

    total_steps=100,

)



for part in recom_chain:

    pass



plt.figure()

nx.draw(

    graph,

    pos={x: x for x in graph.nodes()},

    node_color=[part.assignment[x] for x in graph.nodes()],

    node_size=ns,

    node_shape="s",

    cmap="tab20",

)

plt.savefig("./grids/plots/medium/end_of_tree.png")

plt.close()



print("Finished ReCom")

# ########BUILD SECOND PARTITION



squiggle_partition = Partition(graph, assignment=part.assignment, updaters=updaters)





# ADD CONSTRAINTS

popbound = within_percent_of_ideal_population(squiggle_partition, 1)





# ######BUILD AND RUN SECOND MARKOV CHAIN



#squiggle_chain = MarkovChain(

#    propose_random_flip,

#    Validator([single_flip_contiguous, popbound]),

#    accept=always_accept,

#    initial_state=squiggle_partition,

#    total_steps=100000,

#)





#for part2 in squiggle_chain:

#    pass





#plt.figure()

#nx.draw(

#    graph,

#    pos={x: x for x in graph.nodes()},

#    node_color=[part2.assignment[x] for x in graph.nodes()],

#    node_size=ns,

#    node_shape="s",

#    cmap="tab20",

#)

#plt.savefig("./grids/plots/medium/end_of_boundary.png")

#plt.close()

#print("Finished Squiggling")

# ########BUILD FINAL PARTITION



final_partition = Partition(graph, assignment=grid_partition.assignment, updaters=updaters)





# ADD CONSTRAINTS

popbound = within_percent_of_ideal_population(final_partition, 0.3)





# ########Setup Spectral Proposal





def spectral_cut(G):

    nlist = list(G.nodes())

    n = len(nlist)

    #NLM = (nx.normalized_laplacian_matrix(G)).todense()

    # LM = (nx.laplacian_matrix(G)).todense()



    NLM = (pop_Laplacian(G)).todense()

    NLMva, NLMve = LA.eigh(NLM)

    NFv = NLMve[:, 1]



    #transpose = np.transpose(NFv)

    #print(transpose.dot(NLM.dot(NFv))/(transpose.dot(NFv)))

    #print(NLMva[1])



    xNFv = [NFv.item(x) for x in range(n)]

    node_color = [xNFv[x] > 0 for x in range(n)]



    clusters = {nlist[x]: node_color[x] for x in range(n)}



    return clusters





def propose_spectral_merge(partition):

    edge = random.choice(tuple(partition["cut_edges"]))

    # print(edge)

    et = [partition.assignment[edge[0]], partition.assignment[edge[1]]]

    #print(et)

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

    initial_state=final_partition,

    total_steps=500,

)



cut_edges = []

t = 0

for part3 in final_chain:

    cut_edges.append(len(part3["cut_edges"]))

    print(part3['population'])

    plt.figure()

    nx.draw(

        graph,

        pos,

        node_color=[part3.assignment[x] for x in graph.nodes()],

        node_size=ns,

        node_shape="s",

        cmap="tab20",

    )

    plt.savefig(f"./grids/plots/medium/spectral_step{t:02d}.png")

    plt.close()

    t+=1

    if t%20 == 0:

        print("Finished spectral " + str(t))



print("Finished Spectral")



plt.scatter(range(500), cut_edges,color="b")

plt.show()

