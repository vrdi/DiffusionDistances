# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:09:28 2019

@author: daryl
"""

import networkx as nx
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import label_propagation_communities
from gerrychain import Graph

import geopandas as gpd

G = nx.karate_club_graph()
n=34
#G = nx.grid_graph([5,5])
#n=25

G=Graph.from_json("./County05.json")
n=len(list(G.nodes()))
df = gpd.read_file("./County05.shp")

centroids = df.centroid
c_x = centroids.x
c_y = centroids.y

totpop = 0
for node in G.nodes():
    G.node[node]["TOTPOP"]=int(G.node[node]["TOTPOP"])

    totpop += G.node[node]["TOTPOP"]
    
    


pos = {node:(c_x[node],c_y[node]) for node in G.nodes}   


AM = nx.adjacency_matrix(G)
NLM = (nx.normalized_laplacian_matrix(G)).todense()
LM = (nx.laplacian_matrix(G)).todense()

'''
Labels = [G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes()]


plt.figure()
plt.title("Data Labels")
nx.draw(G, node_color=Labels )
plt.show()
'''
#scipy.linalg.eigh(NLM,eigvals=1)#replace with this for speed
NLMva, NLMve = LA.eigh(NLM)

LMva, LMve = LA.eigh(LM)

Fv = LMve[:,1]
xFv = [Fv.item(x) for x in range(n)]
NFv = NLMve[:,1]
xNFv = [NFv.item(x) for x in range(n)]

plt.figure()
plt.title("Laplacian Eigenvalues")
nx.draw(G,pos=pos, node_color=np.array(xFv) )
plt.show()

plt.figure()
plt.title("Normalized Laplacian Eigenvalues")
nx.draw(G,pos=pos,node_color= xNFv)
plt.show()

plt.figure()
plt.title("Binarized Laplacian Eigenvalues")
nx.draw(G, pos=pos,node_color=[xFv[x] > 0 for x in range(n)] )
plt.show()

cvals = [xFv[x] > 0 for x in range(n)]
cddict = {node:cvals[node] for node in G.nodes()}

df["clusters"]=df.index.map(cddict)
df.plot(column="clusters")
plt.axis('off')
plt.show()

plt.figure()
plt.title("Binarized Normalized Laplacian Eigenvalues")
nx.draw(G,pos=pos,node_color= [xNFv[x] > 0 for x in range(n)])
plt.show()


sc = SpectralClustering(2, affinity='precomputed', assign_labels='discretize')

sc.fit(AM)

plt.figure()
plt.title("Scikit Spectral Clustering")
nx.draw(G,pos=pos,node_color= sc.labels_)
plt.show()


"""
c = list(greedy_modularity_communities(G))


ncs = [0 for x in G.nodes()]

nlist = list(G.nodes())

for i in range(len(c)):
    for j in range(len(nlist)):
        if nlist[j] in c[i]:
            ncs[j]=i
            
         
plt.figure()
plt.title("Modularity")
nx.draw(G,pos=pos,node_color= ncs)
plt.show()       


c = list(label_propagation_communities(G))


ncs = [0 for x in G.nodes()]

nlist = list(G.nodes())

for i in range(len(c)):
    for j in range(len(nlist)):
        if nlist[j] in c[i]:
            ncs[j]=i
            
         
plt.figure()
plt.title("Label Propogation")
nx.draw(G,pos=pos,node_color= ncs)
plt.show()  
"""






