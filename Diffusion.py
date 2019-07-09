# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:05:45 2019

@author: daryl
"""
import geopandas as gpd
from gerrychain import Graph
from gerrychain.tree import recursive_tree_part
import networkx as nx
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import label_propagation_communities
import matplotlib as mpl
import matplotlib.cm as cm
from collections import defaultdict
import random
import math

graph = Graph.from_json("./County05.json")
df = gpd.read_file("./County05.shp")

centroids = df.centroid
c_x = centroids.x
c_y = centroids.y

totpop = 0
for node in graph.nodes():
    graph.node[node]["TOTPOP"]=int(graph.node[node]["TOTPOP"])

    totpop += graph.node[node]["TOTPOP"]
    
#cddict = recursive_tree_part(graph,range(4),totpop/4,"TOTPOP", .01,1)
    


pos = {node:(c_x[node],c_y[node]) for node in graph.nodes}   

n=len(graph.nodes())

num_steps=500
divisor=200

L = (nx.laplacian_matrix(graph)).todense()
L=np.array(L)
E=np.linalg.eig(L)
D=E[0]
V=E[1]



#initial conditions
initial = [random.choice(list(graph.nodes())) for x in range(5)]

starts=set()
for node in initial:
    starts.add(node)
    for nbr in graph.neighbors(node):
        starts.add(nbr)


C0=np.array([5+200*(node in initial) for node in graph.nodes()])

C0=np.array([5+200*(node in starts) for node in graph.nodes()])

#C0 = np.array([random.choice(list(range(10,100))) for node in graph.nodes()])

#C0 = np.array([graph.node[node]["TOTPOP"] for node in graph.nodes()])
#color initialization
max=np.amax(C0)
min=np.amin(C0)
cnorm = mpl.colors.Normalize(vmin=min,vmax=max)
cmap=cm.jet
m = cm.ScalarMappable(norm=cnorm, cmap=cmap)


C0V=np.dot(np.transpose(V),C0)


epsilon=1#.999999999

Phi=np.zeros((n,1))

plotlist=[]


for s in range(0,num_steps):
    t=float(s)/divisor
   
    for i in range(n):
        Phi[i]=C0V[i]*(math.exp((-D[i]*t)))
          
    #Phi=np.add(Phi,beta)
    
    Phi=np.dot(V,Phi)
    #if s==0:
    #    print(Phi)
    
    #norm=np.dot(np.transpose(Phi),Phi)
    
    
    #normalized for hue:
    #for i in range(n):
        #print(Phi[i])
        #Phi[i]=np.divide(Phi[i],norm)
        
     
    d={}
    e={}
    dlist=[]
    elist=[]
    d1=defaultdict(list)
    e1=defaultdict(list)
    nlist = list(graph.nodes())
    for i in range(n):
        
        temp=m.to_rgba(Phi[i])
        dlist.append([temp,i])
    #for k,v in dlist:
    #    d1[k].append(v)
    #d= dict((tuple(k),v) for k,v in d1.iteritems())    
        
        #d[(temp[0][0]*((epsilon)**i),temp[0][1]*((epsilon)**i),temp[0][2]*((epsilon)**i))]=[i]
        d[nlist[i]] = (temp[0][0]*((epsilon)**i),temp[0][1]*((epsilon)**i),temp[0][2]*((epsilon)**i))
    ind=0   
    for j in graph.edges():
        ind=ind+1
        
        
        etemp=(Phi[j[0]]+Phi[j[1]])/2
        
        etemp=m.to_rgba(etemp)
        
        #e[(etemp[0][0]*((epsilon)**ind),etemp[0][1]*((epsilon)**ind),etemp[0][2]*((epsilon)**ind))]=[j]
        e[j] = (etemp[0][0]*((epsilon)**ind),etemp[0][1]*((epsilon)**ind),etemp[0][2]*((epsilon)**ind))
    plt.figure()
    nx.draw(graph,pos=pos,node_color=[d[node] for node in graph.nodes],
            edge_color = [e[edge] for edge in graph.edges()])
    plt.savefig(f"./plots/{s:03d}diff4.png")
    plt.close()
    
    
    #ffmpeg -i %02ddiff.png state_diffusion.gif