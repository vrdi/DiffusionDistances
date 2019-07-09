# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 22:50:03 2019

@author: daryl
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from gerrychain import Graph
import geopandas as gpd


g = nx.grid_graph([10,10])

g = nx.karate_club_graph()

g = Graph.from_json("./County05.json")
df = gpd.read_file("./County05.shp")
centroids = df.centroid
c_x = centroids.x
c_y = centroids.y
shape = True

nlist = list(g.nodes())
n = len(nlist)


pos = nx.kamada_kawai_layout(g)
if shape:
    pos = {node:(c_x[node],c_y[node]) for node in g.nodes}


initial_infection = 2
spontaneous = 0.01
recover = .05
spread = .2
reinfect = False

S={n for n in nlist}
I=set()
R=set()

Ss=[len(g.nodes())]
Is=[]
Rs=[0]

ncs = {x:0 for x in nlist}

num_steps = 100

colordict= {0:'y',1:'r',2:'g'}


plt.figure()
nx.draw(g,pos=pos,node_color=[colordict[ncs[x]] for x in nlist])
plt.savefig("./SIRplots/empty.png")
plt.close()

if shape:
    df["infect"] = df.index.map(ncs)
    df.plot(column = "infect")
    plt.savefig("./SIRplots/dfempty.png")
    plt.close()

for i in range(initial_infection):
    infected = random.choice(nlist)
    ncs[infected] = 1
    I.add(infected)
    S.remove(infected)
    
Ss.append(len(S))
Is.append(len(I))
Rs.append(len(R))

    

    
    
    
plt.figure()
nx.draw(g,pos=pos,node_color=[colordict[ncs[x]] for x in nlist])
plt.savefig("./SIRplots/initial.png")
plt.close()

if shape:
    df["infect"] = df.index.map(ncs)
    df.plot(column = "infect")
    plt.savefig("./SIRplots/dfinitial.png")
    plt.close()


for step in range(num_steps):
    
    for i in range(n):
        if ncs[nlist[i]] == 1:
            if random.random() < recover:
                if reinfect:
                    ncs[nlist[i]] = 0
                    I.remove(nlist[i])
                    S.add(nlist[i])
                else:
                    ncs[nlist[i]] = 2
                    I.remove(nlist[i])
                    R.add(nlist[i])
        elif ncs[nlist[i]] == 0:
            sick = 0
            for neighbor in g.neighbors(nlist[i]):
                if ncs[neighbor]==1:
                    if np.random.binomial(1,spread) ==1:
                        sick = 1  
                        I.add(nlist[i])
                        S.remove(nlist[i])
                        break
            ncs[nlist[i]] = sick
            
    Ss.append(len(S))
    Is.append(len(I))
    Rs.append(len(R))

    #plt.figure()
    nx.draw(g,pos=pos,node_color=[colordict[ncs[x]] for x in nlist])
    plt.savefig(f"./SIRplots/step{step:03d}.png")
    plt.close()
    if shape:
        df["infect"] = df.index.map(ncs)
        df.plot(column = "infect")
        plt.savefig(f"./SIRplots/dfstep{step:03d}.png")
        plt.close()
    
    
    if spontaneous > 0:
        num_infect = np.random.binomial(Ss[-1],spontaneous)
        for j in range(num_infect):
            infected = random.choice(list(S))
            ncs[infected] = 1
            I.add(infected)
            S.remove(infected)

            
    elif Is[-1]==0:
        break

nx.draw(g,pos=pos,node_color=[colordict[ncs[x]] for x in nlist])
plt.show()

if shape:
    df["infect"] = df.index.map(ncs)
    df.plot(column = "infect")
    plt.savefig(f"./SIRplots/dffinal.png")
    plt.close()
    df.plot(column = "infect")
    plt.show()

plt.figure()
plt.plot([x/n for x in Ss], color = 'y', label='S')           
plt.plot([x/n for x in Is], color = 'r', label='I')   
plt.plot([x/n for x in Rs], color = 'g', label='R')    
plt.legend()
plt.ylabel("% of nodes")
plt.xlabel("Time Step")
plt.savefig("./SIRplots/proportions.png")  
plt.close()
   
plt.figure()
plt.plot([x/n for x in Ss], color = 'y', label='S')           
plt.plot([x/n for x in Is], color = 'r', label='I')   
plt.plot([x/n for x in Rs], color = 'g', label='R') 
plt.ylabel("% of nodes")
plt.xlabel("Time Step")   
plt.legend()
plt.show()

        
        











