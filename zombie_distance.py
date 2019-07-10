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
from gerrychain.tree import recursive_tree_part
from gerrychain.partition import Partition
from gerrychain.updaters import Tally, cut_edges


g = nx.grid_graph([10,10])

g = nx.karate_club_graph()

ns=50

g = Graph.from_json("./Data/BG05.json")
df = gpd.read_file("./Data/BG05.shp")
centroids = df.centroid
c_x = centroids.x
c_y = centroids.y
shape = True

nlist = list(g.nodes())
n = len(nlist)

totpop = 0
for node in g.nodes():
    g.node[node]["TOTPOP"]=int(g.node[node]["TOTPOP"])

    totpop += g.node[node]["TOTPOP"]
    

pos = nx.kamada_kawai_layout(g)
if shape:
    pos = {node:(c_x[node],c_y[node]) for node in g.nodes}


cddict1 = recursive_tree_part(g,range(4),totpop/4,"TOTPOP", .01,1)

cddict2 = recursive_tree_part(g,range(4),totpop/4,"TOTPOP", .01,1)

def step_num(partition):
    parent = partition.parent
    if not parent:
        return 0
    return parent["step_num"] + 1

def b_nodes_bi(partition):
    return {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})   


updaters = {
    "population": Tally("TOTPOP"),
    "cut_edges": cut_edges,
    "step_num": step_num,
    'b_nodes': b_nodes_bi
}


Part1 = Partition(g,cddict1,updaters)
Part2 = Partition(g,cddict2,updaters)



initial_infection = len(Part1['b_nodes'])

spontaneous = 0.01
recover = .01
spread = .05
reinfect = False

num_attempts = 100

totals=[]

for zz in range(num_attempts):

    S={n for n in nlist if n not in Part1['b_nodes']}
    I=Part1['b_nodes'].copy()
    R=set()
    
    Ss=[len(S)]
    Is=[len(I)]
    Rs=[0]
    
    ncs = {x:0 for x in nlist}
    
    
    colordict= {0:'y',1:'r',2:'g'}
    
    to_hit = Part2['b_nodes'].copy()
    
    for infected in I:
        ncs[infected] = 1
        to_hit.discard(infected)
    
    """
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
    """
    
    
    
       
    Ss.append(len(S))
    Is.append(len(I))
    Rs.append(len(R))
    
        
    
        
        
        
    plt.figure()
    nx.draw(g,pos=pos,node_color=[colordict[ncs[x]] for x in nlist],node_size=ns)
    plt.savefig("./SIRplots/initial.png")
    plt.close()
    
    #if shape:
    #    df["infect"] = df.index.map(ncs)
    #    df.plot(column = "infect")
    #    plt.savefig("./SIRplots/dfinitial.png")
    #    plt.close()
    
    
    step = 0
    while to_hit:
        step+=1
        
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
                            to_hit.discard(nlist[i])
                            break
                ncs[nlist[i]] = sick
                
        Ss.append(len(S))
        Is.append(len(I))
        Rs.append(len(R))
    
        #plt.figure()
        """
        nx.draw(g,pos=pos,node_color=[colordict[ncs[x]] for x in nlist])
        plt.savefig(f"./SIRplots/step{step:03d}.png")
        plt.close()
        """
        #if shape:
        #    df["infect"] = df.index.map(ncs)
        #    df.plot(column = "infect")
        #    plt.savefig(f"./SIRplots/dfstep{step:03d}.png")
        #    plt.close()
        
        
        if spontaneous > 0:
            num_infect = np.random.binomial(Ss[-1],spontaneous)
            for j in range(num_infect):
                infected = random.choice(list(S))
                ncs[infected] = 1
                I.add(infected)
                S.remove(infected)
                to_hit.discard(infected)
    
                
        elif Is[-1]==0:
            break
    
    
    print("Hit all in ", step, " steps")
    totals.append(step)
    
    nx.draw(g,pos=pos,node_color=[colordict[ncs[x]] for x in nlist],node_size=ns)
    plt.show()
    
    #if shape:
    #    df["infect"] = df.index.map(ncs)
    #    df.plot(column = "infect")
    #    plt.savefig(f"./SIRplots/dffinal.png")
    #    plt.close()
    #    df.plot(column = "infect")
    #    plt.show()
    
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

        
        











