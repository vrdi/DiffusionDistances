# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:45:20 2019

@author: daryl
"""

def greedy_hamming(base_partition, new_partition):

   names = [j for j in base_partition.parts]

   new_names = {}

   for i in new_partition.parts:

       intersection_sizes = {}

       for name in names:

           intersection_sizes.update({len(set(base_partition.assignment.parts[name]).intersection(set(new_partition.assignment.parts[i]))): name})

       new_names.update({intersection_sizes[max(intersection_sizes.keys())]: i})

       names.remove(intersection_sizes[max(intersection_sizes.keys())])

   tot_nodes = len(new_partition.assignment)

   final_int_sizes = []

   for i in base_partition.parts:

       x = len(set(new_partition.assignment.parts[new_names[i]]).intersection(set(base_partition.assignment.parts[i])))

       final_int_sizes.append(x)

   ham_dist = tot_nodes - sum(final_int_sizes)

   return new_names, ham_dist;


def greedy_hamming_pop(base_partition, new_partition):

   names = [j for j in base_partition.parts]

   new_names = {}

   for i in new_partition.parts:

       intersections = {}

       intersection_pops = {}

       for name in names:

           intersections.update({name: set(base_partition.assignment.parts[name]).intersection(set(new_partition.assignment.parts[i]))})

           intersection_pops.update({sum([new_partition.graph.nodes[node]["TOTPOP"] for node in intersections[name]]): name})

       new_names.update({intersection_pops[max(intersection_pops.keys())]: i})

       names.remove(intersection_pops[max(intersection_pops.keys())])

   tot_pop = sum(base_partition["population"].values())

   final_int_pops = []

   for i in base_partition.parts:

       intersection_set = set(new_partition.assignment.parts[new_names[i]]).intersection(set(base_partition.assignment.parts[i]))

       intersection_list = list(intersection_set)

       intersection_pops = sum([new_partition.graph.nodes[node]["TOTPOP"] for node in intersection_list])

       final_int_pops.append(intersection_pops)

   ham_dist_pop = tot_pop - sum(final_int_pops)

   return new_names, ham_dist_pop;