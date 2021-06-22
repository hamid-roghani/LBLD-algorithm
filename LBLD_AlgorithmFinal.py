import time
from itertools import groupby
from math import ceil
from operator import itemgetter
import numpy as np
from numpy import loadtxt
from pandas import DataFrame
import collections
from sklearn.metrics.cluster import normalized_mutual_info_score
# ---------------------------------- Load Dataset -------------------------------------------
dataset_name = "karate" # name of dataset
path = "./datasets/" + dataset_name + ".txt" # path to dataset
iteration = 1           # number of iterations for label selection step (mostly is set to 1 or 2)
merge_flag = 1         # merge_flag=0 -> do not merge //  merge_flag=1 -> do merge
write_flag = 0        # 1 means write nodes labels to file. 0 means do not write
modularity_flag = 1  # 1 means calculate modularity. 0 means do not calculate modularity
NMI_flag = 1        # 1 means calculate NMI. 0 means do not calculate NMI
# ------------------------- compute nodes neighbors and nodes degree --------------------------
nodes_neighbors = {}

i=0
with open(path) as f:

    for line in f:
        row = str(line.strip()).split('\n')[0].split('\t')
        temp_arrey = []
        for j in row:
            if j == '':
                temp_arrey.append(-1)
            if j != '':
                temp_arrey.append(int(j))
        nodes_neighbors.setdefault(i, []).append(temp_arrey)

        if nodes_neighbors[i][0][0] != -1:
            nodes_neighbors.setdefault(i, []).append(len(nodes_neighbors[i][0]))

        elif nodes_neighbors[i][0][0] == -1:
            nodes_neighbors.setdefault(i, []).append(0)
        i = i+1  
N = i # number of nodes
start_time = time.time()
# -----------------------------Compute node importance------------------------------

for i in range(N):
    CN_sum = 0
    temp = []
    d = {}
    if nodes_neighbors[i][1] > 1:

        for neighbor in nodes_neighbors[i][0]:

            intersect = len(list(set(nodes_neighbors[i][0]) & set(nodes_neighbors[neighbor][0])))
            union = nodes_neighbors[i][1] + nodes_neighbors[neighbor][1] - intersect

            if nodes_neighbors[i][1] > nodes_neighbors[neighbor][1]:
                difResult = 1 + len(set(nodes_neighbors[neighbor][0]).difference(set(nodes_neighbors[i][0])))
            else:
                difResult = 1 + len(set(nodes_neighbors[i][0]).difference(set(nodes_neighbors[neighbor][0])))

            CN_sum = CN_sum + ((intersect / (intersect + union)) * (intersect / difResult))
            d[neighbor] = (neighbor, ((intersect / (intersect + union)) * (intersect / difResult)))

    elif nodes_neighbors[i][1] == 1:
        CN_sum = 0
        d[nodes_neighbors[i][0][0]] = (nodes_neighbors[i][0][0], 0)

    elif nodes_neighbors[i][1] == 0:
        CN_sum = 0
        d[-1] = (-1,-1)

    nodes_neighbors.setdefault(i, []).append(list(max(d.values(), key=itemgetter(1))))
    nodes_neighbors.setdefault(i, []).append(([CN_sum, i, 0]))

nodes_neighbors = {k: v for k, v in sorted(nodes_neighbors.items(), key=lambda item: item[1][3][0], reverse=True)}

# --------------------------Select most similar neighbor-----------------------

for i in range(N):
    if nodes_neighbors[i][1] > 1:

        if nodes_neighbors[i][2][1] == 0:  # if similarity is equal to 0, we select neighbor with highest degree
            neighbors_degree = []

            for j in nodes_neighbors[i][0]:
                neighbors_degree.append((j, nodes_neighbors[j][1]))

            max_degree_neighbor = max(neighbors_degree, key=itemgetter(1))[0]
            nodes_neighbors.setdefault(i, []).append([max_degree_neighbor, -1])
            nodes_neighbors[i][3][1] = max_degree_neighbor

            continue
        elif nodes_neighbors[i][2][1] != 0:

            if nodes_neighbors[i][3][0] > nodes_neighbors[nodes_neighbors[i][2][0]][3][0]:
                nodes_neighbors.setdefault(i, []).append([i, nodes_neighbors[i][2][0]])
                nodes_neighbors[i][3][1] = i
            else:
                nodes_neighbors.setdefault(i, []).append([nodes_neighbors[i][2][0], nodes_neighbors[i][2][0]])
                nodes_neighbors[i][3][1] = nodes_neighbors[i][2][0]
    else:
        nodes_neighbors.setdefault(i, []).append([i, -1])
        nodes_neighbors[i][3][1] = i


for i in range(N):
    nodes_neighbors[i][3][1] = nodes_neighbors[nodes_neighbors[i][4][0]][3][1]
# ---------------------------- Top 5 percent important nodes -----------------------------
top_5percent = ceil(N * 5 / 100)
most_important = {}

dict_items = nodes_neighbors.items()
selected_items = list(dict_items)[:top_5percent]  # take top 5% items

for i in range(top_5percent):
    most_important[selected_items[i][0]] = (nodes_neighbors[selected_items[i][0]][4][1])

for i in most_important:
    temp_label = []
    if nodes_neighbors[i][3][0] >= nodes_neighbors[most_important[i]][3][0]:
        temp_label = nodes_neighbors[i][3][1]
        nodes_neighbors[most_important[i]][3][1] = temp_label
    else:
        temp_label = nodes_neighbors[most_important[i]][3][1]
        nodes_neighbors[i][3][1] = temp_label

    CN = []
    CN = list(set(nodes_neighbors[i][0]) & set(nodes_neighbors[most_important[i]][0]))

    for j in CN:
        nodes_neighbors[j][3][1] = temp_label
        nodes_neighbors[j][3][2] = 1

        nodes_neighbors[i][3][2] = 1
        nodes_neighbors[most_important[i]][3][2] = 1


del most_important
del CN
# --------------------------------- Balanced Label diffusion ------------------------------------------------
flag_lock = 1
counter = 1
high = 0
low = N - 1
nodes_key = list(nodes_neighbors.keys())

while counter < (N + 1):

    if flag_lock == 1:
        current_node = nodes_key[high]
        high = high + 1
        flag_lock = 0

        if nodes_neighbors[current_node][1] > 1:

            if nodes_neighbors[current_node][3][2] == 0:

                current_node_neighbor = []
                for j in nodes_neighbors[current_node][0]:
                    current_node_neighbor.append((j, nodes_neighbors[j][3][1]))

                sorted_input = sorted(current_node_neighbor, key=itemgetter(1))
                groups = groupby(sorted_input, key=itemgetter(1))

                neighbors_influence = []

                for i in groups:
                    sum_values = 0
                    for j in i[1]:

                        sum_values = sum_values + nodes_neighbors[j[0]][3][0]
                    neighbors_influence.append((i[0], sum_values))
                nodes_neighbors[current_node][3][1] = max(neighbors_influence, key=itemgetter(1))[0]

    elif flag_lock == 0:

        current_node = nodes_key[low]
        low = low - 1
        flag_lock = 1

        if nodes_neighbors[current_node][1] > 1:

            if nodes_neighbors[current_node][3][2] == 0:

                current_node_neighbor = []
                for j in nodes_neighbors[current_node][0]:
                    current_node_neighbor.append((j, nodes_neighbors[j][3][1]))

                sorted_input = sorted(current_node_neighbor, key=itemgetter(1))
                groups = []
                groups = groupby(sorted_input, key=itemgetter(1))

                neighbors_influence = []

                for i in groups:
                    sum_values = 0
                    for j in i[1]:
                        sum_values = sum_values + (nodes_neighbors[current_node][1] * nodes_neighbors[j[0]][1])
                    neighbors_influence.append((i[0], sum_values))
                nodes_neighbors[current_node][3][1] = max(neighbors_influence, key=itemgetter(1))[0]
    counter += 1
del groups
del neighbors_influence

# ----------------------------- Give labels to nodes with degree=1 ---------------------------------
        
for i in range(N):
    if nodes_neighbors[i][1] == 1:
        nodes_neighbors[i][3][1] = nodes_neighbors[nodes_neighbors[i][0][0]][3][1]
# ---------------------Label selection step (the iterative part of algorithm) ---------------------

for itter in range(iteration):
    for i in range(N):
        if nodes_neighbors[i][1] > 1:
            current_node_neighbor = []

            for j in nodes_neighbors[i][0]:
                current_node_neighbor.append((j, nodes_neighbors[j][3][1]))  # neighbors with their label
            sorted_input = sorted(current_node_neighbor, key=itemgetter(1))
            groups = []
            groups = groupby(sorted_input, key=itemgetter(1))  # nodes are grouped based on their community label

            neighbors_frequency = []
            for j in groups:
                neighbors_frequency.append((j[0], len(list(j[1]))))  # Labels frequency

            temp_max = max(neighbors_frequency, key=itemgetter(1))  # label with highest frequency
            indices = []
            indices = [x for x, y in enumerate(neighbors_frequency) if y[1] == temp_max[1]]

            selected_label = []
            if len(indices) == 1:
                selected_label = temp_max[0]
            else:
                final_max = []
                max_influence = []
                for x in indices:
                    final_max.append(neighbors_frequency[x][0])  # stores only labels with highest frequency

                for x in final_max:
                    temp_influence = 1
                    for y in current_node_neighbor:
                        if y[1] == x:
                            temp_influence = temp_influence * nodes_neighbors[y[0]][3][0]
                    max_influence.append((x, temp_influence))

                selected_label = max(max_influence, key=itemgetter(1))[0]
            nodes_neighbors[i][3][1] = selected_label
# ---------------------------------- Merge Small communities -------------------------------------------------
if merge_flag == 1:

    nodes_labels = DataFrame.from_dict(nodes_neighbors, orient='index')
    nodes_labels.columns = ['Neighbor','Degree', 'max_Similar', 'NI_Label', 'node_NeighborLabel']
    unique_labels = nodes_labels['NI_Label'].apply(lambda x: x[1]).unique()

    communities_group = {}
    for i in unique_labels:
        communities_group[i] = []

    for i in range(N):
        communities_group[nodes_neighbors[i][3][1]].append(i)  # nodes are grouped their communities

    unique_labels_array = []
    for i in communities_group:
        temp_len = len(communities_group[i])
        if temp_len > 1:
            unique_labels_array.append((i, temp_len))

    max_community = max(unique_labels_array, key=itemgetter(1))[1]  # community with biggest size
    average_size = (N - max_community) / (len(unique_labels_array) - 1)  # average size of communities

    less_than_average_communities = []
    less_than_average_communities = list(filter(lambda x: x[1] < average_size, unique_labels_array))

    if less_than_average_communities:

        for i in less_than_average_communities:
            temp_small_communities = []

            for j in communities_group[i[0]]:
                temp_small_communities.append((j, nodes_neighbors[j][1] + nodes_neighbors[j][3][0]))

            candidate_node = []
            candidate_node = max(temp_small_communities, key=itemgetter(1))[0]  # candidate node of community

            temp_neighbors = []
            for j in nodes_neighbors[candidate_node][0]:
                temp_neighbors.append((j, nodes_neighbors[j][1] + nodes_neighbors[j][3][0]))  # neighbors with their score

            max_neighbor_community = max(temp_neighbors, key=itemgetter(1))[0]  # neighbor with maximum score
            selected_label = []
            if nodes_neighbors[max_neighbor_community][3][1] != nodes_neighbors[candidate_node][3][1]:
                if nodes_neighbors[max_neighbor_community][1] >= nodes_neighbors[candidate_node][1]:
                    selected_label = nodes_neighbors[max_neighbor_community][3][1]
            if selected_label:
                for j in temp_small_communities:
                    nodes_neighbors[j[0]][3][1] = selected_label

# -------------------------------Total Time of Algorithm----------------------------------------------------------------------
print("--- Total Execution time %s seconds ---" % (time.time() - start_time))
#----------------------------------- Write to Disk ------------------------------------------------------
ordered_nodes_neighbors = collections.OrderedDict(sorted(nodes_neighbors.items()))
if write_flag == 1:
    with open('./results/' + dataset_name + '.txt', 'w') as filehandle:
        for i in ordered_nodes_neighbors:
            filehandle.write('%s\n' % ordered_nodes_neighbors[i][3][1])

# ---------------------------------- Number of communities --------------------------------
nodes_labels = []
nodes_labels = DataFrame.from_dict(nodes_neighbors, orient='index')
nodes_labels.columns = ['Neighbor','Degree', 'max_Similar', 'NI_Label', 'node_NeighborLabel']
number_of_communities = nodes_labels['NI_Label'].apply(lambda x: x[1]).unique()
print("Number of Communities: ", len(number_of_communities))
# ----------------------------------- Modularity -------------------------------------------------------
if modularity_flag ==1:
    t = 0
    for i in nodes_neighbors:
        t = t + nodes_neighbors[i][1]
    edges = t / 2
    modu = 0
    are_neighbor = []

    for i in range(N):

        for j in range(N):
            if nodes_neighbors[i][3][1] == nodes_neighbors[j][3][1]:
                if nodes_neighbors[i][1] >= 1:

                    if j in nodes_neighbors[i][0]:
                        are_neighbor = 1
                    else:
                        are_neighbor = 0
                    modu = modu + (are_neighbor - ((nodes_neighbors[i][1] * nodes_neighbors[j][1]) / (2 * edges)))

    modularity_final = modu / (2 * edges)
    print('Modularity:  {}'.format(modularity_final))
# ------------------------------- NMI ---------------------------------------
if NMI_flag ==1:
    real_labels= loadtxt("./groundtruth/"+dataset_name+"_real_labels.txt", comments="#", delimiter="\t", unpack=False)
    detected_labels = []
    if dataset_name in ('karate','dolphins','polbooks','football'):
   
        for i in ordered_nodes_neighbors:
            detected_labels.append(ordered_nodes_neighbors[i][3][1])
    
        detected_labels=np.array(detected_labels)
        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels,detected_labels)))
    
    else:
        nodes_map = loadtxt("./datasets/nodes_map/" + dataset_name + "_nodes_map.txt", comments="#", delimiter="\t", unpack=False)

        for i in nodes_map:
            detected_labels.append(nodes_neighbors[i][3][1])
    
        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels,detected_labels)))





