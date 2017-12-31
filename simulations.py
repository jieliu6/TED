# simulate different dependence structures and apply the multiple testing procedures

#import functions
import inputs
from random import randint, random, uniform, gauss
from math import exp, log
import pprint

# set the ground truth of the parameters
edge_type_num = 2
phi_true = {1:0.6,2:0.8}
pi_true = 0.49
mu_true = 2.5
simulation_num = 20
mcmc_num = 1000

# a 10-by-10 grid
grid_strc_size = 10
node_num = grid_strc_size*grid_strc_size
# set the edges to different types
# edges_grid is for the edges
edges_grid = {1:[],2:[]}
for i in range(grid_strc_size):
  for j in range(grid_strc_size-1):
    edge_1_type = randint(1,2)
    edge_2_type = randint(1,2)
    node_index_1_1 = i*grid_strc_size + j
    node_index_1_2 = i*grid_strc_size + j + 1
    node_index_2_1 = i + j*grid_strc_size
    node_index_2_2 = i + (j + 1)*grid_strc_size
    edges_grid[edge_1_type].append((node_index_1_1,node_index_1_2)) 
    edges_grid[edge_2_type].append((node_index_2_1,node_index_2_2)) 

#pprint.pprint(edges_grid)

# write the simulated structure into a file which can be used as one of the input files
f = open("data/simulated_data/simulated_edges.txt", "w+")
for key, value in edges_grid.items():
    for pair in value:
        (node1, node2) = pair
        data = [str(node1), str(node2), str(key)]
        f.write(",".join(data) + "\n")
f.close()

# graph is for recording the neighbors of each node
# typeList is the list of all edge types
# edgeList are the edges (same as edges_grid above) 
graph, typeList, edgeList = inputs.readEdges("simulated_data/simulated_edges.txt")

#pprint.pprint(graph)

# initialization of the hidden state
theta_s = [[randint(0, 1) for i in range(node_num)] for j in range(simulation_num)]
# MCMC
for k in range(mcmc_num):
    for i in range(simulation_num):
        for j in range(node_num):
            for key, value in graph.items():
                #likelihood_0, likelihood_1
                likelihood_0 = 0
                likelihood_1 = 0
                for edgeType, neighbors in value.items():
                    for neighbor in neighbors:
                        if (theta_s[i][neighbor] == 0):
                            likelihood_0 += log(phi_true[int(edgeType)])
                            likelihood_1 += log(1 - phi_true[int(edgeType)])
                        else:
                            likelihood_0 += log(1 - phi_true[int(edgeType)])
                            likelihood_1 += log(phi_true[int(edgeType)])
                likelihood_0 = exp(likelihood_0) * (1-pi_true)
                likelihood_1 = exp(likelihood_1) * pi_true
                ratio = likelihood_0 / (likelihood_0 + likelihood_1)
            if (random() < ratio):
                theta_s[i][key] = 0
            else:
                theta_s[i][key] = 1

# generate statistics and ground truth
for i in range(simulation_num):
    f1 = open("data/simulated_data/statistics_s"+str(i)+".txt", "w+")
    f2 = open("data/simulated_data/groundTruth_s"+str(i)+".txt", "w+")
    for j in range(node_num):
        stat_current = gauss(0,1)+theta_s[i][j]*mu_true
        f1.write(str(stat_current)+"\n")
        f2.write(str(theta_s[i][j]) + "\n")
    f1.close()
    f2.close()




