from inputs import *
from functions import *

import pprint

mu = 2.0
nparticle = 100
n = 10
epislon = 0.0001
pi = 0.495
option = "parametric"

filenames = []

for i in range(20):
   filenames.append('simulated_data/statistics_s' + str(i) + '.txt')

graph, typeList, edgeList = readEdges('simulated_data/simulated_edges.txt')

for filename in filenames:
    x_s = readNodes(filename)
    print(filename)
    theta, phi, particles = initialization(x_s, typeList, nparticle)
    theta, phi, mu, pi = EM(graph, x_s, edgeList, phi, nparticle, n, mu, pi, epislon, option)
    print(phi, mu, pi)

