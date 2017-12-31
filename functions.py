# parametric procedure assumes f1 follows a normalized non-centered gaussian distribution with \mu(unknown) and \sigma = 1
# the edges of the same type share the same strength
# E step infer the hidden state of every hypothesis
# M step estimate the parameters in the model, including \mu in f1 and strength of every edge type

# initialization, randomly assign 0 or 1 to each hypothesis(node), assign a value between 0.5 and 0.8 to each edge type \psi_k

# E step iterates all hypothesis(in random or order)
# for hypothesis i, first find all neighbors connected, then calculate the likelihood of being 0 & being 1.
# Assign the hidden state to 0 or 1 according to the two likelihood.

# M step 
# \mu = avg(statistics of nodes that are 1)
# estimate psi, PCD(persistent contrastive divergence)
from copy import deepcopy
import random
import math
from statistics import mean
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity

def initialization(node, typeList, nparticle):
    nodeValue = [random.randint(0, 1) for i in node]
    typeValue = {i: random.uniform(0.5, 0.8) for i in typeList}
    particles = [[random.randint(0, 1) for i in node] for j in range(nparticle)]
    return nodeValue, typeValue, particles

def converge(list1, list2, epislon):
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > epislon:
            return False
    return True

def EStep(graph, node, nodeValue, typeValue, mu, pi, option):
    for key, value in graph.items():
        #likelihood_0, likelihood_1
        likelihood_0 = 0
        likelihood_1 = 0
        for edgeType, neighbors in value.items():
            for neighbor in neighbors:
                if (nodeValue[neighbor] == 0):
                    likelihood_0 += math.log(typeValue[edgeType])
                    likelihood_1 += math.log(1 - typeValue[edgeType])
                else:
                    likelihood_0 += math.log(1 - typeValue[edgeType])
                    likelihood_1 += math.log(typeValue[edgeType])
        likelihood_0 += math.log(norm.pdf(node[key]))
        if (option == 'parametric'):
            likelihood_1 += math.log(norm.pdf(node[key]-mu))
        elif (option == 'semiparametric'):
            #TODO: KDE for semiparametric
            #sklearn.neighbors.KernelDensity, kernel = ’epanechnikov’
            pass
        likelihood_0 = math.exp(likelihood_0) * (1-pi)
        likelihood_1 = math.exp(likelihood_1) * pi
        ratio = likelihood_0 / (likelihood_0 + likelihood_1)
        if (random.random() < ratio):
            nodeValue[key] = 0
        else:
            nodeValue[key] = 1
    return nodeValue

def MStep(graph, nodeValue, node, typeValue, edgeList, particles, nparticle,  n, pi, mu, t):
    new_mu = mean([node[i] for i in range(len(nodeValue)) if (nodeValue[i] == 1) ])
    for i in range(len(particles)):
        for j in range(n):
            for key, value in graph.items():
                #likelihood_0, likelihood_1
                likelihood_0 = 0
                likelihood_1 = 0
                for edgeType, neighbors in value.items():
                    for neighbor in neighbors:
                        if (particles[i][neighbor] == 0):
                            likelihood_0 += math.log(typeValue[edgeType])
                            likelihood_1 += math.log(1 - typeValue[edgeType])
                        else:
                            likelihood_0 += math.log(1 - typeValue[edgeType])
                            likelihood_1 += math.log(typeValue[edgeType])
                likelihood_0 = math.exp(likelihood_0) * (1-pi)
                likelihood_1 = math.exp(likelihood_1) * pi
                ratio = likelihood_0 / (likelihood_0 + likelihood_1)
            if (random.random() < ratio):
                particles[i][key] = 0
            else:
                particles[i][key] = 1
    
    count_equal_observe = {i: [0, 0] for i in typeValue.keys()}
    count_equal_particle = deepcopy(count_equal_observe)

    pi_observe = sum(nodeValue) / len(nodeValue)
    pi_particle = sum([sum(i) for i in particles]) / (nparticle * len(nodeValue))

    for key, value in edgeList.items():
        for (node1, node2) in value:
            if (nodeValue[node1] == nodeValue[node2]):
                count_equal_observe[key][0] += 1
            count_equal_observe[key][1] += 1
            for i in range(len(particles)):
                if (particles[i][node1] == particles[i][node2]):
                    count_equal_particle[key][0] += 1
                count_equal_particle[key][1] += 1

    likelihood_observe = {i: (count_equal_observe[i][0] / count_equal_observe[i][1]) for i in count_equal_observe.keys()}
    likelihood_particle = {i: (count_equal_particle[i][0] / count_equal_particle[i][1]) for i in count_equal_particle.keys()}
    
    new_typeValue = { i: (typeValue[i] + (1 / (t+1000)) * (likelihood_observe[i] - likelihood_particle[i])) for i in typeValue.keys()}
    new_pi = pi + (1/(t+1000)) * (pi_observe - pi_particle)
    return new_mu, new_typeValue, new_pi

def EM(graph, node, edgeList, typeList, nparticle, n, mu, pi, epislon, option):
    nodeValue, typeValue, particles = initialization(node, typeList, nparticle)
    t = 0
    while True:
        nodeValue = EStep(graph, node, nodeValue, typeValue, mu, pi, option)
        new_mu, new_typeValue, new_pi = MStep(graph, nodeValue, node, typeValue, edgeList, particles, nparticle, n, mu, pi, t)
        if converge([new_mu, new_pi] + list(new_typeValue.values()), [mu, new_pi] + list(typeValue.values()), epislon):
            mu = new_mu
            typeValue = new_typeValue
            pi = new_pi
            break
        else:
            mu = new_mu
            typeValue = new_typeValue
            pi = new_pi
            t += 1
    nodeValue = EStep(graph, node, nodeValue, typeValue, mu, pi, option)


