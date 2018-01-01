# parametric procedure assumes $f_1$ follows a standardized (i.e. \sigma^2=1) non-centered gaussian distribution with \mu(unknown)
# the edges of the same type share the same strength

from copy import deepcopy
from random import randint, random, uniform
from math import exp, log
from statistics import mean
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity

# initialization, randomly assign 0 or 1 to the hidden state of each hypothesis, assign a value between 0.5 and 0.8 to each edge type \phi_k
def initialization(x_s, typeList, nparticle):
    # the hidden states of the hypotheses
    theta = [randint(0, 1) for i in x_s]
    # edge potential parameter in the Markov random field, namely \phi's in the paper (e.g., Figure 1)
    phi = {i: uniform(0.5, 0.8) for i in typeList}
    # particles used in persistent contrastive divergence algorithm
    particles = [[randint(0, 1) for i in x_s] for j in range(nparticle)]
    return theta, phi, particles

# check whether the parameters converge
def converge(list1, list2, epislon):
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > epislon:
            return False
    return True

# E step infer the hidden state of every hypothesis
# E step iterates all hypothesis (in a random order or a specific order)
# For hypothesis i, first find all neighbors connected, then calculate the likelihood of being 0 & being 1.
# Assign the hidden state (\theta_i in the paper) to 0 or 1 according to the two likelihood values.
# graph : the dependence graph
# x_s : test statistics 
# theta: hidden state of the hypotheses
# phi, mu, pi: parameters of the model
# option: parametric or semiparametric
def EStep(graph, x_s, theta, phi, mu, pi, option):
    for key, value in graph.items():
        #likelihood_0, likelihood_1
        likelihood_0 = 0
        likelihood_1 = 0
        for edgeType, neighbors in value.items():
            for neighbor in neighbors:
                if (theta[neighbor] == 0):
                    likelihood_0 += log(phi[edgeType])
                    likelihood_1 += log(1 - phi[edgeType])
                else:
                    likelihood_0 += log(1 - phi[edgeType])
                    likelihood_1 += log(phi[edgeType])
        likelihood_0 += log(norm.pdf(x_s[key]))
        if (option == 'parametric'):
            likelihood_1 += log(norm.pdf(x_s[key]-mu))
        elif (option == 'semiparametric'):
            #TODO: KDE for semiparametric
            #sklearn.neighbors.KernelDensity, kernel = 'epanechnikov'
            pass
        likelihood_0 = exp(likelihood_0) * (1-pi)
        likelihood_1 = exp(likelihood_1) * pi
        ratio = likelihood_0 / (likelihood_0 + likelihood_1)
        if (random() < ratio):
            theta[key] = 0
        else:
            theta[key] = 1
    return theta

# M step estimate the parameters in the model, including \mu in f1, strength of every edge type (\phi) and \pi
# \mu = avg(statistics of nodes whose hidden states are current assigned to be 1)
# estimate phi and pi via PCD-n (persistent contrastive divergence)
def MStep(graph, theta, x_s, phi, edgeList, particles, nparticle, n, pi, mu, t):
    new_mu = mean([x_s[i] for i in range(len(theta)) if (theta[i] == 1) ])
    for i in range(len(particles)):
        for j in range(n):
            for key, value in graph.items():
                #likelihood_0, likelihood_1
                likelihood_0 = 0
                likelihood_1 = 0
                for edgeType, neighbors in value.items():
                    for neighbor in neighbors:
                        if (particles[i][neighbor] == 0):
                            likelihood_0 += log(phi[edgeType])
                            likelihood_1 += log(1 - phi[edgeType])
                        else:
                            likelihood_0 += log(1 - phi[edgeType])
                            likelihood_1 += log(phi[edgeType])
                likelihood_0 = exp(likelihood_0) * (1-pi)
                likelihood_1 = exp(likelihood_1) * pi
                ratio = likelihood_0 / (likelihood_0 + likelihood_1)
            if (random() < ratio):
                particles[i][key] = 0
            else:
                particles[i][key] = 1
    
    count_equal_observe = {i: [0, 0] for i in phi.keys()}
    count_equal_particle = deepcopy(count_equal_observe)

    pi_observe = sum(theta) / len(theta)
    pi_particle = sum([sum(i) for i in particles]) / (nparticle * len(theta))

    for key, value in edgeList.items():
        for (node1, node2) in value:
            if (theta[node1] == theta[node2]):
                count_equal_observe[key][0] += 1
            count_equal_observe[key][1] += 1
            for i in range(len(particles)):
                if (particles[i][node1] == particles[i][node2]):
                    count_equal_particle[key][0] += 1
                count_equal_particle[key][1] += 1

    likelihood_observe = {i: (count_equal_observe[i][0] / count_equal_observe[i][1]) for i in count_equal_observe.keys()}
    likelihood_particle = {i: (count_equal_particle[i][0] / count_equal_particle[i][1]) for i in count_equal_particle.keys()}
    
    new_phi = {i: (phi[i] + (1 / (t+1000)) * (likelihood_observe[i] - likelihood_particle[i])) for i in phi.keys()}
    new_pi = pi + (1/(t+1000)) * (pi_observe - pi_particle)
    return new_mu, new_phi, new_pi

def EM(graph, x_s, edgeList, typeList, nparticle, n, mu, pi, epislon, option):
    theta, phi, particles = initialization(x_s, typeList, nparticle)
    t = 0
    while True:
        theta = EStep(graph, x_s, theta, phi, mu, pi, option)
        new_mu, new_phi, new_pi = MStep(graph, theta, x_s, phi, edgeList, particles, nparticle, n, pi, mu, t)
        if converge([new_mu, new_pi] + list(new_phi.values()), [mu, new_pi] + list(phi.values()), epislon):
            mu = new_mu
            phi = new_phi
            pi = new_pi
            break
        else:
            mu = new_mu
            phi = new_phi
            pi = new_pi
            t += 1
    theta = EStep(graph, x_s, theta, phi, mu, pi, option)
    return theta, phi, mu, pi


