import pandas as pd
import numpy as np
import pprint

def readNodes(filename):
    # all input files should be placed in data folder
    file = './data/' + filename
    # input has no headers
    node = pd.read_csv(file, sep=',', header=None, dtype = np.float32)
    print(node[0].tolist())
    return node[0].tolist()

def readEdges(filename):
    file = './data/' + filename
    # input format will be 1st node, 2nd node and edge type, no headers
    headers = ['node1', 'node2', 'type']
    data = pd.read_csv(file, sep=',', names=headers, header=None)
    node1 = list(data['node1'].astype(np.int32))
    node2 = list(data['node2'].astype(np.int32))
    edgeType = data['type'].astype(str)
    nNode = len(node1)
    if (len(node2) == nNode & len(edgeType) == nNode):
        return processEdges(node1, node2, edgeType, nNode)
    else:
        raise ValueError('Invalid input for edges: Empty values')

def addNewNode(edgeTypes):
    # each node will have a list of connected nodes for each edge type
    return { x: [] for x in edgeTypes }

def processEdges(node1, node2, edgeType, n):
    edges = {}
    typeList = np.unique(edgeType)
    for i in range(n):
        # add new node to the graph
        if node1[i] == node2[i]:
            raise ValueError('Invalid Graph')
        if node1[i] not in edges.keys():
            edges[node1[i]] = addNewNode(typeList)
        edges[node1[i]][edgeType[i]].append(node2[i])
        if node2[i] not in edges.keys():
            edges[node2[i]] = addNewNode(typeList)
        edges[node2[i]][edgeType[i]].append(node1[i])
    return edges

readNodes('nodes.txt')
pprint.pprint(readEdges('edges.txt'))

