
import numpy as np
import time
from itertools import ifilter
from heapq import heappush, heappop
from collections import deque

def read_numbers(data_file):
    input_data_file = open(data_file, 'r')
    input_data = input_data_file.readlines()
    input_data_file.close()

    numbers = np.array([])
    for i_line in xrange(len(input_data)):
        entries = input_data[i_line].split()
        entries = filter(None, entries) # remove empty entries
        line_numbers = [ float(x) if x.lower != "inf" else float("inf") for x in entries ]
        numbers = np.append(numbers, line_numbers)
    return numbers

def read_data(data_file):
    numbers = read_numbers(data_file)
    cur_entry = 0

    # number of nodes
    n = int(numbers[cur_entry])
    cur_entry += 1

    # init graph
    neighbors = [None] * n
    weights = [None] * n

    # construct the graph
    for i_node in xrange(n):
        num_neighbors = int(numbers[cur_entry])
        cur_entry += 1
        cur_neighbors = np.zeros(num_neighbors, dtype = 'int32')
        cur_weights = np.zeros(num_neighbors, dtype = 'float')
        for i_neighbor in xrange(num_neighbors):
            cur_neighbors[i_neighbor] = int(numbers[cur_entry])
            cur_entry += 1
            cur_weights[i_neighbor] = numbers[cur_entry]
            cur_entry += 1
        neighbors[i_node] = cur_neighbors
        weights[i_node] = cur_weights

    # get pairs of nodes to compute distances
    num_pairs_of_interest = int(numbers[cur_entry])
    cur_entry += 1

    node_pairs = np.zeros( (num_pairs_of_interest, 2), dtype = 'int32' )
    for i_pair in xrange(num_pairs_of_interest):
        node_pairs[i_pair][0] = int(numbers[cur_entry])
        cur_entry += 1
        node_pairs[i_pair][1] = int(numbers[cur_entry])
        cur_entry += 1

    return neighbors, weights, node_pairs
   


###############################
'Min heap implementation functions'
'These functions are used for implementing a min-heap / priority queue \
from heapq https://docs.python.org/2/library/heapq.html'
###############################

REMOVED= '<removed-task>'

def add_node(node, distance, pq, entry_finder):
    'Add a new node or update the distance of an existing node'
    if node in entry_finder:
        remove_node(node, entry_finder)
    entry = [distance, node]
    entry_finder[node]=entry
    heappush(pq, entry)
    
def remove_node(node, entry_finder):
    'Mark an existing node as REMOVED. Raise KeyError if not found.'
    entry = entry_finder.pop(node)
    entry[-1] = REMOVED
    
def pop_node(pq, entry_finder):
    'Remove and return the lowest distance node. Raise KeyError if empty.'
    while pq:
        distance, node = heappop(pq)
        if node is not REMOVED:
            del entry_finder[node]
            return node
    raise KeyError('pop from an empty priority queue')
    
###########################################
    'Algorithms'
###########################################
    
def dijkstra(neighbors, weights, s):
    '''
    NB Graph cannot contain negative weights
    :param neighbors: array of arrays where the index i of the outer array corresponds to
    the node i and the integers in the inner arrays represent the neighbors of node i
    :param weights: array of arrays where the index i of the outer array corresponds to
    the node i and the float j in the inner array i represents the weight of the edge between 
    node i and its neighbor number j (refering to neighbors)
    :param s: an int representing the starting node of the algorithm
    :return: an array of the shortest distances between s and all the other nodes
    '''
    # Check all weights are positive
    if (np.any(weights<0)):
        raise Exception('All weights must be postive to run Dijkstra\'s algorithm')
        
    # min heap implementation, see https://docs.python.org/2/library/heapq.html
    Q = [] # corresponds to the heap
    entry_finder = {} # a mapping between nodes and distances
    
    # initialise distance array with infinity
    distance = np.zeros(len(neighbors))
    distance.fill(float('inf'))
    distance[s]=0.0
    
    #Initialise min heap with all nodes
    for i in range(len(neighbors)):
        add_node(i, float('inf'), Q, entry_finder)
    add_node(s,0.0,Q,entry_finder)
    
    while entry_finder:
        i = pop_node(Q, entry_finder)
        # j index is used to get the index of node j amoungst successors of i
        # this is necessary to then get the weight of the  node        
        for (j_index,j) in enumerate(neighbors[i]):
            j=neighbors[i][j_index]
            if j in entry_finder:
                if distance[j] > distance[i] + weights[i][j_index]:
                    distance[j] = distance[i] + weights[i][j_index]
                    add_node(j, distance[j], Q, entry_finder)
    return distance            
            
def BFS(neighbors, weight, s, t):
    '''
    NB Used for constant weight graph only (test4)
    :param neighbors: array of arrays where the index i of the outer array corresponds to
    the node i and the integers in the inner arrays represent the neighbors of node i
    :param weights: array of arrays where the index i of the outer array corresponds to
    the node i and the float j in the inner array i represents the weight of the edge between 
    node i and its neighbor number j (refering to neighbors)
    :param s: an int representing the starting node of the algorithm
    :param t: an int representing the target node of the algorithm
    :return: a float represeting the minimum distance between s and t
    '''
    
    # initialise distance array with infinity
    distance = np.zeros(len(neighbors))
    distance.fill(float('inf'))
    distance[s]=0.0
    
    #Create queue and enqueue root
    Q = deque()
    Q.append(s)
    
    while distance[t]==float('inf'):
        i = Q.popleft()       
        for j in ifilter(lambda x: distance[x]==float('inf'), neighbors[i]):
            distance[j] = distance[i] + weight
            Q.append(j)
    return distance[t]
    
def BellmanFord(neighbors, weights, s):
    '''
    :param neighbors: array of arrays where the index i of the outer array corresponds to
    the node i and the integers in the inner arrays represent the neighbors of node i
    :param weights: array of arrays where the index i of the outer array corresponds to
    the node i and the float j in the inner array i represents the weight of the edge between 
    node i and its neighbor number j (refering to neighbors)
    :param s: an int representing the starting node of the algorithm
    :return: an array of the shortest distances between s and all the other nodes
    '''
    # initialise distance array with infinity
    distance = np.zeros(len(neighbors))
    distance.fill(float('inf'))
    distance[s]=0.0
        
    # relax edges repeatedly
    for i in range(len(neighbors)):
        for (u,u_neighbors) in enumerate(neighbors):
            for (v_index,v) in enumerate(u_neighbors):
                w = weights[u][v_index]
                if distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w

    # check for negative-weight cycles
    for (u,u_neighbors) in enumerate(neighbors):
            for (v_index,v) in enumerate(u_neighbors):
                w = weights[u][v_index]
                if distance[u] + w < distance[v]:
                    raise Exception('Graph contains a negative-weight cycle between '+ str(u) + ' and ' + str(v))
                    
    return distance          
                    
#################################################
'Solutions'
'Code for calculating solution to each of the problems'
#################################################

def solution1():
    input_file="task1_test1.txt"
    output="task1_test1_solution.txt"
    (neighbors, weights, node_pairs) = read_data(input_file)
    num_pairs = len(node_pairs)
    answer=np.empty(num_pairs)
    start_time=time.time()
    for i, node_pair in enumerate(node_pairs):
        answer[i]=dijkstra(neighbors,weights,node_pair[0])[node_pair[1]]
    execution_time=time.time()-start_time
#    f = open(output, 'w')
#    for i in range(len(answer)):
#        f.write(str(answer[i]))
#        f.write("\n")
#    f.close()
    print(execution_time)
    return answer
    
def solution2():
    input_file="task1_test2.txt"
    output="task1_test2_solution.txt"
    (neighbors, weights, node_pairs) = read_data(input_file)
    num_pairs = len(node_pairs)
    answer=np.empty(num_pairs)
    start_time=time.time()
    for i, node_pair in enumerate(node_pairs):
        answer[i]=dijkstra(neighbors,weights,node_pair[0])[node_pair[1]]
    execution_time=time.time()-start_time
#    f = open(output, 'w')
#    for i in range(len(answer)):
#        f.write(str(answer[i]))
#        f.write("\n")
#    f.close()
    print(execution_time)
    return answer
    
def solution3():
    input_file="task1_test3.txt"
    output="task1_test3_solution.txt"
    (neighbors, weights, node_pairs) = read_data(input_file)
    num_pairs = len(node_pairs)
    answer=np.empty(num_pairs)
    start_time=time.time()
    for i, node_pair in enumerate(node_pairs):
        answer[i]=dijkstra(neighbors,weights,node_pair[0])[node_pair[1]]
    execution_time=time.time()-start_time
#    f = open(output, 'w')
#    for i in range(len(answer)):
#        f.write(str(answer[i]))
#        f.write("\n")
#    f.close()
    print(execution_time)
    return answer
    
def solution4():
    input_file="task1_test4.txt"
    output="task1_test4_solution.txt"
    (neighbors, weights, node_pairs) = read_data(input_file)
    num_pairs = len(node_pairs)
    answer=np.empty(num_pairs)
    start_time=time.time()
    for i, node_pair in enumerate(node_pairs):
        answer[i] = BellmanFord(neighbors,weights,node_pair[0])[node_pair[1]]
        print('answer '+str(i))
    execution_time=time.time()-start_time
#    f = open(output, 'w')
#    for i in range(len(answer)):
#        f.write(str(answer[i]))
#        f.write("\n")
#    f.close()
    print(execution_time)
    return answer
    
def solution5():
    input_file="task1_test5.txt"
    output="task1_test5_solution_test.txt"
    (neighbors, weights, node_pairs) = read_data(input_file)
    #Check all weights are constant and equal to weight
    weight=weights[0]
    if (np.any(weights!=weight)):
        return Exception('This is not a constant weighted graph')
    num_pairs = len(node_pairs)
    answer=np.empty(num_pairs)
    start_time=time.time()
    for i, node_pair in enumerate(node_pairs):
        answer[i] = BFS(neighbors,2,node_pair[0],[node_pair[1]])
    execution_time=time.time()-start_time
#    f = open(output, 'w')
#    for i in range(len(answer)):
#        f.write(str(answer[i]))
#        f.write("\n")
#    f.close()
    print(execution_time)
    return answer

import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        neighbors, weights, node_pairs = read_data(file_location)
        if file_location=='task1_test1.txt':
            answer=solution1()
        elif file_location=='task1_test2.txt':
            answer=solution2()
        elif file_location=='task1_test3.txt':
            answer=solution3()
        elif file_location=='task1_test4.txt':
            answer=solution4()
        elif file_location=='task1_test5.txt':
            answer=solution5()
        else:
            print('You must call a valid file')
        print '\n'.join(map(str, answer))
    else:
        print 'This script requires an input file as command line argument.'
