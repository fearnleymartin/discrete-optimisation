# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 22:21:07 2016

@author: Fearnley
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tsp import intersections, twoOptSwap

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

    # number of points
    num_points = int(numbers[cur_entry])
    cur_entry += 1

    # get data on the points
    points = np.zeros((num_points, 2))
    for i_point in xrange(num_points):
        points[i_point, 0] = float(numbers[cur_entry])
        cur_entry += 1
        points[i_point, 1] = float(numbers[cur_entry])
        cur_entry += 1

    return points
    
def dist(A, B):
    return math.sqrt( (A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) )


def check_tsp_solution( solution, points ):
    num_points = points.shape[0]
    visited_nodes = np.zeros(num_points, dtype=bool)
    path_length = dist( points[solution[0]], points[solution[-1]] )
    for i_point in xrange(num_points-1):
        visited_nodes[i_point] = True
        path_length += dist( points[solution[i_point]], points[solution[i_point+1]] )

    is_valid_solution = False in visited_nodes
    return is_valid_solution, path_length


def plot_tsp_solution(solution, points):
    is_valid_solution, path_length = check_tsp_solution( solution, points )

    x = np.hstack((points[solution][:,0], points[solution[0]][0]))
    y = np.hstack((points[solution][:,1], points[solution[0]][1]))

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    solution_quality = ['Inconsistent', 'Valid']
    plt.title( '%s solution; %d points; length = %f'%(solution_quality[is_valid_solution], len(points), path_length) )
    plt.show(block=True)

def write_tsp_solution_to_file(file_location, solution, points, seed=None):
    is_valid_solution, path_length = check_tsp_solution( solution, points )
#    fi = open(file_location, 'r')  # Create file if doesn't already exist
#    fi.close()
    with open(file_location, 'r+') as f:
        lines = f.readlines()
        if len(lines) == 0:
            print('entered ' + str(seed))
            f.write(str(path_length))
            f.write('\n')
            for i in solution:
                f.write(str(i))
                f.write(" ")
            f.close()
        else:
            cur_best_distance = float(lines[0])
            if path_length < cur_best_distance:
                print('new best solution found, seed = ' +  str(seed))
                f.seek(0)
                f.write(str(path_length))
                f.write('\n')
                for i in solution:
                    f.write(str(i))
                    f.write(" ")
                f.close()
                
def load_solution(file_location):
    with open(file_location, 'r') as f:
        lines = f.readlines()
        route_list = lines[1].split(" ")[:-1]
        route = np.arange(len(route_list))
        for i in range(len(route_list)):
            route[i]=int(route_list[i])       
    return route
    
def removeIntersectionsAux(route, points):
    """
    Although uses efficient algorithm for detection of intersections,
    Direction of travel not taken into account yet, temporary solution is to try 
    both directions of travel when doing the 2 opt swaps
    """
    inters = intersections(route, points)
    inters = list(set(inters))
#    print(inters)
    for i in range(len(inters)):
        # i loops through the intersections
        best_route = route
        best_distance = check_tsp_solution(route, points)[1]
        for j in range(2):
            for k in range(2):
                point1=inters[i][0][j]
                point2=inters[i][1][k] 
                new_route = twoOptSwap(route,route.tolist().index(point1),route.tolist().index(point2) )
                is_valid_solution, distance = check_tsp_solution(new_route, points)
                if distance < best_distance:
                    best_distance = distance
                    best_route = new_route
        for j in range(2):
            for k in range(2):
                point1=inters[i][0][j] 
                point2=inters[i][1][k] 
                new_route = twoOptSwap(route,route.tolist().index(point2),route.tolist().index(point1) )
                is_valid_solution, distance = check_tsp_solution(new_route, points)
                if distance < best_distance:
                    best_distance = distance
                    best_route = new_route
        route = best_route
    return route
    
def dist_matrix(points):
    """
    :return list_ext: a matrix of the distance between all points
    Each row i (i.e. distance of point i to all other points) in sorted in reverse order
    """
    num_points = points.shape[0]
    list_ext = []
    for i in range(num_points):
        list_int = []
        
        for j in range(num_points):
            list_int.append((j,dist(points[i],points[j])))
        sorted_list_int = sorted(list_int, key=lambda tup: tup[1])[1:]
        list_ext.append(sorted_list_int)
    return list_ext
    
def sort_route_by_distance(route, points):
    """
    Sorts the route points according to the distance to the successors
    """
    num_points = points.shape[0]
    list_dist = []
    for i in range(num_points-1):
        list_dist.append((route[i],dist(points[route[i]],points[route[i+1]])))
    list_dist.append((route[-1], dist(points[route[-1]],points[route[0]])))
    list_dist = sorted(list_dist, key=lambda tup: tup[1], reverse=True)
    return list_dist