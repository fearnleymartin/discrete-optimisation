
import numpy as np
import math
import sys

import random
import time

from tsp_aux import (
    read_data, 
    dist,
    check_tsp_solution,
    plot_tsp_solution,
    write_tsp_solution_to_file,
    load_solution,
    removeIntersectionsAux,
    dist_matrix,
    sort_route_by_distance,
    )

import bentley_ottman # Third party intersection library

def nearest_neighbors(points, start_point=None):
    """
    :param points: np array of point coordinates [[x1,y1], [x2,y2], ...]
    :param start_point: index (int) of starting point for building greedy solution
    Greedy algorithm: builds solution step by step by choosing closest point at each step
    Gives a reasonable approximation which can be used as a starting point for local search
    :returns solution: np array with the indices of the points to visit in order
    """
    # Initialise solution
    num_points = points.shape[0]
    solution = np.arange(num_points)
    points_to_visit = np.ones(num_points)  
    # we remove points one by one so as not to
    # visit them twice
    if start_point:
        cur_point = start_point
    else:
        # pick random starting point
        cur_point = random.randint(0, num_points-1)
    # set starting point
    solution[0] = cur_point
    
    # Build greedy solution point by point
    for i in range(1,len(points_to_visit)):
        # Find nearest neighbor to cur_point
        min_dist = float("inf")
        min_point = None
        points_to_visit[cur_point]=0
        for (point, active) in enumerate(points_to_visit):
            if active:
                dis = dist(points[cur_point], points[point])
                if dis < min_dist:
                    min_dist = dis
                    min_point = point
        
        solution[i]=min_point
        cur_point = min_point
    
    return solution
        
        
def twoOptSwap(route, i, k):
    """
    :param route: a possible solution (list of point indices in order)
    :param i: point indice  of beginning of swap
    :param k: point indice of end of swap
    :return newRoute: route where the pth between i and k has been inverted
    https://en.wikipedia.org/wiki/2-opt
    """
    newRoute = np.copy(route)
    for j in range(k-i+1):
        newRoute[k-j]=route[i+j]
    return newRoute
    

    
def removeIntersections(route, points):
    """
    Runs the remove intersections algorithms until no intersections are left
    (removing on intersection can introduce another)
    """  
    inters = intersections(route,points)
    while (len(inters)>0):
        route = removeIntersectionsAux(route, points)
        inters = intersections(route, points)
    return route

def intersections(route, points):
    """
    Uses lsi library to calculate intersections with Bentley-Ottman algo
    :param route: the list of points
    :param points: the corrdinates of chosen points
    the library takes as input a list of tuples
    :return: This function returns a dictionary of intersection points (keys) and a list of their associated segments (values).
    """
    # Create a n-uple of segments from the route
    S = [(tuple(points[i]),tuple(points[j])) for (i,j) in zip((route[:-1]),route[1:])]
    i,j = (route[-1],route[0])
    S.append((tuple(points[i]),tuple(points[j])))
    S = tuple(S)
    # Detect intersections (return coordindates of points)    
    inters = bentley_ottman.isect_segments(S)
    
    # Get the point indices of these coordinates and order in sense of travel
    proc_inters = []
    
    for seg in inters:
        # Process results to get point indices
        pl = points.tolist()
        s1_point_index_1 = pl.index(list(seg[0][0]))
        s1_point_index_2 = pl.index(list(seg[0][1]))
        s2_point_index_1 = pl.index(list(seg[1][0]))
        s2_point_index_2 = pl.index(list(seg[1][1]))
        
        # make sure segments are return in the direction of the route
        rl =route.tolist()
        if rl.index(s1_point_index_1)<rl.index(s1_point_index_2):
            s1 = (s1_point_index_1,s1_point_index_2)
        else:
            s1 = (s1_point_index_2,s1_point_index_1)
        if rl.index(s2_point_index_1)<rl.index(s2_point_index_2):
            s2 = (s2_point_index_1,s2_point_index_2)
        else:
            s2 = (s2_point_index_2,s2_point_index_1)
        
        # Return segments in the order of the sense of travel
        if rl.index(s1[0]) < rl.index(s2[0]):
            proc_inters.append((s1,s2))
        else:
            proc_inters.append((s2,s1))
        
    return proc_inters
    
def twoOptSwapComplete(route, points):
    """
    A basic two opt solution which tries all possible combinations of two opts
    Restart whenever a better route is found
    If after the 2 for loop, no new solutions have been found, the while loop terminates
    """
    is_valid_solution, best_distance = check_tsp_solution( route, points )
    while True:
        for i in range(len(route)):
            for k in range(i+1,len(route)):
                new_route = twoOptSwap(route,i,k)
                is_valid_solution, new_distance = check_tsp_solution(new_route, points)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    break
        else:
            return route
            


def sim_anneal1(route, points):
    """
    A metropolis heuristic with simulated annealing
    A two opt local search where sometimes non optimal solutions are excepted
    (in the aim of escaping local minima)
    The simulated annealing sets this probability according to the distance between
    the new found distance and the best distance, as well as a temperature that we 
    dynamically change during the search (from random to hill climbing)
    Tries first to swap long edges with short edges
    """
    num_points = points.shape[0]
    start_time = time.time()
    
    t=10 #  Temperature
    gradient = 1.2  # Cooling schedule
    max_run_time = 90  # Seconds
    reheat_coef = 1.01  # Reheating
    dist_mat = dist_matrix(points)
    points_vois = int(num_points) # look at neighboring points up to this limit
    is_valid_solution, best_distance = check_tsp_solution( route, points )
    overall_best_distance = best_distance
    overall_best_route = route
    while True:
        points_by_segment_length = sort_route_by_distance(route, points)
        # start with points who are joined to far away points
        for i_tuple in points_by_segment_length:
            i = i_tuple[0]
            gen = (k_tuple for k_tuple in dist_mat[i][:points_vois] if k_tuple[0] \
            >i)
            # only searches among neighboring points, we will try swapping on close points
            for k_tuple in gen:
                k=k_tuple[0]
                new_route = twoOptSwap(route,i,k)
                is_valid_solution, new_distance = check_tsp_solution(new_route, points)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    if best_distance < overall_best_distance:
                        overall_best_distance = best_distance
                        overall_best_route = route
                        global best_seed 
                        best_seed = seed
                        print('found new best ' + str(overall_best_distance))
                        start_time=time.time()
                else:
                    if time.time()-start_time > max_run_time:
                        return removeIntersections(overall_best_route,points)
                    # Select bad solutions with a probability threshold
                    threshold = math.exp(-(new_distance-best_distance)/t)
                    rand_num = random.uniform(0,1)
                    if rand_num < threshold:
                        route = new_route
                        t=t/gradient  # Cooling
                        print("t: " + str(t))
                        
        else:
            if time.time()-start_time > max_run_time:
                return removeIntersections(overall_best_route)
            else:
                t=t*reheat_coef  # Cooling
                print("t: " + str(t))
                
def sim_anneal2(route, points):
    """
    A metropolis heuristic with simulated annealing
    A two opt local search where sometimes non optimal solutions are excepted
    (in the aim of escaping local minima)
    The simulated annealing sets this probability according to the distance between
    the new found distance and the best distance, as well as a temperature that we 
    dynamically change during the search (from random to hill climbing)
    Takes random edges each time for two opt
    """
    num_points = points.shape[0]
    start_time = time.time()
    t=5 #  Temperature
    gradient = 1.01  # Cooling schedule
    max_run_time = 60  # Seconds
    is_valid_solution, best_distance = check_tsp_solution( route, points )
    overall_best_distance = best_distance
    overall_best_route = route
    while True:
        i = random.randint(0, num_points-1)
        k = random.randint(0,num_points-1)
        new_route = twoOptSwap(route,i,k)
        is_valid_solution, new_distance = check_tsp_solution(new_route, points)
        if new_distance < best_distance:
            route = new_route
            best_distance = new_distance
            if best_distance < overall_best_distance:
                overall_best_distance = best_distance
                overall_best_route = route
                global best_seed 
                best_seed = seed
                print('found new best ' + str(overall_best_distance))
                start_time = time.time()
        elif new_distance==best_distance:
            pass
        else:
            if time.time()-start_time > max_run_time:
                return removeIntersections(overall_best_route,points)
            # Select bad solutions with a probability threshold
            threshold = math.exp(-(new_distance-best_distance)/t)
            rand_num = random.uniform(0,1)
            if rand_num < threshold:
                route = new_route
                t=t/gradient  # Cooling
                print("t: " + str(t))
            else:
                pass
    
def get_best_nn(points):
    num_points = points.shape[0]
    best_route = None
    best_distance = float('inf')
    for i in xrange(num_points):
        route = nearest_neighbors(points,i)
        distance = check_tsp_solution(route, points)[1]
        if distance < best_distance:
            best_distance = distance
            best_route = route
    return removeIntersections(best_route, points)
        


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        points = read_data(file_location)

        solution = nearest_neighbors(points)
        solution = removeIntersections(solution, points)
        solution, seed = sim_anneal1(solution,points)

        solution_value = check_tsp_solution(solution, points)[1]
        plot_tsp_solution(solution, points)

        print solution_value
        print ' '.join(map(str, solution))
    else:
        print 'This script requires an input file as command line argument.'
