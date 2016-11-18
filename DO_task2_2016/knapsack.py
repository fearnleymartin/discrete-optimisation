
import numpy as np
import time
from memoize import memoized
from collections import deque
import copy
import Queue
import sys

#########################
# DATA I/O
#########################

def read_numbers(data_file):
    input_data_file = open(data_file, 'r')
    input_data = input_data_file.readlines()
    input_data_file.close()

    numbers = np.array([])
    for i_line in xrange(len(input_data)):
        entries = input_data[i_line].split()
        entries = filter(None, entries) # remove empty entries
        line_numbers = [ int(x) for x in entries ]
        numbers = np.append(numbers, line_numbers)
    return numbers


def read_data(data_file):
    numbers = read_numbers(data_file)
    cur_entry = 0

    # number of nodes
    num_items = int(numbers[cur_entry])
    cur_entry += 1
    
    # maximum capacity of the knapsack
    capacity = int(numbers[cur_entry])
    cur_entry += 1
    
    # get data on the items
    value = np.zeros(num_items, dtype='int')
    size = np.zeros(num_items, dtype='int')
    for i_item in xrange(num_items):
        value[i_item] = int(numbers[cur_entry])
        cur_entry += 1
        size[i_item] = int(numbers[cur_entry])
        cur_entry += 1
        
    return value, size, capacity

##########################
# KNAPSACK ALGORITHMS
##########################

def knapsack(value, size, capacity):
    """
    Dynamic programming - Bottom up approach
    \min  \sum_{i=1}^n value[i] x_i
    s.t.  \sum_{i=1}^n size[i] x_i <= capacity
      x_i \in \{ 0, 1 \}
    :param value: array of ints where the item i corresponds to the value of item i
    :param size: array of ints where the item i corresponds to the size of item i
    :param capacity: constraint such that sum(xi*value[i])<capacity, xi \in {0,1}
    :return solution_value, solution_items: tuple with the maximum possible value
    and an array of the items to take where 1 means take the item
    """
    num_items = len(value)
    solution_items = np.zeros(num_items, 'int')
    
    m = np.zeros((num_items+1, capacity+1), 'int')
    
    for i in range(1, num_items+1):  # i means up to first i items
        for j in range(capacity+1):
            if size[i-1] > j:
                m[i, j] = m[i-1, j]
            else:
                m[i, j] = max(m[i-1, j], m[i-1, j-size[i-1]] + value[i-1])
    
    # get items used     
    i = num_items
    j = capacity
    while i > 0:
        if m[i, j] != m[i-1, j]:  # this means item i-1 was used
            solution_items[i-1] = 1
            j -= size[i-1]
            i -= 1
        else:
            solution_items[i-1] = 0
            i -= 1
            
    solution_value = m[num_items,capacity]
    return solution_value, solution_items


def knapsack_mem(value, size, capacity):
    """
    Dynamic programming - Top down approach with memoisation, only calculates the needed values
    \min  \sum_{i=1}^n value[i] x_i
    s.t.  \sum_{i=1}^n size[i] x_i <= capacity
      x_i \in \{ 0, 1 \}
    :param value: array of ints where the item i corresponds to the value of item i
    :param size: array of ints where the item i corresponds to the size of item i
    :param capacity: constraint such that sum(xi*value[i])<capacity, xi \in {0,1}
    :return solution_value, solution_items: tuple with the maximum possible value
    and an array of the items to take where 1 means take the item
    """
    num_items = len(value)
    solution_items = np.zeros(num_items, 'int')
    
    @memoized
    def m(i, j):
        if i == 0:
            return 0
        if size[i-1] > j:
            return m(i-1, j)
        else:
            return max(m(i-1, j), m(i-1, j-size[i-1]) + value[i-1])
            
    # get items used     
    i = num_items
    j = capacity
    while i > 0:
        if m(i, j) != m(i-1, j):  # this means item i-1 was used
            solution_items[i-1] = 1
            j -= size[i-1]
            i -= 1
        else:
            solution_items[i-1] = 0
            i -= 1
    print(m.cache)
    solution_value = m(num_items, capacity)
    
    return solution_value, solution_items


class Node:
    """
    A state representation node for the branch and bound method of solving the knapsack problem
    """
    def __init__(self, value, size, upper_bound, level, fixed_variables):
        self.value = value  # value of current state
        self.size = size  # size of current state
        self.upper_bound = upper_bound  # upper bound of current state
        self.level = level  # The distance from the root node of the tree
        self.fixed_variables = fixed_variables  # The items fixed for this particular state
        
    def __repr__(self):
        return ('value: ' + str(self.value) +
         ', size: ' + str(self.size) +
         ', upper bound: ' + str(self.upper_bound) +
         ', level: ' + str(self.level) +
         ', fixed variables: ' + str(self.fixed_variables))

    def __cmp__(self, other):
        return -cmp(self.upper_bound, other.upper_bound)  # the priority is given to the larger item


def upper_bound(node, value, size, capacity, items):
    """
    solves the continuous knapsack problem (where xi \in [0,1])
    i.e. you take take part of an item
    :param node: the node/state for which the upper bound should be calculated
    :param value: array of ints where the item i corresponds to the value of item i
    :param size: array of ints where the item i corresponds to the size of item i
    :param capacity: constraint such that sum(xi*value[i])<capacity, xi \in [0,1]
    :param items: items sorted by weight to size ratio
    :return: optimal solution to the problem and the amount of each item taken
    """

    selected_items = list(node.fixed_variables)
    selected_value = int(node.value)
    selected_size = int(node.size)

    is_integer_solution = False  # is the upper bound an integer solution ?
    extra_items = 0  # the amount of extra items that can be fitted into the knapsack
    
    for item in items[node.level:]:  # take the items one by one amongst items that haven't already been \
                                    # fixed the the current state
        extra_items += 1
        if size[item]+selected_size <= capacity:  # fill up the knapsack as much as possible
            selected_items.append(1)
            selected_value += value[item]
            selected_size += size[item]
        else:
            is_integer_solution = False
            remaining_size = capacity - selected_size
            amount_of_item_to_take = float(remaining_size)/float(size[item])  # take a fraction of the item \
            # to fill up remaining space
            selected_items.append(amount_of_item_to_take)
            selected_size += int(amount_of_item_to_take * size[item])
            selected_value += amount_of_item_to_take * value[item]
            break  # can no longer take any more items

    return int(selected_value), is_integer_solution, extra_items  # selected value represents upper bound
    # can round down to int


def knapsack_bb1(value, size, capacity):
    """
    Branch and bound algorithm for knapsack problem, depth first search
    \min  \sum_{i=1}^n value[i] x_i
    s.t.  \sum_{i=1}^n size[i] x_i <= capacity
      x_i \in \{ 0, 1 \}
    :param value: array of ints where the item i corresponds to the value of item i
    :param size: array of ints where the item i corresponds to the size of item i
    :param capacity: constraint such that sum(xi*value[i])<capacity, xi \in {0,1}
    :return tuple with the maximum possible value
    and an array of the items to take where 1 means take the item
    """
    num_items = len(value)
    
    # sort items by value to weight ratio
    items = [i for i in range(num_items)]
    items = sorted(items, key=lambda k: float(value[k])/float(size[k]), reverse=True)
    
    initial_node = Node(0, 0, 0, 0, [])
    initial_node.upper_bound = upper_bound(initial_node, value, size, capacity, items)[0]
    
    curr_best_node = Node(0, 0, 0, 0, [])
    
    Q = deque()
    Q.append(initial_node)
    count = 0
    while Q:
        # count += 1
        # if count % 100 == 0:
        #     print(count)
        curr_node = Q.popleft()
        # curr_node.print_n()

        if curr_node.level != num_items:  # if node  if a leaf, then stop
            next_item = items[curr_node.level]

            # we take the next item
            left_node = Node(curr_node.value + value[next_item],
                             curr_node.size + size[next_item],
                             curr_node.upper_bound,
                             curr_node.level+1,
                             list(curr_node.fixed_variables))

            if left_node.size <= capacity:
                # upper bound for left_node is the same as for curr_node
                left_node.fixed_variables.append(1)
                if left_node.value > curr_best_node.value:  # is it the best solution ?
                    curr_best_node = copy.copy(left_node)  # if yes, update the current best solution
                    print(curr_best_node)
                if left_node.upper_bound >= curr_best_node.value:
                    Q.append(left_node)

            # we don't take the next item
            right_node = Node(curr_node.value,
                              curr_node.size,
                              curr_node.upper_bound,
                              curr_node.level+1,
                              list(curr_node.fixed_variables))

            if right_node.size <= capacity:
                right_node.upper_bound, is_integer_solution, extra_items = upper_bound(right_node, value, size, capacity, items)
                right_node.fixed_variables.append(0)
                if right_node.value > curr_best_node.value:  # is it the best solution ?
                    curr_best_node = copy.copy(right_node)  # if yes, update the current best solution
                    print(curr_best_node)
                if is_integer_solution:  # have found optimal solution of branch
                    if right_node.upper_bound > curr_best_node.value:
                        best_node_of_branch = Node(right_node.upper_bound, capacity, right_node.upper_bound, num_items,
                                                   list(right_node.fixed_variables)+([1]*extra_items)) # add zeros ?
                        best_node_of_branch.fixed_variables += [0]*(num_items-len(best_node_of_branch.fixed_variables))
                        curr_best_node = copy.copy(best_node_of_branch)
                        print(curr_best_node)
                if not is_integer_solution:
                    if right_node.upper_bound >= curr_best_node.value:
                        Q.append(right_node)

    curr_best_node.fixed_variables += [0]*(num_items-len(curr_best_node.fixed_variables))

    if check_solution(curr_best_node, value, size, capacity):  # check solution is viable
        print('solution OK')
    else:
        print('solution not OK')
    # sort items back into original order
    items_to_take = sort_solution(curr_best_node, value, items)
    return curr_best_node.value, items_to_take


def knapsack_bb_pq(value, size, capacity):
    """
    Branch and bound algorithm for knapsack problem, best first search
    \min  \sum_{i=1}^n value[i] x_i
    s.t.  \sum_{i=1}^n size[i] x_i <= capacity
      x_i \in \{ 0, 1 \}
    :param value: array of ints where the item i corresponds to the value of item i
    :param size: array of ints where the item i corresponds to the size of item i
    :param capacity: constraint such that sum(xi*value[i])<capacity, xi \in {0,1}
    :return tuple with the maximum possible value
    and an array of the items to take where 1 means take the item
    """
    num_items = len(value)
    # sort items by value to weight ratio
    items = [i for i in range(num_items)]
    items = sorted(items, key=lambda k: float(value[k])/float(size[k]), reverse=True)

    initial_node = Node(0, 0, 0, 0, [])
    initial_node.upper_bound = upper_bound(initial_node, value, size, capacity, items)[0]

    curr_best_node = Node(0, 0, 0, 0, [])

    Q = Queue.PriorityQueue()
    Q.put(initial_node)
    while not Q.empty():
        curr_node = Q.get()
        if curr_node.level != num_items:  # if node  if a leaf, then stop
            next_item = items[curr_node.level]
            # we take the next item
            left_node = Node(curr_node.value + value[next_item],
                             curr_node.size + size[next_item],
                             curr_node.upper_bound,
                             curr_node.level+1,
                             list(curr_node.fixed_variables))
            if left_node.size <= capacity:
                left_node.fixed_variables.append(1)
                if left_node.value > curr_best_node.value:  # is it the best solution ?
                    curr_best_node = copy.copy(left_node)  # if yes, update the current best solution
                    print(curr_best_node)
                if left_node.upper_bound >= curr_best_node.value:
                    Q.put(left_node)

            # we don't take the next item
            right_node = Node(curr_node.value,
                              curr_node.size,
                              curr_node.upper_bound,
                              curr_node.level+1,
                              list(curr_node.fixed_variables))

            if right_node.size <= capacity:
                right_node.upper_bound, is_integer_solution, extra_items = upper_bound(right_node, value, size, capacity, items)
                right_node.fixed_variables.append(0)
                if right_node.value > curr_best_node.value:  # is it the best solution ?
                    curr_best_node = copy.copy(right_node)  # if yes, update the current best solution
                    print(curr_best_node)
                if is_integer_solution:  # have found optimal solution of branch
                    if right_node.upper_bound > curr_best_node.value:
                        best_node_of_branch = Node(right_node.upper_bound, capacity, right_node.upper_bound, num_items,
                                                   list(right_node.fixed_variables)+([1]*extra_items)) # add zeros ?
                        best_node_of_branch.fixed_variables += [0]*(num_items-len(best_node_of_branch.fixed_variables))
                        curr_best_node = copy.copy(best_node_of_branch)
                        print(curr_best_node)
                if not is_integer_solution:
                    if right_node.upper_bound >= curr_best_node.value:
                        Q.put(right_node)

    curr_best_node.fixed_variables += [0]*(num_items-len(curr_best_node.fixed_variables))

    if check_solution(curr_best_node, value, size, capacity):
        print('solution OK')
    else:
        print('solution not OK')
    items_to_take = sort_solution(curr_best_node, value, items)

    return curr_best_node.value, items_to_take


#######################
# AUXILIARY FUNCTIONS
#######################
def check_solution(node, value, size, capacity):
    """
    For checking a solution to branch and bound is correct
    :param node: the node to check
    :param value: array of ints where the item i corresponds to the value of item i
    :param size: array of ints where the item i corresponds to the size of item i
    :param capacity: constraint such that sum(xi*value[i])<capacity, xi \in {0,1}
    :return: true is the solution is valid, false otherwise
    """
    num_items = len(value)

    items = [i for i in range(num_items)]
    items = sorted(items, key=lambda k: float(value[k])/float(size[k]), reverse=True)

    total_value = 0
    total_size = 0

    for (index, xi) in enumerate(node.fixed_variables):
        total_value += xi*value[items[index]]
        total_size += xi*size[items[index]]

    if total_size <= capacity and total_value == node.value:
        return True
    else:
        return False


def sort_solution(node, values, items):
    """
    Sorts the solution items obtained by branch and bound back to original order
    :param node: node containing items to sort back to original order
    :param values: array of ints where the item i corresponds to the value of item i
    :param items: array of ints representing the items ordered by value to size ratio
    :return: array of items to take sorted in their orginal order
    """
    num_items = len(values)
    result = [0]*num_items
    for (i, xi) in enumerate(node.fixed_variables):
        result[items[i]] = xi
    return result

                
def write_solution_to_file(task_number):
    """
    :param task_number: the task to treat
    :return: creates file with solution for the chosen task
    """
    input_file = 'task2_test%d.txt' % task_number
    output_file = 'task2_test%d_solution_bb_pq.txt' % task_number
    
    value, size, capacity = read_data(input_file)    
    start_time = time.time()
    solution_value, solution_items = knapsack_bb_pq(value, size, capacity)
    execution_time = time.time()-start_time
    print(execution_time)
    f = open(output_file, 'w')
    f.write(str(solution_value))
    f.write('\n')
    for i in range(len(solution_items)):
        f.write(str(solution_items[i])+' ')
    f.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        value, size, capacity = read_data(file_location)
        if file_location == 'task2_test1.txt':
            solution_value, solution_items = knapsack_mem(value, size, capacity)
        elif file_location == 'task2_test2.txt':
            solution_value, solution_items = knapsack_mem(value, size, capacity)
        elif file_location == 'task2_test3.txt':
            solution_value, solution_items = knapsack_bb1(value, size, capacity)
        elif file_location == 'task2_test4.txt':
            solution_value, solution_items = knapsack_bb_pq(value, size, capacity)
        elif file_location == 'task2_test5.txt':
            solution_value, solution_items = knapsack_bb_pq(value, size, capacity)
        else:
            print 'Invalid file name'
        print solution_value
        print ' '.join(map(str, solution_items))
    else:
        print 'This script requires an input file as command line argument.'
