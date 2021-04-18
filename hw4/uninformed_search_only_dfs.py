# Name: Giang Duong
# CS 156 SPRING 2021
# Professor Sanjoy Paul
# Homework 4

import data_structures

def dfs(problem):
    """
    Depth first graph search algorithm - implemented for you
    returns a solution for the given search problem
    :param
    problem (a Problem object) representing the quest
            see Problem class definition in spartanquest.py)
    :return: list of actions representing the solution to the quest
                or None if there is no solution
    """
    closed = set()  # keep track of our explored states
    fringe = data_structures.Stack() # for dfs, the fringe is a stack
    state = problem.start_state()
    root = data_structures.Node(state, None, None)
    fringe.push(root)
    while not fringe.is_empty():
        node = fringe.pop()
        if problem.is_goal(node.state):
            return node.solution()  # we found a solution
        if node.state not in closed:  # we are implementing graph search
            closed.add(node.state)
            for child_state, action, action_cost in problem.expand(node.state):
                child_node = data_structures.Node(child_state, node, action)
                fringe.push(child_node)
    return None  # Failure -  no solution was found

def bfs(problem):
    """
    Breadth first graph search algorithm
    returns a solution for the given search problem
    :param
    problem (a Problem object) representing the quest
            see Problem class definition in spartanquest.py)
    :return: list of actions representing the solution to the quest
            or None if there is no solution
    """
    closed = set()  # keep track of our explored states
    fringe = data_structures.Queue()  # for bfs, the fringe is a queue
    state = problem.start_state()
    root = data_structures.Node(state, None, None)
    fringe.push(root)
    while not fringe.is_empty():
        node = fringe.pop()
        if problem.is_goal(node.state):
            return node.solution()  # we found a solution
        if node.state not in closed:  # we are implementing graph search
            closed.add(node.state)
            for child_state, action, action_cost in problem.expand(node.state):
                child_node = data_structures.Node(child_state, node, action)
                fringe.push(child_node)
    return None

def ucs(problem):
    """
    Uniform cost first graph search algorithm
    returns a solution for the given search problem
    :param
    problem (a Problem object) representing the quest
            see Problem class definition in spartanquest.py)
    :return: list of actions representing the solution to the quest
    """
    closed = set()  # keep track of our explored
    fringe = data_structures.PriorityQueue() # for ucs, the fringe is a Priority queue
    state = problem.start_state()
    root = data_structures.Node(state, None, None)
    fringe.push(root, root.cumulative_cost)
    while not fringe.is_empty():
        node = fringe.pop()
        if problem.is_goal(node.state):
            return node.solution()  # we found a solution
        if node.state not in closed:  # we are implementing graph search
            closed.add(node.state)
            for child_state, action, action_cost in problem.expand(node.state):
                child_node = data_structures.Node(child_state, node, action, node.cumulative_cost + action_cost)
                fringe.push(child_node, child_node.cumulative_cost)
    return None
