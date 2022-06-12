# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def get_action_sequence(parent, curr_node):
    action_sequence = []
    while True:
        parent_val = parent[curr_node]
        if parent_val is None: 
            action_sequence.reverse()
            return action_sequence
        curr_node, action = parent_val
        action_sequence.append(action)

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class VisitStatus:
    REMAINING = 'remaining'
    VISITED = 'visited'
    IN_FRONTIER = 'in_frontier'

class Node(object):
    def __init__(self, state, parent=None, action=None, path_cost=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

def bestFirstSearch(problem, eval_func):
    initial_state = problem.getStartState()
    node = Node(initial_state)
    frontier = util.PriorityQueue()
    reached = {}
    reached[initial_state] = node
    

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    visit_status = {}
    parent = {}
    start_state = problem.getStartState()
    search_datastruct = util.Stack()
    search_datastruct.push(start_state)
    visit_status[start_state] = VisitStatus.IN_FRONTIER
    parent[start_state] = None
    action = None

    while not search_datastruct.isEmpty():
        curr_node = search_datastruct.pop()
        if problem.isGoalState(curr_node): 
            return get_action_sequence(parent, curr_node)
        successors = problem.getSuccessors(curr_node)
        for successor_node, action, cost in successors:
            if successor_node not in visit_status:
                search_datastruct.push(successor_node)
                visit_status[successor_node] = VisitStatus.IN_FRONTIER
                parent[successor_node] = (curr_node, action)
        visit_status[curr_node] = VisitStatus.VISITED
    raise ValueError("Invalid board config. No solution found by dfs.")

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = set({})
    parent = {}
    start_state = problem.getStartState()
    search_datastruct = util.Queue()
    search_datastruct.push(start_state)
    visited.add(start_state)
    parent[start_state] = None
    action = None

    while not search_datastruct.isEmpty():
        curr_node = search_datastruct.pop()
        if problem.isGoalState(curr_node): 
            return get_action_sequence(parent, curr_node)
        successors = problem.getSuccessors(curr_node)
        for successor_node, action, cost in successors:
            if successor_node not in visited:
                search_datastruct.push(successor_node)
                visited.add(successor_node)
                parent[successor_node] = (curr_node, action)

    raise ValueError("Invalid board config. No solution found by bfs.")

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = set({})
    parent = {}
    path_cost = {}
    start_state = problem.getStartState()
    search_datastruct = util.PriorityQueue()
    search_datastruct.push(start_state, 0)
    visited.add(start_state)
    parent[start_state] = None
    path_cost[start_state] = 0
    action = None

    while not search_datastruct.isEmpty():
        curr_node = search_datastruct.pop()
        if problem.isGoalState(curr_node): 
            return get_action_sequence(parent, curr_node)
        successors = problem.getSuccessors(curr_node)
        for successor_node, action, cost in successors:
            if successor_node not in visited or path_cost[curr_node] + cost < path_cost[successor_node]:
                search_datastruct.push(successor_node, path_cost[curr_node] + cost)
                visited.add(successor_node)
                parent[successor_node] = (curr_node, action)
                path_cost[successor_node] = path_cost[curr_node] + cost

    raise ValueError("Invalid board config. No solution found by bfs.")

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
