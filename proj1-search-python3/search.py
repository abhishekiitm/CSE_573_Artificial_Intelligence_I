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
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

# eval_func should return +1 for BFS, -1 for DFS and edge weight for UCS

class SearchType:
    BFS = "bfs"
    DFS = "dfs"
    UCS = "ucs"

class Search(object):
    def __init__(self, search_type):
        self.search_type = search_type

    # def is_ancestor(self, node, reached):
        
    #     while node.parent:

        
    def eval_path_cost(self, action_cost):
        if self.search_type == SearchType.BFS:
            return 1
        elif self.search_type == SearchType.DFS:
            return -1
        elif self.search_type == SearchType.UCS:
            return action_cost
    
    def expand(self, problem, node):
        children = []
        s = node.state
        successors = problem.getSuccessors(s)
        ancestors = set()

        expand_node = node
        if self.search_type == SearchType.DFS:
            while node:
                ancestors.add(node.state)
                node = node.parent

        for successor_state, action, cost in successors:
            path_cost = expand_node.path_cost + self.eval_path_cost(cost)
            if successor_state not in ancestors:
                children.append(Node(successor_state, parent=expand_node, action=action, path_cost=path_cost))
        return children

    def bestFirstSearch(self, problem):
        initial_state = problem.getStartState()
        node = Node(initial_state)
        frontier = util.PriorityQueue()
        frontier.push(node, node.path_cost)
        reached = {}
        reached[initial_state] = node

        while not frontier.isEmpty():
            node = frontier.pop()
            if problem.isGoalState(node.state): return node
            for child in self.expand(problem, node):
                s = child.state
                if s not in reached:
                    reached[s] = child
                    frontier.push(child, child.path_cost)
                elif child.path_cost < reached[s].path_cost:
                    frontier.update(child, child.path_cost)


        raise ValueError(f"Invalid board config. No solution found by {self.search_type}.")

def action_sequence(node):
    seq = []
    while node.parent:
        seq.append(node.action)
        node = node.parent
    seq.reverse()
    return seq

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
    dfs_search = Search(SearchType.DFS)
    node = dfs_search.bestFirstSearch(problem)
    seq = action_sequence(node)
    return seq


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    bfs_search = Search(SearchType.BFS)
    node = bfs_search.bestFirstSearch(problem)
    seq = action_sequence(node)
    return seq

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    ucs_search = Search(SearchType.UCS)
    node = ucs_search.bestFirstSearch(problem)
    seq = action_sequence(node)
    return seq

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
