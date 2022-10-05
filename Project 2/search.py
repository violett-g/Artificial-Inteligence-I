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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

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
    print("problem", problem)
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    queue = Stack() #stack to keep the nodes(positions) of the maze
    #use a LIFO implementation in order to explore first the deppest node in the current fringe which was added last in the stack
    visited_nodes = {} #dictionary to mark visited nodes, (dictionary not list in order to be accesed by key not index)
    node = problem.getStartState() #pacman first position
    cost = 0 #total cost till the current position of pacman
    path = [] #a list of the action required to reach the current node form the starting position
    queue.push((node,path))

    while not queue.isEmpty(): #loop while tha stack is not empty
        node = queue.pop() #get the first item from the stack 
        if node[0] not in visited_nodes: #else check if the postion of the item removed is visited before or not
            visited_nodes[node[0]] = True #mark as visited
            if problem.isGoalState(node[0]): #check if the poped item is the "goal" position
                return node[1] #return the list of actions required to reach that position
            for item in problem.getSuccessors(node[0]): #get its succesors
                successor = item[0]
                succ_path = item[1]
                if successor and successor not in visited_nodes: #check id succesors are visited before
                    queue.push((successor,node[1]+[succ_path])) #if not push to the stack  
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    queue = Queue() #keep the accesible nodes(position of the maze)
    #using FIFO in order to explore first the shalowest node in the current fringe which was added first in the queue
    #visited = {} #dictionary to mark visited nodes(positions)
    visited = []
    current_node = problem.getStartState() #pacman first position
    #print("starting state from bfs", current_node)
    cost = 0 #total cost till the current position of pacman
    path = []  #a list of the action required to reach the current node form the starting position
    queue.push((current_node,path,cost))

    while not queue.isEmpty():
        current_node = queue.pop()
        if current_node[0] not in visited: #if not check if visited
            visited.append(current_node[0])
            if problem.isGoalState(current_node[0]): #check if current node is the goal position
                return current_node[1]
            for item in problem.getSuccessors(current_node[0]):
                successor = item[0]
                succ_path = item[1]
                succ_cost = item[2]
                if successor and successor not in visited:
                    queue.push((successor,current_node[1]+[succ_path],current_node[2]+succ_cost))      
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue
    queue = PriorityQueue() #we need to pop the node with the least cost (high priority)
    visited = {}
    current_node = problem.getStartState() #pacman first position
    path = []  #a list of the action required to reach the current node form the starting position
    cost = 0
    if problem.isGoalState(current_node):
        return []
    queue.push((current_node,path), cost)

    while not queue.isEmpty():#iterate through priority queue

        current_node, path = queue.pop()#pop the lement with the higher priorityy
        if current_node not in visited:#test if it is visited before and if not push to dictionary of visited
            visited[current_node] = True
        if problem.isGoalState(current_node):#check if it is the goal state
            return path
        
        for item in problem.getSuccessors(current_node):#if it isnt the goal state find iths succesors
            successor = item[0]
            succ_path = item[1]
            if successor:
                if successor not in visited and (successor not in (state[2][0] for state in queue.heap)):#successor is visited for the first time
                    new_path = path+[succ_path]
                    priority = problem.getCostOfActions(new_path)
                    queue.push((successor,new_path),priority)#push the new item on the heap with its path and priority 
                elif successor not in visited and (successor in (state[2][0] for state in queue.heap)):#successor is revisited so it exist on heap
                    for state in queue.heap:#find the successor on the heap
                       if state[2][0] == successor:
                            old_priority = problem.getCostOfActions(state[2][1])
                            new_priority = problem.getCostOfActions(path + [succ_path])

                            if old_priority > new_priority:#check if its priority is higher than the old one
                                new_path = path+ [succ_path]
                                queue.update((successor,new_path),new_priority)#update the priority and the path
                                # using the given update function when we search for a state , even if the state exist with different priority and path of actions
                                # wont be found. This cause we are searching for a tulpe (state, path of actions) where state is the same but path of actions is different.
                                # In this way update will add the element as new and in the second iteration will find it and the priority will be the same 
                                # and nothing will change.
                                #  In this problem it works but in general a good aproach would be to search the element using only the state
                                # (not a tuple with both state and path of actions) and always delete the old same state that has a lower priority.
                                
    

            
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def update(queue,item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(queue.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del queue.heap[index]
                queue.heap.append((priority, c, item))
                import heapq
                heapq.heapify(queue.heap)
                break

    #aStarSearch is similar to UCS but in this case the cost is a function f(n)=g(n)+h(n) 
    # where g(n) is the cost from start to current node and h(n) from current node to goal. 
    from util import PriorityQueue
    queue = PriorityQueue() #we need to pop the node with the least cost (high priority)
    #visited = {}
    visited = []
    current_node = problem.getStartState() #pacman first position
    path = []  #a list of the action required to reach the current node form the starting position
    cost = 0
    if problem.isGoalState(current_node):
        return []
    queue.push((current_node,path), cost+heuristic(current_node,problem))

    while not queue.isEmpty():#iterate through priority queue

        current_node, path = queue.pop()#pop the lement with the higher priorityy
        if current_node not in visited:#test if it is visited before and if not push to dictionary of visited
            #visited[current_node] = True
            visited.append(current_node)
        if problem.isGoalState(current_node):#check if it is the goal state
            return path
        
        for item in problem.getSuccessors(current_node):#if it isnt the goal state find iths succesors
            successor = item[0]
            succ_path = item[1]
            if successor:
                if successor not in visited and (successor not in (state[2][0] for state in queue.heap)):#successor is visited for the first time
                    new_path = path+[succ_path]
                    priority = problem.getCostOfActions(new_path)+heuristic(successor,problem)
                    queue.push((successor,new_path),priority)#push the new item on the heap with its path and priority 
                elif successor not in visited and (successor in (state[2][0] for state in queue.heap)):#successor is revisited so it exist on heap
                    for state in queue.heap:#find the successor on the heap
                       if state[2][0] == successor:
                            old_priority = problem.getCostOfActions(state[2][1])+heuristic(state[2][0],problem)
                            new_priority = problem.getCostOfActions(path + [succ_path])+heuristic(successor,problem)

                            if old_priority > new_priority:#check if its priority is higher than the old one
                                new_path = path+ [succ_path]
                                queue.update((successor,new_path),new_priority)#update the priority and the path
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
