# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions, GameStateData
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
    
        lowest_eval = -10000000000000
        #if pacman has to stop that means that action takes tha lowest evaluation value (-infint)
        if action == 'Stop':
            return lowest_eval
        #if a ghost is in the same postiiton as pacman and pacman has not an active capsule it menas game is lost and so this action takes the lowest evaluation value
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0 and manhattanDistance(ghost.getPosition(), newPos) <2:
                return lowest_eval
        
        #get ghost positions
        closest_ghost = 100000000000000000
        for ghost in newGhostStates:
            closest_ghost = min(closest_ghost, manhattanDistance(newPos,ghost.getPosition()))

        #find closest food distance from pacman next postion
        min_food_distance = 10000000000000 #starting with a very big value in order to be sure that a real value of distance will be choosen
        for food in newFood.asList():
            min_food_distance = min(min_food_distance, manhattanDistance(food,newPos))

        #closer ghost, further food = lower evaluation value / furhter ghost, closer food = higher evaluation value
        return successorGameState.getScore() + closest_ghost / min_food_distance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #returns pacman best possible action 
        return self.max_value(gameState, 0, 0)[0]

    
    def minmax_value(self, gameState, agentIndex, currDepth):

            """minmax_value = UTILITY -> if final state
                            = max(minmax_value) -> if is max node(agent) = pacman
                            = min(minmax_value) -> if is a min node(agent) = ghost"""

            if currDepth == self.depth * gameState.getNumAgents()  or gameState.isLose() or gameState.isWin(): #final state
                return "",self.evaluationFunction(gameState)

            if agentIndex == 0: #pacman = max agent, call the funstion that returns the action that gives the possible maximmum value
                return self.max_value(gameState, agentIndex, currDepth)
            else: #ghost = min agent, #pacman = max agent, call the funstion that returns the action that gives the possible maximmum value
                return self.min_value(gameState, agentIndex, currDepth)
    
    
    def max_value(self, gameState, agentIndex, currDepth): #returns the maximum possible value an agent can get from the state
        MAX = ("action",-100000000000000000000)
        successorIndex = agentIndex
        for action in gameState.getLegalActions(agentIndex):#for each possible action get its value
            successorIndex = agentIndex + 1
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
            succ_eval= self.minmax_value(gameState.generateSuccessor(agentIndex, action), successorIndex, currDepth+1)[1]           
            successor = (action,succ_eval)
            MAX = max(MAX, successor,key=lambda x:x[1]) #keep the maximun value (using lamba function which take an argument x and return the second elemnt of the tuple to be used later as key)
        return MAX 
   
            
        
    def min_value(self, gameState, agentIndex, currDepth): #returns the minimum possible value an agent can get from the state
        MIN = ("action",1000000000000000)
        for action in gameState.getLegalActions(agentIndex): #for each possible action get the corresponding value
            successorIndex = agentIndex + 1
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
            succ_eval = self.minmax_value(gameState.generateSuccessor(agentIndex, action), successorIndex, currDepth+1)[1]
            successor =(action,succ_eval)
            MIN  = min(MIN, successor, key=lambda x:x[1]) #keep the minimum value
        return MIN
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """ The idea is similar to min max algorithm with some changes on min(max)_value functions as shown in project presenation"""
        #returns pacman best possible action 
        alpha = -1000000000000
        beta = 10000000000
        return self.max_value(gameState, 0, 0, alpha , beta)[0]

    
    def alphabeta_value(self, gameState, agentIndex, currDepth, alpha, beta):

            """alphabeta_value = UTILITY -> if final state
                            = max(alphabeta_value) -> if is max node(agent) = pacman
                            = min(alphabeta_value) -> if is a min node(agent) = ghost"""

            if currDepth == self.depth * gameState.getNumAgents()  or gameState.isLose() or gameState.isWin(): #final state
                return "", self.evaluationFunction(gameState)

            if agentIndex == 0: #pacman = max agent, call the funstion that returns the action that gives the possible maximmum value
                return self.max_value(gameState, agentIndex, currDepth, alpha, beta)
            else: #ghost = min agent, #pacman = max agent, call the funstion that returns the action that gives the possible maximmum value
                return self.min_value(gameState, agentIndex, currDepth, alpha, beta)
    
    
    def max_value(self, gameState, agentIndex, currDepth, alpha, beta): #returns the maximum possible value an agent can get from the state
        MAX = ("action",-100000000000000000000)
        successorIndex = agentIndex
        for action in gameState.getLegalActions(agentIndex):#for each possible action get its value
            
            successorIndex = agentIndex + 1
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
            successor_val = self.alphabeta_value(gameState.generateSuccessor(agentIndex, action), successorIndex, currDepth+1, alpha , beta)[1]          
            successor = (action,successor_val)
            MAX = max(MAX, successor,key=lambda x:x[1]) #keep the maximun value
            
            # Prunning procedure
            if MAX[1] > beta: return MAX
            else: alpha = max(alpha,MAX[1])
        
        return MAX 
   
            
        
    def min_value(self, gameState, agentIndex, currDepth, alpha, beta): #returns the minimum possible value an agent can get from the state
        MIN = ("action",1000000000000000)
        for action in gameState.getLegalActions(agentIndex): #for each possible action get the corresponding value
            
            successorIndex = agentIndex + 1
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
            
            succ_val = self.alphabeta_value(gameState.generateSuccessor(agentIndex, action), successorIndex, currDepth+1, alpha, beta)[1]
            successor = (action,succ_val)
            MIN  = min(MIN, successor, key=lambda x:x[1]) #keep the minimum value
            
            # Prunning procedure
            if MIN[1] < alpha: return MIN
            beta = min(beta,MIN[1])

        return MIN
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        """expectminmax = UTILITY -> if final state
                            = max(expectminamax) -> if is max node(agent) = pacman
                            = min(expectminamax) -> if is a min node(agent) = ghost
                            = Σ Propability * expectminmax-> if is a "chance" node """
        """since we have to deal with ghosts that make random moves we will have only 3 of the four mentioned functions UTILITY, max_value function and expected_value function
        (basically mmin_value function is replaced by expected_value"""

        return self.expectminmax(gameState, 0 , 0)[0]

    def expectminmax(self, gameState, agentIndex, currDepth):
        
        # Final states:
        if len(gameState.getLegalActions(agentIndex)) == 0 or currDepth == self.depth or gameState.isLose() or gameState.isWin():
            return ("", self.evaluationFunction(gameState))

        if agentIndex == 0: #Pacman
            return self.max_value(gameState, agentIndex, currDepth)
        else: #Ghost
            return self.expected_value(gameState, agentIndex, currDepth)

    def max_value(self, gameState, agentIndex, currDepth):
        MAX = ("",-10000000000000000000000000)

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = currDepth

            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.expectminmax(successor, successor_index, successor_depth)[1]
            current = (action,current_value)
            MAX = max(MAX,current, key=lambda x:x[1])#keep the maximum value

        return MAX

    def expected_value(self, gameState, agentIndex, currDepth):
        expected_value = 0
        expected_action = ""
        current_value = 0
        EXPETC = ("",0)

        successor_probability = 1.0 / len(gameState.getLegalActions(agentIndex))#since all the moves has equal propability

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = currDepth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.expectminmax(successor, successor_index, successor_depth)[1]
            expected_value += successor_probability * current_value #calculating value 
            EXPETC = (expected_action, expected_value)
        return EXPETC


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # "*** YOUR CODE HERE ***"
    
    #extract basic information from current game state
    pacmanPos = currentGameState.getPacmanPosition() #pacman current position
    ghostStates = currentGameState.getGhostStates()
    ghostPos = currentGameState.getGhostPositions() #ghosts current positions
    food = currentGameState.getFood().asList() #list of food positions
    capsules = currentGameState.getCapsules() #list of capsules positions
    score = currentGameState.getScore() 

    #get ghost positions
    closest_ghost = 100000000000000000
    for ghost in ghostPos:
        closest_ghost = min(closest_ghost, manhattanDistance(pacmanPos,ghost))

    #find closest food distance from pacman next postion
    closest_food = 10000000000000 #starting with a very big value in order to be sure that a real value of distance will be choosen
    for food in food:
        closest_food = min(closest_food, manhattanDistance(food,pacmanPos))

    #evaluation
        """ The closer the food,the lesser the amount of food and capsules, the greater the distance between pacman and closest ghost,
                and the higher the current score --> the better (higher evaluation value) """
    
    #if a ghost is in the same postiiton as pacman and pacman has not an active capsule it menas game is lost and so this action takes the lowest evaluation value
    for ghost in ghostStates:
        if ghost.scaredTimer == 0 and manhattanDistance(ghost.getPosition(), pacmanPos) <2:
            return -100000000

    food_distance_eval = (1.0/closest_food) * 10
    food_eval = len(food) * -100
    capsules_eval = len(capsules) * -10
    score_eval = score * 1000

    evaluation = food_distance_eval + food_eval + capsules_eval+score_eval 

    return evaluation   


# Abbreviation
better = betterEvaluationFunction
