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
from game import Directions
import random, util

from game import Agent, Actions

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        # print "succeesorGameState:", successorGameState
        # print "newPos" , newPos
        # print "newFood" ,newFood
        # print "new Ghost states:" , newGhostStates
        # print "new Scared TIme" , newScaredTimes
        fearDistance = 3
        ghostDistance = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        maxGhostDistance = max(ghostDistance)
        ghostScore = min(maxGhostDistance, fearDistance)
        foodList = newFood.asList()
        foodDistance = [util.manhattanDistance(newPos, food) for food in foodList]
        if (len(foodDistance) == 0 or ghostScore < fearDistance):
          minFoodDistance = 0
        else:
          minFoodDistance = min(foodDistance)

        if ( successorGameState.getNumFood() < currentGameState.getNumFood() and ghostScore == fearDistance):
          minFoodDistance = -1000

        return successorGameState.getScore() + ghostScore - minFoodDistance

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
        """
        "*** YOUR CODE HERE ***"
      
        return self.value(gameState, 0, self.depth + 1)[1]


    def value(self, gameState, currentAgentIndex, depth):
      if currentAgentIndex == 0:
        depth -= 1
      if ( depth <= 0 or len(gameState.getLegalActions(0)) == 0 ):
        return (self.evaluationFunction(gameState), None)

      if currentAgentIndex == 0:
        return self.maxValue(gameState, currentAgentIndex, depth)
      return self.minValue(gameState, currentAgentIndex, depth)

    def maxValue(self, gameState, currentAgentIndex, depth):
      v = None
      a = None
      for action in gameState.getLegalActions(currentAgentIndex):
        successor = gameState.generateSuccessor(currentAgentIndex, action)
        nextValue = self.value(successor, (currentAgentIndex + 1) % gameState.getNumAgents(), depth)[0]
        if v is None or v < nextValue:
          v = nextValue
          a = action
      return (v, a)

    def minValue(self, gameState, currentAgentIndex, depth):
      v = None
      a = None
      for action in gameState.getLegalActions(currentAgentIndex):
        successor = gameState.generateSuccessor(currentAgentIndex, action)
        nextValue = self.value(successor, (currentAgentIndex + 1) % gameState.getNumAgents(), depth)[0]
        if v is None or v > nextValue:
          v = nextValue
          a = action
      return (v, a)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.value(gameState, 0, self.depth + 1, None, None)[1]


    def value(self, gameState, currentAgentIndex, depth, alpha, beta):
      if currentAgentIndex == 0:
        depth -= 1
      if depth <= 0 or len(gameState.getLegalActions(0)) == 0:
        return (self.evaluationFunction(gameState), None)
      if currentAgentIndex == 0:
        return self.maxValue(gameState, currentAgentIndex, depth, alpha, beta)
      return self.minValue(gameState, currentAgentIndex, depth, alpha, beta)

    def maxValue(self, gameState, currentAgentIndex, depth, alpha, beta):
      v = None
      a = None
      for action in gameState.getLegalActions(currentAgentIndex):
        successor = gameState.generateSuccessor(currentAgentIndex, action)
        nextValue = self.value(successor, (currentAgentIndex + 1) % gameState.getNumAgents(), depth, alpha, beta)[0]
        if v is None or v < nextValue:
          v = nextValue
          a = action
        if beta != None and v > beta:
            return (v, a)
        if alpha is None:
            alpha = v
        else:
            alpha = max(alpha, v)
      return (v, a)

    def minValue(self, gameState, currentAgentIndex, depth, alpha, beta):
      v = None
      a = None
      for action in gameState.getLegalActions(currentAgentIndex):
        successor = gameState.generateSuccessor(currentAgentIndex, action)
        nextValue = self.value(successor, (currentAgentIndex + 1) % gameState.getNumAgents(), depth, alpha, beta)[0]
        if v is None or v > nextValue:
          v = nextValue
          a = action
        if alpha != None and v < alpha:
            return (v, a)
        if beta is None:
            beta = v
        else:
            beta = min(beta, v)
      return (v, a)

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
        return self.value(gameState, 0, self.depth + 1)[1]


    def value(self, gameState, currentAgentIndex, depth):
        if currentAgentIndex == 0:
            depth -= 1
        if ( depth <= 0 or len(gameState.getLegalActions(0)) == 0 ):
            return (self.evaluationFunction(gameState), None)
        if currentAgentIndex == 0:
            return self.maxValue(gameState, currentAgentIndex, depth)
        return self.expValue(gameState, currentAgentIndex, depth)

    def maxValue(self, gameState, currentAgentIndex, depth):
        v = None
        a = None
        for action in gameState.getLegalActions(currentAgentIndex):
          successor = gameState.generateSuccessor(currentAgentIndex, action)
          nextValue = self.value(successor, (currentAgentIndex + 1) % gameState.getNumAgents(), depth)[0]
          if v is None or v < nextValue:
            v = nextValue
            a = action
        return (v, a)

    def expValue(self, gameState, currentAgentIndex, depth):
        v = 0.0
        a = None
        legalActions = gameState.getLegalActions(currentAgentIndex)
        for action in legalActions:
            successor = gameState.generateSuccessor(currentAgentIndex, action)
            nextAgent = (currentAgentIndex + 1) % gameState.getNumAgents()
            p = 1.0 / len(legalActions)
            v += p * self.value(successor, nextAgent, depth)[0]
        return (v, a)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function uses a "fear" factor to move away
      from nearby ghosts. When out of range of the ghost, this stays the same.
      We also use the MazeDistance to the nearest piece of food to incentivize
      moving towards food.

    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    fearDistance = 3
    ghostDistance = [util.manhattanDistance(position, ghost.getPosition()) for ghost in ghostStates]
    maxGhostDistance = max(ghostDistance)
    ghostScore = min(maxGhostDistance, fearDistance)

    foodList = currentGameState.getFood().asList()
    foodDistance = [mazeDistance(position, food, currentGameState) for food in foodList]
    if (len(foodDistance) == 0 or ghostScore < fearDistance):
      minFoodDistance = 0
    else:
      minFoodDistance = min(foodDistance)

    return currentGameState.getScore() - (3 * minFoodDistance) + ghostScore


class PositionSearchProblem():
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue()
    closed_set = set()
    fringe.push((problem.getStartState(), []))
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            node[1].reverse()
            return node[1]
        if node[0] not in closed_set:
            closed_set.add(node[0])
            for child_node in problem.getSuccessors(node[0]):
                path = list()
                path.append(child_node[1])
                path = path + node[1]
                fringe.push((child_node[0], path))
    return None


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))

# Abbreviation
better = betterEvaluationFunction

