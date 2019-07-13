# myAgents.py
# -----------

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint

class MyAgentFactory(AgentFactory):

    def __init__(self, isRed):
        AgentFactory.__init__(self, isRed)
        self.agentsCreated = 0

    def getAgent(self, index):
        upperFood = True
        if self.agentsCreated % 2 == 0:
            upperFood = False
        self.agentsCreated += 1
        return MyOffensiveAgent(index, upperFood)


class MyAgent(CaptureAgent):

    def __init__(self, index, upperFood):
        CaptureAgent.__init__(self, index)
        self.gamma = 0.5
        self.alpha = 0.0001
        self.epsilon = 0.0
        self.weights = util.Counter()
        self.newWeights = util.Counter()
        self.initializeWeights()
        self.lastState = None
        self.lastAction = None
        self.upperFood = upperFood

    def getAction(self, state):
        """
        Get the action the agent should take
        """
        legalActions = state.getLegalActions(self.index)
        action = None
        if len(legalActions) == 0:
            return action

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action


    def computeActionFromQValues(self, currentState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = currentState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.getQValue(currentState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        # Generate a list of actions with the maximum value
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        action = random.choice(bestActions)
        #self.myObservationFunction(currentState)
        self.lastState = currentState
        self.lastAction = action

        return action

    def getQValue(self, state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(state, action)
        return features * self.weights

    def getValue(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        maxQValue = None
        for action in state.getLegalActions(self.index):
          qvalue = self.getQValue(state, action)
          if maxQValue is None or qvalue > maxQValue:
            maxQValue = qvalue

        return 0.0 if maxQValue is None else maxQValue

    def getWeights(self):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        util.raiseNotDefined()

    def getFeatures(self, state, action):
        """
        Returns a counter of features for the state
        """
        util.raiseNotDefined()

    def getSuccessor(self, state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    def getDistanceToFood(self, state):
        """
        Compute distance to the nearest food
        """
        foodList = self.getFood(state).asList()
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = state.getAgentState(self.index).getPosition()
            if not state.getAgentState(self.index).isPacman:
                if self.upperFood:
                    upperY = min([food[1] for food in foodList])
                    distances = [self.getMazeDistance(myPos, food) for food in foodList if food[1] == upperY]
                    return min(distances)
                else:
                    lowerY = max([food[1] for food in foodList])
                    distances = [self.getMazeDistance(myPos, food) for food in foodList if food[1] == lowerY]
                    return min(distances)

            distances = [self.getMazeDistance(myPos, food) for food in foodList]
            return min(distances)


    def getDistanceToTeammate(self, state):
        """
        Compute the distance from this agent to the closest teammate
        """
        myPos = state.getAgentState(self.index).getPosition()
        teammateLocations = [state.getAgentState(agentIndex).getPosition()
                for agentIndex in self.getTeam(state)
                if agentIndex != self.index]
        teamDistances = [self.getMazeDistance(myPos, teammate) for teammate in teammateLocations]
        return 0.0 if len(teamDistances) == 0 else min(teamDistances)

    def getOnDefense(self, state):
        """
        Return whether or not this agent is on offense. 1 for true, 0 for false
        """
        if state.getAgentState(self.index).isPacman:
            return 0.0
        else:
            return 1.0

    def getDistanceToDefender(self, state):
        """
        Return the distance to the closest opponent that is currently a ghost
        """
        myPos = state.getAgentState(self.index).getPosition()
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        defenders = [a for a in enemies if not a.isPacman and a.scaredTimer <= 0 and a.getPosition() != None]
        # scaredDefenders = [a for a in enemies if not a.isPacman and a.scaredTimer > 5 and a.getPosition() != None]
        # if len(scaredDefenders) > 0:
        #     # Figure out how close the scared defender is and have a large negative
        #     # to promote going and eating it
        #     distances = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
        #     closestScaredDistance = min(distances)
        #     if closestScaredDistance <= 1:
        #         return -closestScaredDistance * 5.0
        if len(defenders) > 0:
            # Return the distance of a ghost if it is within 5 units
            distances = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
            closestDistance = min(distances)
            if closestDistance <= 5:
                return closestDistance
        return 6.0

    def getDistanceToInvader(self, state):
        """
        Return the distance to the closest opponent that is currently a ghost
        """
        myPos = state.getAgentState(self.index).getPosition()
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if not state.getAgentState(self.index).isPacman and len(invaders) > 0:
            distances = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            return min(distances)
        else:
            return 0.0

    def getNumInvaders(self, state):
        """
        Return the number of invaders on our side
        """
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        return len(invaders)

    def getDistanceToCapsule(self, state):
        """
        Return the distance to the pellet on the other team's side
        """
        myPos = state.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(state)
        if len(capsules) > 0:
            distances = [self.getMazeDistance(myPos, capsule) for capsule in capsules]
            return min(distances)
        else:
            return 0.0

    def getInDeadEnd(self, state):
        """
        Return whether or not the current position is a dead end
        """
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 2:
            # can stop or move one direction
            return 1.0
        else:
            return 0.0

    def getNumFood(self, state):
        """
        Return the number of food we still need to eat
        """
        return len(self.getFood(state).asList())

    def getNumCapsules(self, state):
        """
        Return the number of capsules we still want to eat
        """
        return len(self.getCapsules(state))

    def getScore(self, state):
        """
        Override the getScore function to include some more Information
        """
        score = CaptureAgent.getScore(self, state)
        score -= len(self.getCapsules(state))
        return score

    def getDead(self, state):
        """
        Return 1 if the pacman agent just died, 0 Otherwise
        """
        if self.lastState != None:
            myPos = state.getAgentState(self.index).getPosition()
            previousPos = self.lastState.getAgentState(self.index).getPosition()
            if util.manhattanDistance(myPos, previousPos) > 1:
                # We died
                return 1
        return 0

    def update(self, state, action, nextState, reward):
        """
        Should update weights based on transition
        """
        difference = reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action)
        features = self.getFeatures(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]
        print "Difference " + str(difference)
        print(self.weights)

    def myObservationFunction(self, state):
        """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward =  self.getScore(state) - self.getScore(self.lastState)
            print "Reward:" + str(reward)
            self.update(self.lastState, self.lastAction, state, reward)
        return state

class MyOffensiveAgent(MyAgent):

    def initializeWeights(self):
        self.weights['successorScore']     = 100.0
        self.weights['distanceToFood']     = -1.0
        self.weights['distanceToTeammate'] = 0.7
        self.weights['distanceToDefender'] = 3.0
        self.weights['distanceToInvader']  = -1.0
        #self.weights['numFood']            = -100.0
        #self.weights['numCapsules']        = -100.0
        self.weights['stop']               = -100.0
        #self.weights['dead']               = -100.0
        #self.weights['distanceToCapsule']  = -3.0
        #self.weights['deadEnd']            = -0.0
        # self.weights['reverse']            = -2.0

    def getFeatures(self, state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(state, action)
        if action == Directions.STOP: features['stop'] = 1
        # rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
        # if action == rev: features['reverse'] = 1
        features['successorScore']     = self.getScore(successor)
        features['distanceToFood']     = self.getDistanceToFood(successor)
        features['distanceToTeammate'] = self.getDistanceToTeammate(successor)
        features['distanceToDefender'] = self.getDistanceToDefender(successor)
        features['distanceToInvader']  = self.getDistanceToInvader(successor)
        features['distanceToCapsule']  = self.getDistanceToCapsule(successor)
        features['deadEnd']            = self.getInDeadEnd(successor)
        features['numFood']            = self.getNumFood(successor)
        features['numCapsules']        = self.getNumCapsules(successor)
        features['dead']               = self.getDead(successor)
        return features


class MyDefensiveAgent(MyAgent):

    def getWeights(self):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        weights = util.Counter()
        weights['stop']               = -100.0
        weights['onDefense']          = 100.0
        weights['distanceToInvader']  = -10.0
        weights['numInvaders']        = -1000.0
        weights['reverse']            = -2.0
        return weights

    def getFeatures(self, state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(state, action)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        features['onDefense']          = self.getOnDefense(successor)
        features['distanceToInvader']  = self.getDistanceToInvader(successor)
        features['numInvaders']        = self.getNumInvaders(successor)
        return features
