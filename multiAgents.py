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
        foodlist = newFood.asList()
        closest_food = float("inf")
        closest_ghost = float("inf")
        #Time penalty means that stopping is not a good, return the lowest number possible so Pac-Man never stops
        if(action == 'Stop'):
            return -float("inf")
        #check where the ghosts will be
        for ghoststate in newGhostStates:
            #stay at least 2 blocks away from any non-edible ghost
            if((ghoststate.scaredTimer == 0) and (manhattanDistance(newPos, ghoststate.getPosition()) < 2)):
                return -float("inf")
            #get the distance to where the closest ghost will be
            elif((manhattanDistance(newPos, ghoststate.getPosition())) < closest_ghost):
                closest_ghost = manhattanDistance(newPos, ghoststate.getPosition())
        #find the closest food pellet
        for food in foodlist:
            closest_food = min(closest_food, manhattanDistance(newPos, food))
        #return score + closest ghost/closest food so that when many food pellets are a similar distance
        #Pac-Man moves closer to the one farther away from the ghost and doesn't end up stuck in loops as often
        #If 2 scores achieved by 2 different actions are equal, the action which leads to higher distance from the closest
        #ghost and lower distance to the closest food will return a greater value
        return successorGameState.getScore() + closest_ghost/closest_food

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
    #Loop through legal actions for minimizer and choose the one that produces the lowest score
    def min_value(self, depth, index, gameState):
            lowest = float("inf")
            decision = ""
            for action in gameState.getLegalActions(index):
                #new_index = (depth + 1) % gameState.getNumAgents()
                new_state = gameState.generateSuccessor(index, action)
                new_score = self.minimax(depth, (index + 1), new_state)[0]
                if(new_score < lowest):
                    lowest = new_score
                    decision = action
            return lowest, decision
    #Loop through legal actions for maximizer and choose the one that produces the highest score
    def max_value(self, depth, index, gameState):
            highest = -float("inf")
            decision = ""
            for action in gameState.getLegalActions(index):
                #new_index = (depth + 1) % gameState.getNumAgents()
                new_state = gameState.generateSuccessor(index, action)
                new_score = self.minimax(depth, (index + 1), new_state)[0]
                if(new_score > highest):
                    highest = new_score
                    decision = action
            return highest, decision
    #increase depth once all agents have had a turn, recurring until self.depth is met or some terminal condition
    def minimax(self, depth, index, gameState):
        if(index == gameState.getNumAgents()):
            depth += 1
            index = 0
        if((depth == self.depth) or (gameState.isLose()) or (gameState.isWin())):
            return self.evaluationFunction(gameState), None
        if(index == 0):#Pacman
            return self.max_value(depth, index, gameState)
        else:#Ghost
            return self.min_value(depth, index, gameState)
        

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
        return self.minimax(0, 0, gameState)[1]
        util.raiseNotDefined()

        

        

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def beta(self, alpha, beta, depth, index, gameState):
        lowest = float("inf")
        decision = ""
        for action in gameState.getLegalActions(index):
            new_state = gameState.generateSuccessor(index, action)
            new_score = self.alpha_beta(alpha, beta, depth, (index + 1), new_state)[0]
            if(new_score < lowest):
                lowest = new_score
                decision = action
            beta = min(beta, lowest)
            if(beta < alpha):
                break
        return lowest, decision
    #Loop through legal actions for maximizer/minimizer and choose the one that causes alpha to be greater than beta
    #alpha represents the best value the maximizer can find, and beta the best value the minimizer can find
    #once alpha > beta, exploring through the rest of the possible actions is likely to yeild worse results
    def alpha(self, alpha, beta, depth, index, gameState):
        highest = -float("inf")
        decision = ""
        for action in gameState.getLegalActions(index):
            #new_index = (depth + 1) % gameState.getNumAgents()
            new_state = gameState.generateSuccessor(index, action)
            new_score = self.alpha_beta(alpha, beta, depth, (index + 1), new_state)[0]
            if(new_score > highest):
                highest = new_score
                decision = action
            alpha = max(alpha, highest)
            if(alpha > beta):
                break
        return highest, decision
        #increase depth once all agents have had a turn, recurring until self.depth is met or some terminal condition
        #minimax augmented to accept extra parameters for alpha and beta and use alpha or beta function
    def alpha_beta(self, alpha, beta, depth, index, gameState):
        if(index == gameState.getNumAgents()):
            depth += 1
            index = 0
        if((depth == self.depth) or (gameState.isLose()) or (gameState.isWin())):
            return self.evaluationFunction(gameState), None
        if(index == 0):#Pacman
            return self.alpha(alpha, beta, depth, index, gameState)
        else:#Ghost
            return self.beta(alpha, beta, depth, index, gameState)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alpha_beta(-float("inf"), float("inf"), 0, 0, gameState)[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    #Loop through legal actions for maximizer and choose the one that produces the highest score
    def max_value(self, depth, index, gameState):
        highest = -float("inf")
        decision = ""
        for action in gameState.getLegalActions(index):
            new_state = gameState.generateSuccessor(index, action)
            new_score = self.expectimax(depth, (index + 1), new_state)[0]
            if(new_score > highest):
                highest = new_score
                decision = action
        return highest, decision
    #All legal actions are equally likely, find the one that produces the highest score on average by
    #multiplying the score by the probability of the action leading to that score being taken and adding
    #the result to a total for each legal action. Max value compares the return values of each call to max_value
    #and chooses the action that leads to the highest average score.
    def exp_value(self, depth, index, gameState):
        legal_actions = gameState.getLegalActions(index)
        prob = 1.0/float(len(legal_actions))
        expected = 0
        decision = ""
        for action in legal_actions:
            new_state = gameState.generateSuccessor(index, action)
            new_score = self.expectimax(depth, (index + 1), new_state)[0]
            expected += prob * float(new_score)
        return expected, decision

    #increase depth once all agents have had a turn, recurring until self.depth is met or some terminal condition
    def expectimax(self, depth, index, gameState):
        if(index == gameState.getNumAgents()):
            depth += 1
            index = 0
        if((depth == self.depth) or (gameState.isLose()) or (gameState.isWin())):
            return self.evaluationFunction(gameState), None
        if(index == 0):#Pacman
            return self.max_value(depth, index, gameState)
        else:#Ghost
            return self.exp_value(depth, index, gameState)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(0, 0, gameState)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: gathers all relevant information such as the closest food, closest ghost, closest power pellet
      closest scared ghost, and amount of food remaining. The inverse of each in which a lower number is better
      is multiplied by some constant corresponding to how important that factor, the same is done with closest
      ghost, but with a much lower coefficient so that Pac-Man doesn't stay on the opposite side of the map for too
      long, given that when finding the closest ghost, the lowest possible number is returned if Pac-Man
      is less than 2 blocks away. the coefficient for scared ghosts is so extreme because eating a ghost
      gains a lot of points, this causes Pac-Man to wait close to a power pellet for a ghost to approach. 
      remaining food is multiplied by a negative coefficient to help keep Pac-Man on task
      since the goal is ultimately to eat all the food. The results of these operations are summed along with
      the current score and this is returned.
    """
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    foodlist = Food.asList()
    capsules = currentGameState.getCapsules()
    closest_food = float("inf")
    closest_ghost = float("inf")
    scared = float("inf")
    power = float ("inf")
    #check where the ghosts are
    for ghoststate in GhostStates:
        #stay at least 2 blocks away from any non-edible ghost
        if((ghoststate.scaredTimer == 0) and (manhattanDistance(Pos, ghoststate.getPosition()) < 3)):
            return -float("inf")
        elif(ghoststate.scaredTimer > 0):
            scared = min(scared, manhattanDistance(Pos, ghoststate.getPosition()))
        #get the distance to where the closest ghost will be
        elif((manhattanDistance(Pos, ghoststate.getPosition())) < closest_ghost):
            closest_ghost = manhattanDistance(Pos, ghoststate.getPosition())
    #find the closest food pellet
    for food in foodlist:
        closest_food = min(closest_food, manhattanDistance(Pos, food))
    for capsule in capsules:
        if not capsule in foodlist:
            power = min (power, manhattanDistance(Pos, capsule))
    return currentGameState.getScore() + (10.0/closest_ghost) + (50.0/closest_food) - (40.0 * currentGameState.getNumFood()) + (50.0/power) + (100.0/scared)
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

