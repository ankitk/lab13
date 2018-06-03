# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        states = mdp.getStates()

        for i in range(0,iterations):
            values = util.Counter()
            for state in states:
                action = self.getAction(state)
                if action is not None:
                    values[state] = self.getQValue(state, action)

            self.values = values


    def getValue(self, state):
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        qValue = 0
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for tsp in transitionStatesAndProbs:
            stateFromMDP = tsp[0]
            probability = tsp[1]
            
            value = self.getValue(stateFromMDP)
            reward = self.mdp.getReward(state, action, stateFromMDP)
            
            qValue += ((self.discount * value) + reward) * probability
            
        return qValue

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)

        qValue = self.getQValue(state, possibleActions[0])
        bestAction = possibleActions[0]

        for action in possibleActions:
            value = self.getQValue(state, action)
            if qValue <= value:
                qValue = value
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
