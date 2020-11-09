import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        score = successor_game_state.score

        successor_legal_moves = successor_game_state.get_agent_legal_actions()
        successor_score = [successor_game_state.generate_successor(action=move).score for move in successor_legal_moves]

        if successor_score:
            return max(successor_score)

        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""

        return self._recursive_search(0, game_state)[0]


    def _recursive_search(self, current_depth, game_state):
        if current_depth == self.depth:
            return '', self.evaluation_function(game_state)

        legal_agent_actions = game_state.get_agent_legal_actions()
        agent_scores = []
        if not legal_agent_actions:
            return None, self.evaluation_function(game_state)
        for agent_action in legal_agent_actions:
            game_state1 = game_state.generate_successor(action=agent_action)
            opponent_legal_actions = game_state1.get_opponent_legal_actions()
            opponent_scores = []
            for opponent_action in opponent_legal_actions:
                game_state2 = game_state1.generate_successor(agent_index=1, action=opponent_action)
                opponent_scores.append(self._recursive_search(current_depth + 1, game_state2)[1])

            min_opponent_action = min(opponent_scores)
            agent_scores.append(min_opponent_action)

        idx = np.argmax(agent_scores)
        return legal_agent_actions[idx], agent_scores[idx]




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        return self._alpha_beta_agent(0, game_state, -np.inf, np.inf)[0]

    def _alpha_beta_agent(self, current_depth, game_state, alpha, beta):
        if current_depth == self.depth:
            return '', self.evaluation_function(game_state)

        legal_agent_actions = game_state.get_agent_legal_actions()
        max_agent_score = -np.inf
        max_agent_action = None

        if not legal_agent_actions:
            return None, self.evaluation_function(game_state)

        for agent_action in legal_agent_actions:
            game_state1 = game_state.generate_successor(action=agent_action)
            min_opponent_score = self._alpha_beta_opponent(current_depth, game_state1, alpha, beta)
            if min_opponent_score > max_agent_score:
                max_agent_score = min_opponent_score
                max_agent_action = agent_action

            if max_agent_score > beta:
                return max_agent_action, max_agent_score

            alpha = max(alpha, max_agent_score)

        return max_agent_action, max_agent_score

    def _alpha_beta_opponent(self, current_depth, game_state, alpha, beta):
        opponent_legal_actions = game_state.get_opponent_legal_actions()
        min_opponent_score = np.inf
        for opponent_action in opponent_legal_actions:
            game_state2 = game_state.generate_successor(agent_index=1, action=opponent_action)
            min_opponent_score = min(self._alpha_beta_agent(current_depth + 1, game_state2, alpha, beta)[1],
                                     min_opponent_score)

            if min_opponent_score < alpha:
                return min_opponent_score

            beta = min(beta, min_opponent_score)

        return min_opponent_score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        return self.agent0_action(game_state, self.depth)[1]
        # util.raiseNotDefined()

    def agent0_action(self, game_state, depth):
        # if depth==0:
        #     return Action.STOP
        max = float('-inf')
        max_action = None
        for action in game_state.get_legal_actions(0):
            state = game_state.generate_successor(0, action)
            score = self.agent1_action(state, depth)
            if score > max:
                max = score
                max_action = action
        return max, max_action

    def agent1_action(self, game_state, depth):
        min = float('inf')
        min_action = None
        scores =[]
        for action in game_state.get_legal_actions(1):
            state = game_state.generate_successor(1, action)
            action0 = None
            if depth > 1:
                score, action0 = self.agent0_action(state, depth - 1)
            if action0 is None:
                score = self.evaluation_function(state)

            scores.append(score)
            # min_action = action

        # scores.sort()
        # return scores[int(len(scores)/2)]
        return sum(scores)/len(scores)
        #return min



def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <I used 4 different heuristics in addition to using score. The heuristics where built from the results
    of 2 papers:
        http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf
        https://home.cse.ust.hk/~yqsong/teaching/comp3211/projects/2017Fall/G11.pdf
        >
    """
    "*** YOUR CODE HERE ***"
    score = current_game_state.score
    corner = corner_heuristic(current_game_state)
    monotonicity = monotonicity_heuristic(current_game_state)
    adjacent = adjacent_heuristic(current_game_state)
    space = space_heuristic(current_game_state)
    weights = np.array([1000, 100, 1, 100, 50])
    values = np.array([score, corner, monotonicity, adjacent, space])
    return np.sum(np.multiply(weights, values))


def corner_heuristic(current_game_state):
    weights = np.array([[64, 32, 16, 8],
                        [32, 16, 8, 4],
                        [16, 8, 4, 2],
                        [8, 4, 2, 1]])
    return np.sum(np.multiply(weights, current_game_state.board))


def monotonicity_heuristic(current_game_state):
    weights = np.array([[32768, 16384, 8192, 4096],
                        [2048, 1024, 512, 256],
                        [128, 64, 32, 16],
                        [8, 4, 2, 1]])
    return np.sum(np.multiply(weights, current_game_state.board))


def adjacent_heuristic(current_game_state):
    board = current_game_state.board
    shape_of_board = board.shape
    penalty = 0
    for i in range(shape_of_board[0]):
        for j in range(shape_of_board[1]):
            if i != 0 and board[i - 1, j] != board[i, j]:
                penalty += 1
            if i != shape_of_board[0] - 1 and board[i + 1, j] != board[i, j]:
                penalty += 1
            if j != 0 and board[i, j - 1] != board[i, j]:
                penalty += 1
            if j != shape_of_board[1] - 1 and board[i, j + 1] != board[i, j]:
                penalty += 1
    return -1 * penalty


def space_heuristic(current_game_state):
    return -1 * len(current_game_state.get_empty_tiles())


# Abbreviation
better = better_evaluation_function
