import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class TicTacToe:
    """Tic Tac Toe board game class."""
    def __init__(self, player1, player2, player1_turn_in = random.choice([True, False]) ):
        """Initialize game and players."""
        self.board = [' ']*9
        self.player1, self.player2 = player1, player2
        self.player1_turn = player1_turn_in

    def play_game(self):
        """Start a game and play until board is full or a player wins"""
        self.player1.start_game('X')
        self.player2.start_game('O')
        while True:
            # initialize players and characters
            if self.player1_turn:
                player, char, other_player = self.player1, 'X', self.player2
            else:
                player, char, other_player = self.player2, 'O', self.player1
            # only display board if human is playing
            if player.agent_type == "human":
                self.display_board()
            # get player move
            space = player.move(self.board)
            # end game and give rewards if illegal move
            if self.board[space] != ' ': 
                if player.agent_type == "human":
                    print("Illegal Move!")
                player.reward(-99, self.board) # score of shame
                break
            # add corresponding player char to board
            self.board[space] = char
            # if winning move, end game and give rewards
            if self.player_wins(char):
                self.display_board()
                player.reward(1, self.board)
                other_player.reward(-1, self.board)
                break
            # if board full, tie game
            if self.board_full():
                self.display_board()
                player.reward(0.5, self.board)
                other_player.reward(0.5, self.board)
                break
            # change turn
            other_player.reward(0, self.board)
            self.player1_turn = not self.player1_turn

    def player_wins(self, char):
        """Check if a player wins"""
        for a,b,c in [(0,1,2), (3,4,5), (6,7,8),
                      (0,3,6), (1,4,7), (2,5,8),
                      (0,4,8), (2,4,6)]:
            if char == self.board[a] == self.board[b] == self.board[c]:
                return True
        return False

    def board_full(self):
        """Returns True if board is full and False otherwise."""
        return not any([space == ' ' for space in self.board])

    def display_board(self):
        """ Prints the current board screen. """
        print(' ' + self.board[6] + ' | ' + self.board[7] + ' | ' + self.board[8])
        print('-----------')
        print(' ' + self.board[3] + ' | ' + self.board[4] + ' | ' + self.board[5])
        print('-----------')
        print(' ' + self.board[0] + ' | ' + self.board[1] + ' | ' + self.board[2])  


class Player(object):
    """Human controlled agent."""
    def __init__(self):
        self.agent_type = "human"

    def start_game(self, char):
        print ("\nNew game!")

    def move(self, board):
        return int(input("Your move? "))

    def reward(self, value, board):
        print ("{} rewarded: {}".format(self.agent_type, value))

    def available_moves(self, board):
        return [i for i in range(0,9) if board[i] == ' ']


class RandomPlayer(Player):
    """Random agent. Players random moves."""
    def __init__(self):
        self.agent_type = "random"

    def reward(self, value, board):
        pass

    def start_game(self, char):
        pass

    def move(self, board):
        return random.choice(self.available_moves(board))


class MinimaxPlayer(Player):
    """Minimax agent. Plays the ideal move."""
    def __init__(self):
        self.agent_type = "minimax"
        self.best_moves = {}

    def start_game(self, char):
        self.me = char
        self.enemy = self.other(char)

    def other(self, char):
        return 'O' if char == 'X' else 'X'

    def move(self, board):
        if tuple(board) in self.best_moves:
            return random.choice(self.best_moves[tuple(board)])
        if len(self.available_moves(board)) == 9:
            return random.choice([0,2,6,8])
        best_yet = -2
        choices = []
        for move in self.available_moves(board):
            board[move] = self.me
            optimal = self.minimax(board, self.enemy, -2, 2)
            board[move] = ' '
            if optimal > best_yet:
                choices = [move]
                best_yet = optimal
            elif optimal == best_yet:
                choices.append(move)
        self.best_moves[tuple(board)] = choices
        return random.choice(choices)

    def minimax(self, board, char, alpha, beta):
        if self.player_wins(self.me, board):
            return 1
        if self.player_wins(self.enemy, board):
            return -1
        if self.board_full(board):
            return 0
        for move in self.available_moves(board):
            board[move] = char
            val = self.minimax(board, self.other(char), alpha, beta)
            board[move] = ' '
            if char == self.me:
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    return beta
            else:
                if val < beta:
                    beta = val
                if beta <= alpha:
                    return alpha
        if char == self.me:
            return alpha
        else:
            return beta

    def player_wins(self, char, board):
        for a,b,c in [(0,1,2), (3,4,5), (6,7,8),
                      (0,3,6), (1,4,7), (2,5,8),
                      (0,4,8), (2,4,6)]:
            if char == board[a] == board[b] == board[c]:
                return True
        return False

    def board_full(self, board):
        return not any([space == ' ' for space in board])

    def reward(self, value, board):
        pass


class QLearningPlayer(Player):
    """QLearning agent. Uses reinforcement learning to find the ideal move."""
    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
        self.agent_type = "Qlearner"
        self.q = {} # (board, action) keys: Q values
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards

    def start_game(self, char):
        self.last_board = (' ',)*9
        self.last_move = None

    def getQ(self, board, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((board, action)) is None:
            self.q[(board, action)] = 1.0
        return self.q.get((board, action))

    def move(self, board):
        self.last_board = tuple(board)
        actions = self.available_moves(board)

        if random.random() < self.epsilon: # explore!
            self.last_move = random.choice(actions)
            return self.last_move

        qs = [self.getQ(self.last_board, a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]
        return actions[i]

    def reward(self, value, board):
        if self.last_move:
            self.learn(self.last_board, self.last_move, value, tuple(board))

    def learn(self, board, action, reward, result_board):
        prev = self.getQ(board, action)
        maxqnew = max([self.getQ(result_board, a) for a in self.available_moves(board)])
        self.q[(board, action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)

if __name__ == '__main__':
    p1 = QLearningPlayer()
    p2 = MinimaxPlayer()
    hist = []
    for i in range(0,1000):
        print("Training #{}".format(i))
        t = TicTacToe(p1, p2)
        t.play_game()
        hist.append(t.player_wins('O'))
    win_rate = []
    num_true = 0
    seen = 0
    # generate win rates for graphing
    for idx, val in enumerate(hist):
        if val:
            num_true += 1
        seen += 1
        win_rate.append(float(num_true)/seen)
    print(win_rate[-1])
    plt.plot(win_rate)
    plt.ylabel('Win Rate')
    plt.xlabel('Games Played')
    plt.show()
    
    p1 = Player()
    p2.epsilon = 0

    while True:
        t = TicTacToe(p1, p2)
        t.play_game()
        
        print(t.player_wins('X'))

