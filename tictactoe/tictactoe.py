import random
import copy

class Game: 
    ''' A single player implementation of a tic-tac-toe game in which a human plays against an AI. '''
    
    def __init__(self, char = 'x', turn = random.randint(0, 1), initial_state = [' ']*9):
        ''' 
        A game state is a list of 9 characters. The user is allowed to choose which character they want to
        use ('x' or 'o') and who plays first (human first = 0, AI first = 1).
        The board index is as follows:
        6 | 7 | 8
        _________
        3 | 4 | 5
        _________
        0 | 1 | 2
        '''
        self.state = initial_state
        self.turn = turn
        if char == 'x':
            self.player_char = 'x'
            self.ai_char = 'o'
        else:
            self.player_char = 'o'
            self.ai_char = 'x'
            
    def draw_board(self):
        ''' Prints the current board screen. '''
        print(' ' + self.state[6] + ' | ' + self.state[7] + ' | ' + self.state[8])
        print('-----------')
        print(' ' + self.state[3] + ' | ' + self.state[4] + ' | ' + self.state[5])
        print('-----------')
        print(' ' + self.state[0] + ' | ' + self.state[1] + ' | ' + self.state[2])
        
    def is_free(self, move): 
        ''' Checks if a spot on the board is free. '''
        return self.state[move]
        
    def one_turn(self):
        ''' One turn in a game. Either the AI or human will choose a move. '''
        if self.turn % 2 == 0:
            print("Your turn: ")
            self.draw_board()
            move = 10
            print("free moves: ", self.get_moves())
            while move not in self.get_moves():
                move = int(input("Enter a move: "))

        else:
            print("AI's turn: ")
            self.draw_board()
            move = self.ai_choose()
            print("AI chooses ", move)
        
        self.do_move(move)

    def get_new_state(self, move):
        ''' Returns a clone of this instance with one move applied. '''
        new_state = copy.deepcopy(self)
        new_state.do_move(move)
        return new_state
        
        
    def game_over(self):
        ''' Returns 1 if game is over and 0 otherwise. '''
        return (
        self.is_winner(0) or 
        self.is_winner(1) or 
        not self.get_moves())
        
    def score(self):
        ''' Calculates a score from the AI's perspective. Shorter games are scored higher. '''
        if self.is_winner(1):
            return 10 - self.turn
        elif self.is_winner(0):
            return self.turn - 10
        else:
            return 0
        
    def minimax(self):
        ''' Returns a tuple containing max/min (depending on turn) score and choice. '''
        if self.game_over():
            return(self.score(), 0)
        moves = []
        scores = []
        
        for move in self.get_moves():
            possible_game = self.get_new_state(move)
            scores.append(possible_game.minimax()[0])
            moves.append(move)
        
        # print("moves: ", moves)
        # print("scores: ", scores)
        if self.turn % 2 == 1:
            choice = moves[scores.index(max(scores))]
            return (max(scores), choice)
            
        else:
            choice = moves[scores.index(min(scores))] 
            return (min(scores), choice)
            
    def ai_choose(self):  
        ''' Returns the move with the best score for the AI. '''
        score, choice = self.minimax()
        print(score, choice)
        return choice
        
    def is_winner(self, player):
        ''' Checks if there is are 3 in a row. '''
        char = [self.player_char, self.ai_char]
        return (
        (self.state[0] == self.state[1] == self.state[2] == char[player]) or
        (self.state[3] == self.state[4] == self.state[5] == char[player]) or
        (self.state[6] == self.state[7] == self.state[8] == char[player]) or
        (self.state[0] == self.state[3] == self.state[6] == char[player]) or
        (self.state[1] == self.state[4] == self.state[7] == char[player]) or
        (self.state[2] == self.state[5] == self.state[8] == char[player]) or
        (self.state[0] == self.state[4] == self.state[8] == char[player]) or
        (self.state[2] == self.state[4] == self.state[6] == char[player]))
        
    def do_move(self, move):
        ''' Inserts chosen move into board. '''
        if self.is_free(move):
            if self.turn % 2 == 0:
                self.state[move] = self.player_char
            else:
                self.state[move] = self.ai_char
        self.turn += 1
            
    def get_moves(self):
        ''' Returns list of available moves. '''
        moves = []
        for i in range(9):
            if self.state[i] == ' ':
                moves.append(i)
        return moves
    
    def play_game(self):
        ''' Begins a full game of tic tac toe. Ends when either player wins or the board is full. '''
        zcopy = copy.deepcopy(self)
        while ' ' in self.state:
            self.one_turn()
            print("turn: ", self.turn)
            print("score: ", self.score())
            if self.is_winner((self.turn - 1)%2):
                print("Winning board: ")
                message = ["The AI wins!", "You win!"]
                self.draw_board()
                print(message[self.turn%2])
                return
        print("Tie game!")
        self.draw_board()

if __name__ == '__main__':
    turn = 2
    char = ''
    while turn not in (0, 1):
        turn = int(input("Input first player [0:you / 1:bot]: "))
    while char not in ('x', 'o'):
        char = input("Input your character [x/o]: ")
    x = Game(char, turn)
    x.play_game()
