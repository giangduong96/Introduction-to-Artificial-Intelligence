# -----------------------------------------------------------------------------
# Name:        tictac
# Purpose:     Implement a game of Tic Tac Toe
#
# Author:      Rula Khayrallah
# -----------------------------------------------------------------------------
"""
A Tic Tac Toe game implementation.
The user always starts playing first.
"""
import argparse
import adversarial_search
import tkinter as tk


class GameState(object):
    """
    Represent a Tic Tac Toe game state

    Arguments:
        size (int): the number of rows/columns in the game

    Attributes:
        size (int): the number of rows/columns in the game
        board (nested list): represents the n x n grid
        Each list item represents one of the squares and is initialized to None
        Each square is accessed as self.board[row][column]
        When the user marks a square, we change the corresponding
        value to 'user'.
        When the AI marks a square, we change the corresponding
        value to 'AI'.

        moves (int): number of moves
        last_move (tuple):  the row, column corresponding to the last square
            played

    """
    _count = 0  # keep track of the number of nodes expanded

    def __init__(self, size):
        self.size = int(size)
        self.board = [[None for column in range(self.size)]
                      for row in range(self.size)]
        self.moves = 0
        self.last_move = None

    def is_win(self, player):
        """
        Check if there are 3 squares marked by the same player in a line
        We only check the lines that may have been affected by the last move.
        :param player: (string) 'AI' or 'user'
        :return: boolean True if the given player has n (n = size) squares in
            a line, False otherwise
        """
        # check the row affected by the last move
        row, col = self.last_move
        if self.size == self.board[row].count(player):
            return True
        # check the diagonal - if the last move was on a diagonal
        if row == col:  # last move was on first diagonal
            if  self.size == [self.board[i][i]
                              for i in range(self.size)].count(player):
                return True
        if row == self.size - 1 - col:  # last move was on second diagonal
            if self.size ==  [self.board[i][self.size - 1 - i]
                              for i in range(self.size)].count(player):
                return True
        # check the column affected by the last move
        if self.size == [self.board[i][col]
                         for i in range(self.size)].count(player):
                return True
        return False

    def available(self, row, col):
        """
        Check if the square corresponding to the given row and column is
        available
        :param row: integer
        :param col: integer
        :return: boolean
        """
        return self.board[row][col] is None

    def make_move(self, player, row, col):
        """
        Claim the square at the given row, column for the player specified
        :param player: (string) 'AI' or 'user'
        :param row: integer
        :param col: integer
        """
        self.board[row][col] = player
        self.last_move = row, col
        self.moves += 1

    def is_tie(self):
        """
        Check if the game has ended in a tie
        :return: boolean
        """
        return self.moves == self.size ** 2

    def successor(self, move, player):
        """
        return the state resulting from the given player taking the specific
        action of marking the square represented by move
        :param move: (tuple) representing the square marked by the player
        :param player: (string) "AI" or "user"
        :return: a GameState object representing the successor state
        """
        GameState._count += 1
        next_state = GameState(self.size)
        for row in range(self.size):
            for col in range(self.size):
                next_state.board[row][col] = self.board[row][col]
        row, col = move
        next_state.board[row][col] = player
        next_state.moves = self.moves + 1
        next_state.last_move = move
        return next_state

    def possible_moves(self):
        """
        Find all the empty squares that can still be marked
        :return: a list of tuples representing all available squares
        """
        moves = [(row, col) for row in range(self.size)
                 for col in range(self.size)
                 if self.board[row][col] is None]
        return moves

    def eval(self):
        """
        Estimates the 'utility' of the given state to the max agent "AI"
        by computing a value that reflects how desirable a given state is.
        This value is based on the number of rows/columns/diagonals that max
        can still win minus the number of rows/columns/diagonals that min can
        still win.
        :return: (int) the number of rows/columns/diagonals that max can still
           win minus the number of rows/columns/diagonals that min can still
           win
        """
        row, col = self.last_move
        min_count = 0  # rows/columns/diagonals that min can still win
        max_count = 0  # rows/columns/diagonals that max can still win
        for row in range(self.size):
            if 'AI' not in self.board[row]:
                min_count += 1
            if 'user' not in self.board[row]:
                max_count += 1
        for col in range(self.size):
            line = [self.board[i][col] for i in range(self.size)]
            if 'AI' not in line:
                min_count += 1
            if 'user' not in line:
                max_count += 1

        diag1 = [self.board[i][i] for i in range(self.size)]
        if 'AI' not in diag1:
            min_count += 1
        if 'user' not in diag1:
            max_count += 1

        diag2 = [self.board[i][self.size - 1 - i]
                 for i in range(self.size)]
        if 'AI' not in diag2:
            min_count += 1
        if 'user' not in diag2:
            max_count += 1
        return max_count - min_count

class Game(object):
    """
    Game class for a Tic Tac Toe game.
    The user is X and the AI is O.

    Arguments:
    parent: the root window object
    size (int): the number of rows/columns in the game
    search (string): the adversarial search algorithm to be used by our
        AI agent: "rand", "minimax", "alphabeta" or "abdl"
    depth (int):  depth limit for use with the abdl algorithm only

    Attributes:
    canvas (Canvas): the tkinter widget corresponding to the tic tac toe board
    message (Label): a tkinter Label widget to display the win/lose/tie message
    size (int): the number of rows/columns in the game
    search (string): the adversarial search algorithm to be used by our
        AI agent: "rand", "minimax", "alphabeta" or "abdl"
    depth (int):  depth limit for use with the abdl algorithm only
    state (Gamestate): object to keep track of undeerlying state of the game
    """
    square_size = 100  # length in pixels of the side of a grid square

    def __init__(self, parent, size, search, depth):

        self.size = int(size)
        self.search = search
        self.state = GameState(size)
        self.depth = depth

        parent.title('CS 156 Tic Tac Toe')
        self.canvas = tk.Canvas(parent,
                                     width=self.size * self.square_size,
                                     height=self.size * self.square_size)
        self.canvas.grid()

        # create the squares on the canvas
        for row in range(self.size):
            for column in range(self.size):
                self.canvas.create_rectangle(column * self.square_size,
                                             row * self.square_size,
                                             (column + 1) * self.square_size,
                                             (row + 1) * self.square_size,
                                             fill = 'blue',
                                             outline = 'white')
        self.message = tk.Label(parent)
        self.message.grid()

        self.canvas.bind("<Button-1>", self.play)
        self.message.configure(text='')

    def play(self, event):
        """
        Implement the basic controls of the game.
        The user's click is immediately followed by the AI move.
        """
        row = event.y // self.square_size      # row corresponding to the click
        column = event.x // self.square_size   # col corresponding to the click
        if self.state.available(row, column):  # If that square is unmarked
            self.mark(row, column, 'user')
            self.state.make_move('user', row, column)
            if self.state.is_win('user'):
                self.gameover('You won!')
            elif not self.state.is_tie():
                self.ai_move()
            else:
                self.gameover("It's a tie!")

    def gameover(self, outcome):
        """
        Display the game outcome and disable further clicks
        :param outcome: (string) 'You won', 'You lost' or "It's a tie"
        """
        self.message.config(text=outcome)
        self.canvas.unbind("<Button-1>")


    def mark(self, row, column, player):
        """
        Mark a given square for the given player.

        :param row: (int) row corresponding to the square to be marked
        :param column: (int) column corresponding to the square to be marked
        :param player: (string)  'AI' (O) or 'user' (X)
        """
        if player == 'user':  # user is X
            self.canvas.create_line((column + 0.25)  * self.square_size,
                                    (row + 0.25)  * self.square_size,
                                         (column + 0.75) * self.square_size,
                                         (row + 0.75) * self.square_size,
                                         fill='yellow',
                                    width = 5)
            self.canvas.create_line((column + 0.75) * self.square_size,
                                    (row + 0.25) * self.square_size,
                                    (column + 0.25) * self.square_size,
                                    (row + 0.75) * self.square_size,
                                         fill='yellow',
                                    width = 5)
        else:  # AI is O
            self.canvas.create_oval((column + 0.25) * self.square_size,
                                    (row + 0.25) * self.square_size,
                                         (column + 0.75) * self.square_size,
                                         (row + 0.75) * self.square_size,
                                         outline='yellow',
                                    width = 5 )

    def ai_move(self):
        """
        Invoke the adversarial search algorithm specified to mark a square
        for the AI agent.
        """
        search_function = getattr(adversarial_search, self.search)
        if self.search == "abdl":
            row, column = search_function(self.state, self.depth)
        else:
            row, column = search_function(self.state)
        self.mark(row, column, 'AI')
        self.state.make_move('AI', row, column)
        if self.state.is_win('AI'):
            self.gameover('You lost!')
        elif self.state.is_tie():
            self.gameover("It's a tie!")
        print(f'Number of nodes evaluated so far: {self.state._count:,}')

def get_arguments():
    """
    Parse and validate the command line arguments
    :return: (tuple containing the size of the game and the algorithm
    """
    #run tictactoe
    #parser = argparse.ArgumentParser()
    #parser.add_argument("echo")
    #args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('size',
                        help='number of rows and columns',
                        )
    parser.add_argument('search',
                        help='rand, minimax, alphabeta or abdl ',
                        choices=['rand', 'minimax', 'alphabeta', 'abdl'])
    parser.add_argument('depth',
                        help='how many levels',
                        )
    arguments = parser.parse_args()
    print(arguments)
    print("hi")
    size = arguments.size
    print('size = ', arguments.size)
    print(type (size))
    depth = arguments.depth
    print('depth =', arguments.depth)
    print('search = ', arguments.search)
    search = str(arguments.search)
    print('search = ', search)
    return size, search, depth

def main():
    size, search, depth = get_arguments()
    # Instantiate a root window
    root = tk.Tk()
    # Instantiate a Game object
    my_game = Game(root, size, search, depth)
    # Enter the main event loop
    root.mainloop()
    print(f'Total number of nodes evaluated: {my_game.state._count:,}')

if __name__ == '__main__':
    main()
