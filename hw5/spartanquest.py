
"""
Main program to guide Sammy the Spartan on a quest within a given maze
Usage:  spartanquest.py maze_file search_algoritm
The maze_file is a text file such as SJSU.txt
The search_algorithm in homework 2 is:
    dfs: for depth first search
    bfs: for breadth first search
    ucs for uniform cost search
Example:  spartanquest.py SJSU.txt dfs
The uninformed search algorithms functions are implemented in the file
uninformed_search.py.# -----------------------------------------------------------------------------
# Name:     spartanquest
# 
dfs has been implemented for you.
Your task for homework 2 is to implement bfs and ucs.
"""
import time
import argparse
import uninformed_search
import graphics

class Maze(object):
    """
    Represent the maze layout: its width, height and walls
    Arguments:
        width (int):  the width of the maze
        height (int): the height of the maze
    Attributes:
        width (int):  the width of the maze
        height (int): the height of the maze
        walls (two dimensional list of booleans):
            Each element represents a position in the maze.
            True indicates that there is a wall in that position.
            False indicates the absence of a wall.
            self.walls[r][c] indicates the presence or absence of a wall
            at row r and column c in the maze.
            Note that the top left corner of the maze is at row 0 and column 0
    """
    def __init__(self, width, height):
        self.walls = [[False for col in range(width)]
                      for row in range(height)]
        self.width = width
        self.height = height

    def add_wall(self, position):
        """
        Add a wall in the specified position
        :param position: tuple (row, column) representing a maze position
        :return: None
        """
        row, col = position
        self.walls[row][col] = True

    def is_wall(self, position):
        """
        Is there a wall in the given position?
        :param position: tuple (row, column) representing a maze position
        :return:
        (Boolean) True if there is a wall in that position, False otherwise
        """
        row, col = position
        return self.walls[row][col]

    def within_bounds(self, position):
        """
        Is the given position within the maze?
        :param position: tuple (row, column) representing a position
        :return:
        (Boolean) True if the position is inside the maze, False otherwise
        """
        row, col = position
        return row in range(self.height) and col in range(self.width)


class Problem(object):
    """
    Represent our search problem at any point in the quest
    Arguments:
        mazefile (file):  text file containing the maze info
    Attributes:
        maze (Maze object):  the maze for this quest
            The maze is constant throughout the quest
        mascot_position (tuple of integers - (row, column)):
            The current position of Sammy in the maze -
            Note that the origin is the upper left corner of the maze
        medals (a set of tuples):
            A set containing the positions of the remaining medals in the quest
    """
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"
    moves = {WEST: (0, -1), SOUTH: (1, 0), EAST: (0, 1), NORTH: (-1, 0)}

    # The cost associated with each move.  Moving backward is the most costly
    cost = {WEST: 1, SOUTH: 1, EAST: 6, NORTH: 2}

    def __init__(self, mazefile):
        self._nodes_expanded = 0 # private variable to keep track of the search
        self.medals = set()
        self.read_quest(mazefile)

    def read_quest(self, mazefile):
        """
        Read the maze file specified and encode the information in the object
        The length of the first line in the file is used to determine the
        width of the maze.  So the last position in the first row of the maze
        must be represented by a character in the first line (not a space).
        X or x:  represent a wall
        M or m: represent the presence of a medal at that position
        S or s: represent the starting position of our Spartan mascot, Sammy
        Any other character: a vacant maze position
        :param
        mazefile (file):  the text file containing the maze info
        :return: None
        """
        layout = mazefile.readlines()
        width = len(layout[0].strip()) # the first line
        height = len(layout) # the number of lines represents the height
        self.maze = Maze(width, height)
        row = 0
        for line in layout:
            col = 0
            for char in line:
                if char in {'X', 'x'}: # X represents a wall
                    self.maze.add_wall((row, col))
                elif char in {'M', 'm'}: # M represents a medal position
                    self.add_medal((row, col))
                elif char in {'S', 's'}: # S represents Sammy's position
                    self.add_mascot((row, col))
                col += 1 # anything else is a vacant maze position
            row += 1
        mazefile.close()


    def add_mascot(self, position):
        """
        Save the mascot's position
        :param position: tuple (row, column) representing a maze position
        :return: None
        """
        self.mascot_position = position

    def add_medal(self, position):
        """
        Add the specified position to the set containing all the medals
        :param position: tuple (row, column) representing a maze position
        :return: None
        """
        self.medals.add(position)

    def is_goal(self, state):
        """
        Is the state specified a goal state?
        The state is a goal state when there are no medals left to collect.
        :param
        state - A state is represented by a tuple containing two other tuples:
                the current position (row, column) of Sammy the Spartan
                a tuple containing the positions of the remaining medals
        :return: Boolean - True if this is a goal state and False otherwise
        """
        position, medals_left = state
        return not medals_left

    def start_state(self):
        """
        Return the start state in this quest
        The start state is identified by the mascot's position and the initial
        distribution of the medals in the maze.
        represented by a tuple containing two other tuples:
                the current position (row, column) of Sammy teh Spartan
                a tuple containing all the positions of the remaining medals
        :return:
        state - The start state in the quest
                A state is represented by a tuple containing two other tuples:
                the current position (row, column) of Sammy the Spartan
                a tuple containing the positions of the remaining medals
        """
        return self.mascot_position, tuple(self.medals)

    def successors(self, state):
        """
        Return a list of tuples representing all states that are reachable
        from the current state with their corresponding action and costs
        :param
        state - A state is represented by a tuple containing two other tuples:
                the current position (row, column) of Sammy the Spartan
                a tuple containing the positions of the remaining medals
        :return:
        a list of tuples representing all states that are reachable
        from the current state with their corresponding action and cost
        """
        result = []
        self._nodes_expanded += 1 # update private variable
        position, current_medals = state
        current_row, current_col = position
        for action in self.moves:
            new_position = (current_row + self.moves[action][0],
                            current_col + self.moves[action][1])
            # if the move in that direction is valid
            if self.maze.within_bounds(new_position) and \
                not self.maze.is_wall(new_position):
                # Remove medal- if any - from that position
                new_medals = set(current_medals)
                new_medals.discard(new_position)
                new_state = (new_position, tuple(new_medals))
                result.append((new_state, action, self.cost[action]))
        return result


    def path_cost(self, actions):
        """
        Return the total cost of a sequence of actions/moves
        :param
        actions (list) - A list of actions/moves
        :return (int):  the total cost of these actions
        """
        return sum(self.cost[action] for action in actions)

    def nodes_expanded(self):
        return self._nodes_expanded


def get_arguments():
    '''
    Parse and validate the command line arguments
    :return: (tuple containing a file object and a string)
            the maze file specified  and the search algoeithm specified
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('maze_file',
                        help='name of the text file containing the maze info',
                        type=argparse.FileType('r')) # open the file
    parser.add_argument('search_algorithm',
                        help='dfs, bfs or ucs?',
                        choices=['dfs', 'bfs', 'ucs'])
    arguments = parser.parse_args()

    maze_file = arguments.maze_file
    search = arguments.search_algorithm
    print('maze file: ', maze_file)
    print('search: ', search)
    return maze_file, search

def main():
    maze_file, search = get_arguments()
    quest = Problem(maze_file)  # Initialize our search problem for this quest
    search_function = getattr(uninformed_search, search)
    start_time = time.time()
    solution = search_function(quest)  # Invoke the search algorithm specified
    elapsed_time = time.time() - start_time

    # Print some statistics
    if solution is not None:
        print('Path length: ', len(solution))
        print('Path cost: ', quest.path_cost(solution))
    else:
        print('The quest failed!')
    print('Number of nodes expanded: ',(quest.nodes_expanded()))
    print('Processing time: {:.4f} (sec)'.format(elapsed_time))

    graphics.Display(quest, solution)  # Visualize the solution


if __name__ == '__main__':
    main()
