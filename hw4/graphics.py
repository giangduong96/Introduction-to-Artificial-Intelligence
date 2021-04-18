# -----------------------------------------------------------------------------
# Name:     graphics
# Purpose:  Visualization of the maze and solution for the Spartan's quest
#
# Author:   Rula Khayrallah
#
# Copyright ©  Rula Khayrallah, 2018
# -----------------------------------------------------------------------------
"""
Class definition to visualize the quest
"""
import tkinter
import time


class Display(object):
    """
    Visualization of a given solution to a quest
    Arguments:
        problem (a Problem object) representing the quest
            see Problem class definition in spartanquest.py)
        solution:  list of actions representing the solution to the quest
    Attributes:
        problem (a Problem object) representing the quest
            see Problem class definition in spartanquest.py)
        path:  list of actions representing the solution to the quest
        canvas:  tkinter widget used to visualize the solution
        medal_icon: set of Canvas ovals representing the remaining medals
        mascot: Canvas image representing Sammy the Spartan
    """
    time_interval = 0.1  # specify 0.1 for faster animation or 0.5 for slower
    size = 40  # size in pixel for each grid position

    def __init__(self, problem, solution):
        self.medal_icon = {}
        self.path = solution
        self.problem = problem # save the problem info

        root = tkinter.Tk()
        root.title('Go Spartans!')
        self.canvas = tkinter.Canvas(root,
                                     width=self.size * problem.maze.width,
                                     height=self.size * problem.maze.height)

        self.canvas.grid()
        sammy_row, sammy_col = problem.mascot_position

        for row in range(problem.maze.height):
            for col in range(problem.maze.width):
                if problem.maze.is_wall((row, col)):
                    fill_color = 'blue'
                else:
                    fill_color = 'black'
                self.canvas.create_rectangle(col * self.size,
                                             row * self.size,
                                             (col + 1) * self.size,
                                             (row + 1) * self.size,
                                             fill=fill_color,
                                             outline="")

        # load the .gif image file
        sammy = tkinter.PhotoImage(file='sammy.gif')

        self.mascot = self.canvas.create_image((sammy_col + 0.5) * self.size,
                                               (sammy_row + 0.5) * self.size,
                                               image=sammy)
        for each_medal_row, each_medal_col in problem.medals:
            self.medal_icon[(each_medal_row, each_medal_col)] = \
                self.canvas.create_oval((each_medal_col + 0.25) * self.size,
                                        (each_medal_row + 0.25) * self.size,
                                        (each_medal_col + 0.75) * self.size,
                                        (each_medal_row + 0.75) * self.size,
                                        fill="gold",
                                        outline="")

        if solution is not None:
            root.after(1, self.animate)
        root.mainloop()


    def animate(self):
        """
        Animate the tkinter window to visualize Sammy's path through the maze
        :return: None
        """
        for action in self.path:
            time.sleep(self.time_interval)
            row, col = self.problem.mascot_position
            new_row = row + self.problem.moves[action][0]
            new_col = col + self.problem.moves[action][1]
            position = (new_row, new_col)

            if not self.problem.maze.within_bounds(position):
                raise Exception('Falling off the maze....')
            elif self.problem.maze.is_wall(position):
                raise Exception('Crash!  Wall encountered')
            elif position in self.problem.medals:
                self.problem.medals.discard(position)
                self.canvas.itemconfigure(self.medal_icon[position],
                                          fill="")
            self.problem.mascot_position = position

            move_y, move_x = self.problem.moves[action]
            self.canvas.move(self.mascot,
                             move_x * self.size,
                             move_y * self.size)
            self.canvas.create_line((col + 0.5) * self.size,
                                    (row + 0.5) * self.size,
                                    (new_col + 0.5) * self.size,
                                    (new_row + 0.5) * self.size,
                                    arrow = tkinter.LAST,
                                    width = 3,
                                    fill = "yellow")

            self.canvas.update()