# Name: Giang Duong
# CS 156 SPRING 2021
# Professor Sanjoy Paul
# Homework 7
"""
Module to track the belief distribution over all possible grid positions

Your task for homework 8 is to implement:
1.  update
2.  recommend_sensing
"""
import utils


class Belief(object):
    """
    Belief class used to track the belief distribution based on the
    sensing evidence we have so far.
    Arguments:
    size (int): the number of rows/columns in the grid
    Attributes:
    open (set of tuples): set containing all the positions that have not
        been observed so far.
    current_distribution (dictionary): probability distribution based on
        the evidence acquired so far.
        The keys of the dictionary are the possible grid positions
        The values represent the (conditional) probability that the
        treasure is found at that position given the evidence
        (sensor data) accumulated so far.
    """

    def __init__(self, size):
        # Initially all positions are open - have not been observed
        self.open = {(x, y) for x in range(size)
                     for y in range(size)}
        # Initialize to a uniform distribution
        self.current_distribution = {pos: 1 / (size ** 2) for pos in self.open}

    def get_beliefs(self):
        """
        Accessor method for the belief distribution
        Note: to be used in treasurehunt.py only for a cleaner interface
        You do not need to use it inside the Belief class.
        :return:
          dictionary representing the belief distribution given the evidence
          accumulated so far.
        """
        return self.current_distribution

    def update(self, color, sensor_position, model):
        """
        Update the belief distribution based on new evidence:  our agent
        detected the given color at sensor location: sensor_position.
        :param color: (string) color detected
        :param sensor_position: (tuple) position of the sensor
        :param model (Model object) models the relationship between the
             treasure location and the sensor data
        :return: None
        """
        # Iterate over ALL positions in the grid and update the
        # probability of finding the treasure at that position - given
        # the new evidence.
        # The probability of the evidence given the Manhattan distance
        # to the treasure id given by calling model.psonargivendist.
        # Don't forget to normalize.
        # Don't forget to update self.open since sensor_position has
        # now been observed.
        self.open.remove(sensor_position)
        sumProb = float(0)
        for x, y in self.current_distribution:
            dist = utils.manhattan_distance((x, y), sensor_position)
            # P(S|T), float probability
            cp = model.psonargivendist(color, dist)
            self.current_distribution[x, y] *= cp
            sumProb += self.current_distribution[x, y]
        for w, z in self.current_distribution:
            self.current_distribution[w, z] = self.current_distribution[w, z] / sumProb

    def recommend_sensing(self):
        """
        Recommend where we should take the next measurement in the grid.
        The position should be the most promising unobserved location.
        If all remaining unobserved locations have a probability of 0,
        return the unobserved location that is closest to the (observed)
        location with he highest probablity.
        If there are no remaining unobserved locations return the
        (observed) location with the highest probability.
        :return: tuple representing the position where we should take
            the next measurement
        """
        # Enter your code and remove the statement below
        # for perc in self.open:
        #    prob = self.current_distribution[perc]

        # finds highest observed location
        highObserved = max(self.current_distribution, key=self.current_distribution.get)

        # finds highest unobserved location
        recommend = max(self.open, key=self.current_distribution.get)

        # how many unobserved spots are left
        unobs = len(self.open)

        # checks for remaining unobserved locations
        if unobs > 0:
            # check if probabilities above 0
            if self.current_distribution.get(recommend) > 0:
                return recommend
            else:
                closest = utils.closest_point(highObserved, self.open);
                return closest
        else:
            return highObserved
