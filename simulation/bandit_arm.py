#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm in the classic k-armed bandit RL problem.

Last modified on Mon Apr 11

@author: Taylor D. Burke
"""
import numpy as np


class Arm:
    """Arm within a k-armed bandit problem.

    Attributes:
        label (String): the name of the arm
        mean (float): the mean reward of the normal reward distribution.
        variance (float): the variance of the normal reward distribution.
        reward_distribution (np.array<float>): the normal reward
            distribution.
        num_pulls (integer): number of times the arm has been pulled.
    """

    def __init__(self, label, mean, variance, timesteps):
        """Initialize arm.

        Args:
            mean (float): the mean reward of the normal reward distribution.
            variance (float): the variance of the normal reward distribution.
            timesteps (integer): number of times the arm can be pulled.

        Returns:
            None.
        """
        self.label = label
        self.mean = mean
        self.variance = variance
        self.reward_distribution = np.random.normal(
            self.mean, np.sqrt(self.variance), timesteps
        )
        self.num_pulls = 0

    def get_reward(self):
        """Return the next reward.

        Gets called when the agent has selected this arm at some timestep.
        Updates the number of arm pulls.

        Returns:
            (float): a single reward drawn from the reward distribution.
        """
        return self.reward_distribution[self.num_pulls - 1]

    def __repr__(self):
        """Override the built in representation function.

        Returns:
            None.

        """
        return f"Arm('{self.label}', {self.mean}, {self.variance}, pulls: {self.num_pulls})"

    def __str__(self):
        """Override the built in toString function.

        Returns:
            (string): the formatted version of the arm's necessary information.

        """
        return f"Arm (mean: {self.mean}, variance: {self.variance}, pulls: {self.num_pulls})"

    def __eq__(self, other):
        """Ovverride the built in equals function.

        Args:
            other (Arm): arm to compare to self.

        Returns:
            (Boolean): True if both Arms have the same mean and variance,
                else False.

        """
        return (
            (self.label == self.label)
            and (self.mean == other.mean)
            and (self.variance == other.variance)
        )
