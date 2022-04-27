#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approximate bayesian agent in the classic k-armed bandit RL problem.

Last modified on Mon Apr 11

@author: Taylor D. Burke
"""

import numpy as np


class BayesianMultiArmedBandit:
    """Approximate bayesian agent within a k-armed bandit problem.

    Attributes:
        arms (array<Arm>): collection of all the Arms the agent must chose from.
        num_arms (Integer): number of all the Arms the agent must chose from.
        true_means (array<float>): all the true means, specific to each
            arm's reward distribution
        true_variances (array<float>): all the true variances, specific to each
            arm's reward distribution
        estimate_means (array<float>): all the estimate means, specific to each
            arm's reward distribution
        estimate_variances (array<float>): all the estimate variances, specific
            to each arm's reward distribution
        timesteps (integer):  number of pulls the agent can make.
    """

    def __init__(self, arms, timesteps):
        """Initialize BayesianMultiArmedBandit.

        Args:
            arms (Array<Arm>): collection of all the Arms the agent must chose
                from.
            timesteps (Integer): number of pulls the agent mnust make.

        Returns:
            None.

        """
        self.arms = arms
        self.num_arms = len(self.arms)

        self.true_means = [arm.mean for arm in arms]
        self.true_variances = [arm.variance for arm in arms]

        self.estimate_means = np.zeros((self.num_arms, timesteps + 1))
        self.estimate_variances = np.zeros((self.num_arms, timesteps + 1))
        self.estimate_variances[:, 0] = np.copy(self.true_variances)

        self.timesteps = timesteps
        self.num_optimal_actions = 0

    def update_estimates(self, arm_selected, reward_received, timestep):
        """Update beliefs using Kalman filtering equations.

        Args:
            arm_selected (Integer): arm index agent chose.
            reward_received (Float): reward agent recieved.
            timestep (Integer): total number of pulls an agent has made.

        Returns:
            None.

        """
        true_variance = self.true_variances[arm_selected]

        # Get agent's prior belief about the selected arm's reward distribution.
        prior_estimate_mean = self.estimate_means[arm_selected][timestep]
        prior_estimate_variance = self.estimate_variances[arm_selected][timestep]

        # Implement Kalman filtering equations.
        learning_rate = prior_estimate_variance / (
            prior_estimate_variance + true_variance
        )
        posterior_estimate_variance = (
            prior_estimate_variance - learning_rate * prior_estimate_variance
        )
        prediction_error = reward_received - prior_estimate_mean
        posterior_estimate_mean = prior_estimate_mean + learning_rate * prediction_error

        # Carry over the posteriors into priors.
        self.estimate_variances[:, timestep + 1] = np.copy(
            self.estimate_variances[:, timestep]
        )
        self.estimate_means[:, timestep + 1] = np.copy(self.estimate_means[:, timestep])

        # Only the arm that was selected gets belief updating for new priors.
        self.estimate_variances[arm_selected][
            timestep + 1
        ] = posterior_estimate_variance
        self.estimate_means[arm_selected][timestep + 1] = posterior_estimate_mean

    def evaluate_action(self, arm_selected):
        """Evaluate agent's previous action.

        Updates the agent's total number of optimal actions taken. An action is
        considered optimal if the agent chose the arm with the highest true
        mean.

        Args:
            arm_selected (integer): arm that the agent last selected.
            timestep (integer): total number of pulls an agent has made.

        Returns:
            None.

        """
        self.num_optimal_actions += int(
            self.arms[arm_selected].mean == np.max(self.true_means)
        )

    # TODO: create a default select arm function that selected the arm with the
    # argmax nean

    def pull_arm(self, arm_selected, timestep):
        """Pull the specified arm.

        This updates the number of times that the selected arm has been pulled
        and dispenses a reward. This calls on other functions to update the
        agent's belief about the selected arm's reward distribution
        and evaluate the agent's selection.

        Args:
            arm_selected (integer): arm that the agent last selected.

        Returns:
            None.
        """
        self.arms[arm_selected].num_pulls += 1
        reward_received = self.arms[arm_selected].get_reward()

        self.update_estimates(arm_selected, reward_received, timestep)
        self.evaluate_action(arm_selected)

    def run_trials(self):
        """Simulate an approximate bayesian bandit sequentially selecting arms.

        Returns:
            None.
        """
        for timestep in range(self.timesteps):
            self.select_arm(timestep)

    def __str__(self):
        """Override the built in toString function.

        Returns:
            information (string): the formatted version of the
                BayesianMultiArmedBandit's necessary information.
        """
        information = f"A bandit with the following {self.num_arms} arms:\n"
        for num_arm in range(len(self.arms_means)):
            information += f"{num_arm + 1}: {str(self.arms[num_arm])} \n"

        return information
