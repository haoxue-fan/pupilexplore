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
        true_arm_means (array<float>): all the true means, specific to each
            arm's reward distribution
        true_arm_variances (array<float>): all the true variances, specific to each
            arm's reward distribution
        mean_estimates (array<float>): all the estimate means, specific to each
            arm's reward distribution
        variance_in_estimates (array<float>): all the estimate variances, specific
            to each arm's reward distribution
        timesteps (integer):  number of pulls the agent can make.
    """

    def __init__(
        self, arms, timesteps, prior_mean_estimates, prior_variance_in_estimates
    ):
        """Initialize BayesianMultiArmedBandit.

        Args:
            arms (Array<Arm>): collection of all the Arms the agent must chose
                from.
            timesteps (Integer): number of pulls the agent mnust make.
            #TODO: fill out
            prior_mean_estimates (Array<Float>): DESCRIPTION.
            prior_estimate_variance (Array<Float>): DESCRIPTION.

        Returns:
            None.

        """
        self.arms = arms
        self.num_arms = len(self.arms)
        self.timesteps = timesteps
        self.num_optimal_actions = 0
        self.rewards = np.zeros(timesteps)

        self.true_arm_means = [arm.mean for arm in arms]
        self.true_arm_variances = [arm.variance for arm in arms]

        self.mean_estimates = np.zeros((self.num_arms, timesteps))
        self.variance_in_estimates = np.zeros((self.num_arms, timesteps))

        self.mean_estimates[:, 0] = prior_mean_estimates
        self.variance_in_estimates[:, 0] = prior_variance_in_estimates

    def update_estimates(self, arm_selected, reward_received, timestep):
        """Update beliefs using Kalman filtering equations.

        Args:
            arm_selected (Integer): arm index agent chose.
            reward_received (Float): reward agent recieved.
            timestep (Integer): total number of pulls an agent has made.

        Returns:
            None.

        """
        true_arm_variance = self.true_arm_variances[arm_selected]

        # Get agent's prior belief about the selected arm's reward distribution.
        prior_mean_estimate = self.mean_estimates[arm_selected][timestep]
        prior_variance_in_estimate = self.variance_in_estimates[arm_selected][timestep]

        # Implement Kalman filtering equations.
        learning_rate = prior_variance_in_estimate / (
            prior_variance_in_estimate + true_arm_variance
        )
        posterior_variance_in_estimate = (
            prior_variance_in_estimate - learning_rate * prior_variance_in_estimate
        )
        prediction_error = reward_received - prior_mean_estimate
        posterior_mean_estimate = prior_mean_estimate + learning_rate * prediction_error

        # Carry over this trial's posteriors into next trial's priors if this
        # is not the last timestep. Subtract one because timestep is zero
        # indexed.
        if timestep != self.timesteps - 1:
            self.variance_in_estimates[:, timestep + 1] = np.copy(
                self.variance_in_estimates[:, timestep]
            )
            self.mean_estimates[:, timestep + 1] = np.copy(
                self.mean_estimates[:, timestep]
            )
            self.variance_in_estimates[arm_selected][
                timestep + 1
            ] = posterior_variance_in_estimate
            self.mean_estimates[arm_selected][timestep + 1] = posterior_mean_estimate

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
            self.arms[arm_selected].mean == np.max(self.true_arm_means)
        )

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
        self.rewards[timestep] = reward_received

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
        for num_arm in range(self.num_arms):
            information += f"{num_arm + 1}: {str(self.arms[num_arm])} \n"

        return information
