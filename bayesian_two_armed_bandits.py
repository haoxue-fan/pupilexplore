#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:31:53 2022

@author: taylorburke
"""

import numpy as np
from scipy.stats import norm
from bayesian_multi_armed_bandit import BayesianMultiArmedBandit


class BayesianTwoArmedBandit(BayesianMultiArmedBandit):
    """Approximate baysian agent with only two arms.

    Attributes:
        choices (<Array<Integerr>>): array where 0s indicate choosing the left
            arm and 1s indicate choosing the right arm
        choice_probabilities (Array<Float>): probability of choosing arm 1
            across trials
    """

    def __init__(self, arms, timesteps):
        """Initialize BayesianTwoArmedBandit.

        Args:
            arms (Array<Arm>): collection of all the Arms the agent must chose
                from.
            timesteps (Integer): number of pulls the agent mnust make.

        Returns:
            None.

        """
        self.choices = np.zeros(timesteps + 1)
        self.choice_probabilities = np.zeros(timesteps)
        BayesianMultiArmedBandit.__init__(self, arms, timesteps)


class UCBBayesianTwoArmedBandit(BayesianTwoArmedBandit):
    """Approximate bayesian agent with an UCB selection algorithm.

    Attributes:
        uncertainty_bonus (Integer): amount to bonus uncertainty during , gamma.
        choice_stochasticity (Integer): analogous to temperature in the softmax
            policy, lambda.
    """

    def __init__(self, uncertainty_bonus, choice_stochasticity, arms, timesteps):
        """Initialize UCBBayesianTwoArmedBandit.

        Args:
            uncertainty_bonus (Integer): amount to bonus uncertainty during
                directed exploration, gamma.
            choice_stochasticity (Integer): analogous to temperature in the
                softmax policy, lambda.
            arms (Array<Arm>): collection of all the Arms the agent must chose
                from.
            timesteps (Integer): number of pulls the agent mnust make.

        Returns:
            None.
        """
        self.uncertainty_bonus = uncertainty_bonus
        self.choice_stochasticity = choice_stochasticity
        BayesianTwoArmedBandit.__init__(self, arms, timesteps)

    def select_arm(self, timestep):
        """Select an arm according to the Gaussian CDF (probit) policy.

        Args:
            timestep (Integer): total number of pulls an agent has made.

        Returns:
            None.
        """
        bonused_reward_estimates = self.estimate_means[
            :, timestep
        ] + self.uncertainty_bonus * np.sqrt(self.estimate_variances[:, timestep])
        choice_probability = norm.cdf(
            (bonused_reward_estimates[0] - bonused_reward_estimates[1])
            / self.choice_stochasticity
        )
        self.choice_probabilities[timestep] = choice_probability

        random_threshold = np.random.uniform(0, 1, 1)

        if random_threshold < choice_probability:
            arm_selected = 0
        else:
            arm_selected = 1
            self.choices[timestep] = 1

        self.pull_arm(arm_selected, timestep)

    def __repr__(self):
        """Override the built in representation function.

        Returns:
            None.

        """
        return f"UCB Bandit({self.num_arms} arms)"


class ThompsonBayesianTwoArmedBandit(BayesianTwoArmedBandit):
    """Approximate bayesian agent with Thompson sampling selection algorithm.

    Attributes:
        choice_probabilities (Array<Float>): probability of choosing arm 1
            across trials
    """

    def __init__(self, arms, timesteps):
        """Initialize ThompsonBayesianTwoArmedBandit.

        Args:
            arms (Array<Arm>): collection of all the Arms the agent must chose
                from.
            timesteps (Integer): number of pulls the agent mnust make.

        Returns:
            None.

        """
        BayesianTwoArmedBandit.__init__(self, arms, timesteps)

    def select_arm(self, timestep):
        """Select an arm according to the Thompson Sampling policy.

        Args:
            timestep (integer): total number of pulls an agent has made.

        Returns:
            None.
        """
        choice_probability = norm.cdf(
            (self.estimate_means[0][timestep] - self.estimate_means[1][timestep])
            / np.sqrt(
                self.estimate_variances[0][timestep]
                + self.estimate_variances[1][timestep]
            )
        )
        self.choice_probabilities[timestep] = choice_probability

        random_threshold = np.random.uniform(0, 1, 1)

        if random_threshold < choice_probability:
            arm_selected = 0
        else:
            arm_selected = 1
            self.choices[timestep] = 1

        self.pull_arm(arm_selected, timestep)

    def __repr__(self):
        """Override the built in representation function.

        Returns:
            None.

        """
        return f"Thompson Bandit({self.num_arms} arms)"


class HybridBayesianTwoArmedBandit(BayesianTwoArmedBandit):
    """Approximate bayesian agent with an hybrid selection algorithm.

    Attributes:
        uncertainty_bonus (Integer): amount to bonus uncertainty during
            directed exploration, gamma.
        balance_factor (Integer): balance between directed and random
            exploration, beta.
    """

    def __init__(self, uncertainty_bonus, balance_factor, arms, timesteps):
        """Initialize HybridBayesianTwoArmedBandit.

        Args:
            uncertainty_bonus (Integer): amount to bonus uncertainty during
                directed exploration, gamma.
            balance_factor (Integer): balance between directed and random
                exploration, beta.
            arms (Array<Arm>): collection of all the Arms the agent must chose
                from.
            timesteps (Integer): number of pulls the agent mnust make.

        Returns:
            None.

        """
        self.uncertainty_bonus = uncertainty_bonus
        self.balance_factor = balance_factor
        BayesianTwoArmedBandit.__init__(self, arms, timesteps)

    def select_arm(self, timestep):
        """Select an arm according to the hyrbrid policy.

        The hybrid policy balances between directed and random exploration.

        Args:
            timestep (integer): total number of pulls an agent has made.

        Returns:
            None.
        """
        random_exploration_probability = (
            self.balance_factor
            * (self.estimate_means[0][timestep] - self.estimate_means[1][timestep])
            / np.sqrt(
                self.estimate_variances[0][timestep]
                + self.estimate_variances[1][timestep]
            )
        )
        directed_exploration_probability = self.uncertainty_bonus * (
            np.sqrt(self.estimate_variances[0][timestep])
            - np.sqrt(self.estimate_variances[1][timestep])
        )

        choice_probability = norm.cdf(
            random_exploration_probability + directed_exploration_probability
        )
        self.choice_probabilities[timestep] = choice_probability

        random_threshold = np.random.uniform(0, 1, 1)

        if random_threshold < choice_probability:
            arm_selected = 0
        else:
            arm_selected = 1
            self.choices[timestep] = 1

        self.pull_arm(arm_selected, timestep)

    def __repr__(self):
        """Override the built in representation function.

        Returns:
            None.

        """
        return f"Hybrid Bandit({self.num_arms} arms)"
