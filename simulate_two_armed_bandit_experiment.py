#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:14:03 2022

@author: Taylor Denee Burke
"""

from bandit_arm import Arm
import numpy as np
import pandas as pd

from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import permutations
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations_with_replacement
from bayesian_two_armed_bandits import (
    UCBBayesianTwoArmedBandit,
    ThompsonBayesianTwoArmedBandit,
    HybridBayesianTwoArmedBandit,
)


class TwoArmedBanditExperiment:
    """Two armed bandit experiment witg an approximate bayesian agent.

    Atttributes:
        experiment_specs (Dictionary): provided experiment specifications
        num_participants (Integer): number of participants
        num_blocks (Integer): number of blocks per particpant
        num_trials_per_blocks (Integer): number of trials within every block
        resample_means (Boolean): whether or not to resample the means of the
            arms at the beginning of every block.
        reward_mean (Integer):
        reward_variance (Integer):
        arms (Array<Arm>):
        conditions (Array<Tuple<Integer>):
        exploration_strategy (String): determines how the bandit will select
            arms; can either be 'UCB', 'Thompson Sampling', of 'Hybrid'.
    """

    def __init__(self, experiment_specs):
        """Initialize TwoArmedBanditExperiment.

        If the length of the conditions array is zero, all possible conditions
        will be assigned.

        Args:
            experiment_specs (Dictionary): all experiment specifications,
            construction as such:
                {
                    "num_participants": INTEGER,
                    "num_blocks": INTEGER,
                    "num_trials_per_block": INTEGER,
                    "conditions": ARRAY<TUPLE<STRING>>,
                    "reward_distribution": {
                        "resample_means": BOOLEAN,
                        "mean": INTEGER,
                        "variance": INTEGER},
                    "arm_1": {"label": STRING, "variance": INTEGER},
                    "arm_2": {"label": STRING, "variance": INTEGER},
                    "exploration": {
                        "strategy": STRING,
                        "uncertainty_bonus": INTEGER,
                        "choice_stochasticity": INTEGER,
                        "balance_factor": INTEGER,
                    }
                }

        Returns:
            None.

        """
        self.experiment_specs = experiment_specs

        self.num_participants = experiment_specs["num_participants"]
        self.num_blocks = experiment_specs["num_blocks"]
        self.num_trials_per_block = experiment_specs["num_trials_per_block"]

        # Reward distribution for both arms
        self.resample_means = experiment_specs["reward_distribution"]["resample_means"]
        self.reward_mean = experiment_specs["reward_distribution"]["mean"]
        self.reward_variance = experiment_specs["reward_distribution"]["variance"]

        self.arms = np.empty(2, dtype=Arm)

        if (
            "mean" not in experiment_specs["arm_1"]
            and "mean" not in experiment_specs["arm_2"]
            and not self.resample_means
        ):
            raise KeyError(
                "If not resampling the means for each arm, you need to \
                supply the means for each arm"
            )

        self.init_arms()
        arms_labels = [arm.label for arm in self.arms]

        if len(experiment_specs["conditions"]) == 0:
            self.conditions = set(map(tuple, permutations(arms_labels)))
            self.conditions.update(
                map(tuple, combinations_with_replacement(arms_labels, 2))
            )
            self.conditions = list(self.conditions)
        else:
            conditions = experiment_specs["conditions"]
            for left_arm_label, right_arm_label in conditions:
                if (
                    left_arm_label not in arms_labels
                    and right_arm_label not in arms_labels
                ):
                    raise KeyError(
                        "Each condition should contain only valid arm labels"
                    )
            # Use cast into a set to eliminate condition repeats.
            self.conditions = list(set(conditions))

        assert (
            self.num_blocks % len(self.conditions) == 0
        ), f"The number of blocks ({self.num_blocks}) should evenly distribute across conditions ({self.conditions})"
        num_blocks_per_condition = self.num_blocks // len(self.conditions)
        self.block_condition_assignments = np.repeat(
            range(len(self.conditions)), num_blocks_per_condition
        )

        self.exploration_strategy = experiment_specs["exploration"]["strategy"]
        match self.exploration_strategy:
            case "UCB":
                self.uncertainty_bonus = experiment_specs["exploration"][
                    "uncertainty_bonus"
                ]
                self.choice_stochasticity = experiment_specs["exploration"][
                    "choice_stochasticity"
                ]
                bandit_type = UCBBayesianTwoArmedBandit
            case "Thompson Sampling":
                bandit_type = ThompsonBayesianTwoArmedBandit
            case "Hybrid":
                self.uncertainty_bonus = experiment_specs["exploration"][
                    "uncertainty_bonus"
                ]
                self.balance_factor = experiment_specs["exploration"]["balance_factor"]
                bandit_type = HybridBayesianTwoArmedBandit
            case _:
                raise ValueError(
                    "Your choice of exploration strategies must be either be UCB, Thompson Sampling, or Hybrid"
                )

        self.bandits = defaultdict(
            lambda: np.ndarray(
                (self.num_participants, num_blocks_per_condition), dtype=bandit_type
            )
        )

    def init_arms(self):
        """(Re)initialize the arms for the bandit to select from.

        This is called when self is initialized and, if the arms' mean rewards
        are being resampled, then at the beginning of every block.

        Returns:
            None.

        """
        new_arm_means = np.random.normal(
            self.reward_mean, np.sqrt(self.reward_variance), 2
        )
        for arm in range(2):
            arm_specs = self.experiment_specs[f"arm_{arm + 1}"]
            self.arms[arm] = Arm(
                arm_specs["label"],
                new_arm_means[arm],
                arm_specs["variance"],
                self.num_trials_per_block,
            )

    def run_participant(self, participant):
        """Simulate a single participant completing the experiment.

        Args:
            participant (Integer): the participant's ID number in [0, num_participants)

        Returns:
            None.

        """
        block_condition_counts = defaultdict(lambda: 0)
        for block in range(self.num_blocks):
            if self.resample_means:
                self.init_arms()

            block_arms = np.array(self.arms)
            condition = self.conditions[self.block_condition_assignments[block]]

            for arm in self.arms:
                for i in range(2):
                    if arm.label == condition[i]:
                        block_arms[i] = arm

            match self.exploration_strategy:
                case "UCB":
                    bandit = UCBBayesianTwoArmedBandit(
                        self.uncertainty_bonus,
                        self.choice_stochasticity,
                        block_arms,
                        self.num_trials_per_block,
                    )
                case "Thompson Sampling":
                    bandit = ThompsonBayesianTwoArmedBandit(
                        block_arms, self.num_trials_per_block
                    )
                case "Hybrid":
                    bandit = HybridBayesianTwoArmedBandit(
                        self.uncertainty_bonus,
                        self.balance_factor,
                        block_arms,
                        self.num_trials_per_block,
                    )

            bandit.run_trials()
            block_condition_count = block_condition_counts[condition]
            self.bandits[condition][participant][block_condition_count] = bandit
            block_condition_counts[condition] += 1

    def pilot(self):
        """Simulate every participant completing the experiment.

        Ensures that every participant's block conditions are shuffled.

        Returns:
            None.

        """
        for participant in range(self.num_participants):
            np.random.shuffle(self.block_condition_assignments)
            self.run_participant(participant)

    def plot_p_optimal_across_conditions(self):
        """Plot the probability of selecting the optimal arm across all conditions.

        Returns:
            None.

        """
        for condition in self.conditions:
            mean_num_optimal_actions = np.mean(
                np.vectorize(lambda bandit: bandit.num_optimal_actions)(
                    self.bandits[condition]
                )
            )
            p_optimal = mean_num_optimal_actions / self.num_trials_per_block
            plt.plot(f"{condition[0]}{condition[1]}", p_optimal, "bo")

        plt.xlabel("Condition")
        plt.ylabel("P(optimal)")
        plt.show()

    def plot_relative_uncertainty_across_conditions(self):
        """Plot average relative uncertainty across all conditions.

        Returns:
            None.

        """
        for condition in self.conditions:
            mean_relative_uncertainty = np.mean(
                np.vectorize(
                    lambda bandit: np.mean(
                        np.sqrt(bandit.estimate_variances[0])
                        - np.sqrt(bandit.estimate_variances[1])
                    )
                )(self.bandits[condition])
            )
            plt.plot(f"{condition[0]}{condition[1]}", mean_relative_uncertainty, "bo")

        y_ax_lower_limit = -4.01
        y_ax_upper_limit = 4.01
        plt.yticks(list(range(int(y_ax_lower_limit), int(y_ax_upper_limit) + 1, 2)))
        plt.ylim([y_ax_lower_limit, y_ax_upper_limit])
        plt.xlabel("Condition")
        plt.ylabel("Relative uncertainty (RU)")
        plt.show()

    # TODO: ask Hao why total uncertainty is super low
    def plot_total_uncertainty_across_conditions(self):
        """Plot average total uncertainty across all conditions.

        Returns:
            None.

        """
        for condition in self.conditions:
            mean_total_uncertainty = np.mean(
                np.vectorize(
                    lambda bandit: np.mean(
                        np.sqrt(
                            bandit.estimate_variances[0] + bandit.estimate_variances[1]
                        )
                    )
                )(self.bandits[condition])
            )
            plt.plot(f"{condition[0]}{condition[1]}", mean_total_uncertainty, "bo")

        # y_ax_lower_limit = 1.99
        # y_ax_upper_limit = 7.01
        # plt.yticks(list(range(int(y_ax_lower_limit), int(y_ax_upper_limit) + 1)))
        # plt.ylim([y_ax_lower_limit, y_ax_upper_limit])
        plt.xlabel("Condition")
        plt.ylabel("Total uncertainty (TU)")
        plt.show()

    def regress(self):
        # formula = 'C ~ -1 + cond + cond:V + (-1 + cond + cond:V|S)';
        for condition in self.conditions:
            for participant in range(len(self.bandits[condition])):
                for block in range(len(self.bandits[condition][participant])):
                    bandit = self.bandits[condition][participant][block]
                    # Making the choices array a binary array where 1's indicate
                    # True values of choosing the left arm.
                    arm_1_choice = np.where(bandit.choices == 0, True, False).astype(
                        float
                    )
                    V = bandit.estimate_means[0] - bandit.estimate_means[1]
                    RU = np.sqrt(bandit.estimate_variances[0])
                    -np.sqrt(bandit.estimate_variances[1])
                    VTU = V / np.sqrt(
                        bandit.estimate_variances[0] + bandit.estimate_variances[1]
                    )
                    formula = "arm_1_choice ~ -1 + V + RU + VTU + (-1 + V + RU + VTU)"
                    data = pd.DataFrame(columns=["arm_1_choice", "V" "RU" "VTU"])
                    data["arm_1_choice"] = arm_1_choice
                    data["V"] = V
                    data["RU"] = RU
                    data["VTU"] = VTU
                    model = smf.glm(formula, data=data, family=sm.families.Gaussian())
                    model.fit()
                    print(model.summary())


# TODO: plot the reward distribution
# TODO: plot  choice probability over expected value difference

if __name__ == "__main__":
    experiment_specs = {
        "num_participants": 50,
        "num_blocks": 12,
        "num_trials_per_block": 20,
        "conditions": [],
        "reward_distribution": {"resample_means": True, "mean": 0, "variance": 100},
        "arm_1": {"label": "S", "variance": 0.00001},
        "arm_2": {"label": "R", "variance": 16},
        "exploration": {
            "strategy": "UCB",
            "uncertainty_bonus": 1,
            "choice_stochasticity": 1,  # This is equivelant to lambda.
            "balance_factor": 4,  # This is equivelant to beta.
        },
    }

    experiment = TwoArmedBanditExperiment(experiment_specs)
    experiment.pilot()
    experiment.plot_p_optimal_across_conditions()
    experiment.plot_relative_uncertainty_across_conditions()
    experiment.plot_total_uncertainty_across_conditions()
    experiment.regress()
