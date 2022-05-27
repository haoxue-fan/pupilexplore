#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:14:03 2022

@author: Taylor Denee Burke
"""

from bandit_arm import Arm
import numpy as np
import pandas as pd
from itertools import permutations
from itertools import combinations_with_replacement
from bayesian_two_armed_bandits import (
    UCBBayesianTwoArmedBandit,
    ThompsonBayesianTwoArmedBandit,
    HybridBayesianTwoArmedBandit,
)


def create_experiment_dataframe():
    """Create the dataframe that stores all the data across the experiment.

    Returns:
        pd.DataFrame

    """
    columns = [
        "subject",
        "block",
        "condition",
        "trial",
        "reward",
        "choice_probability",
        "choice",
    ]

    for arm in ["left_arm", "right_arm"]:
        columns.append(arm + "_estimate_mean")
        columns.append(arm + "_true_mean")
        columns.append(arm + "_variance_in_estimate")
        columns.append(arm + "_true_variance")

    return pd.DataFrame(columns=columns)


class TwoArmedBanditExperiment:
    """Two armed bandit experiment with an approximate bayesian agent.

    Atttributes:
        parameters (Dictionary): provided experiment specifications
        num_participants (Integer): number of participants
        num_blocks (Integer): number of blocks per particpant
        num_trials_per_blocks (Integer): number of trials within every block
        reward_mean (Integer):
        reward_variance (Integer):
        prior_mean_estimates ():
        prior_variance_in_estimates ():
        conditions (Array<Tuple<Integer>):
        block_condition_assignments ():
        exploration_strategy (String): determines how the bandit will select
            arms; can either be 'UCB', 'Thompson Sampling', of 'Hybrid'.
        data ():
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
                        "mean": INTEGER,
                        "variance": INTEGER},
                    "arm_1": {"label": STRING,
                              "variance": INTEGER,
                              "prior_mean_estimate": FLOAT,
                              "prior_variance_in_estimate": FLOAT},
                    "arm_2": {"label": STRING,
                              "variance": INTEGER,
                              "prior_mean_estimate": FLOAT,
                              "prior_variance_in_estimate": FLOAT},
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
        self.parameters = experiment_specs

        self.num_participants = experiment_specs["num_participants"]
        self.num_blocks = experiment_specs["num_blocks"]
        self.num_trials_per_block = experiment_specs["num_trials_per_block"]

        # Reward distribution for both arms
        self.reward_mean = experiment_specs["reward_distribution"]["mean"]
        self.reward_variance = experiment_specs["reward_distribution"]["variance"]

        self.prior_mean_estimates = np.zeros(2)
        self.prior_variance_in_estimates = np.zeros(2)
        arms_labels = np.empty(2, dtype=str)

        for num_arm in range(2):
            arm_info = experiment_specs[f"arm_{num_arm + 1}"]
            self.prior_mean_estimates[num_arm] = arm_info["prior_mean_estimate"]
            self.prior_variance_in_estimates[num_arm] = arm_info[
                "prior_variance_in_estimate"
            ]
            arms_labels[num_arm] = arm_info["label"]

        # Create and randomize conditions across all experiment blocks
        # If no conditions are given, create all the possible conditions
        if "conditions" in experiment_specs:
            conditions = experiment_specs["conditions"]
            for left_arm_label, right_arm_label in conditions:
                if (
                    left_arm_label not in arms_labels
                    and right_arm_label not in arms_labels
                ):
                    raise KeyError(
                        "Each condition should contain only valid arm labels"
                    )
            # Casting into a set to eliminate condition repeats.
            self.conditions = list(set(conditions))
        else:
            self.conditions = set(map(tuple, permutations(arms_labels)))
            self.conditions.update(
                map(tuple, combinations_with_replacement(arms_labels, 2))
            )
            self.conditions = sorted(list(self.conditions))

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
            case "Thompson Sampling":
                pass
            case "Hybrid":
                self.uncertainty_bonus = experiment_specs["exploration"][
                    "uncertainty_bonus"
                ]
                self.balance_factor = experiment_specs["exploration"]["balance_factor"]
            case _:
                raise ValueError(
                    "Your choice of exploration strategies must be either be UCB, Thompson Sampling, or Hybrid"
                )

        self.data = create_experiment_dataframe()

    def get_block_arms(self, condition):
        """Get the arms to be used for a block, given the block condition.

        Assumes that every block resamples the means of both arms.

        Args:
            condition (Tuple<String>): label of the left and right arm,
                indicating what the block condition is.

        Returns:
            block_arms (Array<Arm>): the left and right arms to be used on the
                current block.

        """
        new_arm_means = np.random.normal(
            self.reward_mean, np.sqrt(self.reward_variance), 2
        )
        block_arms = np.empty(2, dtype=Arm)
        for num_arm in range(2):
            arm_label = condition[num_arm]
            arm = 1
            if arm_label == self.parameters["arm_2"]["label"]:
                arm = 2
            block_arms[num_arm] = Arm(
                arm_label,
                new_arm_means[num_arm],
                self.parameters[f"arm_{arm}"]["variance"],
                self.num_trials_per_block,
            )

        return block_arms

    def run_participant(self, participant):
        """Simulate a single participant completing the experiment.

        Args:
            participant (Integer): the participant's ID number in [0, num_participants)

        Returns:
            None.

        """

        for block in range(self.num_blocks):

            block_condition = self.conditions[self.block_condition_assignments[block]]
            block_arms = self.get_block_arms(block_condition)

            match self.exploration_strategy:
                case "UCB":
                    bandit = UCBBayesianTwoArmedBandit(
                        self.uncertainty_bonus,
                        self.choice_stochasticity,
                        block_arms,
                        self.num_trials_per_block,
                        self.prior_mean_estimates,
                        self.prior_variance_in_estimates,
                    )
                case "Thompson Sampling":
                    bandit = ThompsonBayesianTwoArmedBandit(
                        block_arms,
                        self.num_trials_per_block,
                        self.prior_mean_estimates,
                        self.prior_variance_in_estimates,
                    )
                case "Hybrid":
                    bandit = HybridBayesianTwoArmedBandit(
                        self.uncertainty_bonus,
                        self.balance_factor,
                        block_arms,
                        self.num_trials_per_block,
                        self.prior_mean_estimates,
                        self.prior_variance_in_estimates,
                    )

            bandit.run_trials()

            block_data = create_experiment_dataframe()
            block_data["trial"] = np.array(range(1, self.num_trials_per_block + 1))
            block_data["subject"] = participant + 1
            block_data["block"] = block + 1
            block_data["condition"] = "" + block_condition[0] + block_condition[1]
            block_data["choice"] = bandit.choices
            block_data["choice_probability"] = bandit.choice_probabilities
            block_data["reward"] = bandit.rewards

            for num_arm, name in zip(range(2), ["left_arm", "right_arm"]):
                block_data[name + "_true_mean"] = bandit.true_arm_means[num_arm]
                block_data[name + "_true_variance"] = bandit.true_arm_variances[num_arm]
                block_data[name + "_estimate_mean"] = bandit.mean_estimates[num_arm]
                block_data[
                    name + "_variance_in_estimate"
                ] = bandit.variance_in_estimates[num_arm]

            self.data = pd.concat([self.data, block_data], ignore_index=True)

    def pilot(self):
        """Simulate every participant completing the experiment.

        Ensures that every participant's block conditions are shuffled.

        Returns:
            None.

        """
        for participant in range(self.num_participants):
            np.random.shuffle(self.block_condition_assignments)
            self.run_participant(participant)

    # TODO: plot the reward distribution
    # TODO: plot  choice probability over expected value difference
    # TODO: change existing plots to not use the stored bandits
    # def plot_p_optimal_across_conditions(self):
    #     """Plot the probability of sekecting the optimal arm across all conditions.

    #     Returns:
    #         None.

    #     """
    #     for condition in self.conditions:

    #         num_optimal = self.data[]
    #         plt.plot(f"{condition[0]}{condition[1]}",
    #                  num_optimal/ self.num_trials_per_block, "bo")

    #     plt.xlabel("Condition")
    #     plt.ylabel("P(optimal")
    #     plt.show()

    # def plot_p_optimal_across_conditions(self):
    #     """Plot the probability of selecting the optimal arm across all conditions.

    #     Returns:
    #         None.

    #     """
    #     for condition in self.conditions:
    #         mean_num_optimal_actions = np.mean(
    #             np.vectorize(lambda bandit: bandit.num_optimal_actions)(
    #                 self.bandits[condition]
    #             )
    #         )
    #         p_optimal = mean_num_optimal_actions / self.num_trials_per_block
    #         plt.plot(f"{condition[0]}{condition[1]}", p_optimal, "bo")

    #     plt.xlabel("Condition")
    #     plt.ylabel("P(optimal)")
    #     plt.show()

    # def plot_relative_uncertainty_across_conditions(self):
    #     """Plot average relative uncertainty across all conditions.

    #     Returns:
    #         None.

    #     """
    #     for condition in self.conditions:
    #         mean_relative_uncertainty = np.mean(
    #             np.vectorize(
    #                 lambda bandit: np.mean(
    #                     np.sqrt(bandit.estimate_variances[0])
    #                     - np.sqrt(bandit.estimate_variances[1])
    #                 )
    #             )(self.bandits[condition])
    #         )
    #         plt.plot(f"{condition[0]}{condition[1]}", mean_relative_uncertainty, "bo")

    #     y_ax_lower_limit = -4.01
    #     y_ax_upper_limit = 4.01
    #     plt.yticks(list(range(int(y_ax_lower_limit), int(y_ax_upper_limit) + 1, 2)))
    #     plt.ylim([y_ax_lower_limit, y_ax_upper_limit])
    #     plt.xlabel("Condition")
    #     plt.ylabel("Relative uncertainty (RU)")
    #     plt.show()

    # def plot_total_uncertainty_across_conditions(self):
    #     """Plot average total uncertainty across all conditions.

    #     Returns:
    #         None.

    #     """
    #     for condition in self.conditions:
    #         mean_total_uncertainty = np.mean(
    #             np.vectorize(
    #                 lambda bandit: np.mean(
    #                     np.sqrt(
    #                         bandit.estimate_variances[0] + bandit.estimate_variances[1]
    #                     )
    #                 )
    #             )(self.bandits[condition])
    #         )
    #         plt.plot(f"{condition[0]}{condition[1]}", mean_total_uncertainty, "bo")

    #     plt.xlabel("Condition")
    #     plt.ylabel("Total uncertainty (TU)")
    #     plt.show()
