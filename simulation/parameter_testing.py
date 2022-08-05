#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:01:25 2022

@author: Taylor D. Burke
"""

import os
import multiprocessing
from simulate_two_armed_bandit_experiment import TwoArmedBanditExperiment


def simulate(experiment_inputs):
    exploration_strategy, num_blocks, variance = experiment_inputs
    experiment_specs = {
        "num_participants": 50,
        "num_blocks": num_blocks,
        "num_trials_per_block": 10,
        "reward_distribution": {
            "resample_means": True,
            "mean": 0,
            "variance": variance,
        },
        "arm_1": {
            "label": "R",
            "variance": 16,
            "prior_mean_estimate": 0,
            "prior_variance_in_estimate": variance,
        },
        "arm_2": {
            "label": "S",
            "variance": 0.00001,
            "prior_mean_estimate": 0,
            "prior_variance_in_estimate": variance,
        },
        "exploration": {
            "strategy": exploration_strategy,
            "uncertainty_bonus": 1,
            "choice_stochasticity": 1,  # This is equivelant to lambda.
            "balance_factor": 1,  # This is equivelant to beta.
        },
    }

    for num_experiment in range(100):
        file_path = f"../data/{exploration_strategy}_{num_blocks}_{variance}_data_{num_experiment + 1}.csv"
        if not os.path.isfile(file_path):
            experiment = TwoArmedBanditExperiment(experiment_specs)
            experiment.pilot()
            experiment.data.to_csv(
                file_path,
                index=False,
            )


experiment_inputs = []
for exploration_strategy in ["UCB", "Thompson Sampling", "Hybrid"]:
    for num_blocks in [16, 20, 24, 28]:
        for variance in [25, 36, 49, 64, 81, 100]:
            experiment_inputs.append(
                (exploration_strategy, num_blocks, variance),
            )


if __name__ == "__main__":
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=5)
    pool.map(simulate, experiment_inputs)
