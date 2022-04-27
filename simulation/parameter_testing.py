#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:01:25 2022

@author: Taylor D. Burke
"""

import numpy as np
from two_armed_bandit_experiment import TwoArmedBanditExperiment

experiment_specs = {
    "num_participants": 50,
    "num_blocks": 0,  # This is being changed on the 2nd for loop.
    "num_trials_per_block": 10,
    "conditions": [],  # If conditions array is empty, all possible conditions will be used.
    "reward_distribution": {
        "resample_means": True,
        "mean": 0,
        "variance": 0,  # This is being changed on the 3rd for loop
    },
    "arm_1": {"label": "R", "variance": 16},
    "arm_2": {"label": "S", "variance": 0.00001},
    "exploration": {
        "strategy": "",  # This is being changed on the 1st for loop
        "uncertainty_bonus": 1,
        "choice_stochasticity": 1,  # This is equivelant to lambda.
        "balance_factor": 1,  # This is equivelant to beta.
    },
}

significant_combinations = []
for exploration_strategy in ["UCB", "Thompson Sampling", "Hybrid"]:
    experiment_specs["exploration"]["strategy"] = exploration_strategy
    for num_blocks in [16, 20, 24, 28]:
        experiment_specs["num_blocks"] = num_blocks
        for variance in [25, 36, 49, 64, 81, 100]:
            experiment_specs["reward_distribution"]["variance"] = variance
            experiment_regressor_p_values = np.ndarray(100)
            for experiment in range(100):
                experiment = TwoArmedBanditExperiment(experiment_specs)
                experiment.pilot()
                experiment_regressor_p_values[experiment] = experiment.regress()
            if len(np.where(experiment_regressor_p_values < 0.05)) >= 95:
                significant_combinations.append(
                    (exploration_strategy, num_blocks, variance)
                )
            break
        break
    break


for exploration_strategy, num_blocks, variance in significant_combinations:
    print(
        f"Strategy: {exploration_strategy}, num_blocks: {num_blocks}, variance: {variance}"
    )
