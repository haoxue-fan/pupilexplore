#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:59:14 2022

@author: taylorburke
"""

import numpy as np
import matplotlib.pyplot as plt
from simulate_two_armed_bandit_experiment import TwoArmedBanditExperiment

condition = ("Arm1", "Arm2")
uncertainty_bonuses = [0, 0.5, 1, 1.5, 2]
experiment_2_specs = {
    "num_participants": 500,
    "num_blocks": 10,
    "num_trials_per_block": 20,
    "conditions": [condition],
    "reward_distribution": {"resample_means": True, "mean": 0, "variance": 100},
    "arm_1": {"label": "Arm1", "variance": 10},
    "arm_2": {"label": "Arm2", "variance": 10},
    "exploration": {
        "strategy": "",  # Being replaced in the outer for loop
        "uncertainty_bonus": 0,  # Being replaced in the inner for loop
        "choice_stochasticity": 1,  # This is equivelant to lambda.
        "balance_factor": 4,  # This is equivelant to beta.
    },
}


for exploration_strategy in ["UCB", "Thompson Sampling", "Hybrid"]:
    experiment_2_specs["exploration"]["strategy"] = exploration_strategy
    p_optimals = []
    for uncertainty_bonus in uncertainty_bonuses:
        experiment_2_specs["exploration"]["uncertainty_bonus"] = uncertainty_bonus
        experiment_2 = TwoArmedBanditExperiment(experiment_2_specs)
        experiment_2.pilot()
        p_optimals.append(experiment_2.get_p_optimal_across_conditions()[condition])

    plt.plot(uncertainty_bonuses, p_optimals, label=exploration_strategy)

plt.xlabel("$\gamma$")
plt.xticks(uncertainty_bonuses)
plt.ylabel("P(optimal)")
plt.yticks(np.arange(0.8, 0.92, 0.02))
plt.axis(ymin=0.8, ymax=0.9)
plt.legend()
plt.show()
