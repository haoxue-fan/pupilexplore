#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:37:25 2022

@author: taylorburke
"""

import numpy as np
import pandas as pd

DATA = pd.read_csv("../results/regression_results.csv")


def create_table():
    """


    Returns:
        table (TYPE): DESCRIPTION.

    """
    table = pd.DataFrame()
    table = DATA[["exploration_strategy", "num_blocks", "variance"]]

    # (1) Relative uncertainty (RU) DOES alter the INTERCEPT of choice
    #   probability (i.e. there is a significant difference between intercepts
    #   of RS and SR conditions) and the rs coefficient is greater than the sr
    #   coefficient
    table["ru_intercept"] = np.array(0)
    table.loc[
        (DATA["rs_vs_sr_p_val"] < 0.05) & (DATA["rs_coef"] > DATA["sr_coef"]),
        "ru_intercept",
    ] = 1

    # (2) Total uncertainty (TU) DOES NOT alter the INTERCEPT of choice
    #   probability (i.e. there is no significant difference between intercepts
    #   of RR and SS conditions)
    table["tu_intercept"] = np.array(0)
    table.loc[DATA["rr_vs_ss_p_val"] >= 0.05, "tu_intercept"] = 1

    # (3) Relative uncertainty (RU) DOES NOT alter the SLOPE of choice
    #   probability (i.e. there is no significant difference between intercepts of
    #   RS and SR conditions when conditioned on estimated_value_difference)
    table["ru_slope"] = np.array(0)
    table.loc[DATA["rs_v_vs_sr_v_p_val"] >= 0.05, "ru_slope"] = 1

    # (4) Total uncertainty (TU) DOES alter the SLOPE of choice
    #   probability (i.e. there is a significant difference between intercepts of
    #   RR and SS conditions when conditioned on estimated_value_difference)
    #   and the coefficient of rr when conditioned on V (slope) is less than the
    #   coefficient of ss when conditioned on V (slope)
    table["tu_slope"] = np.array(0)
    table.loc[
        (DATA["rr_v_vs_ss_v_p_val"] < 0.05) & (DATA["rr_v_coef"] < DATA["ss_v_coef"]),
        "tu_slope",
    ] = 1

    # (5) Estimated value difference has a significant positive effect on
    #   choice probability
    # (6) Relative uncertainty has a significant positive effect on choice
    #   probability
    # (7) Total uncertainty over estimated value difference has a significant
    #   positive effect on choice probability
    predictors = ["v", "ru", "vtu"]

    for predictor in predictors:
        table[f"{predictor}_effect"] = np.array(0)
        table.loc[
            (DATA[f"{predictor}_coef"] >= 0) & (DATA[f"{predictor}_p_val"] < 0.05),
            f"{predictor}_effect",
        ] = 1

    return table


table = create_table()
table.to_csv(
    "../results/hypothesis_testing_table.csv",
    index=False,
)
