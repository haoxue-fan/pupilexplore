#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:28:19 2022

@author: taylorburke
"""

import os
import numpy as np
import pandas as pd

EXP_SPECS = {"num_blocks": 16, "num_trials": 10, "variance": 36}


def create_dataframe():
    """


    Returns:
        TYPE: DESCRIPTION.

    """
    return pd.DataFrame(
        columns=[
            "subject",
            "block",
            "condition",
            "trial",
            "reward",
            "choice",
            "choice_rt",
            "left_arm_estimate_mean",
            "left_arm_true_mean",
            "left_arm_variance_in_estimate",
            "left_arm_true_variance",
            "right_arm_estimate_mean",
            "right_arm_true_mean",
            "right_arm_variance_in_estimate",
            "right_arm_true_variance",
        ]
    )


def extract_data(directory_path, exp_specs):
    """


    Args:
        directory_path (TYPE): DESCRIPTION.
        exp_specs (TYPE): DESCRIPTION.

    Returns:
        all_data (TYPE): DESCRIPTION.

    """
    all_data = create_dataframe()
    num_subject = 1
    for file_name in os.listdir(directory_path):
        if ".csv" in file_name:
            subject_data = pd.read_csv(directory_path + file_name)
            subject_data = subject_data.loc[subject_data["block"] >= 1]
            subject_data.rename(
                columns={
                    "subjectID": "subject",
                    "mu1": "left_arm_true_mean",
                    "mu2": "right_arm_true_mean",
                    "cond": "condition",
                    "choiceRT": "choice_rt",
                },
                inplace=True,
            )

            subject_data[
                [
                    "left_arm_estimate_mean",
                    "left_arm_variance_in_estimate",
                    "left_arm_true_variance",
                    "right_arm_estimate_mean",
                    "right_arm_variance_in_estimate",
                    "right_arm_true_variance",
                ]
            ] = np.nan

            subject_data["condition"] = subject_data["condition"].map(
                {1: "RS", 2: "SR", 3: "RR", 4: "SS"}
            )
            subject_data["choice"] = subject_data["choice"].map(
                {"machine1": 0, "machine2": 1}
            )
            subject_data["choice"] = subject_data["choice"].astype(int)

            subject_data["subject"] = num_subject
            subject_data.loc[
                (subject_data["condition"] == "SR")
                | (subject_data["condition"] == "SS"),
                "left_arm_true_variance",
            ] = 0.00001
            subject_data.loc[
                (subject_data["condition"] == "RS")
                | (subject_data["condition"] == "RR"),
                "right_arm_true_variance",
            ] = 16

            # Handle missed trials
            for column in ["reward", "choice_rt"]:
                subject_data.loc[subject_data[column] == "[]", column] = np.nan
                subject_data[column] = subject_data[column].astype(float)

            # Perform belief updating within a block.
            # Because this is only data wrangling, must process all blocks and
            # all trials before excluding blocks or trials.
            # Meaning, you still want to include missed trials so when we are
            # excluding we can have proper counts.
            for block in range(exp_specs["num_blocks"]):
                block_data = subject_data.loc[subject_data["block"] == block + 1]
                condition = np.unique(block_data["condition"])[0]
                choices = np.array(block_data["choice"])
                rewards = np.array(block_data["reward"])

                mean_estimates = np.empty((2, exp_specs["num_trials"]))
                variance_in_estimates = np.empty((2, exp_specs["num_trials"]))

                # Setting the priors for the first trial
                for arm in range(2):
                    if condition[arm] == "S":
                        variance_in_estimates[arm][0] = 0.00001
                    else:
                        variance_in_estimates[arm][0] = 16
                    mean_estimates[arm][0] = 0

                trial = 1
                prior_mean_estimate = 0
                prior_variance_in_estimate = exp_specs["variance"]
                while trial < exp_specs["num_trials"] - 1:
                    choice = choices[trial]
                    reward = rewards[trial]

                    # If this was a missed trial, do not perform belief updating.
                    if choice == np.nan:
                        trial += 1
                        continue

                    if condition[choice] == "S":
                        true_arm_variance = 0.00001
                    else:
                        true_arm_variance = 16

                    learning_rate = prior_variance_in_estimate / (
                        prior_variance_in_estimate + true_arm_variance
                    )
                    posterior_variance_in_estimate = (
                        prior_variance_in_estimate
                        - learning_rate * prior_variance_in_estimate
                    )
                    prediction_error = reward - prior_mean_estimate
                    posterior_mean_estimate = (
                        prior_mean_estimate + learning_rate * prediction_error
                    )

                    # Carry over this trial's posteriors into next trial's priors
                    variance_in_estimates[:, trial + 1] = np.copy(
                        variance_in_estimates[:, trial]
                    )
                    mean_estimates[:, trial + 1] = np.copy(mean_estimates[:, trial])
                    variance_in_estimates[choice][
                        trial + 1
                    ] = posterior_variance_in_estimate
                    mean_estimates[choice][trial + 1] = posterior_mean_estimate

                    trial += 1
                    prior_mean_estimate = posterior_mean_estimate
                    prior_variance_in_estimate = posterior_variance_in_estimate

                for arm_indx, arm in zip([0, 1], ["left", "right"]):
                    subject_data.iloc[
                        block_data.index, f"{arm}_arm_estimate_mean"
                    ] = prior_mean_estimate[arm_indx]
                    subject_data.iloc[
                        block_data.index, f"{arm}_arm_variance_in_estimate"
                    ] = prior_variance_in_estimate[arm_indx]

            all_data = pd.concat(
                [all_data, subject_data[all_data.columns]], ignore_index=True
            )
            num_subject += 1

    return all_data


if __name__ == "__main__":
    all_data = extract_data("../data/raw/", EXP_SPECS)
    print(all_data.head)
    # all_data.to_csv("../data/data.csv", index=False)
