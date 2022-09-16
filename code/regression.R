library(lme4)
library(hash)
library(rstan)
library(bayesplot)
library(devtools)
library(useful)
library(gtools)
install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
library(cmdstanr)
# Uncomment this if you are re-running the brm models locally
install_github("paul-buerkner/brms")
library(brms)
library(multcomp)
library(lmtest)
library(plm)
library(tidyverse)
library(grid)
library(caret)
library(tidybayes)
library(dplyr)
library(ggplot2)

# Set the working directory to where the file lives
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

get_a_subjects_data = function(file_path) {
  data = read.csv(file_path)
  
  data$Chosen_Arm_Value[data$Chosen_Arm_Value == ""] <- NaN
  if (sum(is.na(data$Chosen_Arm_Value)) > 32) {
    print(sprintf("excluding %s on missing more than 20% of trials across all
                  blocks", 
                  unique(data$SubjectID)))
    return(NaN)
  }
  
  #TODO: record the amount of data omitted so you can report that in the paper
  # as well as the distribution of trials omitted per condition (ensure even 
  # power across condition)
  data <- na.omit(data) 
  
  # Renaming columns and values for consistent variable naming
  data <- rename(data, choice = Chosen_Arm_Value)
  data <- rename(data, condition = Condition)
  data <- rename(data, id = ID)
  data <- rename(data, v = V)
  data <- rename(data, ru = RU)
  data <- rename(data, tu = TU)
  data$condition[data$condition == "Safe/Safe"] <- "SS"
  data$condition[data$condition == "Safe/Risky"] <- "SR"
  data$condition[data$condition == "Risky/Safe"] <- "RS"
  data$condition[data$condition == "Risky/Risky"] <- "RR"
  
  # Since the trial column in the original data file is indexed by 0,
  # increment all values so trial is indexed by 1
  data$trial = data$trial + 1
  
  data$mean_diff = data$Right_Slot_Actual_Mean -
    data$Left_Slot_Actual_Mean
  # if the right arm has a greater value, mean_diff will be positive
  # positive values mean that the selection should be 0 (not choosing the 
  # left arm)
  data$correct_choice[data$mean_diff > 0] = 0
  # If the right arm has a lesser value, mean_diff will be negative
  # negative values mean that the selection should be 1 (choosing the left arm)
  # If there is no correct choice, then they chose "optimally"
  data$correct_choice[data$mean_diff < 0] = 1
  
  # Filter out the data blocks where both the left and right arm 
  # have the same true mean briefly to find subject's accuracy rating
  filtered_data <- data %>% filter(mean_diff != 0)

  # Exclude subjects if they don't have an accuracy rating greater than 60% 
  # (accuracy here is defined as selecting the arm with the greater mean.)
  if (sum(filtered_data$choice == filtered_data$correct_choice)/
      length(filtered_data$choice) < 0.6) {
    print(sprintf("excluding %s on optimal choice exclusion criteria",
                  unique(filtered_data$id)))
    return(NaN)
  }
  
  SS_unique_choices <- (data %>% filter(condition == "SS") %>% 
                          group_by(block) %>% 
                          summarize(unique_choices = 
                                      n_distinct(choice)))$unique_choices
  
  # Exclude subjects if they don't choose more than 1 option in all four of
  # their SS conditioned blocks (not sampling the other arm at all)
  if (all(SS_unique_choices == 1)) {
    print(sprintf("excluding %s for only selecting one arm on all SS 
                  conditions", unique(filtered_data$id)))
    return(NaN)
  }

  data <- data[, c('id', 'block', 'trial', 'condition', 
                   'choice', 'v', 'ru', 'tu', 'trial_baseline')]
  
  subject = unique(data$id)
  subject_pupil_data <- read.csv(paste("../data/raw/merged_pupil_long_AFTER/", 
                                       sprintf("%s_merged_pupil_long_AFTER.csv", 
                                               subject), sep = ""))

  
  # Since the trial column in the original data file is indexed by 0,
  # increment all values so trial is indexed by 1 and will align with the larger
  # combined data file (important for merging the data frames)
  subject_pupil_data$trial = subject_pupil_data$trial + 1
  
  stimulus_pupil_data_frame <- as.data.frame(subject_pupil_data %>% 
    filter(timestamp_locked_at_stimulus_pre_with_fixation_onset 
           %in% (500:2500) ) %>% group_by(block, trial) %>% 
    summarize(stimulus_evoked_avg_pupil =
                mean(remove_baseline_smoothed_interp_pupil_corrected, 
                     na.rm = TRUE), 
              early_key_press = "early_key_press" %in% unique(event), 
              max_stimulus_evoked_pupil = max(
                remove_baseline_smoothed_interp_pupil_corrected, 
                na.rm = TRUE),
              min_stimulus_evoked_pupil = min(
                remove_baseline_smoothed_interp_pupil_corrected,
                na.rm = TRUE)
              ))
  
  data <- merge(data, stimulus_pupil_data_frame, by = c("block", "trial"))
  
  feedback_pupil_data_frame <- as.data.frame(subject_pupil_data %>% 
    filter(timestamp_locked_at_reward_pre_red_fixation_onset 
           %in% (1000:3000) ) %>% group_by(block, trial) %>%
    summarize(feedback_evoked_avg_pupil =
                mean(remove_baseline_smoothed_interp_pupil_corrected, 
                     na.rm = TRUE)))
  
  data <- merge(data, feedback_pupil_data_frame, by = c("block", "trial"))

  return(data)
}


combine_and_save_all_data = function() {
  data_cols = c('subject', 'id', 'block', 'trial', 'condition', 'choice', 
  'early_key_press', 'v', 'ru', 'tu', 'trial_baseline', 
  'stimulus_evoked_avg_pupil', 'feedback_evoked_avg_pupil')
  
  combined_data = data.frame(matrix(nrow = 0, ncol = length(data_cols))) 
  colnames(combined_data) = data_cols
  
  data_file_paths = list.files(path="../data/raw/merged_pupil_first_row_AFTER", 
                                     pattern=".csv", all.files=TRUE, 
                                     full.names=TRUE)
  subject = 1
  for (data_file_path in data_file_paths) {
    subject_data <- get_a_subjects_data(data_file_path)

    # If the subject was excluded, don't process them
    if (class(subject_data) != "data.frame") {
      next
    }
    subject_data$subject <- subject 
    combined_data = rbind(combined_data, subject_data)
    subject <- subject + 1
  }
  
  combined_data$subject <- as.factor(combined_data$subject)
  combined_data$choice <- as.factor(combined_data$choice)
  combined_data$condition <- as.factor(combined_data$condition)
  
  write.csv(combined_data, "../data/combined_data.csv", row.names = FALSE)
}

load_in_data = function() {
  data <- read.csv("../data/combined_data.csv")
  
  data$v_over_tu <- data$v / data$tu
  data$abs_v = abs(data$v)
  data$abs_ru = abs(data$ru)
  data$abs_v_over_tu = abs(data$v_over_tu)
  
  # Encoding directed behavior to be negative when they choose the right arm
  data$directed_v <- with(data, ifelse(choice == 0, -abs_v, abs_v))
  data$directed_ru <- with(data, ifelse(choice == 0, -abs_ru, abs_ru))
  data$directed_v_over_tu <- data$directed_v / data$tu
  
  data$stimulus_evoked_avg_pupil_with_baseline <- data$stimulus_evoked_avg_pupil + data$trial_baseline
  data$feedback_evoked_avg_pupil_with_baseline <- data$feedback_evoked_avg_pupil + data$trial_baseline
  
  # Add in all the posterior data
  data <- data %>% group_by(subject, block) %>%
    arrange(subject, block, trial)
  new_post_columns <- c("post_abs_v", "post_abs_ru",
                        "post_tu", "post_abs_v_over_tu")
  data <- shift.column(data=data, newNames = new_post_columns,
                            columns=c("abs_v", "abs_ru", "tu", "abs_v_over_tu"),
                            len = 1)
  
  #TODO: edge case: there is no posterior for the 10th trial. Thoughts on 
  # going back in the belief update code to grab the posterior?
  data[data$trial == 10, new_post_columns] = NaN 
  
  # Need to group to ungroup again
  data <- data %>% group_by(subject, block) %>% ungroup() 
  
  # Add in the posterior belief directed data 
  data$directed_post_v <- with(data, ifelse(choice == 0, -post_abs_v, post_abs_v))
  data$directed_post_ru <- with(data, ifelse(choice == 0, -post_abs_ru, post_abs_ru))
  data$directed_post_v_over_tu = data$directed_post_v / data$post_tu
  
  # TODO: add in all the decoded data
  
  # Remove the residuals
  model <- readRDS("../models/feedback_baseline.rds")
  baseline_pupil_residuals <- residuals(model, summary = TRUE)[, c("Estimate")]
  
  return(data)
}

run_and_save_choice_models = function() {
  # Note: for some reason the choice_prior_belief model is not running
  # on the new data but I still have it saved from when it converged on the
  # older data files
  model <- readRDS("../models/choice_uncertainty_model_brm.rds")
  print(summary(model))

  # choice model 1
  formula = choice ~ -1 + v + ru + v_over_tu +
    (-1 + v + ru + v_over_tu | subject);
  model <- brm(formula = formula, data = data,
               family = binomial(link = "probit"))
  print(summary(model))
  saveRDS(model, "../models/choice_prior_belief.rds")

  # choice model 2
  formula = choice ~ -1 + condition + condition:v +
    (-1 + condition + condition:v | subject)
  model <- brm(formula = formula, data = data,
               family = binomial(link = "probit"))
  print(summary(model))
  saveRDS(model, "../models/choice_condition.rds")
}

run_condition_behavior_hypothesis_testing_brm = function() {
  model <- readRDS("../models/choice_condition_model_brm.rds")
  
  # Testing two intercept and two slope hypotheses:
  # (1) Relative uncertainty DOES alter the INTERCEPT of choice 
  #   probability (i.e. there is a significant difference between intercepts of 
  #   RS and SR conditions)
  # (2) Total uncertainty DOES NOT alter the INTERCEPT of choice 
  #   probability (i.e. there is no significant difference between intercepts of 
  #   RR and SS conditions)
  # (3) Relative uncertainty DOES NOT alter the SLOPE of choice 
  #   probability (i.e. there is no significant difference between intercepts of 
  #   RS and SR conditions when conditioned on estimated_value_difference) 
  # (4) Total uncertainty DOES alter the SLOPE of choice 
  #   probability (i.e. there is a significant difference between intercepts of 
  #   RR and SS conditions when conditioned on estimated_value_difference) 
  hypotheses = c("ConditionRS - ConditionSR = 0",
                 "ConditionRR - ConditionSS = 0",
                 "ConditionRS:V - ConditionSR:V = 0",
                 "ConditionRR:V - ConditionSS:V = 0")
  for (hypothesis in hypotheses) {
    hypothesis_test_results = hypothesis(model, hypothesis)
    plot(hypothesis_test_results)
    print(hypothesis_test_results)
  }
}

run_and_save_feedback_model_baseline_residuals = function() {
  data <- load_in_data() 
  
  formula = feedback_evoked_avg_pupil_with_baseline ~ trial_baseline
  model <- brm(formula = formula, data = data, family = gaussian, cores = 4)
  print(summary(model))
  saveRDS(model, "../models/feedback_baseline.rds")
}

run_and_save_feedback_model_1 = function() {
  data <- load_in_data()
  
  # feedback model 1
  formula = feedback_evoked_avg_pupil ~ trial_baseline + abs_v + abs_ru + tu +
    abs_v_over_tu + (trial_baseline + abs_v + abs_ru + 
                       tu + abs_v_over_tu | subject)
  model <- brm(formula = formula, data = data, family = gaussian, cores = 4)
  print(summary(model))
  saveRDS(model, "../models/feedback_prior_belief.rds")
}

run_and_save_feedback_model_2 = function() {
  data <- load_in_data()
  
  # feedback model 2
  formula = feedback_evoked_avg_pupil ~ trial_baseline + post_abs_v +
    post_abs_ru + post_tu +
    post_abs_v_over_tu + (trial_baseline + post_abs_v + post_abs_ru + 
                            post_tu + post_abs_v_over_tu | subject)
  model <- brm(formula = formula, data = data, family = gaussian, cores = 4)
  print(summary(model))
  saveRDS(model, "../models/feedback_posterior_belief.rds")
}

run_and_save_feedback_model_3 = function() {
  data <- load_in_data()
  
  # feedback model 3
  formula = feedback_evoked_avg_pupil ~ trial_baseline + directed_v + 
    directed_ru + tu + directed_v_over_tu + (trial_baseline + directed_v +
                                               directed_ru + tu + 
                                               directed_v_over_tu | subject)
  model <- brm(formula = formula, data = data, family = gaussian, cores = 4)
  print(summary(model))
  saveRDS(model, "../models/feedback_directed_belief.rds")
}

run_and_save_feedback_model_4 = function() {
  data <- load_in_data()
  
  # feedback model 4
  formula = feedback_evoked_avg_pupil ~ trial_baseline + directed_post_v + 
    directed_post_ru + tu + directed_post_v_over_tu + 
    (trial_baseline + directed_post_v + directed_post_ru + tu + 
       directed_post_v_over_tu | subject)
  model <- brm(formula = formula, data = data, family = gaussian, cores = 4)
  print(summary(model))
  saveRDS(model, "../models/feedback_directed_posterior_belief.rds")
}

# Create the model description data frame (to store all the models)
ALL_MODEL_DESCRIPTIONS <- hash()
ALL_MODEL_DESCRIPTIONS['test_model_formula'] = "feedback_evoked_avg_pupil ~ trial_baseline"

# Feedback Model 1
ALL_MODEL_DESCRIPTIONS['feedback_prior_belief_formula'] = "
feedback_evoked_avg_pupil ~ trial_baseline + abs_v + abs_ru + tu + 
abs_v_over_tu + (trial_baseline + abs_v + abs_ru + tu + abs_v_over_tu | subject)"

# Feedback Model 2
ALL_MODEL_DESCRIPTIONS['feedback_posterior_belief_formula'] = "
feedback_evoked_avg_pupil ~ trial_baseline + post_abs_v + post_abs_ru + 
post_tu + post_abs_v_over_tu + (trial_baseline + post_abs_v + 
post_abs_ru + post_tu + post_abs_v_over_tu | subject)"

# Attempting to do it with a hashmap instead
# By default, run all the models
run_and_save_models = function(print_results, models_to_run = keys(ALL_MODEL_DESCRIPTIONS)) {
  data <- load_in_data()
  for(model_to_run in models_to_run) {
    print(paste("Running:", model_to_run, sep = " "))
    
    if (!has.key(model_to_run, ALL_MODEL_DESCRIPTIONS)) {
      print(paste(model_to_run, "is not a valid model name", sep = " "))
      next
    }
    
    formula <- ALL_MODEL_DESCRIPTIONS[[model_to_run]]

    # Family depends on formula
    if (grepl("choice ~", formula, fixed = TRUE)) {
      family <- bernoulli(link = "probit")
    } else {
      family <- gaussian
    }

    model <- brm(formula = formula, data = data, family = family, cores = 4)
    if (print_results) {
      print(summary(model))
    }
    saveRDS(model, paste("../models/", model_to_run, ".rds", sep = ""))
  }
}

run_and_save_models(TRUE, c("test", "test_model_formula"))


# plot_feedback_model_results("feedback_directed_posterior_belief")


run_and_save_stimulus_models = function() {
  data <- load_in_data()
  data_no_early_key_presses <- data %>% filter(early_key_press == FALSE)
  
  # # Model 3a
  # formula = stimulus_evoked_avg_pupil ~ trial_baseline + abs_v + abs_ru + tu +
  #   abs_v_over_tu + (trial_baseline + abs_v + abs_ru + 
  #                      tu + abs_v_over_tu | subject)
  # model <- brm(formula = formula, data = data, family = gaussian)
  # print(summary(model))
  # saveRDS(model, "../models/stimulus_prior_belief.rds")
  # 
  # # Model 3b
  # model <- brm(formula = formula, data = data_no_early_key_presses,
  #              family = gaussian)
  # print(summary(model))
  # saveRDS(model, "../models/stimulus_belief_without_early_key_presses.rds")
  #   
  
  # # Model 6a
  # formula = stimulus_evoked_avg_pupil ~ trial_baseline + 
  #   (abs_v + abs_ru + tu + abs_v_over_tu)*choice + 
  #   (trial_baseline + (abs_v + abs_ru + tu + abs_v_over_tu)*choice | subject)
  # model <- brm(formula = formula, data = data, family = gaussian)
  # print(summary(model))
  # saveRDS(model, "../models/stimulus_prior_belief_choice_interaction.rds")
  # 
  # # Model 6b
  # formula = stimulus_evoked_avg_pupil ~ trial_baseline + 
  #   (abs_v + abs_ru + tu + abs_v_over_tu)*choice + 
  #   (trial_baseline + (abs_v + abs_ru + tu + abs_v_over_tu)*choice | subject)
  # model <- brm(formula = formula, data = data_no_early_key_presses, family = gaussian)
  # print(summary(model))
  # saveRDS(model, "../models/stimulus_prior_belief_choice_interaction_without_early_key_presses.rds")
}


plot_choice_estimates = function(show_plot) {
  results = c()
  model <-readRDS("../models/choice_uncertainty.rds")
  
  fixed_effect <- summary(model)$fixed
  fixed_effect <-rename(fixed_effect, lower_credible_interval = "l-95% CI")
  fixed_effect <-rename(fixed_effect, upper_credible_interval = "u-95% CI")
  
  print(fixed_effect)

  plot <- ggplot(fixed_effect, aes(factor(rownames(fixed_effect), 
                                         rownames(fixed_effect)), Estimate)) +
    scale_x_discrete(labels=c('V', 'RU', 'V/TU')) +
    scale_y_continuous(sec.axis = dup_axis(), breaks = seq(0, 3.4, 0.2)) +
    labs(y = 'Regression coefficient', x = '') + geom_point(size=4) + theme_bw() +
    theme(axis.text = element_text(face="bold"),
          axis.ticks.length = unit(-0.20, "cm"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) + 
    geom_errorbar(aes(ymin = lower_credible_interval, 
                      ymax = upper_credible_interval), width=.04)
  
  if (show_plot) {
    plot
  }

  # Save the plot as a png
  ggsave(filename="../figures/choice_estimates.png", plot=plot)
}

plot_slope_intercept_condition_estimates = function() {
  model <-readRDS("../models/choice_condition.rds")
  
  fixed_effect <- summary(model)$fixed
  fixed_effect <-rename(fixed_effect, lower_credible_interval = "l-95% CI")
  fixed_effect <-rename(fixed_effect, upper_credible_interval = "u-95% CI")
  
  print("here")

  print(fixed_effect) 
  
  intercept_df <- fixed_effect %>% slice(seq(1, n()/2))
  intercept_df$condition <- factor(c("RR", "RS", "SR", "SS"), 
                      levels=c("RS", "SR", "RR", "SS"))
  
  intercept_plot <- ggplot(intercept_df, aes(condition, Estimate)) +
    scale_y_continuous(sec.axis = dup_axis(), limits = c(-0.4, 0.5),
                       breaks = seq(-0.4, 0.5, 0.1),
                       labels = scales::number_format(accuracy = 0.1)) + 
    theme_bw() +
    theme(axis.text = element_text(face="bold"), 
          axis.ticks.length = unit(-0.10, "cm"), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), 
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) +
    labs(y = 'Intercept', x = '') + geom_point() +
    geom_errorbar(aes(ymin = lower_credible_interval, 
                      ymax = upper_credible_interval), width=.04)
  
  slope_df <- fixed_effect %>% slice(seq(n()/2 + 1, n()))
  slope_df$condition <- factor(c("RR", "RS", "SR", "SS"), 
                               levels=c("RS", "SR", "RR", "SS"))

  
  slope_plot <- ggplot(slope_df, aes(condition, Estimate)) + 
    scale_y_continuous(sec.axis = dup_axis(), limits = c(0.9, 2.3),
                       breaks = seq(0.7, 2.3, 0.2)) + theme_bw() +
    theme(axis.text = element_text(face="bold"), 
          axis.ticks.length = unit(-0.10, "cm"), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), 
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) +
    labs(y = 'Slope', x = '') + geom_point() +
    geom_errorbar(aes(ymin = lower_credible_interval, 
                      ymax = upper_credible_interval), width=.04)
  
  require(gridExtra)
  combined_plot <- grid.arrange(intercept_plot, slope_plot, ncol=2)
  
  plot
  
  ggsave(filename="../figures/slope_intercept_condition_estimates.png", 
         plot=combined_plot)
}


plot_stimulus_onset_behavioral_plot = function(show_plot) {
  model <-readRDS("../models/stimulus_evoked_model_brm.rds")
  print(summary(model))

  fixed_effect <- tail(summary(model)$fixed, 11)
  fixed_effect <-rename(fixed_effect, lower_credible_interval = "l-95% CI")
  fixed_effect <-rename(fixed_effect, upper_credible_interval = "u-95% CI")
  
  plot <- ggplot(fixed_effect, aes(factor(rownames(fixed_effect), 
                                          rownames(fixed_effect)), Estimate)) +
    # scale_x_discrete(labels=c('|V|', '|RU|', '|V|/TU')) +
    # scale_y_continuous(sec.axis = dup_axis(), breaks = seq(0, 3.4, 0.2)) +
    labs(y = 'Regression coefficient', x = '') + geom_point(size=4) + theme_bw() +
    theme(axis.text = element_text(face="bold"),
          axis.ticks.length = unit(-0.20, "cm"),
          panel.grid.major = element_blank(),  
          panel.grid.minor = element_blank(),
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) + 
    geom_errorbar(aes(ymin = lower_credible_interval, 
                      ymax = upper_credible_interval), width=.04)
  
  if (show_plot) {
    plot
  }
  
}

plot_feedback_model_results = function(model_name) {
  model <- readRDS(paste("../models/", model_name, ".rds", sep = ""))
  fixed_effect <- summary(model)$fixed
  fixed_effect <- tail(fixed_effect, nrow(fixed_effect) - 1)
  fixed_effect <-rename(fixed_effect, lower_credible_interval = "l-95% CI")
  fixed_effect <-rename(fixed_effect, upper_credible_interval = "u-95% CI")
  
  plot <- ggplot(fixed_effect, aes(factor(rownames(fixed_effect), 
                                          rownames(fixed_effect)), Estimate)) +
    geom_point(size=4) + labs(y = 'Regression coefficient', x = '') + 
    theme_bw() + theme(axis.text = element_text(face="bold"),
          axis.ticks.length = unit(-0.20, "cm"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) +
    geom_errorbar(aes(ymin = lower_credible_interval,
                      ymax = upper_credible_interval), width=.04) +
    geom_hline(yintercept=0,linetype=2)

  
  print(plot)
  ggsave(filename = paste("../figures/", model_name, ".png", sep = ""),
         plot = plot)
}

plot_all_feedback_model_results = function() {
  for (model_name in c("feedback_prior_belief", "feedback_posterior_belief", 
                  "feedback_directed_belief")) {
    print(model_name)
    plot_feedback_model_results(model_name)
  }
}

create_all_plots = function(show_plots) {
  plot_choice_estimates_brm(show_plots)
  plot_slope_intercept_condition_estimates_brm(show_plots)
}


analyze_data = function() {
  combine_and_save_all_data()
  run_hypothesis_testing()
  create_all_plots()
}


# print(summary(model))

# plot_min_max_stimulus_evoked_pupil = function() {
#   # also do a T test against all the
# }

# plot_all_feedback_model_results()

# combine_and_save_all_data()
# data = load_in_data()
# run_and_save_brm_regressors()
# what_is_happening_with_the_pupil_data()

#TODO: parameter recovery on your brm model (try the glm as well for the F test results)
#TODO: do a decoded RU value to predict choice behavior
# TODO: note that all the plots will need new model once they are run since
# the model names have changed
# plot_slope_intercept_condition_estimates()
# plot_stimulus_onset_behavioral_plot(TRUE)
# plot_slope_intercept_condition_estimates(TRUE)
# plot_choice_estimates(TRUE)