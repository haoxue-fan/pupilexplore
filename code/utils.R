## Utility functions for pupil project
## Written by Haoxue Fan and Taylor Burke

# load packages -----------------------------------------------------------

if(!require("pacman")) install.packages("pacman") #Install pacman to facilitate package installing/loading
p_load(lme4, hash, rstan, bayesplot, devtools, useful, gtools, brms, multcomp, lmtest, plm, tidyverse, grid, caret, tidybayes, dplyr, ggplot2, Hmisc, parallel, doParallel, ggpubr, data.table)

# we recommend running this is a fresh R session or restarting your current session
Sys.setenv(CMDSTANR_NO_VER_CHECK=TRUE)
if(!require("cmdstanr")){
install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))}
library(cmdstanr)


# parallel computing ------------------------------------------------------

# set up parallel computing (local)
numCores <- detectCores()
registerDoParallel(numCores)

# function ----------------------------------------------------------------

load_data <- function(type='first_row'){
  #' generic function to load pupil data (preprocessed using pypil). Primarily used in plotting
  print(type)
  if(type=='long'){ # warning: this may run slow
    long_file_dir = '../Data_AFTER/merged_pupil_long_AFTER/'
    long_list <- list()
    idx <- 1
    for (long_file in list.files(long_file_dir)){
      if (idx <= 100){
        long_list[[idx]] <- fread(paste0(long_file_dir, long_file))}
      else{
        long_list[[idx]] <- data.frame()  
      }
      print(idx)
      idx <- idx + 1
    }
    # turn the list into a huge data frame
    long_df <- as.data.frame(do.call(rbind, long_list))
    return(long_df)
  }
  
  if(type=='early_keypress'){
    idx <- 1
    extra_df <- data.frame()
    for (long_file in list.files(long_file_dir)){
      if (idx <= 100){
        loop_df <- read.csv(paste0(long_file_dir, long_file))
        extra_df <- rbind(extra_df, 
                          loop_df %>% group_by(identifier) %>% filter(row_number()==1))
      }
      else{
        long_list[[idx]] <- data.frame()  
      }
      print(idx)
      idx <- idx + 1
    }
    # first turn the timestamp to be positive
    # have a column saying when the early key press happens (during fixation)
    extra_df <- extra_df %>% mutate(across(starts_with('timestamp_locked_at'), function(x){return(x*(-1))}),
                                    early_press_stage=ifelse(timestamp_locked_at_early_key_press <= timestamp_locked_at_stimulus_pre_green_fixation_onset, 'real_early', 'invalid'))
    
    return(extra_df) 
  }
  
  if(type=='first_row'){
    first_row_dir = '../Data_AFTER/merged_pupil_first_row_AFTER/'
    first_row_df <- data.frame()
    idx <- 1
    for (first_row_file in list.files(first_row_dir)){
      print(idx)
      first_row_df <- rbind(first_row_df, read.csv(paste0(first_row_dir, first_row_file)))
      idx <- idx + 1
    }
    return(first_row_df)
  }
  
  if(type=='task'){
    task_dir = '../Data_AFTER/task_AFTER/'
    task_df <- data.frame()
    idx <- 1
    for (task_file in list.files(task_dir)){
      print(idx)
      task_df <- rbind(task_df, read.csv(paste0(task_dir, task_file)))
      idx <- idx + 1
    }
    return(task_df)
  }
  
  if(type=='task_posterior'){
    task_dir = '../Data_AFTER/belief_update_per_sub/'
    task_df <- data.frame()
    idx <- 1
    for (task_file in list.files(task_dir)){
      print(idx)
      task_df <- rbind(task_df, read.csv(paste0(task_dir, task_file)))
      idx <- idx + 1
    }
    return(task_df)
  }
  
  if(type=='demo'){
    demo_dir = '../Data_AFTER/Cleaned_Qualtrics_Data_Updated.csv'
    demo_df <- read.csv(demo_dir)
    return(demo_df)
  }
}

load_in_data = function() {
  #' load in preprocessed data (each row corresponds to pupil and choice data per trial)
  data <- read.csv("../data/combined_data.csv")
  
  data <- data %>% group_by(subject) %>% 
    mutate(trial_baseline_mean = mean(trial_baseline),
           trial_baseline_z_withinsub = scale(trial_baseline),
           trial_baseline_nomean = trial_baseline - trial_baseline_mean)
  data$abs_v <- abs(data$v)
  data$abs_ru <- abs(data$ru)
  data$v_over_tu <- data$v / data$tu
  data$abs_v_over_tu <- abs(data$v_over_tu)
  
  # Encoding directed behavior
  # By definition, V = left_V - right_V, RU = left_uncertainty - right_uncertainty
  # Therefore, all directed values of V and RU stay the same when a person 
  # selects the left arm (when choice = 1). However, when a person selects the
  # right arm (choice = 0), the V and RU calculations reverse, so you can 
  # simply flip the sign for V and RU, respectively. This logic is repeated
  # later down with the posterior values.
  data$directed_v <- with(data, ifelse(choice == 0, -v, v))
  data$directed_ru <- with(data, ifelse(choice == 0, -ru, ru))
  data$directed_v_over_tu <- data$directed_v / data$tu
  
  data$stimulus_evoked_avg_pupil_with_baseline <- data$stimulus_evoked_avg_pupil_without_baseline + data$trial_baseline
  data$feedback_evoked_avg_pupil_with_baseline <- data$feedback_evoked_avg_pupil_without_baseline + data$trial_baseline
  
  data$stimulus_evoked_avg_pupil_with_baseline_nomean = data$stimulus_evoked_avg_pupil_with_baseline - data$trial_baseline_mean
  
  # Add in all the posterior data columns
  data <- data %>% group_by(subject, block) %>%
    arrange(subject, block, trial) %>% 
    mutate(next_choice = dplyr::lead(choice))
  
  # Need to group to ungroup again
  data <- data %>% group_by(subject, block) %>% ungroup() 
  
  # Add in the posterior directed belief directed data based on the next 
  # trial's choice
  data$directed_post_v <- with(data, ifelse(next_choice == 0, -post_v, post_v))
  data$directed_post_ru <- with(data, ifelse(next_choice == 0, -post_ru, post_ru))
  data$directed_post_v_over_tu <- data$directed_post_v / data$post_tu
  
  data$post_abs_v <- abs(data$post_v)
  data$post_abs_ru <- abs(data$post_ru)
  data$post_abs_v_over_tu <- data$post_abs_v / data$tu
  
  return(data)
}


combine_and_save_all_data = function() {
  #' generate combined_data.csv
  
  data_cols = c('subject', 'id', 'block', 'trial', 'condition', 'choice', 
                'early_key_press', 'v', 'ru', 'tu', 'trial_baseline', 
                'stimulus_evoked_avg_pupil', 'feedback_evoked_avg_pupil')
  
  combined_data = data.frame(matrix(nrow = 0, ncol = length(data_cols))) 
  colnames(combined_data) = data_cols
  
  data_file_paths = list.files(path="../data/merged_pupil_first_row_AFTER", 
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

get_a_subjects_data = function(file_path) {
  data = read.csv(file_path)
  
  data$Chosen_Arm_Value[data$Chosen_Arm_Value == ""] <- NaN
  if (sum(is.na(data$Chosen_Arm_Value)) > 32) {
    print(sprintf("excluding %s on missing more than 20% of trials across all
                  blocks", 
                  unique(data$SubjectID)))
    return(NaN)
  }
  
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
  subject_pupil_data <- read.csv(paste("../data/merged_pupil_long_AFTER/", 
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
                                                         early_key_press = "early_key_press" %in% unique(event)))
  
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

# exclude perc_miss from first_row_df
# I now do not know whether i am using this or filter? which function I am interested?

calc_decode_ridge <- function(Y, X, beta, beta_intercept, the_beta, lambda ){
  #' decode var from existing regression. support ridge (lambda != 0)
  
  output <- (Y - rowSums(X * rep(beta %>% as.matrix(), each=nrow(X))) - beta_intercept) * the_beta / (the_beta^2 + lambda)
  return(output)
}

convert_decode_var <- function(data, prefix, type='tu'){
  #' transform decoded var to var used in downstream regression
  
  old_col_name = paste0(prefix, '_', type)
  data$working_col = data[old_col_name] %>% as.matrix() %>% as.numeric()
  if (type == 'tu'){
    data <- data %>% 
      mutate(
        # which is the one that we have used? can not remember
        prefix_vtu = v / working_col,
        prefix_tu_cutoff = ifelse(working_col > 0, working_col, min(data[type])),
        prefix_vtu_cutoff = v / prefix_tu_cutoff,
        prefix_vtu_cutoff_z = scale(prefix_vtu_cutoff),
      )
  }
  # for abs RU: calculate RU
  if (type == 'abs_ru'){
    data <- data %>% 
      mutate(
        prefix_ru = ifelse(ru > 0, working_col, -working_col),
        prefix_ru_z = scale(prefix_ru)
      )
  }
  # for chosen RU: calculate RU
  if (type == 'chosen_ru'){
    data <- data %>% 
      mutate(
        prefix_ru = ifelse(choice == 1, working_col, -working_col),
        prefix_ru_z = scale(prefix_ru)
      )
  }
  if (type == 'chosen_v'){
    data <- data %>% 
      mutate(
        prefix_v = ifelse(choice == 1, working_col, -working_col),
        prefix_v_z = scale(prefix_v)
      )
  }
  
  # if the columns already existed, drop them
  output_col_names <- str_replace(names(data)[str_detect(names(data), 'prefix')], 'prefix', prefix) 
  data <- data[, !names(data) %in% output_col_names]
  # change prefix to real prefix
  names(data)[str_detect(names(data), 'prefix')] <- output_col_names
  # drop the working col at the end
  data <- data %>% dplyr::select(-working_col)
  return(data)
  # other type can be added later. scaling is also not present yet.
}

run_condition_behavior_hypothesis_testing_brm = function() {
  model <- readRDS("../models/choice_condition.rds")
  
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
  hypotheses = c("conditionRS - conditionSR = 0",
                 "conditionRR - conditionSS = 0",
                 "conditionRS:v - conditionSR:v = 0",
                 "conditionRR:v - conditionSS:v = 0")
  for (hypothesis in hypotheses) {
    hypothesis_test_results = hypothesis(model, hypothesis)
    plot(hypothesis_test_results)
    print(hypothesis_test_results)
  }
}

run_and_save_models = function(print_results, models_to_run, 
                               sufix_for_file_name = NULL, data_input=NULL, 
                               criterion_type=NULL, iters = 2000) {
  #' run and save models
  
  if (is.null(data_input)){
    data = load_in_data()
    data = filter_data(data)
    data$v_z <- scale(data$v)
    data$ru_z <- scale(data$ru)
    data$v_over_tu_z <- scale(data$v_over_tu)
    #data <- data %>% group_by(subject) %>% 
    #  mutate(
    #    trial_baseline_mean_new = mean(trial_baseline),
    #    trial_baseline_nomean_new = trial_baseline - trial_baseline_mean_new
    #  ) %>% ungroup()
  }else{
    data = data_input
  }
  # after filtering: 
  ## scale v, ru and vtu (to use in decoding)
  
  
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
      init <- 0
    } else {
      family <- gaussian
      init <- NULL
    }
    
    model <- brm(formula = formula, data = data, family = family, cores = 4, 
                 chains = 4, iter = iters, warmup = floor(iters/2), init = init, 
                 backend = "cmdstanr")
    if (!is.null(criterion_type)){
      model = model %>% add_criterion(criterion_type)
    }
    if (print_results) {
      print(summary(model))
    }
    saveRDS(model, paste("../models/", model_to_run,sufix_for_file_name, ".rds", 
                         sep = ""))
  }
}

# exclude perc_miss from first_row_df
proc_exclude <- function(data){
  if(!'trial_baseline_z' %in% names(data)){
    data <- data %>% group_by(ID) %>% mutate(trial_baseline_z_within_sub = scale(trial_baseline))
  }
  # put all the exclusion criteria here
  print(paste0('before exclusion, nrow=',nrow(data)))
  good_data <- data %>% 
    filter(perc_miss <= .4) %>% 
    filter(!is.na(trial_baseline)) %>% 
    filter(abs(trial_baseline_z_within_sub) <= 3) %>% 
    filter(!is.na(Chosen_Arm_Value))
  print(paste0('after exclusion, nrow=',nrow(good_data)))
  return(good_data)
}

filter_data <- function(data){
  #' framework to do preprocess & verbose the process
  
  # missing raw data for more than 40% of the time
  print('Criteria: remove trials with NA trial baseline')
  filtered <- data %>% filter(is.na(trial_baseline)) 
  print(paste0('nrow = ', nrow(filtered), ', ', 
               round(nrow(filtered)/nrow(data),4) * 100, '%'))
  print(paste0('nsub = ', length(unique(filtered$subject)), ', ', 
               round(length(unique(filtered$subject))/nrow(data),4) * 100, 
               '% of all subjects'))
  
  print('Criteria: remove trials with > 40% missing raw data')
  filtered <- data %>% filter(perc_miss > .4) 
  print(paste0('nrow = ', nrow(filtered), ', ', 
               round(nrow(filtered)/nrow(data),4) * 100, '% of all trials'))
  print(paste0('nsub = ', length(unique(filtered$subject)), ', ', 
               round(length(unique(filtered$subject))/nrow(data),4) * 100, 
               '% of all subjects'))
  
  print('Criteria: remove trials where no choice was made')
  filtered <- data %>% filter(is.na(choice)) 
  print(paste0('nrow = ', nrow(filtered), ', ', 
               round(nrow(filtered)/nrow(data),4) * 100, '% of all trials'))
  print(paste0('nsub = ', length(unique(filtered$subject)), ', ', 
               round(length(unique(filtered$subject))/nrow(data),4) * 100, 
               '% of all subjects'))
  
  print('Criteria: remove trials where trial baseline is outside of 3SD')
  filtered <- data %>% filter(abs(trial_baseline_z_withinsub) > 3) 
  print(paste0('nrow = ', nrow(filtered), ', ', 
               round(nrow(filtered)/nrow(data),4) * 100, '% of all trials'))
  print(paste0('nsub = ', length(unique(filtered$subject)), ', ', 
               round(length(unique(filtered$subject))/nrow(data),4) * 100, 
               '% of all subjects'))
  
  print('Note: Criteria remove participant whose correct rate < 60%')
  data <- data %>% 
    mutate(
      correct = ((Left_Slot_Actual_Mean >= Right_Slot_Actual_Mean) & 
                   choice == 1) | 
        ((Left_Slot_Actual_Mean <= Right_Slot_Actual_Mean) & choice == 0)) %>% 
    group_by(subject) %>% 
    mutate(perc_correct = mean(correct, na.rm=TRUE))
  filtered <- data %>% filter(perc_correct < 0.6)  
  print(paste0('nrow = ', nrow(filtered), ', ', 
               round(nrow(filtered)/nrow(data),4) * 100, '% of all trials'))
  print(paste0('nsub = ', length(unique(filtered$subject)), ', ', 
               round(length(unique(filtered$subject))/nrow(data),4) * 100, 
               '% of all subjects'))
  
  data <- data %>% 
    filter(!is.na(trial_baseline)) %>% 
    filter(perc_miss <= .4) %>% 
    filter(!is.na(choice)) %>% 
    filter(abs(trial_baseline_z_withinsub) <= 3) %>% 
    filter(perc_correct >= 0.6) %>% 
    ungroup()
  
  print(paste0('Filtered data: nrow = ', nrow(data)))
  
  return (data)
}


augment_model_with_multiple_periods = function(print_results, multiple_period_models, 
                                               vars_to_include, 
                                               new_models_to_run = augment_model_multiple, sufix_for_augmented_model='') {
  # This function is not in active use for now. Still keep it here as a reference
  
  # example use syntax: augment_model_with_multiple_periods(TRUE, paste0('decode_VRUTU',c(baseline_models[1], evaluation_models[1])), c('decode_ru_z','decode_v_over_tu_z'), double_models_noV, sufix_for_augmented_model='directed_baseline_evaluation')
  
  idx <- 1
  for(multiple_period_model in multiple_period_models) {
    print(paste("Using variables present in :", multiple_period_model, sep = " "))
    
    
    # load in the fitted model
    model <- readRDS(paste("../models/", multiple_period_model, ".rds", sep = ""))
    # decode var
    if (idx==1){
      data <- model$data
    }
    
    for(var_to_include in vars_to_include){
      
      if (nrow(data) != nrow(model$data)){
        Error('The data dimension extracted from different models is not the same!')
      }
      model_data = model$data
      data[paste0(var_to_include,'_',idx)] = model_data[var_to_include]
      
    }
    idx = idx + 1
    # run decode model
    
  }
  print(data %>% names)
  run_and_save_models(TRUE, new_models_to_run,
                      sufix_for_augmented_model, data_input=data)
}

decode_from_models = function(print_results, models_to_decode_from, 
                              vars_to_decode, lambda = 1, 
                              new_models_to_run = decode_models) {
  #' wrapper to decode a list of var from a list of models
  
  data <- load_in_data()
  data <- filter_data(data)
  data$v_z <- scale(data$v)
  data$ru_z <- scale(data$ru)
  data$v_over_tu_z <- scale(data$v_over_tu)

  for(model_to_decode_from in models_to_decode_from) {
    print(paste("Decoding from:", model_to_decode_from, sep = " "))
    
    if (!has.key(model_to_decode_from, ALL_MODEL_DESCRIPTIONS)) {
      print(paste(model_to_decode_from, "is not a valid model name", sep = " "))
      next
    }
    # load in the fitted model
    model <- readRDS(paste("../models/", model_to_decode_from, ".rds", sep = ""))
    # decode var
    for(var_to_decode in vars_to_decode){
      data[paste0('decode_', var_to_decode)] = decode_sub_specific(
        model, data, var_to_decode, lambda)
      data = convert_decode_var(data, prefix = 'decode', type = var_to_decode)
    }
    # run decode model
    run_and_save_models(TRUE, new_models_to_run,
                        model_to_decode_from, data_input=data)
  }
}

decode_sub_specific = function(model, data, decode_name, lambda){
  #' decode from mixed model using random effects
  
  curr_est <- coef(model)$subject[,1,] %>% as.matrix()
  y_name <- formula.tools::lhs.vars(as.formula(model$formula))
  intercept_name <- 'Intercept'
  other_name <- colnames(curr_est)[decode_name != colnames(curr_est) & 
                                     intercept_name != colnames(curr_est)]
  data$output <- rep(NaN, each=nrow(data))
  for(i in c(1:nrow(curr_est))){
    loop_est <- curr_est[i,]
    data_loop <- data[data$subject == rownames(curr_est)[i],]
    data$output[data$subject == rownames(curr_est)[i]] <- calc_decode_ridge(
      data_loop[y_name], data_loop[other_name], loop_est[other_name], 
      loop_est[intercept_name], loop_est[decode_name], lambda) %>% as.matrix()
  }
  return(data$output)
}

output_criterion = function(models_to_add_criterion, sufixs_for_file_name, output=NULL, show_model_coef=TRUE, mdl_dir='../models/'){
  #' read in model fit metric from existing model var
  
  idx <- 1
  for (sufix_for_file_name in sufixs_for_file_name){
    for (model_to_add_criterion in models_to_add_criterion){
      model <- readRDS(paste(mdl_dir, model_to_add_criterion, sufix_for_file_name, ".rds", sep = ""))
      
      print(paste0(sufix_for_file_name, '-',model_to_add_criterion, ': ',
                   round((model$criteria$loo$estimates)[3,1], 3)
      ))
      if (show_model_coef){
        print(model)
      }
      if (!is.null(output)){
        output[idx] <- model$criteria
        idx <- idx + 1
      }
    }
  }
  if (!is.null(output)){
    return(output)
  }
}

add_criterion_to_model = function(show_result, models_to_add_criterion, 
                                  sufixs_for_file_name, criterion_type=NULL){
  # sufix for file name: indicate which model the decode variables came from
  #' add criterion of given type and resave the models (primarily used for decoded models)
  
  if (!is.null(criterion_type)){
    for (sufix_for_file_name in sufixs_for_file_name){
      print(paste0('loop through decoded models from: ', sufix_for_file_name))
      for (model_to_add_criterion in models_to_add_criterion){
        print(paste0('adding criteria for model: ', model_to_add_criterion))
        model <- readRDS(paste("../models/", model_to_add_criterion, 
                               sufix_for_file_name, ".rds", sep = ""))
        model <- model %>% add_criterion(criterion_type)
        if (show_result){
          print(model$criteria)
        }
        saveRDS(model, paste("../models/", model_to_add_criterion, 
                             sufix_for_file_name, ".rds", 
                             sep = ""))
      }
    }
  }
}


plot_choice_estimates_no_intercept = function(model, show_plot=TRUE) {
  #' generic plot func for bayesian lme (no intercept)
  
  
  results = c()
  
  fixed_effect <- summary(model)$fixed[rownames(summary(model)$fixed) != 'Intercept',]
  fixed_effect <-rename(fixed_effect, lower_credible_interval = "l-95% CI")
  fixed_effect <-rename(fixed_effect, upper_credible_interval = "u-95% CI")
  fixed_effect <- fixed_effect %>% filter(Estimate <= 100)
  print(fixed_effect)
  
  plot <- ggplot(fixed_effect, aes(factor(rownames(fixed_effect), 
                                          rownames(fixed_effect)), Estimate)) +
    labs(y = 'Regression coefficient', x = '') + geom_point(size=4) + pupilexplore_theme +
    geom_errorbar(aes(ymin = lower_credible_interval, 
                      ymax = upper_credible_interval), width=.04) +
    geom_hline(yintercept=0, linetype='dashed')
  
  if (show_plot) {
    plot
  }
}

plot_slope_intercept_condition_estimates = function() {
  model <-readRDS("../models/choice_condition.rds")
  
  fixed_effect <- summary(model)$fixed
  fixed_effect <-rename(fixed_effect, lower_credible_interval = "l-95% CI")
  fixed_effect <-rename(fixed_effect, upper_credible_interval = "u-95% CI")
  
  print(fixed_effect) 
  
  intercept_df <- fixed_effect %>% slice(seq(1, n()/2))
  intercept_df$condition <- factor(c("RR", "RS", "SR", "SS"), 
                                   levels=c("RS", "SR", "RR", "SS"))
  
  intercept_plot <- ggplot(intercept_df, aes(condition, Estimate)) +
    scale_y_continuous(limits = c(-0.41, 0.5),
                       breaks = seq(-0.4, 0.5, 0.2),
                       labels = scales::number_format(accuracy = 0.1)) +
    labs(y = 'Intercept', x = '') + geom_point(size=2.5) +
    geom_errorbar(aes(ymin = lower_credible_interval, 
                      ymax = upper_credible_interval), width=.1, size=0.4) 
  
  
  
  slope_df <- fixed_effect %>% slice(seq(n()/2 + 1, n()))
  slope_df$condition <- factor(c("conditionRR", "conditionRS", "conditionSR", "conditionSS"), 
                               levels=c("conditionRS", "conditionSR", "conditionRR", "conditionSS"),
                               labels=c("RS", "SR", "RR", "SS"))
  
  
  slope_plot <- ggplot(slope_df, aes(condition, Estimate)) + 
    scale_y_continuous(limits = c(0.1, 0.3),
                       breaks = seq(0.1, 0.3, 0.05)) + 
    labs(y = 'Slope', x = '') + geom_point(size=2.5) +
    geom_errorbar(aes(ymin = lower_credible_interval, 
                      ymax = upper_credible_interval), width=.1, size=0.4) 
  
  
  require(gridExtra)
  combined_plot <- grid.arrange(intercept_plot+pupilexplore_theme, slope_plot+pupilexplore_theme, ncol=2)
  
  plot
  
  ggsave(filename="../figures/slope_intercept_condition_estimates.png", 
         plot=combined_plot, width=8, height=4)
}

add_line <- function(p, xlab_name='Time from Stimulus Onset', ylab_name='Pupil Size'){
  p1 <- p + xlab(xlab_name) +
    ylab(ylab_name) +
    geom_vline(xintercept = -1000, linetype = 'dotted') +
    geom_vline(xintercept = 0, linetype = 'dotted') +
    geom_vline(xintercept = 1000, linetype = 'dotted') +
    geom_vline(xintercept = 3000, linetype = 'dotted') +
    geom_vline(xintercept = 4500, linetype = 'dotted') +
    narrative_theme
  return(p1)
}
# var ---------------------------------------------------------------------

pupilexplore_theme <- theme_bw() +
  theme(axis.text = element_text(face="bold"),
        axis.ticks.length = unit(-0.20, "cm"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x.top = element_blank(),
        axis.text.y.right = element_blank(),
        axis.title.x.top = element_blank(),
        axis.title.y.right = element_blank(),
        axis.text.x = element_text(size=10),
        axis.text.y = element_text(size=15),
        axis.title.y = element_text(size=15),
  )

narrative_theme <-   theme_pubr(legend = "bottom")+
  theme(text = element_text(size=18),
        axis.text=element_text(size=18),
        legend.text = element_text(size = 15),
        axis.line = element_line(size = 0.7),
        legend.margin = margin(0, -10, 0, 0),
        legend.box.margin=margin(-15,-10,0,-10),
        legend.key.size = unit(2,"line"))

# Condition Model (Trial Baseline)

ALL_MODEL_DESCRIPTIONS <- hash::hash()

# Choice Model 1 (the not standardized version does not esaily converge)
ALL_MODEL_DESCRIPTIONS["choice_prior_belief"] = "choice ~ -1 + v_z + ru_z + 
v_over_tu_z + (-1 + v_z + ru_z + v_over_tu_z | subject);"

# Choice Model 2
ALL_MODEL_DESCRIPTIONS["choice_condition"] = "choice ~ -1 + condition + 
condition:v + (-1 + condition + condition:v | subject)"

# Baseline Model 0
ALL_MODEL_DESCRIPTIONS['baseline_baseline'] = "
trial_baseline_nomean ~ 1 + (1 | subject)"

# Baseline Model 1
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief'] = "
trial_baseline_nomean ~ abs_v + abs_ru + 
tu + abs_v_over_tu + (abs_v + abs_ru + tu + 
abs_v_over_tu | subject)"

# Baseline Model 2
ALL_MODEL_DESCRIPTIONS['baseline_directed_prior_belief'] = "
trial_baseline_nomean ~ directed_v + directed_ru + 
tu + directed_v_over_tu + (directed_v + directed_ru + tu + 
directed_v_over_tu | subject)"

# Baseline Model 1 - V
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_V'] = "
trial_baseline_nomean ~ abs_v + (abs_v | subject)"

# Baseline Model 1 - RU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_RU'] = "
trial_baseline_nomean ~ abs_ru + (abs_ru | subject)"

# Baseline Model 1 - TU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_TU'] = "
trial_baseline_nomean ~ tu + (tu | subject)"

# Baseline Model 1 - VTU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_VTU'] = "
trial_baseline_nomean ~ abs_v_over_tu + (abs_v_over_tu | subject)"

# Baseline Model 1 - TU + V
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_TU_V'] = "
trial_baseline_nomean ~ abs_v + 
tu + (abs_v + tu | subject)"

# Baseline Model 1 - TU + RU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_TU_RU'] = "
trial_baseline_nomean ~ abs_ru + 
tu + (abs_ru + tu | subject)"

# Baseline Model 1 - TU + VTU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_TU_VTU'] = "
trial_baseline_nomean ~ 
tu + abs_v_over_tu + (tu + 
abs_v_over_tu | subject)"

# Baseline Model 1 - V + RU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_RU_V'] = "
trial_baseline_nomean ~ 
abs_v + abs_ru + (abs_v + abs_ru | subject)"

# Baseline Model 1 - V + VTU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_VTU_V'] = "
trial_baseline_nomean ~ 
abs_v + abs_v_over_tu + (abs_v + abs_v_over_tu | subject)"

# Baseline Model 1 - RU + VTU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_VTU_RU'] = "
trial_baseline_nomean ~ 
abs_ru + abs_v_over_tu + (abs_ru + abs_v_over_tu | subject)"

# Baseline Model 1 - TU + VTU + V
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_TU_VTU_V'] = "
trial_baseline_nomean ~ abs_v +  
tu + abs_v_over_tu + (abs_v + tu + 
abs_v_over_tu | subject)"

# Baseline Model 1 - TU + VTU + RU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_TU_VTU_RU'] = "
trial_baseline_nomean ~ abs_ru + 
tu + abs_v_over_tu + (abs_ru + tu + 
abs_v_over_tu | subject)"

# Baseline Model 1 - V + VTU + RU
ALL_MODEL_DESCRIPTIONS['baseline_prior_belief_VTU_RU_V'] = "
trial_baseline_nomean ~ abs_v + 
tu + abs_v_over_tu + (abs_v + tu + 
abs_v_over_tu | subject)"

# Baseline Model 2
ALL_MODEL_DESCRIPTIONS['baseline_directed_prior_belief'] = "
trial_baseline_nomean ~ directed_v + directed_ru + 
tu + directed_v_over_tu + (directed_v + directed_ru + tu + 
directed_v_over_tu | subject)"

# Baseline Model 2 - V
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_V'] = "
trial_baseline_nomean ~ directed_v + (directed_v | subject)"

# Baseline Model 2 - RU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_RU'] = "
trial_baseline_nomean ~ directed_ru + (directed_ru | subject)"

# Baseline Model 2 - TU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_TU'] = "
trial_baseline_nomean ~ tu + (tu | subject)"

# Baseline Model 2 - VTU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_VTU'] = "
trial_baseline_nomean ~ directed_v_over_tu + (directed_v_over_tu | subject)"

# Baseline Model 2 - TU + V
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_TU_V'] = "
trial_baseline_nomean ~ directed_v + 
tu + (directed_v + tu | subject)"

# Baseline Model 2 - TU + RU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_TU_RU'] = "
trial_baseline_nomean ~ directed_ru + 
tu + (directed_ru + tu | subject)"

# Baseline Model 2 - TU + VTU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_TU_VTU'] = "
trial_baseline_nomean ~ 
tu + directed_v_over_tu + (tu + 
directed_v_over_tu | subject)"

# Baseline Model 2 - V + RU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_RU_V'] = "
trial_baseline_nomean ~ 
directed_v + directed_ru + (directed_v + directed_ru | subject)"

# Baseline Model 2 - V + VTU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_VTU_V'] = "
trial_baseline_nomean ~ 
directed_v + directed_v_over_tu + (directed_v + directed_v_over_tu | subject)"

# Baseline Model 2 - RU + VTU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_VTU_RU'] = "
trial_baseline_nomean ~ 
directed_ru + directed_v_over_tu + (directed_ru + directed_v_over_tu | subject)"

# Baseline Model 2 - TU + VTU + V
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_TU_VTU_V'] = "
trial_baseline_nomean ~ directed_v +  
tu + directed_v_over_tu + (directed_v + tu + 
directed_v_over_tu | subject)"

# Baseline Model 2 - TU + VTU + RU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_TU_VTU_RU'] = "
trial_baseline_nomean ~ directed_ru + 
tu + directed_v_over_tu + (directed_ru + tu + 
directed_v_over_tu | subject)"

# Baseline Model 2 - V + VTU + RU
ALL_MODEL_DESCRIPTIONS['baseline_directed_belief_VTU_RU_V'] = "
trial_baseline_nomean ~ directed_v + 
tu + directed_v_over_tu + (directed_v + tu + 
directed_v_over_tu | subject)"

# Choice Model 1 - with intercept
# Note that here variables are standardized because the same thing has been done for the decoding part. It would not really change the model fit for the original choice_prior_belief one, but for the previous one we just use them on the original scale (since now we do not suffer from convergence issue)

ALL_MODEL_DESCRIPTIONS["choice_plain_z"] = "choice ~ v_z + ru_z + 
v_over_tu_z + (v_z + ru_z + v_over_tu_z | subject);"

# Choice Model Augmented 1 - decoded_RU 
ALL_MODEL_DESCRIPTIONS["decode_RU"] = "choice ~ v_z + ru_z + 
v_over_tu_z + decode_ru_z + (v_z + ru_z + v_over_tu_z + decode_ru_z  | subject);"

# Choice Model Augmented 2 - decoded_TU 
ALL_MODEL_DESCRIPTIONS["decode_TU"] = "choice ~ v_z + ru_z + 
v_over_tu_z + decode_v_over_tu_z + (v_z + ru_z + v_over_tu_z + decode_v_over_tu_z | subject);"

# Choice Model Augmented 3 - decoded_V
ALL_MODEL_DESCRIPTIONS["decode_V"] = "choice ~ v_z + ru_z + 
v_over_tu_z + decode_v_z + (v_z + ru_z + v_over_tu_z + decode_v_z | subject);"

# Choice Model Augmented 4 - decoded_RU + decoded_TU 
ALL_MODEL_DESCRIPTIONS["decode_RUTU"] = "choice ~ v_z + ru_z + 
v_over_tu_z + decode_ru_z + decode_v_over_tu_z + (v_z + ru_z + 
decode_ru_z + v_over_tu_z  + decode_v_over_tu_z | subject);"

# Choice Model Augmented 5 - decoded_RU + decoded_V
ALL_MODEL_DESCRIPTIONS["decode_VRU"] = "choice ~ v_z + ru_z + v_over_tu_z + 
decode_ru_z + decode_v_z + (v_z + ru_z + v_over_tu_z + decode_ru_z + 
decode_v_z | subject);"

# Choice Model Augmented 6 - decoded_V + decoded_TU 
ALL_MODEL_DESCRIPTIONS["decode_VTU"] = "choice ~ v_z + ru_z + v_over_tu_z + 
decode_v_z + decode_v_over_tu_z + (v_z + ru_z + decode_v_z + v_over_tu_z  + 
decode_v_over_tu_z | subject);"

# Choice Model Augmented 7 - decoded_V + decoded_TU 
ALL_MODEL_DESCRIPTIONS["decode_VRUTU"] = "choice ~ v_z + ru_z + v_over_tu_z + 
decode_v_z + decode_ru_z + decode_v_over_tu_z + (v_z + ru_z + v_over_tu_z  + 
decode_v_z + decode_ru_z + decode_v_over_tu_z | subject);"

all_models <- keys(ALL_MODEL_DESCRIPTIONS) # see model description in utils.R
choice_models <- base::Filter(function(model) grepl("choice", model, fixed = TRUE), all_models) # all models is defined in utils.R
baseline_models <- base::Filter(function(model) grepl("baseline", model, fixed = TRUE), all_models)
baseline_model_everything <- 'baseline_prior_belief'
decode_models <- base::Filter(function(model) grepl("decode", model, 
                                                    fixed = TRUE), all_models)
