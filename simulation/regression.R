library(dplyr)
library(multcomp)
library(lmtest)
library(plm)

# Set the working directory to where the file lives
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

get_experiment = function(experiment_file_path) {
  experiment = read.csv(experiment_file_path)
  
  # Add in the estimated value difference to the data file
  experiment$estimated_value_difference <- experiment$left_arm_estimate_mean - 
    experiment$right_arm_estimate_mean
    
  # Add in the relative uncertainty to the data file
  experiment$relative_uncertainty <- sqrt(experiment$left_arm_variance_in_estimate) - 
    sqrt(experiment$right_arm_variance_in_estimate)
    
  # Add in the total uncertainty to the data file
  experiment$total_uncertainty <- sqrt(experiment$left_arm_variance_in_estimate +
                                          experiment$right_arm_variance_in_estimate)
  
  # Add in estimated_value_difference over total_uncertainty to the data file
  experiment$value_over_total_uncertainty = experiment$estimated_value_difference / experiment$total_uncertainty
  
  experiment$choosing_left_arm <- with(experiment, ifelse(choice == 0, 1, 0))
  
  # Consider the condition and choosing the left arm data columns as factors
  # This ensures choosing the left arm and condition are treated as "dummy" 
  # variables and are considered categorical rather than numeric values
  experiment$choosing_left_arm <- as.factor(experiment$choosing_left_arm)
  experiment$condition <- as.factor(experiment$condition)

  return(experiment)
}

run_glm_regressor = function(formula, experiment, num_expected_coefs) {
  results = c()
  model = glm(formula = formula, data = experiment, binomial(link = "probit"))
  model_results = summary(model)
  coefficients = model_results$coefficients
  
  for (num_coefficient in (1:num_expected_coefs)) {
    coef_val = coefficients[num_coefficient, 1]
    p_value = coefficients[num_coefficient, 4]
    results = append(results, coef_val)
    results = append(results, p_value)
  }
  return(results)
}

regress_on_experiment = function(experiment) {
  results = c()

  formula = choosing_left_arm ~ -1 + estimated_value_difference + relative_uncertainty + value_over_total_uncertainty;
  results = append(results, run_glm_regressor(formula, experiment, 3))
  
  formula = choosing_left_arm ~ -1 + condition + condition:estimated_value_difference
  results = append(results, run_glm_regressor(formula, experiment, 8))
  
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
                 "conditionRS:estimated_value_difference - conditionSR:estimated_value_difference = 0", 
                 "conditionRR:estimated_value_difference - conditionSS:estimated_value_difference = 0")
  model = glm(formula = formula, data = experiment, binomial(link = "probit"))
  for (hypothesis in hypotheses) {
    hypothesis_test_results = summary(glht(model, linfct = hypothesis), test = Ftest())
    results = append(results, hypothesis_test_results$test$pvalue)
  }
  
  return(results)
}

regress_on_all_experiments = function(data_folder){
  first_save_data = TRUE
  for (exploration_strategy in list("Hybrid", "Thompson Sampling", "UCB")){
    for (num_blocks in list(16, 20, 24, 28)) {
      for (variance in list(25, 36, 49, 64, 81, 100)) {
        print(sprintf("working on %s_%i_%i", exploration_strategy, num_blocks, 
                      variance))
        exp_regression_results <- data.frame(exploration_strategy = character(0),
                                             num_blocks = integer(0),
                                             variance = integer(0),
                                             experiment = integer(0),
                                             # Results from the choice/uncertainty model
                                             v_coef = numeric(0), 
                                             v_p_val = numeric(0), 
                                             ru_coef = numeric(0), 
                                             ru_p_val = numeric(0), 
                                             vtu_coef = numeric(0), 
                                             vtu_p_val = numeric(0), 
                                             # Results from the choice/condition model
                                             rr_coef = numeric(0), 
                                             rr_p_val = numeric(0), 
                                             rs_coef = numeric(0), 
                                             rs_p_val = numeric(0), 
                                             sr_coef = numeric(0), 
                                             sr_p_val = numeric(0), 
                                             ss_coef = numeric(0), 
                                             ss_p_val = numeric(0), 
                                             rr_v_coef = numeric(0), 
                                             rr_v_p_val = numeric(0),
                                             rs_v_coef = numeric(0), 
                                             rs_v_p_val = numeric(0), 
                                             sr_v_coef = numeric(0), 
                                             sr_v_p_val = numeric(0), 
                                             ss_v_coef = numeric(0), 
                                             ss_v_p_val = numeric(0), 
                                             # Results from testing intercept/slope hypotheses
                                             rs_vs_sr_p_val = numeric(0),
                                             rr_vs_ss_p_val = numeric(0), 
                                             rs_v_vs_sr_v_p_val = numeric(0), 
                                             rr_v_vs_ss_v_p_val = numeric(0)
        )
        for (num_experiment in (1:10)) {
          experiment_file_path = sprintf("../%s/%s_%i_%i_data_%i.csv", 
                              data_folder,
                              exploration_strategy, 
                              num_blocks, variance, 
                              num_experiment)
          regression_results = regress_on_experiment(get_experiment(experiment_file_path))
          
          exp_regression_results[nrow(exp_regression_results) + 1, ] = c(exploration_strategy,
                                                       num_blocks, variance,
                                                       num_experiment,
                                                       regression_results)
        }
        
        if (first_save_data) {
          write.csv(exp_regression_results,"../results/regression_results.csv",
                    row.names = FALSE)
          first_save_data = FALSE
        } else {
          write.table(exp_regression_results,"../results/regression_results.csv",
                      append = TRUE, quote = FALSE, row.names = FALSE,
                      sep = ",", col.names = FALSE)
        }
      }
    }
  }
}

regress_on_all_experiments("data")