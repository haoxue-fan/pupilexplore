# All of these 

run_and_save_glme_regressors = function() {
  belief_update_data <- read.csv("../data/belief_update.csv")
  
  formula = Choice ~ -1 + V + RU + VOverTU +
    (-1 + V + RU + VOverTU | Subject);
  model = glmer(formula = formula, data = belief_update_data,
                binomial(link = "probit"),
                glmerControl(optimizer="bobyqa",
                             optCtrl = list(maxfun = 10000000)))
  saveRDS(model, "../models/choice_uncertainty_model.rds")
  
  formula = Choice ~ -1 + Condition + Condition:V +
    (-1 + Condition + Condition:V | Subject)
  model = glmer(formula = formula, data = belief_update_data,
                binomial(link = "probit"),
                glmerControl(optimizer="bobyqa",
                             optCtrl = list(maxfun = 10000000)))
  print(summary(model))
  saveRDS(model, "../models/choice_condition_model.rds")
}

run_hypothesis_testing = function() {
  model <- readRDS("../models/choice_condition_model.rds")
  
  print(summary(model))
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
    hypothesis_test_results = summary(glht(model, linfct = hypothesis), 
                                      test = Ftest())
    print(hypothesis_test_results)
  }
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

glm_regress_on_experiment = function(combined_data_file_path, 
                                     results_data_file_path) {
  data <- read.csv(combined_data_file_path)
  for (subject in 1:max(as.integer(data$Subject))){
    subject_data = subset(data, Subject == subject) 
    regression_results = c()
    
    formula = Choice ~ -1 + V + RU + VOverTU;
    regression_results = append(regression_results, 
                                run_glm_regressor(formula, subject_data, 3))
    
    formula = Choice ~ -1 + Condition + Condition:V;
    regression_results = append(regression_results, 
                                run_glm_regressor(formula, subject_data, 8))
    
    # Testing two intercept and two slope hypotheses:
    # (1) Relative uncertainty DOES alter the INTERCEPT of choice 
    #   probability (i.e. there is a significant difference between intercepts 
    #   of RS and SR conditions)
    # (2) Total uncertainty DOES NOT alter the INTERCEPT of choice 
    #   probability (i.e. there is no significant difference between intercepts 
    #   of RR and SS conditions)
    # (3) Relative uncertainty DOES NOT alter the SLOPE of choice 
    #   probability (i.e. there is no significant difference between intercepts 
    #   of RS and SR conditions when conditioned on estimated_value_difference) 
    # (4) Total uncertainty DOES alter the SLOPE of choice 
    #   probability (i.e. there is a significant difference between intercepts 
    #   of RR and SS conditions when conditioned on estimated_value_difference) 
    hypotheses = c("ConditionRS - ConditionSR = 0",
                   "ConditionRR - ConditionSS = 0",
                   "ConditionRS:V - ConditionSR:V = 0",
                   "ConditionRR:V - ConditionSS:V = 0")
    model = glm(formula = formula, data = subject_data, 
                binomial(link = "probit"))
    for (hypothesis in hypotheses) {
      hypothesis_test_results = summary(glht(model, linfct = hypothesis), 
                                        test = Ftest())
      regression_results = append(regression_results, 
                                  hypothesis_test_results$test$pvalue)
    }
    
    exp_regression_results <- data.frame(subject = numeric(0),
                                         id = character(),
                                         # Results from the uncertainty model
                                         v_coef = numeric(0), 
                                         v_p_val = numeric(0), 
                                         ru_coef = numeric(0), 
                                         ru_p_val = numeric(0), 
                                         vtu_coef = numeric(0), 
                                         vtu_p_val = numeric(0), 
                                         # Results from the condition model
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
                                         # Results from intercept/slope hypotheses
                                         rs_vs_sr_p_val = numeric(0),
                                         rr_vs_ss_p_val = numeric(0), 
                                         rs_v_vs_sr_v_p_val = numeric(0), 
                                         rr_v_vs_ss_v_p_val = numeric(0)
    )
    exp_regression_results[nrow(exp_regression_results) + 1, ] = c(subject, 
                                                                   unique(subject_data$ID), 
                                                                   regression_results)
    
    if (subject == 1) {
      write.csv(exp_regression_results, results_data_file_path, 
                row.names = FALSE)
    } else {
      write.table(exp_regression_results, results_data_file_path,
                  append = TRUE, quote = FALSE, row.names = FALSE,
                  sep = ",", col.names = FALSE)
    }
  }
}

plot_choice_estimates_glme = function(show_plot) {
  results = c()
  model <-readRDS("../models/choice_uncertainty_model.rds")
  coefficients = (summary(model))$coefficients
  
  coef_df <- data.frame(coef = factor(rownames(coefficients), 
                                      rownames(coefficients)), 
                        estimate = coefficients[, 1],
                        sd = coefficients[, 2],
                        p_val = coefficients[, 4])
  
  plot <- ggplot(coef_df, aes(coef, estimate)) +
    scale_x_discrete(labels=c('V', 'RU', 'V/TU')) +
    scale_y_continuous(sec.axis = dup_axis(), breaks = seq(0, 3.4, 0.2)) +
    labs(y = 'Regression coefficient', x = '') + geom_point(size=4) + 
    theme(axis.text = element_text(face="bold"), 
          axis.ticks.length = unit(-0.20, "cm"), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), 
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) +
    geom_errorbar(aes(ymin = estimate - sd, ymax = estimate + sd), width=.04) 
  
  if (show_plot) {
    plot
  }
  
  # Save the plot as a png
  ggsave(filename="../figures/choice_estimates.png", plot=plot)
}

plot_slope_intercept_condition_estimates = function(show_plot) {
  model <-readRDS("../models/choice_condition.rds")
  coefficients = (summary(model))$coefficients

  coef_df <- data.frame(condition = factor(c("RR", "RS", "SR", "SS"),
                                           levels=c("RS", "SR", "RR", "SS")),
                        estimate = coefficients[, 1],
                        sd = coefficients[, 2],
                        p_val = coefficients[, 4])

  intercept_df <- coef_df %>% slice(seq(1, n()/2))

  intercept_plot <- ggplot(intercept_df, aes(condition, estimate)) +
    scale_y_continuous(sec.axis = dup_axis(), limits = c(-0.3, 0.4),
                       breaks = seq(-0.3, 0.4, 0.1),
                       labels = scales::number_format(accuracy = 0.1)) +
    theme(axis.text = element_text(face="bold"),
          axis.ticks.length = unit(-0.10, "cm"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) +
    labs(y = 'Intercept', x = '') + geom_point() +
    geom_errorbar(aes(ymin = estimate - sd, ymax = estimate + sd), width=.04)

  slope_df <- coef_df %>% slice(seq(n()/2 + 1, n()))

  slope_plot <- ggplot(slope_df, aes(condition, estimate)) +
    scale_y_continuous(sec.axis = dup_axis(), limits = c(0.9, 1.9),
                       breaks = seq(0.7, 2.0, 0.2)) +
    theme(axis.text = element_text(face="bold"),
          axis.ticks.length = unit(-0.10, "cm"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.text.x.top = element_blank(),
          axis.text.y.right = element_blank(),
          axis.title.x.top = element_blank(),
          axis.title.y.right = element_blank()) +
    labs(y = 'Slope', x = '') + geom_point() +
    geom_errorbar(aes(ymin = estimate - sd, ymax = estimate + sd), width=.04)

  require(gridExtra)
  combined_plot <- grid.arrange(intercept_plot, slope_plot, ncol=2)

  if (show_plot) {
    plot
  }

  ggsave(filename="../figures/slope_intercept_condition_estimates.png",
         plot=combined_plot)
}
