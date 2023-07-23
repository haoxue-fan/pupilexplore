# Script to reproduce plots for pupil vs. explore project
# Written by Haoxue Fan and Taylor Burke

# Set working directory and source utils ------------------------------------------------------------

path <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(path)
source('utils.R')

# Load in data (only needed for Fig 6-8) --------------------------------------------
first_row_df <- load_data('first_row') %>% proc_exclude() %>% group_by(ID) %>% 
  mutate(trial_baseline_mean = mean(trial_baseline),
         trial_baseline_nomean = trial_baseline - trial_baseline_mean)

long_df <- load_data('long') # this may take a while
long_df <- long_df %>% as.data.table
long_df <- long_df %>% merge(first_row_df %>% dplyr::select(identifier, Condition, V, RU, TU, trial_baseline, trial_baseline_mean), by.x='identifier', by.y='identifier')
long_df$remove_meanbaseline_smoothed_interp_pupil_corrected <- long_df$smoothed_interp_pupil_corrected - long_df$trial_baseline_mean
long_df <- long_df %>% mutate(bin_TU = ifelse(TU <=3, 1, 
                                              ifelse(TU <= 6, 2, 3)) %>% 
                                factor(levels=c(1:3), labels=c('low (0,3]', 'med (3,6]', 'high (6, Inf)'))) %>% as.data.table


# Figure --------------------------------------------

## Figure3 --------------------------------------------
run_condition_behavior_hypothesis_testing_brm() 

## Figure4 --------------------------------------------
list_criterion_vanilla <- output_criterion('choice_plain_z', show_model_coef = FALSE, output=list(), mdl_dir = '../models/', sufixs_for_file_name = '')
LOOIC_vanilla <- list_criterion_vanilla[[1]]$estimates[3,1]           

list_criterion_baseline_models <- c(
  output_criterion('',baseline_models, show_model_coef = FALSE, output=list(), mdl_dir = '../models/'))

list_rowname = c(baseline_models)

df_criterion_baseline_models <- list_criterion_baseline_models %>% lapply(function(x){return(x$estimates[3,1])}) %>% data.frame() %>% t %>%  magrittr::set_colnames('LOOIC') %>% data.frame() %>% 
  mutate(LOOIC_relative = LOOIC - LOOIC_vanilla)

df_criterion_baseline_models <- df_criterion_baseline_models %>% magrittr::set_rownames(list_rowname) %>% rownames_to_column(var='model') %>% separate(model, into=c('period','type1','type2','var'), extra = 'merge')

df_criterion_baseline_models <- df_criterion_baseline_models %>% dplyr::select(-type2, -LOOIC_relative) %>% 
  mutate(var=ifelse(var=='belief'|is.na(var), 'full', var))

df_criterion_baseline_models %>% mutate(var=factor(var, levels=c('V','RU','TU','VTU','TU_RU','TU_V','RU_V','VTU_RU','VTU_V','TU_VTU','TU_VTU_RU','TU_VTU_V','VTU_RU_V','full'), labels=c(1:14))) %>% 
  filter(type1 != 'baseline') %>% ggplot(aes(x=var, LOOIC, mean, group=type1, color=type1), var, y=LOOIC) + 
  geom_point(size=3, shape=19)+
  xlab('Model Number') +
  ylab('Leave-oue-out Cross Validation')+
  scale_color_discrete(labels=c('directed model','absolute model'), name='')+
  theme(axis.text.x = element_text(size=15, angle=30),
        axis.text.y = element_text(size=15),
        axis.title = element_text(size=20))+
  narrative_theme

## Figure5 --------------------------------------------
abs_coef_name <- c('|V|','|RU|','TU','|V|/TU')
mdl <- readRDS('../models/baseline_prior_belief.rds') 
plot_choice_estimates_no_intercept(mdl) + scale_x_discrete(labels = abs_coef_name)

## Figure6 --------------------------------------------
long_df_sub_removebaseline_binTU <-
  long_df[,
          .(mean_pupil=mean(remove_meanbaseline_smoothed_interp_pupil_corrected, na.rm=T)),
          by=.(bin_TU, ID.x, timestamp_locked_at_stimulus_pre_with_fixation_onset)
  ]

p <- long_df_sub_removebaseline_binTU[timestamp_locked_at_stimulus_pre_with_fixation_onset >= -1500 & timestamp_locked_at_stimulus_pre_with_fixation_onset <= 7500,] %>% ggplot(aes(x=timestamp_locked_at_stimulus_pre_with_fixation_onset, y = mean_pupil, group=bin_TU, color=bin_TU, fill=bin_TU)) +
  stat_summary(geom='point')+
  stat_summary(geom='ribbon', alpha=0.1)
p <- add_line(p) +
  scale_color_manual(values=colorRampPalette(c('blue','red'))(3))+
  scale_fill_manual(values=colorRampPalette(c('blue','red'))(3)) 
p

## Figure7 --------------------------------------------

long_df_sub_removebaseline_cond_trial1 <-
  long_df[,.(mean_pupil=mean(remove_meanbaseline_smoothed_interp_pupil_corrected, na.rm = T)),
          by=.(trial, Condition, ID.x, timestamp_locked_at_stimulus_pre_with_fixation_onset)]

p <- 
  long_df_sub_removebaseline_cond_trial1[
    timestamp_locked_at_stimulus_pre_with_fixation_onset >= -1500 & timestamp_locked_at_stimulus_pre_with_fixation_onset <= 7500,] %>% 
  ggplot(aes(x=timestamp_locked_at_stimulus_pre_with_fixation_onset, y = mean_pupil, group=Condition, color=Condition, fill=Condition)) +
  stat_summary(geom='point')+
  stat_summary(geom='ribbon', alpha=0.1)
p <- add_line(p)
p

p <- first_row_df %>% 
  group_by(ID, Condition) %>% 
  summarise(trial_baseline_nomean = mean(trial_baseline_nomean)) %>% 
  ggplot(aes(x=Condition, y=trial_baseline_nomean, group=Condition, color=Condition)) + 
  stat_summary(geom = 'point') + stat_summary(geom='errorbar', width=0.04) + 
  ylab('Trial Baseline Pupil Size') + 
  narrative_theme +
  theme(legend.position = "none")
p


## Figure8 --------------------------------------------
first_row_df %>% group_by(ID, Condition, trial) %>% dplyr::summarise(trial_baseline_nomean=mean(trial_baseline_nomean, na.rm=T)) %>% mutate(trial=trial+1) %>%  
  ggplot(aes(x=trial, y=trial_baseline_nomean)) +
  stat_summary(geom='point')+
  stat_summary(geom='errorbar')+
  facet_wrap(~Condition) +
  narrative_theme+
  scale_x_continuous(limits = c(1,10), breaks=c(1,5,10))+
  ylab('Trial Baseline Pupil Size')

## Figure9 --------------------------------------------
list_criterion_decode_models <- c(output_criterion(decode_models, 'baseline_prior_belief', show_model_coef = FALSE, output=list(), mdl_dir = '../models/'))

list_rowname = c(paste0(decode_models, '_baseline'))

list_criterion_vanilla <- output_criterion('choice_plain_z', show_model_coef = FALSE, output=list(), mdl_dir = '../models/', sufixs_for_file_name = '')
LOOIC_vanilla <- list_criterion_vanilla[[1]]$estimates[3,1]

df_criterion_decode_models <- list_criterion_decode_models %>% lapply(function(x){return(x$estimates[3,1])}) %>% data.frame() %>% t %>% 
  magrittr::set_rownames(list_rowname) %>% magrittr::set_colnames('LOOIC') %>% data.frame() %>% rownames_to_column(var='model') %>% 
  mutate(LOOIC_relative = LOOIC - LOOIC_vanilla)

# try to plot list1 and list_criterion_decode_models together by making it into one graph
df_criterion_decode_models <- df_criterion_decode_models %>% separate(model, c("placeholder","var")) 

df_criterion_decode_models %>% 
  ggplot(aes(x=forcats::fct_reorder(factor(var), LOOIC_relative, mean), y=LOOIC_relative, group=var, color='black', fill='black')) + 
  geom_bar(stat='identity', position='dodge', aes(group=var), alpha=0.3) +
  narrative_theme+
  xlab('Var decoded from the Abs model')+
  ylab('Relative LOOIC')+
  theme(legend.position = 'none')




