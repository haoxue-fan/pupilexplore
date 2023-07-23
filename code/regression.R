# Pupil regression 
# .R file to compile pupil analysis for evaluation and trial baseline period
# Written by Haoxue and Taylor


# load packages and set dir -----------------------------------------------------------

# Set the working directory to where the file lives
path <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(path)
source('utils.R')

# run regression (choice as DV) --------------------------------------------
# make sure to have 'combined_data.csv' in the ../data folder (can be downloaded from osf https://osf.io/q795b/)
run_and_save_models(TRUE, choice_models[3])

# run regression (baseline pupil as DV) --------------------------------------------
# absolute & directed models
run_and_save_models(TRUE, baseline_models)
add_criterion_to_model(TRUE, '', baseline_models, 'loo')
# ~ Condition
# ~ trial

# decode analysis (choice as DV) --------------------------------------------
decode_from_models(TRUE, baseline_model_everything, 
                   vars_to_decode = c('abs_ru','abs_v','tu'), lambda = 0,
                   decode_models)
add_criterion_to_model(TRUE, decode_models, baseline_model_everything) # what is decode models?


