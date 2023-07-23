#!/usr/bin/env python
# coding: utf-8

# # BELIEF UPDATE PROCESS
# 
# #Input: This code takes in behavioral dataframes for each participant from the gambling task, namely:
#     #1. The participant's subject ID
#     #2. The block number [1,16]
#     #3. The condition of the corresponding block
#     #4. The reward the participant received from each trial's chosen slot
#     #5. The slot the participant chose
#     #6. The type of slot correspond to the left and right slot each trial (stays constant throughout a block, changes each block)
#     #7. The actual means of both the left and right slots in each block
#     
#         #*Note on inputs: as the code is written, your participants' files should be located on your desktop; change 'username' in the code below to your laptop's username for the code to access your files, or change the path to where your files are located 
#     
# #Output: The code outputs dataframes for each participants, including:
#     #1. The participant's subject ID
#     #2. The block number
#     #3. The trial number
#     #4. The condition of the corresponding block 
#     #5. The arm chosen each trial and its condition (risky/safe)
#     #6. The actual means of both the left and right slots in each block
#     #7. The rewards of each slot in each block
#     #8. The prior value and variance estimates of both slots in each trial
#     #9. The posterior value and variance estimates of both slots in each trial
#     #10. The prior and posterior value differences each trial 
#     #11. The prior and posterior relative uncertainty each trial
#     #12. The prior and posterior total uncertainty each trial
#     
# #Dependency:
#     #Packages Used:
#         #1. pandas
#         #2. numpy
#         #3. math
#     #Version of Python: Python 3.9.15
#     
# #This code uses Kalman filtering equations to estimate each participant's belief and uncertainty about the value of each slot for each trial of the task. Our use of Kalman filtering equations is based on the belief update model used by Gershman (2019), which may be found at https://gershmanlab.com/pubs/Gershman19_uncertainty.pdf.   
# 

# In[ ]:


#INSTRUCTIONS:

#1. ADD THE CSV FILES FOR EACH PARTICIPANT TO THE FILE LIST

#2. IDENTIFY THE CORRESPONDING NAMES OF THE PARAMETERS IN YOUR DATAFRAME

  #subject_column — the column with the subject IDs
  #block_column — the column for which block the participant is on
  #condition_column — the column for the block's condition (Safe/Risky, Risky/Safe, Safe/Safe, or Risky/Risky)
  #reward_received_column — the column for the participant's received reward each trial
  #choice_column — the column for which slot (or lack of slot) the participant chose
  #left_machine_name — the name under which the left slot is listed
  #right_machine_name — the name under which the right slot is listed

#3. ADD THE NAMES INTO THE FUNCTION AND RUN THE CODE


# In[1]:


import pandas as pd
import numpy as np
import math 


# In[2]:


#MAKE A LIST OF ALL THE NAMES OF THE CSV FILES WE WANT TO LOOP THROUGH
file_list = ['taskData264COY_2022_06_28.csv', 'taskData29XXOO_2022_06_13.csv', 'taskData34BRN8_2022_07_15.csv', 'taskData42I6EI_2022_07_14.csv', 'taskData4J0VS0_2022_06_23.csv', 'taskData7DL514_2022_06_14.csv', 'taskData7T9M4M_2022_06_28.csv', 'taskData80MSTS_2022_07_07.csv', 'taskData8OX7U6_2022_06_15.csv', 'taskData9W0DJC_2022_07_01.csv', 'taskDataA98DB9_2022_06_09.csv', 'taskDataAJA1KZ_2022_07_14.csv', 'taskDataBZR5YS_2022_07_14.csv', 'taskDataCCV1AT_2022_07_11.csv', 'taskDataCM1TZG_2022_06_23.csv', 'taskDataCWYKY2_2022_06_30.csv', 'taskDataDA2GA3_2022_06_14.csv', 'taskDataDK88ZQ_2022_07_19.csv', 'taskDataDU5ZOC_2022_06_21.csv', 'taskDataEPSUO5_2022_07_18.csv', 'taskDataESCFOV_2022_06_22.csv', 'taskDataFDT8PT_2022_06_16.csv', 'taskDataFNZ0EF_2022_06_28.csv', 'taskDataGB0NP3_2022_07_08.csv', 'taskDataHGK949_2022_07_12.csv', 'taskDataN0WPBH_2022_06_17.csv', 'taskDataNQ0XLV_2022_07_20.csv', 'taskDataOVJJA1_2022_06_13.csv', 'taskDataQO4J04_2022_06_14.csv', 'taskDataQZ1BQR_2022_06_16.csv', 'taskDataRW8QQ1_2022_06_30.csv', 'taskDataSAIERP_2022_07_20.csv', 'taskDataU33EHJ_2022_06_21.csv', 'taskDataUP2Q6V_2022_06_28.csv', 'taskDataUWQ7RD_2022_06_30.csv', 'taskDataVBGL6F_2022_07_08.csv', 'taskDataWZH973_2022_06_15.csv', 'taskDataYB70MS_2022_06_13.csv', 'taskDataYI5H7A_2022_06_22.csv', 'taskDataYUVTXN_2022_07_01.csv', 'taskData597FR1_2022_07_21.csv', 'taskDataXS5T9Y_2022_07_21.csv', 'taskDataX41LW9_2022_07_21.csv', 'taskDataAWCP7P_2022_07_22.csv', 'taskDataFFY4DW_2022_07_22.csv', 'taskDataBPUN96_2022_07_22.csv', 'taskData67EUQB_2022_07_28.csv', 'taskDataXZTAUG_2022_08_02.csv', 'taskData5C00RP_2022_08_03.csv', 'taskDataAGO8L7_2022_08_03.csv', 'taskDataE2U81V_2022_08_04.csv', 'taskDataCZ3GMF_2022_08_04.csv', 'taskDataESXQC9_2022_08_05.csv', 'taskData75L9HL_2022_08_05.csv']



# In[9]:


def belief_update_process(file_list, subject_column, block_column, condition_column, reward_received_column, choice_column, left_machine_name, right_machine_name, left_mean_column, right_mean_column, left_reward_column, right_reward_column):

  #INITIALIZE VARIABLES
  initial_variance = 36
  initial_mean = 0
  tau_squared_risky = 16
  tau_squared_safe = 0.00001
  initial_risky_learning_rate = (initial_variance) / (initial_variance + tau_squared_risky) 
  initial_safe_learning_rate = (initial_variance) / (initial_variance + tau_squared_safe)

  #INITIALIZE DATAFRAME LIST
  dataframe_list = []

  #LOAD IN THE DATA FROM ONE CSV FILE IN THE LIST
  for i in file_list:
    df = pd.read_csv('/Users/username/Desktop/' + i) 

    df[reward_received_column].replace('[]', np.nan, inplace=True)
    df[reward_received_column] = df[reward_received_column].astype(float)

    #INITIALIZE LISTS TO APPEND TO DATAFRAME LATER
    list_of_prior_means_left = []
    list_of_prior_means_right = []  
    list_of_prior_variances_left = []
    list_of_prior_variances_right = []
    list_of_posterior_means_left = []
    list_of_posterior_means_right = []
    list_of_posterior_variances_left = []
    list_of_posterior_variances_right = []
    block = []
    trial = [] 
    choice = []
    choice_value = []
    condition = []
    condition_value = []
    subject = []
    left_actual_mean = []
    right_actual_mean = []
    left_reward = []
    right_reward = []

    #FOR LOOP TO LOOP THROUGH EACH BLOCK FOR SELECTED PARTICIPANT
    block_number = df[block_column]
    block_number, ind = np.unique(block_number, return_index=True)
    block_number = block_number[np.argsort(ind)]
    block_number = list(block_number)
    block_list = []
    for k in block_number:
      if k != -1: 
        block_list.append(k)

    for j in block_list:
      filt = (df[block_column] == j)
      block_subset = df[filt] 

      #INITIALIZE VARIABLES
      posterior_mean_left = initial_mean
      posterior_mean_right = initial_mean
      posterior_variance_left = initial_variance
      posterior_variance_right = initial_variance 

      #APPEND ACTUAL MEANS/REWARDS/CONDITION TO LISTS
      for z in range(len(block_subset)):
        left_actual_mean.append(block_subset[left_mean_column].iloc[z])
        right_actual_mean.append(block_subset[right_mean_column].iloc[z])
        condition_value.append(block_subset[condition_column].iloc[z])
        left_reward.append(block_subset[left_reward_column].iloc[z])
        right_reward.append(block_subset[right_reward_column].iloc[z]) 

      #BLOCK CONDITION IS RISKY/SAFE
      if block_subset[condition_column].iloc[0] == 1:

        #FOR LOOP FOR UPDATING MEANS/VARIANCE
        for n in range(len(block_subset)):

          #APPEND BLOCK, TRIAL, AND SUBJECT VALUES TO LISTS
          block.append(block_subset[block_column].iloc[n])
          trial.append(n + 1)
          subject.append(block_subset[subject_column].iloc[n])
          condition.append('Risky/Safe') 

          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE LEFT RISKY ARM
          if block_subset[choice_column].iloc[n] == left_machine_name: 
            choice.append('Left_R')
            choice_value.append(1)

            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0:

                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance) 

                #UPDATE MEAN/VARIANCE FOR LEFT (CHOSEN) ARM
                posterior_mean_left = initial_mean + initial_risky_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_left = initial_variance - initial_risky_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(initial_mean)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(initial_variance)

            #ELSE STATEMENT SAYING THESE WERE DECISIONS FOLLOWING THE FIRST
            else: 

                #APPEND PREVIOUS POSTERIOR TO PRIOR LIST FOR RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right)
                list_of_prior_variances_right.append(posterior_variance_right)

                #UPDATE MEAN/VARIANCE FOR LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left)
                list_of_prior_variances_left.append(posterior_variance_left)
                risky_learning_rate = (posterior_variance_left) / (posterior_variance_left + tau_squared_risky) 
                posterior_mean_left = posterior_mean_left + risky_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_left) 
                posterior_variance_left = posterior_variance_left - risky_learning_rate * (posterior_variance_left)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)
                

          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE RIGHT SAFE ARM
          if block_subset[choice_column].iloc[n] == right_machine_name:
            choice.append('Right_S')
            choice_value.append(0) 

            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0:

                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance)

                #UPDATE MEAN/VARIANCE FOR RIGHT ARM
                posterior_mean_right = initial_mean + initial_safe_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_right = initial_variance - initial_safe_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(initial_mean)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(initial_variance)
                list_of_posterior_variances_right.append(posterior_variance_right)

            #ELSE STATEMENT SAYING THESE WERE DECISIONS FOLLOWING THE FIRST
            else:

                #LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left) 
                list_of_prior_variances_left.append(posterior_variance_left)
                
                #RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right)
                list_of_prior_variances_right.append(posterior_variance_right)
                safe_learning_rate = (posterior_variance_right) / (posterior_variance_right + tau_squared_safe)
                posterior_mean_right = posterior_mean_right + safe_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_right)
                posterior_variance_right = posterior_variance_right - safe_learning_rate * (posterior_variance_right) 
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)
                
          #ELSE STATEMENT SAYING THE PARTICIPANT DID NOT MAKE A DECISION
          if block_subset[choice_column].iloc[n] == '[]':

            choice.append('-')
            choice_value.append('-')

            #APPEND PRIORS TO LISTS
            list_of_prior_means_left.append(posterior_mean_left)
            list_of_prior_means_right.append(posterior_mean_right)
            list_of_prior_variances_left.append(posterior_variance_left)
            list_of_prior_variances_right.append(posterior_variance_right)
            
            #APPEND POSTERIORS TO LISTS
            list_of_posterior_means_left.append(posterior_mean_left)
            list_of_posterior_means_right.append(posterior_mean_right)
            list_of_posterior_variances_left.append(posterior_variance_left)
            list_of_posterior_variances_right.append(posterior_variance_right)


#_____________________________________________________



      #BLOCK CONDITION IS SAFE/RISKY
      if block_subset[condition_column].iloc[0] == 2:

        for n in range(len(block_subset)): 

          #APPEND BLOCK, TRIAL, AND SUBJECT VALUES TO LISTS
          block.append(block_subset[block_column].iloc[n])
          trial.append(n + 1)
          subject.append(block_subset[subject_column].iloc[n])
          condition.append('Safe/Risky')

          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE LEFT SAFE ARM
          if block_subset[choice_column].iloc[n] == left_machine_name:
            choice.append('Left_S') 
            choice_value.append(1) 
            
            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0:
                
                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance)

                #UPDATE MEAN/VARIANCE FOR LEFT ARM
                posterior_mean_left = initial_mean + initial_safe_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_left = initial_variance - initial_safe_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(initial_mean)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(initial_variance)

            #ELSE STATEMENT SAYING THESE WERE DECISIONS FOLLOWING THE FIRST
            else:

                #RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right)
                list_of_prior_variances_right.append(posterior_variance_right)

                #LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left)
                list_of_prior_variances_left.append(posterior_variance_left)
                safe_learning_rate = (posterior_variance_left) / (posterior_variance_left + tau_squared_safe) 
                posterior_mean_left = posterior_mean_left + safe_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_left) 
                posterior_variance_left = posterior_variance_left - safe_learning_rate * (posterior_variance_left)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)
               
          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE RIGHT RISKY ARM
          if block_subset[choice_column].iloc[n] == right_machine_name:
            choice.append('Right_R')
            choice_value.append(0)

            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0:

                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance)

                #UPDATE MEAN/VARIANCE FOR RIGHT ARM
                posterior_mean_right = initial_mean + initial_risky_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_right = initial_variance - initial_risky_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(initial_mean)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(initial_variance)
                list_of_posterior_variances_right.append(posterior_variance_right)

            #ELSE STATEMENT SAYING THESE WERE THE DECISIONS FOLLOWING THE FIRST
            else:

                #LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left)
                list_of_prior_variances_left.append(posterior_variance_left)
                
                #RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right)
                list_of_prior_variances_right.append(posterior_variance_right)
                risky_learning_rate = (posterior_variance_right) / (posterior_variance_right + tau_squared_risky)
                posterior_mean_right = posterior_mean_right + risky_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_right)
                posterior_variance_right = posterior_variance_right - risky_learning_rate * (posterior_variance_right)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)

          #ELSE STATEMENT SAYING THE PARTICIPANT DID NOT MAKE A DECISION
          if block_subset[choice_column].iloc[n] == '[]':

            choice.append('-')
            choice_value.append('-')

            #APPEND PRIORS TO LISTS
            list_of_prior_means_left.append(posterior_mean_left)
            list_of_prior_means_right.append(posterior_mean_right)
            list_of_prior_variances_left.append(posterior_variance_left)
            list_of_prior_variances_right.append(posterior_variance_right)

            #APPEND POSTERIORS TO LISTS
            list_of_posterior_means_left.append(posterior_mean_left)
            list_of_posterior_means_right.append(posterior_mean_right)
            list_of_posterior_variances_left.append(posterior_variance_left)
            list_of_posterior_variances_right.append(posterior_variance_right)

#_____________________________________________________



      #BLOCK CONDITION IS RISKY/RISKY
      if block_subset[condition_column].iloc[0] == 3:

        for n in range(len(block_subset)):
          
          #APPEND BLOCK, TRIAL, AND SUBJECT VALUES TO LISTS
          block.append(block_subset[block_column].iloc[n])
          trial.append(n + 1)
          subject.append(block_subset[subject_column].iloc[n])
          condition.append('Risky/Risky')

          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE LEFT RISKY ARM
          if block_subset[choice_column].iloc[n] == left_machine_name: 
            choice.append('Left_R')
            choice_value.append(1)

            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0: 

                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance)

                #LEFT ARM
                posterior_mean_left = initial_mean + initial_risky_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_left = initial_variance - initial_risky_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(initial_mean)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(initial_variance)

            #ELSE STATEMENT SAYING THESE ARE DECISIONS FOLLOWING THE FIRST
            else:

                #RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right)
                list_of_prior_variances_right.append(posterior_variance_right)
                
                #LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left) 
                list_of_prior_variances_left.append(posterior_variance_left)
                left_risky_learning_rate = (posterior_variance_left) / (posterior_variance_left + tau_squared_risky) 
                posterior_mean_left = posterior_mean_left + left_risky_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_left)
                posterior_variance_left = posterior_variance_left - left_risky_learning_rate * (posterior_variance_left) 
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)

          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE RIGHT RISKY ARM
          if block_subset[choice_column].iloc[n] == right_machine_name:
            choice.append('Right_R')
            choice_value.append(0)

            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0:

                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance)

                #RIGHT ARM
                posterior_mean_right = initial_mean + initial_risky_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_right = initial_variance - initial_risky_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(initial_mean)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(initial_variance)
                list_of_posterior_variances_right.append(posterior_variance_right)

            #ELSE STATEMENT SAYING THESE WERE THE DECISIONS FOLLOWING THE FIRST
            else:

                #LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left)
                list_of_prior_variances_left.append(posterior_variance_left)
                
                #RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right)
                list_of_prior_variances_right.append(posterior_variance_right)
                right_risky_learning_rate = (posterior_variance_right) / (posterior_variance_right + tau_squared_risky)
                posterior_mean_right = posterior_mean_right + right_risky_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_right)
                posterior_variance_right = posterior_variance_right - right_risky_learning_rate * (posterior_variance_right) 
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)
            
          #ELSE STATEMENT SAYING THE PARTICIPANT DID NOT MAKE A DECISION
          if block_subset[choice_column].iloc[n] == '[]':

            choice.append('-')
            choice_value.append('-')

            #APPEND PRIORS TO LISTS
            list_of_prior_means_left.append(posterior_mean_left)
            list_of_prior_means_right.append(posterior_mean_right)
            list_of_prior_variances_left.append(posterior_variance_left)
            list_of_prior_variances_right.append(posterior_variance_right)
            
            #APPEND POSTERIORS TO LISTS
            list_of_posterior_means_left.append(posterior_mean_left)
            list_of_posterior_means_right.append(posterior_mean_right)
            list_of_posterior_variances_left.append(posterior_variance_left)
            list_of_posterior_variances_right.append(posterior_variance_right)


#_____________________________________________________



      #BLOCK CONDITION IS SAFE/SAFE
      if block_subset[condition_column].iloc[0] == 4:

        for n in range(len(block_subset)):

          #APPEND BLOCK, TRIAL, AND SUBJECT VALUES TO LISTS
          block.append(block_subset[block_column].iloc[n])
          trial.append(n + 1)
          subject.append(block_subset[subject_column].iloc[n])
          condition.append('Safe/Safe')

          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE LEFT SAFE ARM
          if block_subset[choice_column].iloc[n] == left_machine_name: 
            choice.append('Left_S')
            choice_value.append(1) 

            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0:

                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance)

                #UPDATE MEAN/VARIANCE FOR LEFT ARM
                posterior_mean_left = initial_mean + initial_safe_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_left = initial_variance - initial_safe_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(initial_mean)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(initial_variance)

            #ELSE STATEMENT SAYING THESE WERE THE DECISIONS FOLLOWING THE FIRST
            else:

                #RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right)
                list_of_prior_variances_right.append(posterior_variance_right)
                
                #LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left)
                list_of_prior_variances_left.append(posterior_variance_left)
                left_safe_learning_rate = (posterior_variance_left) / (posterior_variance_left + tau_squared_safe) 
                posterior_mean_left = posterior_mean_left + left_safe_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_left) 
                posterior_variance_left = posterior_variance_left - left_safe_learning_rate * (posterior_variance_left) 
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)

          #IF STATEMENT SAYING THE PARTICIPANT CHOSE THE RIGHT SAFE ARM
          if block_subset[choice_column].iloc[n] == right_machine_name:
            choice.append('Right_S')
            choice_value.append(0)

            #IF STATEMENT SAYING THIS WAS THE PARTICIPANT'S FIRST DECISION
            if n == 0:
              
                #APPEND PRIORS TO LISTS (INITIAL MEAN/VARIANCE IS ALWAYS THE PRIOR FOR FIRST TRIAL)
                list_of_prior_means_left.append(initial_mean)
                list_of_prior_means_right.append(initial_mean)
                list_of_prior_variances_left.append(initial_variance)
                list_of_prior_variances_right.append(initial_variance)

                #UPDATE MEAN/VARIANCE FOR RIGHT ARM
                posterior_mean_right = initial_mean + initial_safe_learning_rate * (block_subset[reward_received_column].iloc[n] - initial_mean)
                posterior_variance_right = initial_variance - initial_safe_learning_rate * (initial_variance)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(initial_mean)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(initial_variance)
                list_of_posterior_variances_right.append(posterior_variance_right)

            #ELSE STATEMENT SAYING THESE WERE THE DECISIONS FOLLOWING THE FIRST
            else:

                #LEFT ARM
                list_of_prior_means_left.append(posterior_mean_left)
                list_of_prior_variances_left.append(posterior_variance_left)
                
                #RIGHT ARM
                list_of_prior_means_right.append(posterior_mean_right) 
                list_of_prior_variances_right.append(posterior_variance_right)               
                right_safe_learning_rate = (posterior_variance_right) / (posterior_variance_right + tau_squared_safe)
                posterior_mean_right = posterior_mean_right + right_safe_learning_rate * (block_subset[reward_received_column].iloc[n] - posterior_mean_right)
                posterior_variance_right = posterior_variance_right - right_safe_learning_rate * (posterior_variance_right)
                
                #APPEND POSTERIORS TO LISTS
                list_of_posterior_means_left.append(posterior_mean_left)
                list_of_posterior_means_right.append(posterior_mean_right)
                list_of_posterior_variances_left.append(posterior_variance_left)
                list_of_posterior_variances_right.append(posterior_variance_right)

          #ELSE STATEMENT SAYING THE PARTICIPANT DID NOT MAKE A DECISION
          if block_subset[choice_column].iloc[n] == '[]':

            choice.append('-')
            choice_value.append('-')

            #APPEND PRIORS TO LISTS
            list_of_prior_means_left.append(posterior_mean_left)
            list_of_prior_means_right.append(posterior_mean_right)
            list_of_prior_variances_left.append(posterior_variance_left)
            list_of_prior_variances_right.append(posterior_variance_right)
            
            #APPEND POSTERIORS TO LISTS
            list_of_posterior_means_left.append(posterior_mean_left)
            list_of_posterior_means_right.append(posterior_mean_right)
            list_of_posterior_variances_left.append(posterior_variance_left)
            list_of_posterior_variances_right.append(posterior_variance_right)

          
#_____________________________________________________



    #CODE FOR CALCULATING V, RU, AND TU:
    #INITIALIZE LISTS TO APPEND TO LATER
    Prior_V_list = []
    Prior_RU_list = []
    Prior_TU_list = []
    Posterior_V_list = []
    Posterior_RU_list = []
    Posterior_TU_list = []

    #CALCULATE V
    for j in range(len(list_of_prior_means_left)):

      Prior_V = (list_of_prior_means_left[j] - list_of_prior_means_right[j])
      Prior_V_list.append(Prior_V)  
        
      Posterior_V = (list_of_posterior_means_left[j] - list_of_posterior_means_right[j])
      Posterior_V_list.append(Posterior_V)

    #CALCULATE RU
    for j in range(len(list_of_prior_variances_left)):

      Prior_RU = (math.sqrt(list_of_prior_variances_left[j]) - math.sqrt(list_of_prior_variances_right[j]))
      Prior_RU_list.append(Prior_RU)
        
      Posterior_RU = (math.sqrt(list_of_posterior_variances_left[j]) - math.sqrt(list_of_posterior_variances_right[j]))
      Posterior_RU_list.append(Posterior_RU)
        
    #CALCULATE TU
    for j in range(len(list_of_prior_variances_left)):

      Prior_TU = math.sqrt(list_of_prior_variances_left[j] + list_of_prior_variances_right[j]) 
      Prior_TU_list.append(Prior_TU)
        
      Posterior_TU = math.sqrt(list_of_posterior_variances_left[j] + list_of_posterior_variances_right[j])
      Posterior_TU_list.append(Posterior_TU)
        
    #CODE FOR CREATING DATAFRAME
    data = {'SubjectID': subject, 'Block': block, 'Condition': condition, 'Condition_Value': condition_value, 'Trial': trial, 'Chosen_Arm': choice, 'Chosen_Arm_Value': choice_value, 'Left_Slot_Actual_Mean': left_actual_mean, 'Right_Slot_Actual_Mean': right_actual_mean, 'Left_Reward' : left_reward, 'Right_Reward': right_reward, 'Prior_Means_Left': list_of_prior_means_left, 'Prior_Means_Right': list_of_prior_means_right, 'Prior_Variances_Left': list_of_prior_variances_left, 'Prior_Variances_Right': list_of_prior_variances_right, 'Posterior_Means_Left': list_of_posterior_means_left, 'Posterior_Means_Right': list_of_posterior_means_right, 'Posterior_Variances_Left': list_of_posterior_variances_left, 'Posterior_Variances_Right': list_of_posterior_variances_right, 'Prior_V': Prior_V_list, 'Prior_RU': Prior_RU_list, 'Prior_TU': Prior_TU_list, 'Posterior_V': Posterior_V_list, 'Posterior_RU': Posterior_RU_list, 'Posterior_TU': Posterior_TU_list}

    dataframe = pd.DataFrame(data)
    dataframe.replace('-', np.nan, inplace=True)
    dataframe_list.append(dataframe)
    
  return dataframe_list 

  
 



# In[10]:


#ACTUALLY RUNNING CODE

dataframes = belief_update_process(file_list, 'subjectID', 'block', 'cond', 'reward', 'choice', 'machine1', 'machine2', 'mu1', 'mu2', 'reward1', 'reward2')


# In[11]:


for i in range(len(dataframes)):
  df_of_interest = dataframes[i]
  filename = df_of_interest['SubjectID'].iloc[0]

  dataframes[i].to_csv(filename + '.csv', index=False)


# In[ ]:




