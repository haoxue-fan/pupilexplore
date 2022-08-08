
# python script to batch preprocess pypil class (part 2)

# 1.  drop_redundant_columns: drop columns that are redundant in the current setting

# 2.  remove_irrelevant_events: drop rows linked with irrelevant events

# 3.  add_first_row_pupil: add the first row file merging task df and summary stats from pupil data

# 4.  add_timestamp: add timestamp columns to merged data

# 5. percent_miss: calculate percent of missing data within each trial

# 6. avg_within_timewindow: a versatile function used to calculate avg over a specified timewindow

# 7. baseline_correct: correct trial pupil data using its corresponding baseline

# written by Haoxue Fan (haoxue_fan@g.harvard.edu) 

import pypil
import pandas as pd
from glob import glob
from itertools import compress
import traceback
import sys

idx_max = 9999 # set this number to restrict analysis to a subset of all data (9999 is an arbitrary large number)

part_dirs = sorted(glob('../../Real_Subject_Data/*/'))
task_dirs = sorted(glob('../../Belief_Update_Process/*'))

overwrite = False
idx = 0
baseline_flag = True

# part_dirs = ['../../Real_Subject_Data/29XXOO_(06-13-2022)/']

for data_dir in part_dirs:
    print(data_dir)
    try: 
        current_file = glob(data_dir + '*asc')[0]
    except Exception:
        continue
    if idx <= idx_max:
        try:
            current_file = glob(data_dir + '*asc')[0]
            print(idx, 'Index of the current preprocessing job:')
            print('    ', current_file)
            pupil_data = pypil.pypil(current_file)
            # find task df
            task_df = list(compress(task_dirs, \
                [pupil_data.subjectid in x for x in task_dirs]))[0]
            pupil_data.read_merged_and_task(task_df, overwrite)
            
            pupil_data.drop_redundant_column()
            pupil_data.remove_irrelevant_events()

            pupil_data.add_first_row_pupil()
            pupil_data.add_timestamp()

            pupil_data.percent_miss()
            
            pupil_data.avg_within_timewindow(\
                timestamp_col='timestamp_locked_at_stimulus_pre_with_fixation_onset',\
                y_col='smoothed_interp_pupil_corrected',\
                timestamp_range=[-1000, 0],\
                new_x_col='trial_baseline')
            pupil_data.baseline_correct()
            pupil_data.write_merged_and_task(overwrite)
            # pupil_data.avg_within_timewindow(\
            #     timestamp_col='timestamp_locked_at_stimulus_pre_with_fixation_onset',\
            #     y_col='remove_baseline_smoothed_interp_pupil_corrected',\
            #     timestamp_range=[500, 2500],\
            #     new_x_col='eval_middle_2000')

            # pupil_data.avg_within_timewindow(\
            #     timestamp_col='timestamp_locked_at_stimulus_pre_with_fixation_onset',\
            #     y_col='remove_baseline_smoothed_interp_pupil_corrected',\
            #     timestamp_range=[1000, 2500],\
            #     new_x_col='eval_middle_1500')

            # pupil_data.avg_within_timewindow(\
            #     timestamp_col='timestamp_locked_at_stimulus_pre_with_fixation_onset',\
            #     y_col='remove_baseline_smoothed_interp_pupil_corrected',\
            #     timestamp_range=[1000, 3000],\
            #     new_x_col='eval_late_2000')


            idx = idx + 1
            print('\n')
        except Exception:
            continue
        

