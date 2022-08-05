
# python script to batch preprocess .asc files using pypil class, the process of which is described in Kret & Sjak-Shie (2018)

# 1.  pypil.prepare_phase(): This method organizes the data into and managable format removing all
#        the unnecessary components produced by the eyetracker

# 2.  pypil.filter_phase(): This method performs a series of filters on the data to smooth the data and start
#        filtering out the bad data points

# 3.  pypil.process_valid_samples(): The final method uses the filtered data to make inferences about which data 
#        points are likely due to noise (based on either physical impossibilities or logic; see Kret & Sjak-Shie, 
#        2018, see Figure 2 and 4). Then removes these values and imputes the value give the autocorrelation and 
#        overall path trajectory.  

# part of the code adapted from Deshawn Sambrano
# written by Haoxue Fan (haoxue_fan@g.harvard.edu) 


import pypil
import pandas as pd
from glob import glob
from itertools import compress

idx_max = 9999 # set this number to restrict analysis to a subset of all data (9999 is an arbitrary large number)

part_dirs = sorted(glob('../../Real_Subject_Data/*/'))
task_dirs = sorted(glob('../../Belief_Update_Process/*'))

overwrite = True
idx = 0
baseline_flag = True

# part_dirs = ['../../Real_Subject_Data/29XXOO_(06-13-2022)/']

for data_dir in part_dirs:
    if idx <= idx_max:
        try:
            current_file = glob(data_dir + '*asc')[0]
            print(idx, 'Index of the current preprocessing job:')
            print('    ', current_file)
            pupil_data = pypil.pypil(current_file)
            pupil_data.prepare_phase(overwrite)
            pupil_data.filter_phase(overwrite)
            pupil_data.process_valid_samples(overwrite)
            # read in task df using subjectid stored in pypil object
            task_df = pd.read_csv(list(compress(task_dirs, \
                [pupil_data.subjectid in x for x in task_dirs]))[0])
            pupil_data.choice_msgpupil_merge(task_df)
            if baseline_flag:
                pupil_data.merged_data = pupil_data.baseline_pupil_data.copy(deep=True)
                pupil_data.filter_phase(overwrite, baseline_flag)
                pupil_data.process_valid_samples(overwrite, baseline_flag)
            idx = idx + 1
            print('\n')
        except Exception:
            continue

