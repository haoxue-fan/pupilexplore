# Pupil vs. Explore Code Repo - Code Folder

This folder contains analysis code

Contributors: Haoxue, Taylor, Deshawn & Emily

Now contains:

1. batch_preprocess.py, batch_preprocess2.py: batch changing EDF to ASC file,
   merge task and pupil data, and calculate pupil size in time windows of interest
2. pypil.py: pypil object to preprocess pupil data following 
    [Kret & Sjak-Shie (2019)](https://link-springer-com.ezp-prod1.hul.harvard.edu/article/10.3758/s13428-018-1075-y)
   (we are working on a [repo](https://github.com/dsambrano/pypil) to provide a more generic version of this code lib.
   stay tuned!)
   
3. Preprocess_cleanup_github.ipynb: notebook containing preprocess steps [H]
4. Belief_Update_Process_Function.ipynb, Belief_Update_Process_Function.py: notebook calculating belief update for behav data 
5. Cleaning_Baseline_Blinks_Functions.ipynb: notebook cleaning baseline blink
   data (anaysis not included in the current paper)
6. plot.R: plot script to reproduce Figures 
7. regression.R: R script to reproduce main results 
8. utils.R: R script for various utility functions and variable definitions
   (required to run plot.R and regression.R)

For any questions, contact Haoxue (haoxue_fan@g.harvard.edu)
