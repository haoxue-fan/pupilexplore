# ----------------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------------

# practice.py: 
# The practice trial to run the two--armed bandit task used in Gershman (2019)
# The default for this practice is without eye link connected

# Some of the functions were adapted from the jspsych version of the task which
# can be found in cognition.run phelpslab account
# Written by Haoxue Fan
# Last updated on May 2022

# This file is run within psychopy 2022.1.3 and pylink downloaded from SR
# research developer kit on a Windows Computer. The integration with eyelink
# hasn't been tested on a mac.

# This file should be used with utils.py and main_config.yaml in the same folder

from asyncio import wait_for
from multiprocessing import dummy

import pylink
import os
import platform
import random
import time
import sys
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, gui, __version__
from PIL import Image  # for preparing the Host backdrop image
from string import ascii_letters, digits
from utils import * 
import numpy as np
import pandas as pd
import yaml


# Switch to the script folder
script_path = os.path.dirname(sys.argv[0])
if len(script_path) != 0:
    os.chdir(script_path)

# Show only critical log message in the PsychoPy console
from psychopy import logging
logging.console.setLevel(logging.CRITICAL)

# Set this variable to True if you use the built-in retina screen as your
# primary display device on macOS. If have an external monitor, set this
# variable True if you choose to "Optimize for Built-in Retina Display"
# in the Displays preference settings.

# get parameters from yaml
with open('main_config.yaml', 'r') as file:
    exp_config = yaml.safe_load(file)

use_retina = exp_config['use_retina']
dummy_mode = exp_config['dummy_mode']
debug_mode = exp_config['debug_mode']
full_screen = exp_config['full_screen']
msgColor = exp_config['msg']['msgColor']
typeColor = exp_config['type']['typeColor']
typeSize = exp_config['type']['typeSize']
typeBold = exp_config['type']['typeBold']
foreground_color = exp_config['calibration']['foreground_color']
background_color = exp_config['calibration']['background_color']
cond = exp_config['cond'][0]
fixSize = exp_config['fixation']['fixSize']
fixLineWidth = exp_config['fixation']['fixLineWidth']
fixColor = exp_config['fixation']['fixColor']
rectWidth = exp_config['rect']['rectWidth']
rectHeight = exp_config['rect']['rectHeight']
rectLineWidth = exp_config['rect']['rectLineWidth']
rectDistCenter = exp_config['rect']['rectDistCenter']
rectColor = exp_config['rect']['rectColor']

sd_mean_mu = exp_config['sd_mean_mu']
sd_observe = exp_config['sd_observe']
sd_rw = exp_config['sd_rw']
labels = exp_config['labels']
scaling_factor = exp_config['scaling_factor']
start_coin = exp_config['start_coin']

if debug_mode:
    n_blocks = exp_config['n_blocks_debug']
    n_trials = exp_config['n_trials_debug']
else:
    n_blocks = exp_config['n_blocks']
    n_trials = exp_config['n_trials']

if debug_mode:
    fixation_length_min = exp_config['trial']['fixation_length_min_debug']
    fixation_length_max = exp_config['trial']['fixation_length_max_debug']
    stimulus_pre_with_fixation_length = exp_config['trial']['stimulus_pre_with_fixation_length_debug']
    stimulus_pre_without_fixation_length_max = exp_config['trial']['stimulus_pre_without_fixation_length_max_debug']
    reward_pre_without_fixation_length = exp_config['trial']['reward_pre_without_fixation_length_debug']
    baseline_length = exp_config['baseline_length_debug']
    extra_fixation_length = exp_config['trial']['extra_fixation_length_debug']
else:
    fixation_length_min = exp_config['trial']['fixation_length_min']
    fixation_length_max = exp_config['trial']['fixation_length_max']
    stimulus_pre_with_fixation_length = exp_config['trial']['stimulus_pre_with_fixation_length']
    stimulus_pre_without_fixation_length_max = exp_config['trial']['stimulus_pre_without_fixation_length_max']
    reward_pre_without_fixation_length = exp_config['trial']['reward_pre_without_fixation_length']
    baseline_length = exp_config['baseline_length']
    extra_fixation_length = exp_config['trial']['extra_fixation_length']

left_key = exp_config['keys']['left_key']
right_key = exp_config['keys']['right_key']

# Set up EDF data file name and local data folder
#
# The EDF data filename should not exceed 8 alphanumeric characters
# use ONLY number 0-9, letters, & _ (underscore) in the filename
edf_fname = 'TEST'

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
# haoxue: put them as they are here for now, and potentially change it when i understand better what dlg_prompt means
dlg_title = 'Enter EDF File Name'
dlg_prompt = 'Please enter a subjectID with 8 or fewer characters\n' + \
             '[letters, numbers, and underscore].'

# test - very inaccurate! 
handness = 'left'



# loop until we get a valid filename
while True:
    dlg = gui.Dlg(dlg_title)
    dlg.addText(dlg_prompt)
    dlg.addField('subjectID:', edf_fname)
    dlg.addField('Right/left handed?', handness, choices=['left','right'])
    dlg.addText('\nPlease fill out the blanks below to complete the experiment setup:')
    dlg.addField('Run on Windows/Mac?', choices=['windows','Mac'])
    dlg.addField('Debug Mode?', choices=['0','1'])
    dlg.addfield('Eyelink connected?', choices=['0','1'])
    
    # show dialog and wait for OK or Cancel
    ok_data = dlg.show()
    if dlg.OK:  # if ok_data is not None
        print('EDF data filename: {}'.format(ok_data[0]))
    else:
        print('user cancelled')
        core.quit()
        sys.exit()

    # get the string entered by the experimenter
    tmp_str = dlg.data[0]
    # strip trailing characters, ignore the ".edf" extension (if there is any)
    edf_fname = tmp_str.rstrip().split('.')[0]

    # check if the filename is valid (length <= 8 & no special char)
    allowed_char = ascii_letters + digits + '_'
    if not all([c in allowed_char for c in edf_fname]):
        print('ERROR: Invalid EDF filename')
    elif len(edf_fname) > 8:
        print('ERROR: EDF filename should not exceed 8 characters')
    else:
        # save all the varables
        handness = ok_data[1]
        use_retina = ok_data[2]
        debug_mode = ok_data[3]
        is_connected = ok_data[4]
        break

# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time
time_str = time.strftime("_%Y_%m_%d", time.localtime())
session_identifier = edf_fname + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)
# name the behavioral data file
data_identifier = os.path.join(session_folder, 'taskData' + session_identifier + '.csv')

# Step 1: Connect to the EyeLink Host PC
#
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"
if dummy_mode:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()

# Step 2: Open an EDF data file on the Host PC
edf_file = edf_fname + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add a header text to the EDF file to identify the current experiment name
# This is OPTIONAL. If your text starts with "RECORDED BY " it will be
# available in DataViewer's Inspector window by clicking
# the EDF session node in the top panel and looking for the "Recorded By:"
# field in the bottom panel of the Inspector.
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# Step 3: Configure the tracker
#
# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
# 5-EyeLink 1000 Plus, 6-Portable DUO
eyelink_ver = 0  # set version to 0, in case running in Dummy mode
if not dummy_mode:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    # print out some version info in the shell
    print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

# File and Link data control
# what eye events to save in the EDF file, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
# what eye events to make available over the link, include everything by default
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# what sample data to save in the EDF data file and to make available
# over the link, include the 'HTARGET' flag to save head target sticker
# data for supported eye trackers
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Optional tracking parameters
# Sample rate, 250, 500, 1000, or 2000, check your tracker specification
if eyelink_ver > 2:
    el_tracker.sendCommand("sample_rate 1000")
# Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
el_tracker.sendCommand("calibration_type = HV9")

# Step 4: set up a graphics environment for calibration
#
# Open a window, be sure to specify monitor parameters
monitor_width = exp_config['monitor']['width']
monitor_dist = exp_config['monitor']['distance']
mon = monitors.Monitor('myMonitor', width=monitor_width, distance=monitor_dist)
win = visual.Window(fullscr=full_screen,
                    monitor=mon,
                    winType='pyglet',
                    units='pix')

# get the native screen resolution used by PsychoPy
scn_width, scn_height = win.size

# resolution fix for Mac retina displays
if 'Darwin' in platform.system():
    if use_retina:
        scn_width = int(scn_width/2.0)
        scn_height = int(scn_height/2.0)

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
# see the EyeLink Installation Guide, "Customizing Screen Settings"
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization, see Data
# Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)
# haoxue: maybe this one also needs a function?

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
print(genv)  # print out the version number of the CoreGraphics library

genv.setCalibrationColors(foreground_color, background_color)


# Set up the calibration target
#
# The target could be a "circle" (default), a "picture", a "movie" clip,
# or a rotating "spiral". To configure the type of calibration target, set
# genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
# genv.setTargetType('picture')
#
# Use gen.setPictureTarget() to set a "picture" target
# genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))
#
# Use genv.setMovieTarget() to set a "movie" target
# genv.setMovieTarget(os.path.join('videos', 'calibVid.mov'))

# Use a picture as the calibration target
genv.setTargetType('circle')

# Configure the size of the calibration target (in pixels)
# this option applies only to "circle" and "spiral" targets
genv.setTargetSize(24)

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('', '', '')

# resolution fix for macOS retina display issues
if use_retina:
    genv.fixMacRetinaDisplay()

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)


# define the plot objects outside of the loop (attributes can be changed within
# the loop)

# fixation in the middle of the screen
fixation = visual.ShapeStim(win,
            vertices   = ((0, -fixSize),(0, fixSize),(0,0),(-fixSize,0),(fixSize, 0)),
            lineWidth  = fixLineWidth,
            closeShape = False,
            lineColor  = fixColor,
            ori = 0
       )

# left slot machine
left_rect = visual.ShapeStim(win,
        vertices  = ((-rectDistCenter-rectWidth, -rectHeight/2), (-rectDistCenter-rectWidth, rectHeight/2),\
             (-rectDistCenter, rectHeight/2), (-rectDistCenter, -rectHeight/2)),
        lineWidth = rectLineWidth,
        closeShape = True,
        lineColor = rectColor,
        ori = 0)

# right slot machine
right_rect = visual.ShapeStim(win,
        vertices  = ((rectDistCenter+rectWidth, -rectHeight/2), (rectDistCenter+rectWidth, rectHeight/2),\
             (rectDistCenter, rectHeight/2), (rectDistCenter, -rectHeight/2)),
        lineWidth = rectLineWidth,
        closeShape = True,
        lineColor = rectColor,
        ori = 0)

# label/reward for the left slot machine
left_type = visual.TextStim(win,
    text = 'D',
    pos = (-rectDistCenter-rectWidth/2, 0),
    color = typeColor,
    height = typeSize,
    bold = typeBold,
)

# label/reward for the right slot machine
right_type = visual.TextStim(win,
    text = 'D',
    pos = (rectDistCenter+rectWidth/2, 0),
    color = typeColor,
    height = typeSize,
    bold = typeBold,
)

block_end_msg = 'This marks the end of this block.\nTake a rest if you need.\nWhen you are ready, press space to proceed.'
baseline_end_msg = 'This marks the end of the baseline measurement period.\nTake a rest if you need.\nWhen you are ready, press space to proceed.'

# Step 5: Set up the camera and calibrate the tracker

def run_calibrate():
    """ function to call if we want to run calibration during the task
    """

    task_msg = ''
    if dummy_mode:
        task_msg = task_msg + '\nNow, press ENTER to start the task'
    else:
        task_msg = task_msg + '\nNow, press ENTER twice to calibrate tracker' 

    show_msg(win, task_msg, msgColor, wait_for_keypress=True, key_list=['return','space'])

    # skip this step if running the script in Dummy Mode
    if not dummy_mode:
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print('ERROR:', err)
            el_tracker.exitCalibration()

def terminate_task(win):
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script

    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected() == 1:
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial(win)

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, msgColor, wait_for_keypress=False)
        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    saveData(data)
    # close the PsychoPy window
    win.close()
    core.quit()

def run_baseline():
    """ Pupil baseline measure
    """

    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()

    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1) # question for Deshawn: should I shut it down between blocks? or actually I do not need?
        # related q: can i do drift check even with the eye track still collecting?
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial(win)
        return pylink.TRIAL_ERROR
    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)

    # introduce them to the fixation page
    msg = 'Now we are going to measure your baseline pupil size.\nThis will take 5 minutes.'+\
    '\nPlease fixate your eyes on the cross in the center of the screen.\nPlease try to keep your eyes open and reduce blinking.'+\
    '\nWe will let you know when this is done and then you can take a rest.'

    show_msg(win, msg, msgColor, wait_for_keypress= True)

    baseline_onset_time = core.getTime()
    el_tracker.sendMessage('baseline_onset') # Haoxue: add trial number?
    while core.getTime() - baseline_onset_time <= baseline_length:
        fixation.draw()
        win.flip()
    
    # end of the baseline screen
    clear_screen(win) 
    show_msg(win, block_end_msg, msgColor, wait_for_keypress=True, key_list=['space'])
    el_tracker.sendMessage('baseline_end')

# Step 6: Run the experimental trials, index all the trials
def run_block(block_pars, block_index, curr_cond, practice_flag=0):
    """ Helper function specifying the events that will occur in a single trial

    """
    
    bandit_type = exp_config['cond'][curr_cond-1]
    
    if debug_mode:
        # during debug mode, print out the reward array
        _, machine1_reward_array = block_pars[0]
        _, machine2_reward_array = block_pars[1]
        print('machine1_reward_array:', machine1_reward_array)
        print('machine2_reward_array:', machine2_reward_array)
        
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()

    # send a "TRIALID" message to mark the start of a trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('BLOCKID %d' % block_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'BLOCK number %d' % block_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # we recommend drift-check at the beginning of each trial
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    #
    # Skip drift-check if running the script in Dummy Mode
    while not dummy_mode:
        # terminate the task if no longer connected to the tracker or
        # user pressed Ctrl-C to terminate the task
        if (not el_tracker.isConnected()) or el_tracker.breakPressed():
            terminate_task(win)
            return pylink.ABORT_EXPT

        # drift-check and re-do camera setup if ESCAPE is pressed
        try:
            error = el_tracker.doDriftCorrect(int(scn_width/2.0),
                                              int(scn_height/2.0), 1, 1)
            # break following a success drift-check
            if error is not pylink.ESC_KEY:
                break
        except:
            pass

    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()

    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1) # question for Deshawn: should I shut it down between blocks? or actually I do not need?
        # related q: can i do drift check even with the eye track still collecting?
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial(win)
        return pylink.TRIAL_ERROR
    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)

    # start of the block screen
    if practice_flag:
        block_start_msg = 'Practice Block'+\
            '\nSlot Machines in this block:'+\
            '\n'+\
            '\n'+bandit_type[0]+' and '+bandit_type[1]+\
            '\n'+\
                '\nPress Space to begin if you are ready.'
    else:
        block_start_msg = 'Block '+str(block_index)+' of '+str(n_blocks)+\
            '\nSlot Machines in this block:'+\
            '\n'+\
            '\n'+bandit_type[0]+' and '+bandit_type[1]+\
            '\n'+\
                '\nPress Space to begin if you are ready.'
    clear_screen(win) 
    show_msg(win, block_start_msg, msgColor, wait_for_keypress=True, key_list=['space'], textHeight=50)
    el_tracker.sendMessage('block_start')

    for trial in range(n_trials): # Haouxe: need to figure out a way to save data
        run_trial(trial, block_pars, bandit_type, curr_cond, block_index)

    # end of the block screen
    clear_screen(win) 
    show_msg(win, block_end_msg, msgColor, wait_for_keypress=True, key_list=['space'])
    el_tracker.sendMessage('block_end')

    # save choice Data at the block end
    saveData(data)

def run_practice():
    """ Helper function to run practice block 
        In the practice block, there is always two arms of different types and
        different mean

    """
    curr_label_logic = [x == labels[1] for x in exp_config['cond'][0]]
    
    # generate a pair of machines with differnt mean and different types
    machine1, machine2 = [[0, 0, 0, 0], [0, 0, 0, 0]]    
    while machine1[0] == machine2[0]:
        machine1 = gen_params(sd_observe, sd_mean_mu, sd_rw, labels, curr_label_logic[0])
        machine2 = gen_params(sd_observe, sd_mean_mu, sd_rw, labels, curr_label_logic[1])

    machine1_array = gen_params_array(machine1, n_trials)
    machine2_array = gen_params_array(machine2, n_trials)

    # run block and pass in the practice_flag as 1 (so the block index will be labeled
    # as -1)
    run_block([machine1_array, machine2_array], block_index=-1, curr_cond=1, practice_flag=1)


def calculate_bonus(data):
    """ Helper function calculating accumulated bonus multiplied by the scaling factor

    """
    bonus = np.floor(data['total_coin'][-1] * scaling_factor)

    # Caution: maximum and minimum bonus is now hard coded! May want to include
    # that in the config file too in the future to increase flexibility
    if bonus < 1: 
        return 1
    if bonus > 5:
        return 5
    return bonus

def saveData(data):
    """
    Collects all the data - experiment, behavioral task, stimuli specs and subject info, converts it to a pandas data frame and saves as a csv.
    """
    # The directory has been taken care of in the main flow of the experiment. 
    # May want to include this check in future versions to increase robustness
    # if not os.path.isdir('data'): os.mkdir('data')
    
    # convert the data type from dictionary to dataframe
    taskData = pd.DataFrame(data)
    taskData.to_csv(data_identifier, index = False, encoding = 'utf-8')


def run_trial(trial_index, block_pars, bandit_type, curr_cond, block_index):
    """ 
    Function to run each trial
    """

    # grab parameters from the function input 
    machine1_mean_array, machine1_reward_array = block_pars[0]
    machine2_mean_array, machine2_reward_array = block_pars[1]
    # make sure the type of each option is up to date
    left_type.text = bandit_type[0]
    right_type.text = bandit_type[1]

    # generate IFI
    fixation_length = np.random.uniform(low=fixation_length_min,high=fixation_length_max)
    
    # append data before the trial starts
    data['subjectID'].append(edf_fname)
    data['psychopyVers'].append(__version__)
    data['codeVers'].append(exp_config['codeVers'])
    data['sd_observe'].append(sd_observe)
    data['sd_mean_mu'].append(sd_mean_mu)
    data['sd_rw'].append(sd_rw)
    data['block'].append(block_index) # can it access to block index here?
    data['trial'].append(trial_index+1)
    data['scaling_factor'].append(scaling_factor)

    data['IFI'].append(fixation_length)
    data['cond'].append(curr_cond)
    
    data['mu1'].append(machine1_mean_array[trial_index])
    data['mu2'].append(machine2_mean_array[trial_index])
    data['reward1'].append(machine1_reward_array[trial_index])
    data['reward2'].append(machine2_reward_array[trial_index])
    data['start_coin'].append(start_coin)

    # append [] to data fields that will be filled during the task
    data['choiceRT'].append([])
    data['choice'].append([])
    data['reward'].append([])
    data['keycode'].append([])
    data['correct'].append([])
    
    # for the first trial in the first block (of practice and main experiment), 
    # total_coin = start_coin. 
    # Otherwise, initialize total_coin as the total_coin of the last trial

    if (block_index == 1 | block_index == -1) & trial_index == 0:
        data['total_coin'].append(data['start_coin'][0])
    else:
        data['total_coin'].append(data['total_coin'][-1])
    
    if machine1_mean_array[trial_index] >= machine2_mean_array[trial_index]:
        data['correctArm'].append('machine1')    
    else:
        data['correctArm'].append('machine2')    

    # part 0 : talk to eye tracker
    el_tracker.sendMessage('TRIALID %d' % trial_index)
    # record_status_message : show some info on the Host PC
    status_msg = 'TRIAL number %d' % trial_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # part 1: fixation (fixation)
    fixation.lineColor = (1, 1, 1)
            
    fixation_onset_time = core.getTime()
    el_tracker.sendMessage('fixation_onset') 
    
    if trial_index == 0:
        fixation_length += extra_fixation_length
        
    while core.getTime() - fixation_onset_time <= fixation_length:
        fixation.draw()
        win.flip()
    
    # part 2: stimulus presentation (fixation + bandits_type)
    stimulus_pre_with_fixation_onset_time = core.getTime()
    el_tracker.sendMessage('stimulus_pre_with_fixation_onset') 
    
    while core.getTime() - stimulus_pre_with_fixation_onset_time  <= stimulus_pre_with_fixation_length: # Haoxue: is core.getTime() an accurate one?
        fixation.draw()
        left_rect.draw()
        right_rect.draw()
        left_type.draw()
        right_type.draw()
        win.flip()
    
    # part 3: stimulus presentation + choice (bandits_type)
    stimulus_pre_without_fixation_onset_time = core.getTime()
    el_tracker.sendMessage('stimulus_pre_without_fixation_onset')
    
    # remove any existing key press
    event.clearEvents() 
    while core.getTime() - stimulus_pre_without_fixation_onset_time <= stimulus_pre_without_fixation_length_max:
        left_rect.draw()
        right_rect.draw()
        left_type.draw()
        right_type.draw()
        fixation.lineColor = (0, 1, 0)
        fixation.draw()
        win.flip()

        # collect choice in part 3
        RT = -1  # keep track of the response time
        get_keypress = False
        choice = -1 # default choice

        while (not get_keypress) or (core.getTime() - stimulus_pre_without_fixation_onset_time <= stimulus_pre_without_fixation_length_max):
            # present the picture for a maximum of 5 seconds
            if core.getTime() - stimulus_pre_without_fixation_onset_time > stimulus_pre_without_fixation_length_max: # Haoxue: is core.getTime() an accurate one?
                el_tracker.sendMessage('time_out')
                break

            # abort the current trial if the tracker is no longer recording
            error = el_tracker.isRecording()
            if error is not pylink.TRIAL_OK:
                el_tracker.sendMessage('tracker_disconnected')
                abort_trial(win)
                return error

            # check keyboard events
            for keycode, modifier in event.getKeys(modifiers=True):
                if keycode == left_key:
                    # send over a message to log the key press
                    el_tracker.sendMessage('key_press left_option_chosen')
                    # get response time in ms, PsychoPy report time in sec
                    RT = int((core.getTime() - stimulus_pre_without_fixation_onset_time )*1000)
                    # record which option is chosen 
                    get_keypress = True
                    choice = 'machine1'
                    left_type.text = machine1_reward_array[trial_index]
                    data['choiceRT'][-1] = RT
                    data['choice'][-1] = choice
                    data['reward'][-1] = machine1_reward_array[trial_index]
                    data['keycode'][-1] = keycode
                    data['correct'][-1] = choice == data['correctArm'][-1]
                    data['total_coin'][-1] += machine1_reward_array[trial_index]
#                    break

                if keycode == right_key:
                    # send over a message to log the key press
                    el_tracker.sendMessage('key_press right_option_chosen')
                    # get response time in ms, PsychoPy report time in sec
                    RT = int((core.getTime() - stimulus_pre_without_fixation_onset_time )*1000)
                    # record which option is chosen
                    get_keypress = True
                    choice = 'machine2'
                    right_type.text = machine2_reward_array[trial_index]
                    data['choiceRT'][-1] = RT
                    data['choice'][-1] = choice
                    data['reward'][-1] = machine2_reward_array[trial_index]
                    data['keycode'][-1] = keycode
                    data['correct'][-1] = choice == data['correctArm'][-1]
                    data['total_coin'][-1] += machine2_reward_array[trial_index]
#                    break

                # Abort a trial if "ESCAPE" is pressed
                if keycode == 'escape':
                    el_tracker.sendMessage('trial_skipped_by_user')
                    # clear the screen
                    clear_screen(win)
                    # abort trial
                    abort_trial(win)
                    return pylink.SKIP_TRIAL

                # Terminate the task if Ctrl-c
                if keycode == 'c' and (modifier['ctrl'] is True):
                    el_tracker.sendMessage('terminated_by_user')
                    terminate_task(win)
                    return pylink.ABORT_EXPT

    # part 4: reward presentation
    reward_pre_without_fixation_onset_time = core.getTime()
    el_tracker.sendMessage('reward_pre_without_fixation_onset')
    while core.getTime() - reward_pre_without_fixation_onset_time <= reward_pre_without_fixation_length:
        left_rect.draw()
        right_rect.draw()
        left_type.draw()
        right_type.draw()
        if data['keycode'][-1] == []:
            fixation.lineColor = (1,0,0)
        fixation.draw()
        win.flip()
    
    # clear the screen
    clear_screen(win)
    el_tracker.sendMessage('blank_screen')
    # send a message to clear the Data Viewer screen as well
    el_tracker.sendMessage('!V CLEAR 128 128 128')

    # record trial variables to the EDF data file, for details, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    # Haoxue: they need to follow the format !V TRIAL_VAR your_var your_value 
    el_tracker.sendMessage('!V TRIAL_VAR condition %s' % curr_cond)
    el_tracker.sendMessage('!V TRIAL_VAR RT %d' % RT)
    el_tracker.sendMessage('!V TRIAL_VAR Choice %s' % choice) # there seem to be some problem in saving this part of the data

    # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    




block_list = np.random.randint(low=1, high=5, size=n_blocks)

# lets start the task!!!
data = initializeData()
seed = np.random.randint(low=1000000, high=9999999, size=1)
np.random.seed(seed)

# put tracker in idle/offline mode before recording
el_tracker.setOfflineMode()

# calibrate
run_calibrate()

# Baseline Measurement
if not dummy_mode:
    run_baseline()

# run practice
run_practice()

fixation.lineColor = (1,1,1)

# run real task
for j in range(len(block_list)):
    # generate index for the current label - 0: not first label; 1: first label
    curr_label_logic = [x == labels[1] for x in exp_config['cond'][block_list[j]-1]]
    
    # generate a pair of machines with differnt mean and different types
    machine1, machine2 = [[0, 0, 0, 0], [0, 0, 0, 0]]    
    while machine1[0] == machine2[0]:
        machine1 = gen_params(sd_observe, sd_mean_mu, sd_rw, labels, curr_label_logic[0])
        machine2 = gen_params(sd_observe, sd_mean_mu, sd_rw, labels, curr_label_logic[1])

    machine1_array = gen_params_array(machine1, n_trials)
    machine2_array = gen_params_array(machine2, n_trials)

    run_block([machine1_array, machine2_array], j+1, block_list[j])

end_msg = 'You have finished the virtual vegas task. Well done!'+\
    '\nPress SPACE to see how much you have earned as a bonus in the task!.'
show_msg(win, end_msg, msgColor, wait_for_keypress=True, key_list=['space'])

bonus = calculate_bonus(data)

bonus_msg = 'Your accumulated reward equals to a reward of $'+str(bonus)+'!'+\
    '\nThe bonus will be paid together with your base rate at the end of the experiment.'+\
    '\nPress let the experimenter know you have finished.'
    
show_msg(win, bonus_msg, msgColor, wait_for_keypress=True, key_list=['space'])


# stop recording; add 100 msec to catch final events before stopping
pylink.pumpDelay(100)
el_tracker.stopRecording()

# TODO: calculate bonus

# Step 7: disconnect, download the EDF file, then terminate the task
terminate_task(win)