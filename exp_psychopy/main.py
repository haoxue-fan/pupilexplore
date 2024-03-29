# ----------------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------------

# main.py:
# The main function to run the two-armed bandit task used in Gershman (2019)

# Some of the functions were adapted from the jspsych version of the task which
# can be found in cognition.run phelpslab account
# Written by Haoxue Fan
# Last updated on June 2022

# This file is run using psychopy 2022.1.4 and pylink on a Windows Computer. 
# The integration with eyelink hasn't been tested on a mac.

# This file should be used with utils.py and main_config.yaml in the same folder

from asyncio import wait_for
from multiprocessing import dummy

import pylink, os, platform, random, time, sys, yaml
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, gui, __version__
from string import ascii_letters, digits
from utils import * 
import numpy as np
import pandas as pd


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

use_retina = exp_config['use_retina'] # 1 if the exp runs on a Mac with retina screen
dummy_mode = exp_config['dummy_mode'] # 1 if the exp runs w/n connecting an eye tracker
debug_mode = exp_config['debug_mode'] # 1 if in the debug mode (fewer trials, shorter)
full_screen = exp_config['full_screen'] # 1 if the exp opens a full screen. 0 will be useful when debugging
msgColor = exp_config['msg']['msgColor'] # color of text msg
typeColor = exp_config['type']['typeColor'] # color of slot machine type
typeSize = exp_config['type']['typeSize'] # size of slot machine type
typeBold = exp_config['type']['typeBold'] # ?
foreground_color = exp_config['calibration']['foreground_color'] # color of the figures
background_color = exp_config['calibration']['background_color'] 
cond = exp_config['cond'][0] # 4 cond. See Gershman (2019) for more info
fixSize = exp_config['fixation']['fixSize'] # size of fixation
fixLineWidth = exp_config['fixation']['fixLineWidth'] 
fixColor = exp_config['fixation']['fixColor']
rectWidth = exp_config['rect']['rectWidth'] # width of slot machine
rectHeight = exp_config['rect']['rectHeight']
rectLineWidth = exp_config['rect']['rectLineWidth']
rectDistCenter = exp_config['rect']['rectDistCenter']
rectColor = exp_config['rect']['rectColor']
blockmsgHeight = exp_config['msg']['blockmsgHeight'] 
fixation_action_color = exp_config['fixation']['fixation_action_color'] # color of the fixation signaling choice phase
fixation_miss_color = exp_config['fixation']['fixation_miss_color'] # color of the fixation signaling missing a trial/too fast
sd_mean_mu = exp_config['sd_mean_mu'] # std of the gaussian distribution that generates slot machines' mean
sd_observe = exp_config['sd_observe'] # std of the gaussian distribution that generats observations
sd_rw = exp_config['sd_rw'] # std of the random walk of the mean
labels = exp_config['labels'] # ?
scaling_factor = exp_config['scaling_factor'] # scaling factor used to generate final reward outcome
start_coin = exp_config['start_coin'] # start coin number (to compromise potential loss)
left_key = exp_config['keys']['left_key'] # keyword for the left slot machine
right_key = exp_config['keys']['right_key'] # keyword for the right slot machine
baseline_start_key = exp_config['keys']['baseline_start_key'] # secret key to start baseline measure
baseline_end_key = exp_config['keys']['baseline_end_key'] # secret key to proceed after baseline measure
before_drift_check_interval = exp_config['before_drift_check_interval'] # loading screen before drift check
handness = exp_config['handness'] # default for participant's handness
glass = exp_config['glass'] # default for participants' glass situation
practice_flag = exp_config['practice_flag'] # default for including practice
calibrate_flag = exp_config['calibrate_flag'] # default for including calibration
baseline_flag = exp_config['baseline_flag'] # default for including baseline
experiment_flag = exp_config['experiment_flag'] # default for main experiment

# Set up EDF data file name and local data folder
#
# The EDF data filename should not exceed 8 alphanumeric characters
# use ONLY number 0-9, letters, & _ (underscore) in the filename
edf_fname = 'TEST'

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
dlg_title = 'Enter EDF File Name'
dlg_prompt = 'Please enter a subjectID with 8 or fewer characters\n' + \
             '[letters, numbers, and underscore].'
start_block_n = 1

# loop until we get a valid filename
while True:
    # draw the dialogue window
    dlg = gui.Dlg(dlg_title)
    dlg.addText(dlg_prompt)
    dlg.addField('subjectID:', edf_fname)
    dlg.addField('Right/left handed?', choices=['left','right'])
    dlg.addField('Glasses/Contact Lens?', choices=['None','Glasses','Contact Lens'])
    dlg.addText('\nPlease fill out the blanks below to complete the experiment setup:')
    dlg.addField('Run on a retina screen?', choices=[0,1])
    dlg.addField('Debug Mode?', choices=[0,1])
    dlg.addField('Eyelink disconnected?', choices=[0,1])
    dlg.addText('\nPlease fill out the blanks below to indicate the included stages of the task:')
    dlg.addField('Practice?', choices = [1,0])
    dlg.addField('Calibrate? (strongly recommend include)', choices = [1,0])
    dlg.addField('Baseline Pupil Size?', choices = [1,0])
    dlg.addField('Experiment?', choices = [1,0])
    dlg.addField('Start from Round?', start_block_n) # could possibly use this to start from a specific round
    
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
    elif edf_fname == 'TEST':
        print('WARNING: EDF filename is the same as TEST participant. Please change')
    else:
        # save all the varables
        handness = ok_data[1]
        glass = ok_data[2]
        use_retina = ok_data[3]
        debug_mode = ok_data[4]
        dummy_mode = ok_data[5]
        practice_flag = ok_data[6]
        calibrate_flag = ok_data[7]
        baseline_flag = ok_data[8]
        experiment_flag = ok_data[9]
        start_block_n = ok_data[10]
        print(ok_data)
        break

# choose between two sets of parameters given debug_mode
if debug_mode:
    n_blocks = exp_config['n_blocks_debug']
    n_trials = exp_config['n_trials_debug']
    fixation_length_min = exp_config['trial']['fixation_length_min_debug']
    fixation_length_max = exp_config['trial']['fixation_length_max_debug']
    stimulus_pre_with_fixation_length = exp_config['trial']['stimulus_pre_with_fixation_length_debug']
    stimulus_pre_green_fixation_length_max = exp_config['trial']['stimulus_pre_green_fixation_length_max_debug']
    reward_pre_red_fixation_length = exp_config['trial']['reward_pre_red_fixation_length_debug']
    baseline_length = exp_config['baseline_length_debug']
    extra_fixation_length = exp_config['trial']['extra_fixation_length_debug']
else:
    n_blocks = exp_config['n_blocks']
    n_trials = exp_config['n_trials']
    fixation_length_min = exp_config['trial']['fixation_length_min']
    fixation_length_max = exp_config['trial']['fixation_length_max']
    stimulus_pre_with_fixation_length = exp_config['trial']['stimulus_pre_with_fixation_length']
    stimulus_pre_green_fixation_length_max = exp_config['trial']['stimulus_pre_green_fixation_length_max']
    reward_pre_red_fixation_length = exp_config['trial']['reward_pre_red_fixation_length']
    baseline_length = exp_config['baseline_length']
    extra_fixation_length = exp_config['trial']['extra_fixation_length']

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
bonus_identifier = os.path.join(session_folder, 'bonus' + session_identifier + '.txt')
# Connect to the EyeLink Host PC
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

block_end_msg = 'This marks the end of this round.'+\
'\n\nDo NOT move your head and Please keep your chin on the chinrest.'+\
'\n\nTake a rest if you need (close your eyes, blinking, etc).'+\
'\nWhen you are ready, press space to proceed.'

baseline_end_msg = 'This marks the end of the baseline measurement. Good job!'+\
'\n\nDo NOT move your head and Please keep your chin on the chinrest.'+\
'\n\nTake a rest if you need (close your eyes, blinking, etc).'+\
'\nPlease wait for the experimenter''s instructions.'


def run_calibrate():
    """ function to call if we want to run calibration during the task
    """

    task_msg = ''
    if dummy_mode:
        # if it is dummy mode, skip calibrate
        task_msg = task_msg + '\nNow, press ENTER to start the task'
    else:
        task_msg = task_msg + '\nNow, press ENTER twice to calibrate tracker' 

    show_msg(win, task_msg, msgColor, wait_for_keypress=True, key_list=['return'])

    # skip this step if running the script in Dummy Mode
    if not dummy_mode:
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print('ERROR:', err)
            el_tracker.exitCalibration()

def terminate_task(win):
    """ Terminate the task gracefully and retrieve the EDF data file
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
        msg = 'EDF data is transferring from EyeLink Host PC, please do not shut down the computer.'
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
        el_tracker.startRecording(1, 1, 1, 1) 
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial(win)
        return pylink.TRIAL_ERROR
    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)

    # introduction page before fixation 
    msg = 'Now we are going to measure your baseline pupil size.\nThis will take 5 minutes.'+\
    '\nPlease fixate your eyes on the cross in the center of the screen.\nPlease try to keep your eyes open and reduce blinking.'+\
    '\nWe will let you know when this is done and then you can take a rest.'+\
    '\nPlease avoid moving during this 5 minute or after. Otherwise we will have to abort the experiment.'

    show_msg(win, msg, msgColor, wait_for_keypress= True, key_list=baseline_start_key) # use baseline start key as a secret key

    # start timing
    baseline_onset_time = core.getTime()
    el_tracker.sendMessage('baseline_onset') 
    while core.getTime() - baseline_onset_time <= baseline_length:
        fixation.draw()
        win.flip()
    
    # end of the baseline screen
    clear_screen(win) 
    show_msg(win, baseline_end_msg, msgColor, wait_for_keypress=True, key_list=baseline_end_key)
    el_tracker.sendMessage('baseline_end')

def run_block(block_pars, block_index, curr_cond, practice_flag=0):
    """ Helper function specifying the events that will occur in a single trial
    block_pars: mean array and reward array for two slot machines
    block_index: block number, starts from 1
    curr_cond: current condition
    practice_flag: indicate whether it is practice block, default = 1
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

    # send a "TRIALID" message to mark the start of a trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('BLOCKID %d' % block_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'BLOCK number %d' % block_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # start of the block screen
    # disclaimer: the layout has only been tested on DELL 21.5 inch screen
    if practice_flag:
        block_start_msg = 'Practice Block'+\
            '\nSlot Machines in this round:'+\
            '\n\n\n\n\n\n\n'+\
            '\n\n\n\n\n\n\nUse left and right arrow key to indicate your decision.'+\
            '\nTry to avoid moving.'+\
            '\nPress Space when you are ready.'

    else:
        block_start_msg = 'Round '+str(block_index)+' of '+str(n_blocks)+\
            '\nSlot Machines in this round:'+\
            '\n\n\n\n\n\n\n'+\
            '\n\n\n\n\n\n\nUse left and right arrow key to indicate your decision.'+\
            '\nTry to keep your eyes open and avoid blinking.'+\
                '\nPress Space to begin calibration.'+\
                '\nOn the calibration page, look at the white dot and press Space.'
        
    # clean the screen before drawing
    clear_screen(win) 
    
    # draw slot machines, type info, and msg
    left_rect.draw()
    right_rect.draw()
    left_type.text = bandit_type[0]
    right_type.text = bandit_type[1]
    left_type.draw()
    right_type.draw()
    show_msg(win, block_start_msg, msgColor, wait_for_keypress=True, key_list=['space'], textHeight=blockmsgHeight, clear_screen_flag=False)
    el_tracker.sendMessage('BLOCKID %d block_start' % block_index)
    
    # we recommend drift-check at the beginning of each trial
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    #
    # Skip drift-check if running the script in Dummy Mode or if it is the practice block
    if not practice_flag:
        while not dummy_mode:
            # terminate the task if no longer connected to the tracker or
            # user pressed Ctrl-C to terminate the task
            if (not el_tracker.isConnected()) or el_tracker.breakPressed():
                terminate_task(win)
                return pylink.ABORT_EXPT

            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                # add an extra page before drift check to help participants be prepared
                before_drift_check_msg = 'Calibration starts in about '+str(before_drift_check_interval)+' seconds...'
    
                show_msg(win, before_drift_check_msg, msgColor, wait_for_keypress=False, textHeight=blockmsgHeight)
                el_tracker.sendMessage('BLOCKID %d before_drift_check' % block_index)
                before_drift_check_onset = core.getTime()
                # set the length of the extra page (uniformly sampled from [before_drift_check_interval-0.5, after_drift_check_interval+0.5])
                before_drift_check_interval_curr = np.random.uniform(low=before_drift_check_interval-0.5,high=before_drift_check_interval+0.5)
                while core.getTime() - before_drift_check_onset <= before_drift_check_interval_curr:
                    continue
                el_tracker.sendMessage('BLOCKID %d start_drift_check' % block_index)
                error = el_tracker.doDriftCorrect(int(scn_width/2.0),
                                                  int(scn_height/2.0), 1, 1)
                
                # break following a success drift-check
                if error is not pylink.ESC_KEY:
                    break
            except:
                pass

    # put tracker in idle/offline mode before recording
#     el_tracker.setOfflineMode()

    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1) 
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial(win)
        return pylink.TRIAL_ERROR
    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)

    for trial in range(n_trials): 
        run_trial(trial, block_pars, bandit_type, curr_cond, block_index)

    # end of the block screen
    clear_screen(win) 
    show_msg(win, block_end_msg, msgColor, wait_for_keypress=True, key_list=['space'])
    el_tracker.sendMessage('BLOCKID %d block_end' % block_index)
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

def run_trial(trial_index, block_pars, bandit_type, curr_cond, block_index):
    """ 
    Function to run each trial
    trial_index: curren trial number
    block_pars: mean array and reward array for two slot machines
    bandit_type: type of two slot machines
    curr_cond: current condition
    block_index: current block number
    """

    # grab parameters from the function input 
    machine1_mean_array, machine1_reward_array = block_pars[0]
    machine2_mean_array, machine2_reward_array = block_pars[1]
    # make sure the type of each option is up to date
    left_type.text = bandit_type[0]
    right_type.text = bandit_type[1]

    # generate IFI for each trial from a uniform distribution [fixation_length_min, fixation_length_max]
    fixation_length = np.random.uniform(low=fixation_length_min,high=fixation_length_max)
    
    # append data before the trial starts to avoid unequal length of the dictionary object data
    data['subjectID'].append(edf_fname)
    data['seed'].append(seed)
    data['psychopyVers'].append(__version__)
    data['codeVers'].append(exp_config['codeVers'])
    data['use_retina'].append(use_retina)
    data['handness'].append(handness)
    data['glass'].append(glass)
    data['debug_mode'].append(debug_mode)
    data['dummy_mode'].append(dummy_mode)
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
    data['early_press'].append([])
    
    # for the first trial in the first block (of practice and main experiment), 
    # total_coin = start_coin. 
    # Otherwise, initialize total_coin as the total_coin of the last trial
    if (block_index == 1 or block_index == -1) and (trial_index == 0):
        data['total_coin'].append(data['start_coin'][0])
    else:
        data['total_coin'].append(data['total_coin'][-1])
    
    if machine1_mean_array[trial_index] >= machine2_mean_array[trial_index]:
        data['correctArm'].append('machine1')    
    else:
        data['correctArm'].append('machine2')    

    # send message to eye tracker to signal the start of the trial
    el_tracker.sendMessage('BLOCKID %d TRIALID %d' % (block_index,trial_index))
    # record_status_message : show some info on the Host PC
    status_msg = 'BLOCK_number %d TRIAL number %d' % (block_index,trial_index)
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)
    # DRAW: fixation (fixation)
    fixation.lineColor = fixColor
            
    fixation_onset_time = core.getTime()
    el_tracker.sendMessage('BLOCKID %d TRIALID %d fixation_onset' % (block_index,trial_index)) 
    
    # first block of the task: increase the fixation length
    if trial_index == 0:
        fixation_length += extra_fixation_length
        
    while core.getTime() - fixation_onset_time <= fixation_length:
        fixation.draw()
        win.flip()
    
    # DRAW: stimulus presentation (fixation + bandits_type)
    #       record button press but continue as usual
    stimulus_pre_with_fixation_onset_time = core.getTime()
    el_tracker.sendMessage('BLOCKID %d TRIALID %d stimulus_pre_with_fixation_onset' % (block_index,trial_index)) 

    while core.getTime() - stimulus_pre_with_fixation_onset_time  <= stimulus_pre_with_fixation_length: 
        fixation.draw()
        left_rect.draw()
        right_rect.draw()
        left_type.draw()
        right_type.draw()
        win.flip()
        
        RT = -1  # keep track of the response time
        get_keypress = False
        choice = '' # default choice

        while (not get_keypress) or (core.getTime() - stimulus_pre_with_fixation_onset_time  <= stimulus_pre_with_fixation_length):
            # present the choise set for a max of stimulus_pre_with_fixation_length
            if core.getTime() - stimulus_pre_with_fixation_onset_time  > stimulus_pre_with_fixation_length:

#                win.getMovieFrame()
#                win.saveMovieFrames('stimulus_pre_with_white_fixation.png')
                break

            # abort the current trial if the tracker is no longer recording
            error = el_tracker.isRecording()
            if error is not pylink.TRIAL_OK:
                el_tracker.sendMessage('tracker_disconnected')
                abort_trial(win)
                return error
            
            if choice == '':
                # check keyboard events
                # no matter what they have pressed, send a message to the eye tracker
                for keycode, modifier in event.getKeys(modifiers=True):
                    
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
                    
                    # send over a message to log the early key press
                    el_tracker.sendMessage('BLOCKID %d TRIALID %d early_key_press %s' % (block_index,trial_index,keycode)) 
                    data['early_press'][-1] = 1
                    break


    
    # DRAW: stimulus presentation + choice (bandits_type, signaling choice phase)
    #         skip part 3 if part 2 is early press
    stimulus_pre_green_fixation_onset_time = core.getTime()
    el_tracker.sendMessage('BLOCKID %d TRIALID %d stimulus_pre_green_fixation_onset' % (block_index,trial_index)) 
      
    # remove any existing key press
    event.clearEvents() 
    while core.getTime() - stimulus_pre_green_fixation_onset_time <= stimulus_pre_green_fixation_length_max:
        left_rect.draw()
        right_rect.draw()
        left_type.draw()
        right_type.draw()
        fixation.lineColor = fixation_action_color
        fixation.draw()
        win.flip()

        # collect choice 
        RT = -1  # keep track of the response time
        get_keypress = False
        choice = '' # default choice

        while (not get_keypress) or (core.getTime() - stimulus_pre_green_fixation_onset_time <= stimulus_pre_green_fixation_length_max):
            # present the picture for a maximum of stimulus_pre_green_fixation_length_max
            if core.getTime() - stimulus_pre_green_fixation_onset_time > stimulus_pre_green_fixation_length_max: 
#                win.getMovieFrame()
#                win.saveMovieFrames('stimulus_pre_with_green_fixation.png')
                el_tracker.sendMessage('time_out')
                break

            # abort the current trial if the tracker is no longer recording
            error = el_tracker.isRecording()
            if error is not pylink.TRIAL_OK:
                el_tracker.sendMessage('tracker_disconnected')
                abort_trial(win)
                return error
            
            if choice == '':
                # check keyboard events
                for keycode, modifier in event.getKeys(modifiers=True):
                    if keycode == left_key:
                        # send over a message to log the key press
                        el_tracker.sendMessage('BLOCKID %d TRIALID %d key_press left_option_chosen' % (block_index,trial_index)) 
                        # get response time in ms, PsychoPy report time in sec
                        RT = int((core.getTime() - stimulus_pre_green_fixation_onset_time )*1000)
                        # record which option is chosen 
                        get_keypress = True
                        choice = 'machine1'
                        if machine1_reward_array[trial_index] > 0:
                            text = '+' + str(machine1_reward_array[trial_index])
                        else: 
                            text = machine1_reward_array[trial_index]
                        left_type.text = text
                        data['choiceRT'][-1] = RT
                        data['choice'][-1] = choice
                        data['reward'][-1] = machine1_reward_array[trial_index]
                        data['keycode'][-1] = keycode
                        data['correct'][-1] = choice == data['correctArm'][-1]
                        data['total_coin'][-1] += machine1_reward_array[trial_index]
                        break
                    if keycode == right_key:
                        # send over a message to log the key press
                        el_tracker.sendMessage('BLOCKID %d TRIALID %d key_press right_option_chosen' % (block_index,trial_index)) 
                        # get response time in ms, PsychoPy report time in sec
                        RT = int((core.getTime() - stimulus_pre_green_fixation_onset_time )*1000)
                        # record which option is chosen
                        get_keypress = True
                        choice = 'machine2'
                        if machine2_reward_array[trial_index] > 0:
                            text = '+' + str(machine2_reward_array[trial_index])
                        else: 
                            text = machine2_reward_array[trial_index]
                        right_type.text = text
                        data['choiceRT'][-1] = RT
                        data['choice'][-1] = choice
                        data['reward'][-1] = machine2_reward_array[trial_index]
                        data['keycode'][-1] = keycode
                        data['correct'][-1] = choice == data['correctArm'][-1]
                        data['total_coin'][-1] += machine2_reward_array[trial_index]
                        break

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
                    

                    el_tracker.sendMessage('BLOCKID %d TRIALID %d early_key_press %s' % (block_index,trial_index,keycode)) 
                    data['early_press'][-1] = 1
                    
    # DRAW: reward presentation or missing trial
    reward_pre_red_fixation_onset_time = core.getTime()
    el_tracker.sendMessage('BLOCKID %d TRIALID %d reward_pre_red_fixation_onset' % (block_index,trial_index)) 
                        
    while core.getTime() - reward_pre_red_fixation_onset_time <= reward_pre_red_fixation_length:
        left_rect.draw()
        right_rect.draw()
        left_type.draw()
        right_type.draw()
        # if no response was recorded or the participant has made an early response
        # change the fixation to the color of fixation_miss_color
        if data['keycode'][-1] == []:
            fixation.lineColor = fixation_miss_color
        fixation.draw()
        win.flip()
     
#    win.getMovieFrame()
#    win.saveMovieFrames('stimulus_pre_with_red_fixation.png')
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
    
def calculate_bonus(data):
    """ Helper function calculating accumulated bonus multiplied by the scaling factor
    data: the data object
    """
    # bonus is calculated by the total coin that they have collected 
    # potential problem may exist for the trials that they have missed
    bonus = np.floor(data['total_coin'][-1] * scaling_factor)
    
    # Bonus: min = 1; max = 5 (now hard coded)
    if bonus < 1: 
        bonus = 1
    if bonus > 5:
        bonus = 5
    with open(bonus_identifier, 'w') as output:
        output.write('bonus: '+str(bonus))

    return bonus

def saveData(data):
    """
    Collects all the data - experiment, behavioral task, stimuli specs and subject info, converts it to a pandas data frame and saves as a csv.
    data: the data object of the task
    """
    # The directory has been taken care of in the main flow of the experiment. 
    # May want to include this check in future versions to increase robustness
    # if not os.path.isdir('data'): os.mkdir('data')
    
    # convert the data type from dictionary to dataframe
    taskData = pd.DataFrame(data)
    taskData.to_csv(data_identifier, index = False, encoding = 'utf-8')

    
block_list = np.repeat([1,2,3,4], repeats=n_blocks/4)
np.random.shuffle(block_list)

# initialize data structure
data = initializeData()

# set the seed of the current task
seed = np.random.randint(low=1000000, high=9999999, size=1)
np.random.seed(seed)

# put tracker in idle/offline mode before recording (i.e., before the whole experiment)
el_tracker.setOfflineMode()

# run practice
if practice_flag:
    run_practice()
    # make sure the fixation is set to default fixColor after practice
    fixation.lineColor = fixColor

# calibrate
if calibrate_flag:
    run_calibrate()

# Baseline Measurement
if not dummy_mode:
    if baseline_flag:
        run_baseline()

# run real task
if experiment_flag:
    
    for j in range(start_block_n-1, len(block_list)):
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

print('BONUS: '+str(bonus))
        
bonus_msg = 'Your accumulated reward equals to a reward of $'+str(bonus)+'!'+\
    '\nThe bonus will be paid together with your base rate at the end of the experiment.'+\
    '\nPress let the experimenter know you have finished.'
    
show_msg(win, bonus_msg, msgColor, wait_for_keypress=True, key_list=['space'])


# stop recording; add 100 msec to catch final events before stopping
pylink.pumpDelay(100)
el_tracker.stopRecording()

# Last step: disconnect, download the EDF file, then terminate the task
terminate_task(win)