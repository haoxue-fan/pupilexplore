# ----------------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------------

# quick_check.py
# A short function to open the eye tracker and quickly check whether it is possible to track the participant's eye. 
# No data needs to be recorded. 

# Written by Haoxue Fan
# Last updated on May 2022

# This file is run within psychopy 2022.1.3 and pylink downloaded from SR
# research developer kit on a Windows Computer. The integration with eyelink
# hasn't been tested on a mac.

# This file should be used with utils.py and main_config.yaml in the same folder

from asyncio import wait_for
from multiprocessing import dummy

import pylink, os, platform, random, time, sys, yaml
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import core, event, monitors, gui, __version__
#from PIL import Image  # for preparing the Host backdrop image
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

use_retina = exp_config['use_retina']
full_screen = exp_config['full_screen']
msgColor = exp_config['msg']['msgColor']
foreground_color = exp_config['calibration']['foreground_color']
background_color = exp_config['calibration']['background_color']
# Step 1: Connect to the EyeLink Host PC
#
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"

try:
    el_tracker = pylink.EyeLink("100.1.1.1")
except RuntimeError as error:
    print('ERROR:', error)
    core.quit()
    sys.exit()

# Step 3: Configure the tracker
#
# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
# 5-EyeLink 1000 Plus, 6-Portable DUO
eyelink_ver = 0  # set version to 0, in case running in Dummy mode
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


# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file

# resolution fix for macOS retina display issues
if use_retina:
    genv.fixMacRetinaDisplay()

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)

def run_calibrate():
    """ function to call if we want to run calibration during the task
    """

    task_msg = '\nNow, press ENTER twice to enter eye tracker set up page'
    show_msg(win, task_msg, msgColor, wait_for_keypress=True, key_list=['return'])

    # skip this step if running the script in Dummy Mode
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

        el_tracker.close()

    # close the PsychoPy window
    win.close()
    core.quit()
# define the plot objects outside of the loop (attributes can be changed within
# the loop)


# we do not need to save data AT ALL for the quick check

#block_list = np.random.randint(low=1, high=5, size=n_blocks)

# put tracker in idle/offline mode before recording
el_tracker.setOfflineMode()

# calibrate
run_calibrate()

# stop recording; add 100 msec to catch final events before stopping
pylink.pumpDelay(100)
el_tracker.stopRecording()

# Step 7: disconnect, download the EDF file, then terminate the task
terminate_task(win)