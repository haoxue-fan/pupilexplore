# Haoxue: try to build a general framework for our task without all the object thing
# the current goal is only use object for the trial part.

# also need to think about how they look when they are put on mac instead of windows

import pylink
import os
import platform
import random
import time
import sys
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, gui
from PIL import Image  # for preparing the Host backdrop image
from string import ascii_letters, digits
import numpy as np

# define a few helper functions for trial handling


def clear_screen(win, background_color = [-1,-1,-1]):
    """ clear up the PsychoPy window"""

    # win.fillColor = genv.getBackgroundColor()
    win.fillColor = background_color
    win.flip()


def show_msg(win, text, msgcolor, wait_for_keypress=True, key_list=None, textHeight=None):
    """ Show task instructions on screen"""
    scn_width, scn_height = win.size
    if textHeight is not None:
        msg = visual.TextStim(win, text,
                          color=msgcolor,
                          wrapWidth=scn_width/2,
                          height=textHeight)
    else:
        msg = visual.TextStim(win, text,
                              color=msgcolor,
                              wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys(keyList=key_list)
        clear_screen(win)



def abort_trial(win):
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR
    
def gen_params(sd_observe, sd_mean_mu, sd_rw, label, machine_iloc = -1):
    if machine_iloc == -1:
        machine_iloc = np.random.randint(low=0, high=2, size=1)
    sd_options = [0, sd_observe]
    sd_rw_options = [0, sd_rw]
    curr_mu = np.random.normal(0, sd_mean_mu)
    curr_sd = sd_options[machine_iloc]
    curr_label = label[machine_iloc]
    curr_sd_rw = sd_rw_options[machine_iloc]
    return curr_mu, curr_sd, curr_label , curr_sd_rw
  
def gen_params_array(machine_params, n_trials):
    mean_array = [0] * n_trials
    reward_array = [0] * n_trials
    mean_array[0] = int(np.round(machine_params[0]))
    reward_array[0] = int(np.round(np.random.normal(machine_params[0], machine_params[1])))
    for j in range(1, n_trials):
        mean_array[j] = int(np.round(np.random.normal(mean_array[j-1], machine_params[3])))
        reward_array[j] = int(np.round(np.random.normal(mean_array[j], machine_params[1])))
    return mean_array, reward_array

    

def initializeData():
    taskData = {
        'subjectID': [],
        'seed': [],
        'psychopyVers': [],
        'codeVers': [],
        'use_retina': [],
        'handness': [],
        'glass': [],
        'debug_mode': [],
        'dummy_mode': [],
        'sd_observe': [],
        'sd_mean_mu': [],
        'sd_rw': [],
        'block': [],
        'trial': [],
        'scaling_factor': [],
        'choiceRT': [],
        'IFI': [],
        'correct': [],
        'correctArm': [],
        'choice': [],
        'keycode': [],
        'cond': [],
        'mu1': [],
        'mu2': [],
        'reward1': [],
        'reward2': [],
        'reward': [],
        'start_coin': [],
        'total_coin': [],
    }   
    return taskData


