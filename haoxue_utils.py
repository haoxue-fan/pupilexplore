# Haoxue: try to build a general framework for our task without all the object thing
# the current goal is only use object for the trial part.

# also need to think about how they look when they are put on mac instead of windows

from __future__ import division
from __future__ import print_function

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


def show_msg(win, text, msgcolor, wait_for_keypress=True, key_list=None):
    """ Show task instructions on screen"""
    scn_width, scn_height = win.size
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


def terminate_task(win):
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
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
        # Haoxue: we want to show this at the very end of the task!
        msg = 'EDF data is transferring from EyeLink Host PC...'
#        show_msg(win, msg, msgColor, wait_for_keypress=False)
        show_msg(win, msg, [1,1,1], wait_for_keypress=False)
        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


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
    
def gen_params(sd_observe, sd_mean_mu, sd_rw, label):
    machine_iloc = np.random.randint(low=0, high=2, size=1)
    sd_options = [0, sd_observe]
    sd_rw_options = [0, sd_rw]
    curr_mu = np.random.normal(0, sd_mean_mu)
    curr_sd = sd_options[machine_iloc]
    curr_label = label[machine_iloc]
    curr_sd_rw = sd_rw_options[machine_iloc]
    return curr_mu, curr_sd, curr_label, curr_sd_rw
  
def gen_params_array(machine_params, n_trials):
    mean_array = [0] * n_trials
    reward_array = [0] * n_trials
    mean_array[0] = int(np.round(machine_params[0]))
    reward_array[0] = int(np.round(np.random.normal(machine_params[0], machine_params[1])))
    for j in range(1, n_trials):
        mean_array[j] = int(np.round(np.random.normal(mean_array[j-1], machine_params[3])))
        reward_array[j] = int(np.round(np.random.normal(mean_array[j], machine_params[1])))
    return mean_array, reward_array




