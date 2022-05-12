# ----------------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------------

# luminance_test.py:
# The light-weighted function to test the luminance for 
# (1) find equal luminence color for the fixation
# (2) luminance magnitude for all possible layout (though still need to figure out the exact font size)

# eye tracker is not needed in this function
# data also does not need to be saved

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
from psychopy import visual, core, event, monitors, gui, __version__
from PIL import Image  # for preparing the Host backdrop image
from string import ascii_letters, digits
from utils import * 
import numpy as np
import pandas as pd

# get parameters from yaml
with open('luminance_config.yaml', 'r') as file:
    exp_config = yaml.safe_load(file)

display_length = exp_config['display_length']
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
blockmsgHeight = exp_config['msg']['blockmsgHeight']

#show_msg(win, bonus_msg, msgColor, wait_for_keypress=True, key_list=['space'])
monitor_width = exp_config['monitor']['width']
monitor_dist = exp_config['monitor']['distance']
mon = monitors.Monitor('myMonitor', width=monitor_width, distance=monitor_dist)
win = visual.Window(fullscr=full_screen,
                    monitor=mon,
                    winType='pyglet',
                    units='pix')

# get the native screen resolution used by PsychoPy
scn_width, scn_height = win.size


# fixation in the middle of the screen
fixation = visual.ShapeStim(win,
            vertices   = ((0, -fixSize),(0, fixSize),(0,0),(-fixSize,0),(fixSize, 0)),
            lineWidth  = fixLineWidth,
            closeShape = False,
            lineColor  = fixColor,
            ori = 0
       )

# left slot machine
# the small thing needs to be fixed!! (not yet)

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

# Step 5: Set up the camera and calibrate the tracker

def show_everything(left_text, right_text, fixation_color, total_time):
    left_type.text = left_text
    right_type.text = right_text
    fixation.lineColor = fixation_color
    onset_time = core.getTime()
    get_keypress = False
    while (not get_keypress) and (core.getTime() - onset_time  <= total_time): # Haoxue: is core.getTime() an accurate one?
        fixation.draw()
        left_rect.draw()
        right_rect.draw()
        left_type.draw()
        right_type.draw()
        win.flip()
        
        for keycode, modifier in event.getKeys(modifiers=True):
            if keycode == 'space':
                get_keypress = True
                break
            if keycode == 'return':
                print('color: '+str(fixation_color))
            if keycode == 'c' and (modifier['ctrl'] is True):
                terminate_task(win)
                 

def terminate_task(win):
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script

    """
    # close the PsychoPy window
    win.close()
    core.quit()

# resolution fix for Mac retina displays
if 'Darwin' in platform.system():
    if use_retina:
        scn_width = int(scn_width/2.0)
        scn_height = int(scn_height/2.0)


#show_everything('R','S',[0.6,0.6,0.6],display_length)
#for i in np.arange(0.9, 0.99, 0.03):
#    for j in np.arange(0.5, 0.8, 0.05):
#        show_everything('R','S',(i ,j,0),display_length)
#for i in np.arange(0.6, 0.8, 0.03):
#    show_everything('R','S',(0,i,0),display_length)

show_everything('R','R',(0.6,0.6,0.6),display_length)
show_everything('R','R',(0.9,0.55,0),display_length)
show_everything('R','R',(0,0.78,0),display_length)

#show_everything('R','S',(0,0,1),display_length)

# Step 7: disconnect, download the EDF file, then terminate the task
terminate_task(win)