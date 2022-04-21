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

import yaml

with open('framework_config.yaml', 'r') as file:
    exp_config = yaml.safe_load(file)
# haoxue: consider put these things in a yaml file.
use_retina = exp_config['use_retina']
dummy_mode = exp_config['dummy_mode']
full_screen = exp_config['full_screen']
msgColor = exp_config['msg']['msgColor']
typeColor = exp_config['type']['typeColor']

# Set up EDF data file name and local data folder
#
# The EDF data filename should not exceed 8 alphanumeric characters
# use ONLY number 0-9, letters, & _ (underscore) in the filename
edf_fname = 'TEST'

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
# haoxue: put them as they are here for now, and potentially change it when i understand better what dlg_prompt means
dlg_title = 'Enter EDF File Name'
dlg_prompt = 'Please enter a file name with 8 or fewer characters\n' + \
             '[letters, numbers, and underscore].'

# loop until we get a valid filename
while True:
    dlg = gui.Dlg(dlg_title)
    dlg.addText(dlg_prompt)
    dlg.addField('File Name:', edf_fname)
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
    # strip trailing characters, ignore the ".edf" extension
    edf_fname = tmp_str.rstrip().split('.')[0]

    # check if the filename is valid (length <= 8 & no special char)
    allowed_char = ascii_letters + digits + '_'
    if not all([c in allowed_char for c in edf_fname]):
        print('ERROR: Invalid EDF filename')
    elif len(edf_fname) > 8:
        print('ERROR: EDF filename should not exceed 8 characters')
    else:
        break

# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time
time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = edf_fname + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

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
# Set a gamepad button to accept calibration/drift check target
# You need a supported gamepad/button box that is connected to the Host PC
# Haoxue: I do not think we have this?
# el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

# Step 4: set up a graphics environment for calibration
#
# Open a window, be sure to specify monitor parameters
# Haoxue: may want to make here customized - width and distance, are they for the monitor? is distance the distance between chiinrest and the screen?
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

#print('scn_width ', scn_width)
#print('scn_height ', scn_height)
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


# foreground_color = (-1, -1, -1)
# background_color = win.color
foreground_color = exp_config['calibration']['foreground_color']
background_color = exp_config['calibration']['background_color']

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


# define a few helper functions for trial handling


def clear_screen(win):
    """ clear up the PsychoPy window"""

    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_msg(win, text, msgcolor, wait_for_keypress=True):
    """ Show task instructions on screen"""
    msg = visual.TextStim(win, text,
                          color=msgcolor,
                          wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

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

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial():
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

# define fixation
# Haoxue: not sure why they do not have color here?


# Haoxue: this is the bread and butter we want to taylor for our own exp
# general idea being that we want a loop, 
cond = exp_config['cond'][0]

fixSize = exp_config['fixation']['fixSize']
fixLineWidth = exp_config['fixation']['fixLineWidth']
fixColor = exp_config['fixation']['fixColor']

rectWidth = exp_config['rect']['rectWidth']
rectHeight = exp_config['rect']['rectHeight']
rectLineWidth = exp_config['rect']['rectLineWidth']
rectDistCenter = exp_config['rect']['rectDistCenter']
rectColor = exp_config['rect']['rectColor']
    
    
def run_trial(trial_pars, trial_index, bandit_type):
    """ Helper function specifying the events that will occur in a single trial

    trial_pars - a list containing trial parameters, e.g.,
                ['cond_1', 'img_1.jpg']
    trial_index - record the order of trial presentation in the task
    """
    

    fixation = visual.ShapeStim(win,
            vertices   = ((0, -fixSize),(0, fixSize),(0,0),(-fixSize,0),(fixSize, 0)),
            lineWidth  = fixLineWidth,
            closeShape = False,
            lineColor  = fixColor,
            ori = 0
       )

    left_rect = visual.ShapeStim(win,
            vertices  = ((-rectDistCenter-rectWidth, -rectHeight/2), (-rectDistCenter-rectWidth, rectHeight/2), (-rectDistCenter, rectHeight/2), (-rectDistCenter, -rectHeight/2)),
            lineWidth = rectLineWidth,
            closeShape = True,
            lineColor = rectColor,
            ori = 0)
    right_rect = visual.ShapeStim(win,
            vertices  = ((rectDistCenter+rectWidth, -rectHeight/2), (rectDistCenter+rectWidth, rectHeight/2), (rectDistCenter, rectHeight/2), (rectDistCenter, -rectHeight/2)),
            lineWidth = rectLineWidth,
            closeShape = True,
            lineColor = rectColor,
            ori = 0)
    
    left_type = visual.TextStim(win,
        text = bandit_type[0],
        pos = (-rectDistCenter-rectWidth/2, 0),
        color = typeColor,
    )
    right_type = visual.TextStim(win,
        text = bandit_type[1],
        pos = (rectDistCenter+rectWidth/2, 0),
        color = typeColor,
    )

    # unpacking the trial parameters
    # cond, pic = trial_pars

    # load the image to display, here we stretch the image to fill full screen
    # img = visual.ImageStim(win,
    #                        image=os.path.join('images', pic),
    #                        size=(scn_width, scn_height))

    # get a reference to the currently active EyeLink connection
    # Haoxue: I actually do not understand the part below - why do i need to get another eyelink? why do I need to put the thing into offline mode? is that i am constantly open-close eyelink in this process?
    el_tracker = pylink.getEYELINK()

    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()

    # clear the host screen before we draw the backdrop
    el_tracker.sendCommand('clear_screen 0')

    # show a backdrop image on the Host screen, imageBackdrop() the recommended
    # function, if you do not need to scale the image on the Host
    # parameters: image_file, crop_x, crop_y, crop_width, crop_height,
    #             x, y on the Host, drawing options
##    el_tracker.imageBackdrop(os.path.join('images', pic),
##                             0, 0, scn_width, scn_height, 0, 0,
##                             pylink.BX_MAXCONTRAST)

    # If you need to scale the backdrop image on the Host, use the old Pylink
    # bitmapBackdrop(), which requires an additional step of converting the
    # image pixels into a recognizable format by the Host PC.
    # pixels = [line1, ...lineH], line = [pix1,...pixW], pix=(R,G,B)
    #
    # the bitmapBackdrop() command takes time to return, not recommended
    # for tasks where the ITI matters, e.g., in an event-related fMRI task
    # parameters: width, height, pixel, crop_x, crop_y,
    #             crop_width, crop_height, x, y on the Host, drawing options
    #
    # Use the code commented below to convert the image and send the backdrop
    # im = Image.open('images' + os.sep + pic)  # read image with PIL
    # im = im.resize((scn_width, scn_height))
    # img_pixels = im.load()  # access the pixel data of the image
    # pixels = [[img_pixels[i, j] for i in range(scn_width)]
    #           for j in range(scn_height)]
    # el_tracker.bitmapBackdrop(scn_width, scn_height, pixels,
    #                           0, 0, scn_width, scn_height,
    #                           0, 0, pylink.BX_MAXCONTRAST)

    # OPTIONAL: draw landmarks and texts on the Host screen
    # In addition to backdrop image, You may draw simples on the Host PC to use
    # as landmarks. For illustration purpose, here we draw some texts and a box
    # For a list of supported draw commands, see the "COMMANDS.INI" file on the
    # Host PC (under /elcl/exe)
    # left = int(scn_width/2.0) - 60
    # top = int(scn_height/2.0) - 60
    # right = int(scn_width/2.0) + 60
    # bottom = int(scn_height/2.0) + 60
    # draw_cmd = 'draw_filled_box %d %d %d %d 1' % (left, top, right, bottom)
    # el_tracker.sendCommand(draw_cmd)

    # send a "TRIALID" message to mark the start of a trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIALID %d' % trial_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'TRIAL number %d' % trial_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # Haoxue: potentially thinking about doing drift check once at the beginning of the block
    #         and the potential problem of changing the recording schedule
    # Haoxue: maybe this is where people talk about doing calibration before each block? i.e., they can do this automatically?
    # drift check
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
            terminate_task()
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
        abort_trial()
        return pylink.TRIAL_ERROR

    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)
    
    # show the image, and log a message to mark the onset of the image
    clear_screen(win)
#    fixation.draw()
    left_rect.draw()
    right_rect.draw()
    left_type.draw()
    right_type.draw()
    win.flip()
    
    el_tracker.sendMessage('fixation_onset') # Haoxue: I may want to add some custom msg here showing which trial it is?
    img_onset_time = core.getTime()  # record the image onset time

    # Send a message to clear the Data Viewer screen, get it ready for
    # drawing the pictures during visualization
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send over a message to specify where the image is stored relative
    # to the EDF data file, see Data Viewer User Manual, "Protocol for
    # EyeLink Data to Viewer Integration"
    # bg_image = '../../images/' + pic
    # imgload_msg = '!V IMGLOAD CENTER %s %d %d %d %d' % (bg_image,
    #                                                     int(scn_width/2.0),
    #                                                     int(scn_height/2.0),
    #                                                     int(scn_width),
    #                                                     int(scn_height))
    # el_tracker.sendMessage(imgload_msg)

    # send interest area messages to record in the EDF data file
    # here we draw a rectangular IA, for illustration purposes
    # format: !V IAREA RECTANGLE <id> <left> <top> <right> <bottom> [label]
    # for all supported interest area commands, see the Data Viewer Manual,
    # "Protocol for EyeLink Data to Viewer Integration"
    # ia_pars = (1, left, top, right, bottom, 'screen_center')
    # el_tracker.sendMessage('!V IAREA RECTANGLE %d %d %d %d %d %s' % ia_pars)

    # show the image for 5-secs or until the SPACEBAR is pressed
#    event.clearEvents()  # clear cached PsychoPy events
    RT = -1  # keep track of the response time
    get_keypress = False
    # Haoxue: not sure where I should specify them...
    left_key = exp_config['keys']['left_key']
    right_key = exp_config['keys']['right_key']
    choice = -1 # default choice

    while not get_keypress:
        # present the picture for a maximum of 5 seconds
        if core.getTime() - img_onset_time >= 5.0: # Haoxue: is core.getTime() an accurate one?
            el_tracker.sendMessage('time_out')
            break

        # abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return error

        # check keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Haoxue: try to make it into choosing left or right here - maybe j and k? 
            if keycode == left_key:
                # send over a message to log the key press
                el_tracker.sendMessage('key_press left_option_chosen')
                # get response time in ms, PsychoPy report time in sec
                RT = int((core.getTime() - img_onset_time)*1000)
                # record which option is chosen - should I take into account cond here?
                get_keypress = True
                choice = 0

            if keycode == right_key:
                # send over a message to log the key press
                el_tracker.sendMessage('key_press right_option_chosen')
                # get response time in ms, PsychoPy report time in sec
                RT = int((core.getTime() - img_onset_time)*1000)
                # record which option is chosen
                get_keypress = True
                choice = 1

            # # Stop stimulus presentation when the spacebar is pressed
            # if keycode == 'space':
            #     # send over a message to log the key press
            #     el_tracker.sendMessage('key_pressed')

            #     # get response time in ms, PsychoPy report time in sec
            #     RT = int((core.getTime() - img_onset_time)*1000)
            #     get_keypress = True

            # Abort a trial if "ESCAPE" is pressed
            if keycode == 'escape':
                el_tracker.sendMessage('trial_skipped_by_user')
                # clear the screen
                clear_screen(win)
                # abort trial
                abort_trial()
                return pylink.SKIP_TRIAL

            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task()
                return pylink.ABORT_EXPT
            print(keycode)

    # clear the screen
    clear_screen(win)
    el_tracker.sendMessage('blank_screen')
    # send a message to clear the Data Viewer screen as well
    el_tracker.sendMessage('!V CLEAR 128 128 128')

    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    el_tracker.stopRecording()

    # record trial variables to the EDF data file, for details, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('!V TRIAL_VAR condition %s' % cond)
    # el_tracker.sendMessage('!V TRIAL_VAR image %s' % pic)
    el_tracker.sendMessage('!V TRIAL_VAR RT %d' % RT)
    el_tracker.sendMessage('!V TRIAL_VAR Choice %d' % choice)

    # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)


# Step 5: Set up the camera and calibrate the tracker

# Show the task instructions
task_msg = 'In the task, you may press the SPACEBAR to end a trial\n' + \
    '\nPress Ctrl-C to if you need to quit the task early\n'
if dummy_mode:
    task_msg = task_msg + '\nNow, press ENTER to start the task'
else:
    task_msg = task_msg + '\nNow, press ENTER twice to calibrate tracker' 

show_msg(win, task_msg, msgColor, wait_for_keypress=True)

# skip this step if running the script in Dummy Mode
if not dummy_mode:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()

# Step 6: Run the experimental trials, index all the trials

# construct a list of 4 trials
# test_list = trials[:]*2

# randomize the trial list
# random.shuffle(test_list)

curr_cond = 1
curr_type = exp_config['cond'][curr_cond]
run_trial(1,1,curr_type)

# trial_index = 1
# for trial_pars in test_list:
#     run_trial(trial_pars, trial_index)
#     trial_index += 1

# Step 7: disconnect, download the EDF file, then terminate the task
terminate_task()
