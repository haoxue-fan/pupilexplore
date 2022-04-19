#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bandit.py

Main code for the experiment, defines all the necessary functions. We heavily rely on Psychopy library, in particular on its iohub module that we use for communicating with the eye tracker.

Updated: Mar 13 2018, Hrvoje Stojic
"""

from __future__ import division

# Importing the PsychoPy libraries that we want to use
from psychopy import core, visual, gui, event, data, monitors, __version__
from psychopy.iohub import (EventConstants, ioHubExperimentRuntime, module_directory, getCurrentDateTimeString)

# importing other needed libraries and functions
import csv, os, time, copy
from collections import Iterable

import random as rd
import math as m
import pandas as pd
import numpy as np

# os.chdir('/home/hstojic/Research/project/gas_MABeye/cExp/software')


## ----------------------------------------------------------------------------
## 1. Defining help functions
## ----------------------------------------------------------------------------

# ----
# Functions for generating the stimuli
# ----

class BernoulliArm:
    def __init__(self, prob, id):
        self._values = [0, 1]
        self._prob = [1 - prob, prob]
        self._id = id
    def play(self):
        return np.random.choice(self._values, size=1, replace=True, p=self._prob)
    def Id(self):
        return self._id

class LotteryArm:
    def __init__(self, values, probs, id):
        self._values = values
        self._probs = probs
        self._id = id
    def play(self):
        return np.random.choice(self._values, size=1, replace=True, p=self._probs)
    def Id(self):
        return self._id

class GaussianArm:
    def __init__(self, img, mu, sigma, ino, decay, id, seed):
        self._img = img
        self._mu = mu
        self._sigma = sigma
        self._ino = ino
        self._decay = decay
        self._id = id
        self._rn = np.random.RandomState(seed)
    def update(self):
        self._mu = self._decay * self._mu + self._rn.normal(0, self._ino, 1)[0]
    def play(self):
        return rd.gauss(self._mu, self._sigma)
    def getImg(self):
        return self._img
    def Id(self):
        return self._id


class Bandit:
    def __init__(self, arms, randomizeArms = None):
        for i in range(len(arms)):
            if not isinstance(arms[i], GaussianArm):
                raise ValueError, "GaussianArm object required."
        self._arms = arms
        if randomizeArms == 'once':
            rd.shuffle(self._arms)
    def noArms(self):
        return len(self._arms)
    def getArm(self, i):
        if isinstance(i, (int,long)):
            return self._arms[i]
        else:
            raise ValueError, "I need an int"
    def play(self, i):
        # play the chosen arm
        reward = self._arms[i].play()
        self.updateArms()
        return reward
    def updateArms(self):
        for i in range(len(self._arms)):
            self._arms[i].update()
    def randomizeArms(self):
        rd.shuffle(self._arms)
    def armIds(self):
        names = []
        for i in range(len(self._arms)):
            names.append(self._arms[i].Id())
        return names
    def armImages(self):
        images = []
        for i in range(len(self._arms)):
            images.append(self._arms[i].getImg())
        return images
    def valExpArms(self):
        values = []
        for i in range(len(self._arms)):
            values.append(self._arms[i]._mu)
        return values
    def valSdArms(self):
        values = []
        for i in range(len(self._arms)):
            values.append(self._arms[i]._sigma)
        return values
    def inoSdArms(self):
        values = []
        for i in range(len(self._arms)):
            values.append(self._arms[i]._ino)
        return values
    def decayArms(self):
        values = []
        for i in range(len(self._arms)):
            values.append(self._arms[i]._decay)
        return values


class MultiGameBandit:
    def __init__(self, games, scalingFactor, exchangeRate, units, randomizeGames = False):
        for i in range(len(games)):
            if not isinstance(games[i]["bandit"], Bandit):
                raise ValueError, "Bandit class required."
            if not isinstance(games[i]['noTrials'], int):
                raise ValueError, "Integer value needed."
        self._games = games
        self._scalingFactor = scalingFactor
        self._exchangeRate = exchangeRate
        self._units = units
        self._currentTrial = 0
        self._currentGame = 0
        self._done = False
        self._gameDone = False
        self._gamesList = range(len(self._games))
        # randomize games if instructed, including units and scaling factors,
        # exchange rate (though these have to be flipped jointly)
        if randomizeGames:
            rd.shuffle(self._games)
            rd.shuffle(self._units)
            rd.shuffle(self._gamesList)
            self._scalingFactor = [x for y, x in sorted(zip(self._gamesList, scalingFactor))]
            self._exchangeRate = [x for y, x in sorted(zip(self._gamesList, exchangeRate))]
    def nextTrial(self):
        if self._currentTrial < self._games[self._currentGame]["noTrials"] - 1:
            self._currentTrial += 1
        elif self._currentTrial == self._games[self._currentGame]["noTrials"] - 1:
            self._gameDone = True
    def nextGame(self):
        if self._currentGame < len(self._games) - 1:
            self._currentGame += 1
            self._currentTrial = 0
            self._gameDone = False
        else:
            self._done = True
    def done(self):
        return self._done
    def gameDone(self):
        return self._gameDone
    def play(self,i):
        out = self._games[self._currentGame]['bandit'].play(i)
        return out
    def randomizeArms(self):
        self._games[self._currentGame]['bandit'].randomizeArms()
    def getGame(self):
        return self._currentGame
    def getGameType(self):
        return self._games[self._currentGame]['gameType']
    def getNoArms(self):
        return self._games[self._currentGame]['noArms']
    def getBalanceInitial(self):
        return self._games[self._currentGame]['balanceInitial']
    def getExchangeRate(self):
        return self._exchangeRate[self._currentGame]
    def getScalingFactor(self):
        return self._scalingFactor[self._currentGame]
    def getRoundDecimal(self):
        return int(self._games[self._currentGame]['roundDecimal'])
    def getUnits(self):
        return self._units[self._currentGame]
    def getTrial(self):
        return self._currentTrial
    def getArmIds(self):
        return self._games[self._currentGame]['bandit'].armIds()
    def getArmImages(self):
        return self._games[self._currentGame]['bandit'].armImages()
    def getValExpArms(self):
        return self._games[self._currentGame]['bandit'].valExpArms()
    def getValSdArms(self):
        return self._games[self._currentGame]['bandit'].valSdArms()
    def getInoSdArms(self):
        return self._games[self._currentGame]['bandit'].inoSdArms()
    def getDecayArms(self):
        return self._games[self._currentGame]['bandit'].decayArms()


# ----
# Aux functions 
# ----

def loadingInstructionsFromTxt(path):
    '''
    Loads instructions from an external textual file from the "path" supplied in the argument. The textual file has to be in certain format: the text between two hash tags "#" will be taken to mean one screen/page of instructions. Most likely this will need a bit of tweaking to show right amount of text on each screen. Paragraphs are assumed to be in a single line in the textual file.
    '''

    # we first read in whole file as a list where each element is 
    # one line from the file, then we go elmt by elmt, clean it up from 
    # enter and space strings, and connect paragraphs to create screen
    instructionText = []
    with open(path) as inputfile:
        results = inputfile.readlines()
        hh=str()
        for i in range(len(results)):
            results[i] = unicode(results[i], 'utf-8')
            if (not results[i][0]=='#' and
                not results[i][0]==' ' and
                not results[i][0]=='\n' and
                not len(results[i][0])==0):
                hh += results[i]+'\n'
            if results[i][0]=='#':
                instructionText.append(hh)
                hh=str()

        # cleaning it up from some remaining weird spaces
        while '' in instructionText:
            instructionText.remove('')

    return instructionText

def flatten(l):
    """
    Flattens an irregular list. It's a generator so list() needs to be called on it to get a simple output.
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def frange(start, stop, step):
    """
    Creates a sequence of real numbers, from start to stop, with step sizes defined by step argument.
    """
    i = start
    res = []
    while i <= stop:
        res.append(i)
        i += step
    return res

def getArmPosition(distance, noArms): 
    """
    Computes the position of arms if we place them in a circle.
    """
    angles = frange(0, 2*m.pi, (2*m.pi/float(noArms)))
    positions = []
    for arm in range(len(angles)):
        positions.append([m.cos(angles[arm])*distance, m.sin(angles[arm])*distance])
    return positions[:-1]


## ----------------------------------------------------------------------------
## 2. Main function that executes the bandit experiment
## ----------------------------------------------------------------------------

class ExperimentRuntime(ioHubExperimentRuntime):

    """
    We create an experiment using psychopy and the ioHub framework by using
    the ioHubExperimentRuntime class as a parent class. We have to overwrite
    the parent method run() that lines up all the components of the experiment. 
    """

    def __init__(self,*args):

        ### Inheritance from the parent class
        ioHubExperimentRuntime.__init__(self, *args)
        
        ### Parameters
        # extract all task parameters from experiment_config.yaml
        self.pars = self.getConfiguration()['parameters']          
        # important modifiers
        self.EyeTracking = self.pars['EyeTracking']
        self.verbose = self.pars['verbose']
        self.practice = False
        self.calibration = 0
        # get info about psychopy and code version
        self.psychopyVers = __version__
        self.codeVersion =  self.getConfiguration()['version']
        # extract subject info and set the randomization seed
        self.subjectID = rd.randint(1000000, 9999999)
        self.seed = int(self.getUserDefinedParameters()['seed'])
        rd.seed(self.seed)

        # clock for total time spent in the experiment
        self.totalTime = core.MonotonicClock()
        self.timeInstructions = 0

        ### Stimuli 
        self.expID = None
        self.expCond = None
        self.noGames = None
        self.practiceImages = None
        self.stimuli = self.generateStimuli()
        self.noArms = self.stimuli._games[0]['noArms']

        ### Data
        self.data = self.initializeData()
        self.subjectData = None
        self.filepath = os.path.join('data', str(self.subjectID) + '_' + \
            data.getDateStr()) + '.csv'

        ### Various stimuli shapes  
        # initialize window
        self.win = self.initializeWindow()
        # circle to use for the Gaze Cursor
        self.gazeDot = visual.GratingStim(self.win, tex = None, mask = "gauss",
            pos = (0,0), color = self.pars['gazeDotColor'],
            size = (self.pars['gazeDotSize'], self.pars['gazeDotSize']))
        # circular AOI for checking fixations
        self.fixAOI = visual.Circle(self.win, radius = self.pars['AOIradius'])
        # A box we use for showing general instructions
        self.instructionsBox = visual.TextStim(self.win, 
            pos = [0,0],
            text = '', 
            height = self.pars['instructionsTextSize'],
            color = self.pars['instructionsTextColor'],
            wrapWidth = self.pars['instructionsWrapWidth']
        )
        # creating a box we will use for showing practice instructions
        self.practiceBox = visual.TextStim(self.win, 
            pos = [-16,0],
            text = '', 
            height = self.pars['practiceTextSize'],
            color = self.pars['practiceTextColor'],
            wrapWidth = self.pars['practiceWrapWidth']
        )
        # few objects useful for distinguishing the practice and normal trials
        self.stimulusTime = self.pars['stimulusTime']
        self.choiceTime = self.pars['choiceTime']
        self.feedbackDuration = self.pars['feedbackDuration']
        # fixation cross 
        self.fixation = visual.ShapeStim(self.win,
            vertices   = ((0, -self.pars['fixSize']),(0, self.pars['fixSize']),(0,0),(-self.pars['fixSize'],0),(self.pars['fixSize'], 0)),
            lineWidth  = self.pars['fixLineWidth'],
            closeShape = False,
            lineColor  = self.pars['fixColor'],
            ori = 0
       )
        # feedback
        self.feedbackText = visual.TextStim(self.win, 
            text = '', 
            color = self.pars['feedbackTextColor'],
            height = self.pars['feedbackTextSize']
        )
        self.feedbackBox = visual.Rect(self.win, 
            fillColor = self.pars['backgroundColor'],
            lineColor = self.pars['backgroundColor'],
            height = self.pars['feedbackTextSize']*1.2,
            width = self.pars['feedbackTextSize']*2.5
        )
        # late choice feedback text
        self.lateChoiceBox = visual.TextStim(self.win, 
            # pos = [0,0],
            text = "You were late with making a choice.\n\n          Please respond faster!", 
            color = self.pars['lateChoiceColor'],
            height = self.pars['instructionsTextSize'],
            alignHoriz = 'center',
            alignVert = 'center',
            wrapWidth = self.pars['instructionsWrapWidth']
        ) 

        # compute arm positions and create arm stimuli objects
        self.armPositions = getArmPosition(self.pars['armDistance'], self.noArms)
        # define stimuli objects
        self.armBoxes = []; self.armImages = [] 
        for i in range(self.noArms):
            self.armImages.append(visual.ImageStim(self.win, image=None, size=self.pars['imgSize'], pos=self.armPositions[i]))
            self.armBoxes.append(visual.Circle(self.win, radius = self.pars['armBoxRadius'], edges = 52, lineWidth = self.pars['armBoxLineWidth'], lineColor = self.pars['armBoxColor'], pos=self.armPositions[i] ))

        # trial and running total boxes, we don't use them at the moment 
        self.trialInfoBox = visual.TextStim(self.win, 
            pos = [0,0],
            text = '',  
            color = self.pars['instructionsTextColor'],
            height = self.pars['instructionsTextSize'],
            alignHoriz = 'center',
            alignVert = 'center'
        )
        correctBox = visual.TextStim(self.win, text = "total: 0",  alignHoriz = 'right', alignVert = 'top', height=self.pars['instructionsTextSize'])


    def run(self,*args):

        """
        This function creates a psychopy.visual.Window object based on settings from the iohub_config.yaml file.
        """

        ## Step 0. Extract some basic info

        # dict with info from experiment_config.yaml
        configInfo = self.getConfiguration()
        expInfo = self.getExperimentMetaData()
        sessionInfo = self.getSessionMetaData()
        userParsInfo = self.getUserDefinedParameters()  # also in sessionInfo

        # access the devices
        kb = self.devices.keyboard
        # mouse = self.devices.mouse


        ## Step 1. Get subject Information, set the seed and initialize data

        self.win.winHandle.set_fullscreen(False) # disable fullscreen
        self.win.winHandle.minimize() # minimise the PsychoPy window
        self.win.flip() # redraw the (minimised) window
        self.getSubjectData()
        self.win.winHandle.set_fullscreen(True) 
        self.win.winHandle.maximize()
        self.win.winHandle.activate()
        self.win.flip()

        ## Step 2. Showing the instructions for the whole experiment

        # brief start of the experiment screen
        txt = 'Press any key when you are ready to start the experiment.'
        self.instructionsBox.setText(txt)
        self.instructionsBox.draw()
        flipTime = self.win.flip()
        self.message("EXPERIMENT_START", flipTime)

        # Wait until a key event
        kb.clearEvents()
        kb.waitForPresses()

        # show instructions
        self.showExperimentInstructions(self.pars['startInstructionsPath'])


        ## Step 3. Practice trials
        
        if self.pars['practice']: 
            self.practice = True
            self.showText(self.pars['practiceStartPath'], 'practiceStart_text')
            self.runTask()
            self.showText(self.pars['practiceEndPath'], 'practiceEnd_text')
            self.practice = False

        
        ## Step 4. Executing the experimental task

        self.runTask()


        ## Step 5. Finishing

        # save the data
        self.saveData()

        # final message, we wait for the experimenter to press the special 
        # key combination to quit the experiment
        txt = 'The experiment is over, thank you for your participation!\n\nPlease contact the experimenter when you are ready.'
        self.instructionsBox.setText(txt)
        self.instructionsBox.draw()
        self.win.setColor(self.pars['backgroundColor'])
        flipTime = self.win.flip()
        self.message("goodbye_draw", flipTime)

        # we check the keys in a loop until we detect abortKeys
        # specified in experiment_config
        kb.clearEvents()
        targetKeysPressed = False
        while not targetKeysPressed:
            if kb.getPresses(keys = [self.pars['abortKeys'][1]], mods = [self.pars['abortKeys'][0]]): targetKeysPressed = True; 
        self.message("goodbye_end")

        # clear the events, QUIT
        self.hub.clearEvents('all')
        self.message("EXPERIMENT_END")
        core.quit()


    def message(self, txt, flipTime = None):
        """
        This function creates a psychopy.visual.Window object based on settings from the iohub_config.yaml file.
        """
        if self.practice: 
            game = "_practice"
        else:
            game = "_"
        self.hub.sendMessageEvent(
            text = str(self.subjectID) + game + txt, 
            sec_time = flipTime
        )

    def messageGame(self, txt, flipTime = None):
        """
        This function creates a psychopy.visual.Window object based on settings from the iohub_config.yaml file.
        """
        if self.practice: 
            game = "_practice_game_"
        else:
            game = "_game_"
        self.hub.sendMessageEvent(
            text = str(self.subjectID) + game + str(self.stimuli.getGame() + 1) + "_" + txt,
            sec_time = flipTime
        )

    def messageTrial(self, txt, flipTime = None):
        """
        This function creates a psychopy.visual.Window object based on settings from the iohub_config.yaml file.
        """
        if self.practice: 
            game = "_practice_game_"
        else:
            game = "_game_"
        self.hub.sendMessageEvent(
            text = str(self.subjectID) + game + str(self.stimuli.getGame() + 1) + "_trial_" + str(self.stimuli.getTrial() + 1) + "_" + txt,
            sec_time = flipTime
        )


    def initializeWindow(self):
        
        """
        Creates a psychopy.visual.Window object based on settings from the iohub_config.yaml file.
        """

        # access the display device
        display = self.devices.display

        # set the monitor information
        pixelResolution = [display.getConfiguration()['pixel_dimensions']['width'], display.getConfiguration()['pixel_dimensions']['height']]
        mon = monitors.Monitor(display.getPsychopyMonitorName())
        mon.setDistance(display.getDefaultEyeDistance())
        mon.setSizePix(pixelResolution)
        mon.setWidth(display.getPhysicalDimensions()['width']/10)  # to get cm

        # now we initialize the screen
        win = visual.Window(
            size = pixelResolution, 
            fullscr = True, 
            allowGUI = False,
            units = self.pars['units'], 
            monitor = mon, 
            screen = display.getDeviceNumber(),
            color = self.pars['backgroundColor'], 
            winType = 'pyglet', 
            allowStencil = True)

        return win


    def getSubjectData(self):
        """
        Shows psychopy GUI asking for some basic data. Subject ID has to be either left blank or an integer.
        """

        flipTime = self.win.flip()
        self.message("subjectdata_draw", flipTime)
        subjectData = {
            'Gender': ['Male', 'Female'], 
            'Age':'', 
            'Eye glasses?': ['Yes', 'No'], 
            'Contact lenses?': ['Yes', 'No'] 
        }
        infoDlg = gui.DlgFromDict(dictionary = subjectData, \
            title = 'Subject Info') 
        self.subjectData = subjectData


    def showExperimentInstructions(self, path):

        """
        Shows instructions text in a psychopy.visual.Window object based on an external textual file.
        """

        # aliases for things that are often called
        debug = self.verbose
        EyeTracking = self.EyeTracking
        pars = self.pars
        win = self.win

        # access the keyboard device
        kb = self.devices.keyboard
        kb.clearEvents()

        # fetch instructions
        instructions = loadingInstructionsFromTxt(path);

        # set the background color for instructions
        # we use the general background color
        win.setColor(pars['backgroundColor'])
        win.flip()

        # read out start time
        startTime = core.getTime()
        if debug: print "Start time: ", startTime

        # show instructions, we iterate over pages of instructions 
        # (defined by the length of the input file).
        noPage = 0
        while not noPage == len(instructions):

            # update the text depending on which page we are
            self.instructionsBox.text = instructions[noPage]
            self.instructionsBox.draw()
            flipTime = win.flip()
            self.hub.sendMessageEvent(str(self.subjectID) + "_instructions_draw_" + str(noPage), 
                sec_time = flipTime)

            # we check the keys in a loop until we detect some of the targeted
            # keys, specified in experiment_config
            kb.clearEvents()
            targetKeysPressed = False
            while not targetKeysPressed:
                if kb.getPresses(keys = pars['backKey']):
                    if noPage > 0: noPage -= 1; targetKeysPressed = True
                if kb.getPresses(keys = pars['continueKey']):
                    noPage += 1; targetKeysPressed = True
                if kb.getPresses(keys = [pars['abortKeys'][1]], mods = [pars['abortKeys'][0]]):
                    print "Aborting the experiment!"
                    core.quit()

        # read out end time
        endTime = core.getTime()
        if debug: print "end time: ", endTime
        self.message("instructions_end", endTime)

        # note the time when instructions finished 
        self.timeInstructions = endTime - startTime
        if debug: 
            print "Time instructions: ", self.timeInstructions
            print "Instructions finished!"

        # clear events
        kb.clearEvents()



    def showText(self, txt, msg, insertInfo = None, path = True):

        """
        Shows text about the performance in the game or the experiment. It first loads (properly formatted) text file indicated in path and then (optionally) inserts information passed through insertInfo argument (e.g. earnings from a game).
        """

        # aliases for things that are often called
        debug = self.verbose
        pars = self.pars

        # access the keyboard device
        kb = self.devices.keyboard
        kb.clearEvents()

        # if path, txt is actually a path and we load a text file
        if path: txt = loadingInstructionsFromTxt(txt);
        
        # read out start time
        startTime = core.getTime()
        if debug: print "Start time: ", startTime

        # display the txt
        if insertInfo:
            self.instructionsBox.setText(txt[0] %insertInfo)
        else:
            self.instructionsBox.setText(txt[0])
        self.instructionsBox.draw()
        flipTime = self.win.flip()
        self.message(msg + "_draw", flipTime)

        # we check the keys in a loop until we detect continueKey
        # specified in experiment_config
        targetKeysPressed = False
        while not targetKeysPressed:
            if kb.getPresses(keys = pars['continueKey']):
                targetKeysPressed = True
            if kb.getPresses(keys = [
                ], mods = [pars['abortKeys'][0]]):
                print "Aborting the experiment!"
                core.quit()

        # read out end time
        endTime = core.getTime()
        if debug: print "end time: ", endTime
        self.message(msg + "_end", endTime)

        # note the time when instructions finished 
        timeInstructions = endTime - startTime
        if debug: 
            print "Time text: ", timeInstructions

        # clear events
        kb.clearEvents()


    def initializeData(self):

        """
        Initializes the task data that will be recorded during the experiment.
        """

        taskData = {
            'subjectID': [],
            'sessionID': [],
            'seed': [],
            'psychopyVers': [],
            'codeVersion': [],
            'expID': [],
            'expCond': [],
            'instTime': [],
            'game': [], 
            'gameType': [], 
            'noArms': [],
            'trial': [],
            'calibration': [],
            'choiceMade': [],
            'chosenArm': [],
            'chosenArmId': [],
            'chosenArmPos': [],
            'choiceRT': [],  
            'feedbackRT': [],  
            'rewardScaled': [],
            'reward': [],
            'rewardExp': [],
            'rewardMaxExp': [],
            'regret': [],
            'regretExp': [],
            'correct': [],
            'correctArm': [],
            'correctArmId': [],
            'chosenRankExp': [],
            'timeFixStim': [],
            'timeFixChoice': [],
            'ITI': [],
            'IFI': [],
            'timeTrial': [],
            'rewardTotal': [],
            'rewardTotalScaled': [],
            'timeTotal': [],
            'exchangeRate': [],
            'scalingFactor': [],
            'units': []
        }

        # add fields that depend on the game characteristics
        for arm in range(1, self.noArms + 1):
            taskData["idArm" + str(arm)] = []
            taskData["imgArm" + str(arm)] = []
            taskData["posArm" + str(arm)] = []
            taskData["valExpArm" + str(arm)] = []
            taskData["valSdArm" + str(arm)] = []
            taskData["inoSdArm" + str(arm)] = []
            taskData["decayArm" + str(arm)] = []

        return taskData


    def saveData(self):
        """
        Collects all the data - experiment, behavioral task, stimuli specs and subject info, converts it to a pandas data frame and saves as a csv.
        """

        if not os.path.isdir('data'): os.mkdir('data')
        taskData = pd.DataFrame(self.data)
        
        # we include subjectData
        length = taskData.shape[0]
        subjectInfo = pd.DataFrame(
            {
            'gender': [self.subjectData['Gender']]*length,
            'age': [self.subjectData['Age']]*length,
            'glasses': [self.subjectData['Eye glasses?']]*length,
            'lenses': [self.subjectData['Contact lenses?']]*length
            }
        )
        data = pd.concat([subjectInfo, taskData], axis = 1) 
        data.to_csv(self.filepath, index = False, encoding = 'utf-8')


    def generateStimuli(self):
        
        """
        Generates stimuli (rewards etc) based on settings loaded from a csv file defined by "choiceSpecsPath".
        """

        # aliases for things that are often called
        debug = self.verbose
        pars = self.pars

        # load stimuli specs as a panda data frame 
        expSpecs = pd.read_csv(pars['choiceSpecsPath'])
        if debug: print "all exp specifications:", expSpecs

        # list of all conditions, as set in the file
        expConditions = expSpecs.expCond.unique()
        if debug: print "randomly assigned condition: ", expConditions

        # random assignment to conditions, between subject type
        self.expCond = rd.choice(expConditions)
        if debug: print "randomly assigned condition: ", self.expCond
        
        # randomize colors of the background 
        rd.shuffle(pars['gameColors'])

        # load the images for the arms and randomize them
        if pars['practice']: 
            imgDir = pars['imgPracticeDir']
            imgDirContent = os.listdir(imgDir)
            self.practiceImages = [os.path.join(imgDir, i) for i in imgDirContent]
            rd.shuffle(self.practiceImages)
        imgDir = pars['imgDir'] 
        imgDirContent = os.listdir(imgDir)
        imgPaths = [os.path.join(imgDir, i) for i in imgDirContent]
        rd.shuffle(imgPaths)


        ### generating the data

        # description of the games
        games = []

        # based on randomization we select only the stimuli specification 
        # for selected condition
        conditionData = expSpecs[expSpecs['expCond'] == self.expCond]
        if debug: print "condition data: ", conditionData
        self.expID = expSpecs.iloc[0]['expID']
        if debug: print "expID: ", self.expID

        # list of all games within condition
        gamesList = conditionData.game.unique()
        self.noGames = len(gamesList)
        if debug: print "list of all game IDs: ", gamesList

        # extracting variables that will be randomized
        scalingFactor = conditionData.scalingFactor.unique()
        exchangeRate = conditionData.exchangeRate.unique()
        units = conditionData.units.unique()

        # for loop that goes over each game, creates Arm class for each 
        # arm and adds additional details
        for gameID in gamesList: 
            # get the game specific data
            gameData = conditionData[conditionData['game'] == gameID]
            noArms = len(gameData['armID'])
            banditType = gameData.iloc[0]['banditType']

            # we put the game characteristics into a dictionary
            game = {}; arms = []
            for arm in range(noArms): 
                if banditType == 'Gaussian':
                    if gameData.iloc[arm]['seed'] == "None":
                        seed = rd.randint(1000, 40000)
                    else:
                        seed = gameData.iloc[arm]['seed']
                    arms.append(GaussianArm(imgPaths.pop(), gameData.iloc[arm]['mean'], gameData.iloc[arm]['sd'],  gameData.iloc[arm]['ino'], gameData.iloc[arm]['decay'], gameData.iloc[arm]['armID'], seed))
            game["bandit"] = Bandit(arms, pars['randomizeArms'])
            game['noArms'] = noArms 
            game['noTrials'] = int(gameData.iloc[0]['noTrials'])
            game['gameType'] = gameData.iloc[0]['gameType'] 
            game['balanceInitial'] = gameData.iloc[0]['balanceInitial']
            game['roundDecimal'] = gameData.iloc[0]['roundDecimal']
            game['banditType'] = banditType 

            # finally, we add the game specs to the games list
            games.append(game)

        # create the whole multi game bandit task
        expTask = MultiGameBandit(games, scalingFactor, exchangeRate, units, pars['randomizeGames'])

        return expTask


    def isFixation(self, positions, duration = 0.1, threshold = 0.95):
        """
        Computes proportion of gazes within object, only if duration is min as prescribed.
        """
        if len(positions) < 2:
            return False
        else:
            dur = 0
            validGazes = [positions[-1][1]]
            for elt in [-1*i for i in range(2, len(positions)+1)]:
                dur += positions[elt+1][0] - positions[elt][0]
                validGazes.append(positions[elt][1])
                if dur >= duration:
                    break
            propOnObject = sum(validGazes)/len(validGazes)
            if (dur >= duration) and (propOnObject > threshold):
                return True 
            else:
                return False


    def isObjectFixated(self, tracker, kb, duration = 0.1, threshold = 0.95):

        """
        Checks whether eyes are fixating on a particular object on the screen for a certain duration.
        """
        tracker.clearEvents()
        kb.clearEvents()
        self.fixAOI.setPos(self.fixation.pos)
        # check if gaze falls into it
        positions = []
        timeStart = core.getTime()
        fixated = False
        while not fixated:
            # Get the latest gaze position in display coord space
            gpos = tracker.getPosition()
            # verify if valid and in AOI surrounding the fix cross
            if isinstance(gpos, (tuple, list)):
                if self.fixAOI.contains(gpos):
                    positions.append((core.getTime(), 1))
                else:
                    positions.append((core.getTime(), 0))
                self.fixation.draw()
                if self.pars['gazeDotShow']:
                    self.gazeDot.setPos(gpos)
                    self.gazeDot.draw()
            else:
                self.fixation.draw()
            # we check if most of the positions within the alloted duration
            # were at the object
            if self.isFixation(positions, duration, threshold):
                fixated = True 
                self.fixation.opacity = self.pars['feedbackOpacity']
            self.practiceBox.draw()
            flipTime = self.win.flip() 

            # we check if calibration is called
            if kb.getPresses(keys = [self.pars['calKeys'][1]], mods = [self.pars['calKeys'][0]]):
                self.calibration = 1
                self.runTrackerCalibration(tracker, kb, intrial = True)  
  
        return {'flipTime': flipTime, 'timeFix': core.getTime() - timeStart}


    def isOptionChosen(self, tracker, kb, mouse, objects, duration):

        """
        Checks whether eyes are fixating on a particular option on the screen and a task key is pressed for choosing it. These actions have to be performed within a certain time limit, indicated with the duration argument.
        """

        # help vars
        optionsBoxes = self.armBoxes
        options = self.armImages
        rangeOptions = range(len(options))

        if self.EyeTracking:
            # clear the buffers
            tracker.clearEvents()
            kb.clearEvents() 
            choiceMade = False
            choiceStartTime = core.getTime()
            while ((core.getTime() - choiceStartTime) < duration) and (not choiceMade):
                # reset the opacities
                for i in rangeOptions: 
                    optionsBoxes[i].opacity = 1.0
                    options[i].opacity = 1.0
                # Get the latest gaze position in display coord space
                gpos = tracker.getPosition()
                kbpresses = kb.getPresses()
                # verify if gaze valid and within one of the arms
                if isinstance(gpos, (tuple, list)):
                    for arm in rangeOptions: 
                        if optionsBoxes[arm].contains(gpos):
                            if self.pars['taskKey'] in kbpresses:
                                choiceEndTime = core.getTime()
                                chosenArm = arm
                                chosenArmPos = copy.copy(optionsBoxes[arm].pos)
                                choiceMade = True
                            optionsBoxes[arm].opacity = self.pars['feedbackOpacity']
                            options[arm].opacity = self.pars['feedbackOpacity']
                    if self.pars['gazeDotShow']:
                        self.gazeDot.setPos(gpos)
                        self.gazeDot.draw()
                    if objects: 
                        [s.draw() for s in objects]
                    [s.draw() for s in options + optionsBoxes]
                else:
                    if objects: 
                        [s.draw() for s in objects]
                    [s.draw() for s in options + optionsBoxes]
                flipTime = self.win.flip() 
                
        else:
            # clear buffers
            event.clearEvents()
            # make mouse cursor visible
            position = (0, 0)
            mouse.setPos(newPos = position)  
            mouse.setVisible(True)
            # we show stimuli until one box is clicked
            choiceMade = False
            choiceStartTime = core.getTime()
            while ((core.getTime() - choiceStartTime) < duration) and (not choiceMade):
                # reset the opacities
                for i in rangeOptions: 
                    optionsBoxes[i].opacity = 1.0
                    options[i].opacity = 1.0
                # detecting mouse clicks
                for arm in rangeOptions: 
                    if optionsBoxes[arm].contains(mouse.getPos()):
                        optionsBoxes[arm].opacity = self.pars['feedbackOpacity']
                        options[arm].opacity = self.pars['feedbackOpacity']
                    if mouse.isPressedIn(optionsBoxes[arm]): 
                        choiceEndTime = core.getTime()
                        chosenArm = arm
                        chosenArmPos = copy.copy(optionsBoxes[arm].pos)
                        choiceMade = True
                if objects: 
                    [s.draw() for s in objects]
                [s.draw() for s in options + optionsBoxes]
                flipTime = self.win.flip()
        
        if choiceMade:
            return {'choiceMade':True, 'choiceEndTime':choiceEndTime, 'chosenArm':chosenArm, 'chosenArmPos':chosenArmPos, 'choiceStartTime':choiceStartTime}
        else:
            return {'choiceMade':False, 'choiceEndTime':None, 'chosenArm':'NA', 'chosenArmPos':'NA', 'choiceStartTime':choiceStartTime}


    def showGazeDot(self, tracker, kb, objects, duration):
        """
        Shows the gaze dot on the screen while tracking eye movements. Useful for debugging.
        """
        tracker.clearEvents()
        kb.clearEvents()
        fixationTime = core.getTime()
        while (core.getTime() - fixationTime) < duration:
            gpos = tracker.getPosition()
            if self.pars['gazeDotShow'] and type(gpos) in [tuple,list]:
                if objects: [s.draw() for s in objects]
                self.gazeDot.setPos(gpos)
                self.gazeDot.draw()
            else:
                if objects: [s.draw() for s in objects]
            flipTime = self.win.flip()

            # we check if calibration is called
            if kb.getPresses(keys = [self.pars['calKeys'][1]], mods = [self.pars['calKeys'][0]]):
                self.calibration = 1
                self.runTrackerCalibration(tracker, kb, intrial = True)

        return flipTime


    def transition(self, objects, mode = 'fadeout', duration = 2, freq = 60, opacityStart = 0, opacityEnd = 1, fixation = True, fixObjects = None):
        """
        Gradually fades out objects over the time specified by the duration argument. It depends on the screen frequency, usually 60 Hz. Fixation cross stays on the screen but rotates to signal the onset of the choice phase.
        """
        nsteps = int(duration*freq)
        opacitySteps = frange(opacityStart, opacityEnd, opacityEnd/(nsteps - 1))
        opacitySteps[-1] = opacityEnd
        oriSteps = frange(0, 45, 45/(nsteps - 1))
        oriSteps[-1] = 45
        if mode == 'fadein': 
            steps = range(len(opacitySteps))
        elif mode == 'fadeout':
            steps = [-1*i for i in range(1, len(opacitySteps) + 1)]
        for s in steps:
            if objects:
                for o in objects:
                    o.opacity = opacitySteps[s]
                    o.draw()
            if fixation:
                self.fixation.ori = oriSteps[s]
                self.fixation.draw()
            if fixObjects:
                [o.draw() for o in fixObjects]
            flipTime = self.win.flip()
        return flipTime


    def getPupilBaseline(self, tracker, kb, intrial = False):
        
        """
        Measures pupil area for a large amount of time to get estimate of an individual's baseliine.
        """
        if not intrial:
            # show the instructions first
            txt = loadingInstructionsFromTxt(self.pars['pupilBaselinePath'])
            self.instructionsBox.setText(txt[0])
            self.instructionsBox.draw()
            flipTime = self.win.flip()
            self.messageGame("pupilbaseline_instructions_draw", flipTime)
            # wait for the continue key
            kb.clearEvents()
            kb.waitForPresses(keys = self.pars['continueKey'])
            flipTime = self.win.flip()
            self.messageGame("pupilbaseline_instructions_end", flipTime)
            # setting the tracker
            tracker.setRecordingState(True)  
            # show the fixation cross
            self.fixation.draw()
            flipTime = self.win.flip()
            self.messageGame("pupilbaseline_fixation_draw", flipTime)
            flipTime = self.showGazeDot(tracker, kb, [self.fixation], self.pars["pupilBaselineDuration"])
            self.messageGame("pupilbaseline_fixation_end", flipTime)
            tracker.setRecordingState(False)
        else:
            self.fixation.draw()
            flipTime = self.win.flip()
            self.messageGame("pupilbaseline_fixation_draw", flipTime)
            flipTime = self.showGazeDot(tracker, kb, [self.fixation], self.pars["pupilBaselineDuration"])
            self.messageGame("pupilbaseline_fixation_end", flipTime)
            if not tracker.isRecordingEnabled(): 
                tracker.setRecordingState(True)  


    def getPupilCamera(self, tracker, kb, intrial = False):
        
        """
        Measures pupil area while the person is looking at the camera, necesseary for getting true pupil diameter.
        """
        if not intrial:
            # show the instructions first
            txt = loadingInstructionsFromTxt(self.pars['pupilCameraPath'])
            self.instructionsBox.setText(txt[0])
            self.instructionsBox.draw()
            flipTime = self.win.flip()
            self.messageGame("pupilcamera_instructions_draw", flipTime)
            # wait for the continue key
            kb.clearEvents()
            kb.waitForPresses(keys = self.pars['continueKey'])
            flipTime = self.win.flip()
            self.messageGame("pupilcamera_instructions_end", flipTime)
            # setting the tracker
            tracker.setRecordingState(True)  
            # show the blank screen
            flipTime = self.win.flip()
            self.messageGame("pupilcamera_fixation_draw", flipTime)
            flipTime = self.showGazeDot(tracker, kb, [], self.pars["pupilCameraDuration"])
            self.messageGame("pupilcamera_fixation_end", flipTime)
            tracker.setRecordingState(False)
        else:
            flipTime = self.win.flip()
            self.messageGame("pupilcamera_fixation_draw", flipTime)
            flipTime = self.showGazeDot(tracker, kb, [], self.pars["pupilCameraDuration"])
            self.messageGame("pupilcamera_fixation_end", flipTime)
            if not tracker.isRecordingEnabled(): 
                tracker.setRecordingState(True)  


    def runTrackerCalibration(self, tracker, kb, intrial = False):
        
        """
        Running the eye tracker default setup procedure.
        The details of the setup procedure (calibration, validation, etc)
        are unique to each implementation of the Common Eye Tracker 
        Interface. All have the common end goal of calibrating 
        the eye tracking system prior to data collection.
        """

        if not intrial:
            # show the instructions first
            txt = loadingInstructionsFromTxt(self.pars['trackerCalibrationPath'])
            self.instructionsBox.setText(txt[0])
            self.instructionsBox.draw()
            flipTime = self.win.flip()
            self.messageGame("trackercalibration_instructions_draw", flipTime)
            # wait for the continue key
            kb.clearEvents()
            kb.waitForPresses(keys = self.pars['continueKey'])
            flipTime = self.win.flip()
            self.messageGame("trackercalibration_instructions_end", flipTime)
            # setting the tracker
            tracker.setRecordingState(True)  
            flipTime = self.win.flip()
            self.messageGame("trackercalibration_start", flipTime)

            self.win.winHandle.set_fullscreen(False) # disable fullscreen
            self.win.winHandle.minimize() # minimise the PsychoPy window
            self.win.flip() # redraw the (minimised) window
            tracker.runSetupProcedure()
            self.win.winHandle.set_fullscreen(True) 
            self.win.winHandle.maximize()
            self.win.winHandle.activate()
            self.win.flip()

            flipTime = self.win.flip()
            self.messageGame("trackercalibration_end", flipTime)
            tracker.setRecordingState(False)
        else:
            flipTime = self.win.flip()
            self.messageGame("trackercalibration_start", flipTime)
            
            self.win.winHandle.set_fullscreen(False) # disable fullscreen
            self.win.winHandle.minimize() # minimise the PsychoPy window
            self.win.flip() # redraw the (minimised) window
            tracker.runSetupProcedure()
            self.win.winHandle.set_fullscreen(True) 
            self.win.winHandle.maximize()
            self.win.winHandle.activate()
            self.win.flip()
            
            flipTime = self.win.flip()
            self.messageGame("trackercalibration_end", flipTime)
            if not tracker.isRecordingEnabled(): 
                tracker.setRecordingState(True)  


    def runTask(self):

        """
        Executes the bandit task, recording the gaze if instructed.
        """

        # aliases for things that are often called
        debug = self.verbose
        EyeTracking = self.EyeTracking
        pars = self.pars
        win = self.win
        kb = self.devices.keyboard
        expTask = self.stimuli

        # setting the tracker if using it, otherwise we use psychopy mouse in
        # pure behavioral mode
        if EyeTracking: 
            tracker = self.devices.tracker
            if debug: print "Tracker connected? ", tracker.isConnected()
        else:
            mouse = event.Mouse(win = win)
            mouse.setVisible(False)
            
        # a list with total rewards from each game
        balance = []
        balanceScaled = []

        # start the task message
        if self.practice: 
            if debug: print '------------\nPractice start!\n------------\n'
            self.message("PRACTICE_START")
        else: 
            if debug: print '------------\nTask start!\n------------\n'
            self.message("TASK_START")


        ### Main loop that goes game by game, trial by trial

        while not expTask.done():
            
            # clearing the buffer
            self.hub.clearEvents('all')

            # set the game color
            win.setColor(pars['gameColors'][expTask.getGame()])
            win.flip()


            ### Showing the game instructions

            if not self.practice:
                # set up the the info to insert into instructions
                gameID = 'Game ' + str(expTask.getGame() + 1)
                units = expTask.getUnits()
                exchangeRate = expTask.getExchangeRate()
                insertInfo = (gameID, units, units, round(100/exchangeRate,2))
                # show the start of the game instructions
                self.showText(pars['startGameReportPath'], 'game_instructions', insertInfo)


            ### Tracker calibration 

            if EyeTracking: 
                self.runTrackerCalibration(tracker, kb)
           

            ### Recording the baseline pupil area

            # we do this only for real trials
            if EyeTracking and not self.practice: 
                self.getPupilCamera(tracker, kb)
                self.getPupilBaseline(tracker, kb)
            
            
            ### Start the game

            # set the initial balance for the game
            balanceGame = expTask.getBalanceInitial()
            balanceGameScaled = expTask.getBalanceInitial() * expTask.getScalingFactor()

            # if in practice mode we use a special counter and load practice 
            # instructions
            if self.practice:
                practiceTrial = 1
                practiceInstructions = loadingInstructionsFromTxt(pars['practiceInstructionsPath'])

            # starting the tracker
            if EyeTracking: tracker.setRecordingState(True)               

            # we go through the trials until the flag that game is done is
            # raised
            while not expTask.gameDone():

                if debug: 
                    print "Game: " + str(expTask.getGame() + 1) + " Trial: " + str(expTask.getTrial() + 1)

                # ----
                # Trial start
                # ----

                # Clear all the events received prior to the trial start
                self.hub.clearEvents('all')

                # set the images of the arms
                if self.practice: 
                    if pars['randomizeArms'] == 'everytrial':
                        rd.shuffle(self.practiceImages)
                    images = self.practiceImages
                else: 
                    if pars['randomizeArms'] == 'everytrial':
                        expTask.randomizeArms()
                    images = expTask.getArmImages()
                for i in range(self.noArms): 
                    self.armImages[i].setImage(images[i])


                ### Inter trial interval 

                # optionally add practice text
                if self.practice and practiceTrial <= pars['noPracticeTrialsInstructions']:
                    self.practiceBox.setText(practiceInstructions[0])
                else:
                    self.practiceBox.setText('')

                # we fade-in the fixation cross
                flipTime = self.transition([self.fixation], 'fadein', pars['fadingDuration'], fixation = False, fixObjects = [self.practiceBox])
                self.messageTrial("ITI_start", flipTime)
                if debug: "ITI_start"

                # randomly drawn from uniform distribution
                ITI = rd.uniform(pars['aITI'], pars['bITI'])
                if EyeTracking:
                    flipTime = self.showGazeDot(tracker, kb, [self.fixation] + [self.practiceBox], ITI)
                else:
                    core.wait(ITI)
                    
                # changing the cross orientation slowly to indicate the 
                # Transition to the stimulus phase
                flipTime = self.transition([], 'fadein', pars['fadingDuration'])
                self.messageTrial("ITI_end", flipTime)
                if debug: "ITI_end"
                
                
                ### Fixate to present the stimuli

                # optionally add practice text
                if self.practice and practiceTrial <= pars['noPracticeTrialsInstructions']:
                    self.practiceBox.setText(practiceInstructions[1])
                else:
                    self.practiceBox.setText('')

                # showing the fixation cross until fixated or for fixed time
                self.practiceBox.draw()
                self.fixation.draw()
                flipTime = win.flip()
                self.messageTrial("stimfixcross_draw", flipTime)
                if EyeTracking:
                    res = self.isObjectFixated(tracker, kb, pars['AOIduration'], pars['AOIthreshold'])
                    flipTime = res['flipTime']
                    timeFixStim = res['timeFix']
                else:
                    core.wait(pars['fixationTime'])
                    timeFixStim = pars['fixationTime']
                    self.practiceBox.draw()
                    self.fixation.opacity = pars['feedbackOpacity']
                    self.fixation.draw()
                    flipTime = win.flip()
                self.fixation.opacity = 1.0
                self.messageTrial("stimfixcross_end", flipTime)
                if debug: print "stimfixcross_end"


                ### Stimuli onset
                
                # optionally add practice text
                if self.practice and \
                    practiceTrial <= pars['noPracticeTrialsInstructions']:
                    self.practiceBox.setText(practiceInstructions[2])
                    self.stimulusTime = pars['stimulusTimePractice']
                else:
                    self.practiceBox.setText('')
                    self.stimulusTime = pars['stimulusTime']

                # showing the stimuli - boxes representing the options
                flipTime = self.transition(self.armBoxes + self.armImages + \
                    [self.practiceBox], 'fadein', pars['fadingDuration'], \
                    fixation = False)
                self.messageTrial("stimuli_draw", flipTime)
                stimuliTime = core.getTime()

                # either fixed waiting time and go signal, no mouse available
                if EyeTracking:
                    flipTime = self.showGazeDot(tracker, kb, \
                        self.armBoxes + self.armImages + [self.practiceBox], \
                        self.stimulusTime)
                else:
                    core.wait(self.stimulusTime)
                    [s.draw() for s in self.armBoxes + self.armImages + \
                        [self.practiceBox]]
                    flipTime = win.flip()


                ### Transition to choice phase

                # fading out arms
                flipTime = self.transition(self.armBoxes + self.armImages, \
                    'fadeout', pars['fadingDuration'])
                self.messageTrial("stimuli_end", flipTime)
                if debug: print "stimuli_end"
                
                # resetting arms opacities 
                [s.setOpacity(1) for s in self.armBoxes + self.armImages]

                
                ### Choice onset              

                # optionally add practice text
                if self.practice and \
                    practiceTrial <= pars['noPracticeTrialsInstructions']:
                    self.practiceBox.setText(practiceInstructions[3])
                else:
                    self.practiceBox.setText('')

                # we remove the arms and show only fixation cross 
                # (with orientation changed to 0 degrees) as a signal 
                # for making a choice, if eye tracking participants have to
                # fixate the cross and options will appear, while behavioral
                # version simply waits for certain duration
                self.fixation.ori = 0
                self.fixation.draw()
                self.practiceBox.draw()
                flipTime = win.flip()
                self.messageTrial("chfixcross_draw", flipTime)            
                if EyeTracking:
                    res = self.isObjectFixated(tracker, kb, pars['AOIduration'], pars['AOIthreshold'])
                    flipTime = res['flipTime']
                    timeFixChoice = res['timeFix']
                else:
                    core.wait(pars['fixationTime'])
                    timeFixChoice = pars['fixationTime']
                    self.fixation.opacity = pars['feedbackOpacity']
                    flipTime = win.flip()
                self.fixation.opacity = 1.0

                # we fade-out the fixation cross
                flipTime = self.transition([self.fixation], 'fadeout', pars['fadingDuration'], fixation = False, fixObjects = [self.practiceBox])
                self.messageTrial("chfixcross_end", flipTime)
                if debug: print "chfixcross_end"

                # optionally add practice text
                if self.practice and practiceTrial <= pars['noPracticeTrialsInstructions']:
                    self.practiceBox.setText(practiceInstructions[4])
                    self.choiceTime = pars['choiceTimePractice']
                else:
                    self.practiceBox.setText('')
                    self.choiceTime = pars['choiceTime']

                # now we wait for a choice
                # if eye tracking choice is made by fixating an option and 
                # pressing taskKey, in behavioral version it is done with a 
                # mouse
                [s.draw() for s in self.armBoxes + self.armImages + [self.practiceBox]]
                flipTime = win.flip()
                self.messageTrial("choice_draw", flipTime)
                if EyeTracking:
                    response = self.isOptionChosen(tracker, kb, [], [self.fixation] + [self.practiceBox], self.choiceTime)
                else:
                    response = self.isOptionChosen([], [], mouse, [self.fixation] + [self.practiceBox], self.choiceTime)
                self.messageTrial("choice_end", response['choiceEndTime'])
                if debug: print "choice_end"

                # useful aliases
                chosenArm = response['chosenArm']
                choiceMade = response['choiceMade']

                
                ### Feedback onset

                # if choice was made we continue with feedback:
                if choiceMade:

                    if debug: 
                        print "chosenArm: " + str(chosenArm)
                        print "chosenArmPos: " + str(response['chosenArmPos'])

                    # optionally add practice text
                    if self.practice and practiceTrial <= pars['noPracticeTrialsInstructions']:
                        self.practiceBox.setText(practiceInstructions[5])
                        self.feedbackDuration = pars['feedbackDurationPractice']
                    else:
                        self.practiceBox.setText('')
                        self.feedbackDuration = pars['feedbackDuration']

                    # modify looks of the chosen arm, 
                    # set the position of the feedback
                    self.armBoxes[chosenArm].opacity = pars['feedbackOpacity']
                    self.armImages[chosenArm].opacity = pars['feedbackOpacity']

                    # show the new "clicked" alternative
                    [s.draw() for s in self.armBoxes + self.armImages + [self.practiceBox]]
                    if not EyeTracking: mouse.setVisible(False)
                    win.flip()
                    # if there is some remaining time in self.choiceTime we
                    # wait until it runs out
                    waitTime = self.choiceTime - (response['choiceEndTime'] \
                        - response['choiceStartTime'])
                    core.wait(waitTime)

                    # we either remove nonchosen arms or not, and we have
                    # jittered waiting period for the feedback, 
                    # randomly drawn from uniform distribution
                    chosenArmObject = [self.practiceBox]
                    nonchosenArmObjects = []
                    for arm in range(len(self.armBoxes)):
                        if arm == chosenArm:
                            chosenArmObject.extend([self.armBoxes[arm], self.armImages[arm]])
                        else:
                            nonchosenArmObjects.extend([self.armBoxes[arm], self.armImages[arm]])
                    if pars['visibility'] == 'chosenArm':
                        flipTime = self.transition(nonchosenArmObjects, 'fadeout', pars['fadingDuration'], fixation = False, fixObjects = chosenArmObject)
                    else:
                        [s.draw() for s in self.armBoxes + self.armImages]
                        flipTime = win.flip()
                    self.messageTrial("FI_start", flipTime)
                    FI = rd.uniform(pars['aFI'], pars['bFI'])
                    core.wait(FI)
                    self.messageTrial("FI_end", flipTime + FI)

                    # get the reward, i.e. pull the arm
                    roundDecimal = expTask.getRoundDecimal()
                    reward = expTask.play(chosenArm)
                    scalingFactor = expTask.getScalingFactor()
                    rewardRound = round(reward*scalingFactor, roundDecimal)
                    if debug: print "reward: " + str(rewardRound)

                    # update the reward text
                    if self.practice:
                        rewardPractice = round(rd.random()*100, roundDecimal)
                        self.feedbackText.setText(rewardPractice)
                    else:
                        self.feedbackText.setText(str(rewardRound))
                    self.feedbackText.pos = response['chosenArmPos']
                    self.feedbackBox.pos = response['chosenArmPos']            

                    # show the feedback 
                    # we either remove nonchosen arms or not
                    # win.flip()
                    if pars['visibility'] == 'chosenArm':
                        self.armBoxes[chosenArm].draw()
                        self.armImages[chosenArm].draw()
                    else:
                        [s.draw() for s in self.armBoxes + self.armImages]
                    self.feedbackBox.draw()
                    self.feedbackText.draw()
                    self.practiceBox.draw()
                    flipTime = win.flip()
                    self.messageTrial("feedback_draw", flipTime)
                    feedbackStartTime = core.getTime()

                    # remove the stimuli from the screen when time runs out
                    core.wait(self.feedbackDuration)
                    if pars['visibility'] == 'chosenArm':
                        flipTime = self.transition(chosenArmObject + [self.feedbackBox] + [self.feedbackText], 'fadeout', pars['fadingDuration'], opacityEnd = pars['feedbackOpacity'], fixation = False)
                    else:
                        flipTime = win.flip()                   
                    self.messageTrial("feedback_end", flipTime)
                    feedbackEndTime = core.getTime()

                # else we interrupt the trial, show a message 
                # about being late and repeat the trial
                else:  
                    if not EyeTracking: mouse.setVisible(False)
                    # update the "late choice" text with penalty amount
                    roundDecimal = expTask.getRoundDecimal()
                    scalingFactor = expTask.getScalingFactor()
                    penaltyRound = round(self.pars['missingFee']* \
                        scalingFactor, roundDecimal)
                    units = expTask.getUnits()
                    self.lateChoiceBox.setText("You were late with making a choice.\n\n         You have lost " + str(penaltyRound) + " " + units + ".\n\n           Please respond faster!")
                    
                    # showing the message for certain duration
                    win.flip()
                    self.lateChoiceBox.draw()
                    flipTime = win.flip()
                    self.messageTrial("choicemissed_draw", flipTime)
                    FI = rd.uniform(pars['aFI'], pars['bFI'])
                    core.wait(FI + self.feedbackDuration)
                    feedbackEndTime = core.getTime()
                    self.messageTrial("choicemissed_end", feedbackEndTime)

                    # creating variables with missing values that will be saved
                    chosenArm = 'NA'
                    chosenArmPos = 'NA'
                    reward = -self.pars['missingFee']
                    rewardRound = -penaltyRound
        

                ### Saving the data

                if choiceMade:
                    # various response times
                    choiceRT = "{:.3f}".format(response['choiceEndTime'] - response['choiceStartTime'])
                    feedbackRT = "{:.3f}".format(feedbackEndTime - feedbackStartTime)
                    timeTrial = "{:.3f}".format(feedbackEndTime - stimuliTime)

                    # various other quantities we want to track
                    armIds = expTask.getArmIds()
                    valExpArms = expTask.getValExpArms()
                    rewardExp = valExpArms[chosenArm]
                    rewardMaxExp = max(valExpArms)
                    regret = rewardMaxExp - reward
                    regretExp = rewardMaxExp - rewardExp
                    chosenArmId = armIds[chosenArm]
                    if regretExp == 0: 
                        correct = 1 
                    else: 
                        correct = 0
                    correctArm = valExpArms.index(rewardMaxExp) 
                    correctArmId = armIds[correctArm]
                    chosenRankExp = sorted(valExpArms, reverse = True).index(rewardExp) 
                    if debug: print "chosen rank: ", chosenRankExp

                    # correct for python 0 starting index
                    chosenArm = chosenArm + 1
                    correctArm = correctArm + 1
                    chosenRankExp = chosenRankExp + 1
                
                # if choice was not made we create different values for some
                # variables
                else:
                    # various response times
                    choiceRT = 'NA'
                    feedbackRT = 'NA'
                    timeTrial = "{:.3f}".format(feedbackEndTime - stimuliTime)

                    # various other quantities we want to track
                    armIds = expTask.getArmIds()
                    valExpArms = expTask.getValExpArms()
                    rewardExp = 'NA'
                    rewardMaxExp = max(valExpArms)
                    regret = 'NA'
                    regretExp = 'NA'
                    chosenArmId = 'NA'
                    correct = 'NA' 
                    correctArm = valExpArms.index(rewardMaxExp) 
                    correctArmId = 'NA'
                    chosenRankExp = 'NA'

                # update the balance for the game
                balanceGame += reward
                balanceGameScaled += rewardRound
                if debug: print "Balance scaled: " +  str(balanceGameScaled)

                # updating the data
                if not self.practice:
                    self.data['subjectID'].append(self.subjectID)
                    self.data['sessionID'].append(self.hub.getSessionID())
                    self.data['seed'].append(self.seed)
                    self.data['psychopyVers'].append(self.psychopyVers)
                    self.data['codeVersion'].append(self.codeVersion)
                    self.data['expID'].append(self.expID)
                    self.data['expCond'].append(self.expCond)
                    self.data['instTime'].append(self.timeInstructions)
                    self.data['game'].append(expTask.getGame() + 1)
                    self.data['gameType'].append(expTask.getGameType())
                    self.data['noArms'].append(expTask.getNoArms())
                    self.data['trial'].append(expTask.getTrial() + 1)
                    self.data['calibration'].append(self.calibration)
                    self.data['choiceMade'].append(choiceMade*1)
                    self.data['chosenArm'].append(chosenArm)
                    self.data['chosenArmId'].append(chosenArmId)
                    self.data['chosenArmPos'].append(response['chosenArmPos'])
                    self.data['choiceRT'].append(choiceRT)
                    self.data['feedbackRT'].append(feedbackRT)
                    self.data['rewardScaled'].append(rewardRound)
                    self.data['reward'].append(reward)
                    self.data['rewardExp'].append(rewardExp)
                    self.data['rewardMaxExp'].append(rewardMaxExp)
                    self.data['regret'].append(regret)
                    self.data['regretExp'].append(regretExp)
                    self.data['correct'].append(correct)
                    self.data['correctArm'].append(correctArm)
                    self.data['correctArmId'].append(correctArmId)
                    self.data['chosenRankExp'].append(chosenRankExp)
                    self.data['timeFixStim'].append(timeFixStim)
                    self.data['timeFixChoice'].append(timeFixChoice)
                    self.data['ITI'].append(ITI)
                    self.data['IFI'].append(FI)
                    self.data['timeTrial'].append(timeTrial)
                    self.data['rewardTotal'].append(balanceGame)
                    self.data['rewardTotalScaled'].append(balanceGameScaled)
                    self.data['timeTotal'].append(self.totalTime.getTime())
                    self.data['exchangeRate'].append(expTask.getExchangeRate())
                    self.data['scalingFactor'].append(expTask.getScalingFactor())
                    self.data['units'].append(expTask.getUnits())
                    
                    # add fields that depend on the game characteristics
                    for arm in range(1, self.noArms + 1):
                        self.data["idArm" + str(arm)].append(expTask.getArmIds()[arm - 1])
                        self.data["imgArm" + str(arm)].append(expTask.getArmImages()[arm - 1][4:])
                        self.data["posArm" + str(arm)].append(self.armPositions[arm - 1])
                        self.data["valExpArm" + str(arm)].append(expTask.getValExpArms()[arm - 1])
                        self.data["valSdArm" + str(arm)].append(expTask.getValSdArms()[arm - 1])
                        self.data["inoSdArm" + str(arm)].append(expTask.getInoSdArms()[arm - 1])
                        self.data["decayArm" + str(arm)].append(expTask.getDecayArms()[arm - 1])
                
                # reset the opacities of the arms and cross orientation
                [s.setOpacity(1) for s in self.armBoxes + self.armImages + [self.feedbackText] + [self.feedbackBox] + [self.fixation] + [self.practiceBox] + [self.trialInfoBox]]
                self.fixation.ori = 0
                self.calibration = 0

                # check the counter for the practice trials
                if self.practice and choiceMade:
                    practiceTrial += 1
                    if practiceTrial > pars['noPracticeTrials']: 
                        expTask._gameDone = True
                    
                # saving partial data and advancing to the next trial
                self.saveData()
                if choiceMade: expTask.nextTrial()


            # ----
            # game done
            # ----

            # stop recording if eye tracking mode
            if EyeTracking: tracker.setRecordingState(False)

            # record the earnings
            balance.append(balanceGame)
            balanceScaled.append(balanceGameScaled)

            # set up the game outcomes
            gameEarnings = balanceScaled[expTask.getGame()]
            roundDecimal = expTask.getRoundDecimal()
            units = expTask.getUnits()
            exchangeRate = expTask.getExchangeRate()
            insertInfo = (round(gameEarnings, roundDecimal), units, round(100/exchangeRate,2), units, round(gameEarnings/exchangeRate,2))

            # if in practice we do not continue to the next game
            if self.practice: expTask._done = True
            
            # show the end of the game report
            if not self.practice:
                self.showText(pars['endGameReportPath'], 'game_report', insertInfo)
                expTask.nextGame()


        ## ----
        ## Step 5. Reporting results of the experiment and goodbye
        ## ----

        # if we were in practice we have to reset the task indicators
        if self.practice:                
            expTask._done = False
            expTask._gameDone = False
            expTask._currentTrial = 0
            expTask._currentGame = 0

        # set the background color 
        win.setColor(pars['backgroundColor'])

        # compute the earnings
        if self.practice:
            chosenGame = 0 
            gameEarnings = 0.0
        else:
            chosenGame = rd.choice(range(self.noGames)) 
            expTask._currentGame = chosenGame
            gameEarnings = balanceScaled[chosenGame]
        exchangeRate = expTask.getExchangeRate()
        realEarnings = gameEarnings/exchangeRate

        # format the insertInfo
        roundDecimal = expTask.getRoundDecimal()
        units = expTask.getUnits()
        insertInfo = (chosenGame + 1, round(gameEarnings, roundDecimal), units, round(100/exchangeRate, 2), units, round(realEarnings, 2), round(pars['showUpFee'], roundDecimal), round(realEarnings + pars['showUpFee'], 2) )

        # report the final earnings for the experimenter
        print "Game to pay out: " + str(chosenGame)
        print "Show up fee: " + str(pars['showUpFee'])
        print "Earnings in the game: " + str(realEarnings)
        
        # show the end of the experiment report, but not in practice mode
        if self.practice:
            if debug: print '------------\nPractice end!\n------------\n'
            self.message("PRACTICE_END")
        else:
            if self.noGames > 1:
                self.showText(pars['endExpReportPath'], 'exp_report', insertInfo)
            if debug: print '------------\nTask end!\n------------\n'
            self.message("TASK_END")
        self.hub.clearEvents('all')


        ## -----
        ## GAME OVER
        ## -----