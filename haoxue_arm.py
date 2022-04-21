#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
haoxue_arm.py

python code storing the code to generate different types of arms 

Modified based on Hrvoje Stojic's code
Last updated by Haoxue Fan on Apr 18, 2022

"""

from __future__ import division

# Importing the PsychoPy libraries that we want to use
# from psychopy import core, visual, gui, event, data, monitors, __version__
# from psychopy.iohub import (EventConstants, ioHubExperimentRuntime, module_directory, getCurrentDateTimeString)
# import pylink
# from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

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
                raise ValueError("GaussianArm object required.")
        self._arms = arms
        if randomizeArms == 'once':
            rd.shuffle(self._arms)
    def noArms(self):
        return len(self._arms)
    def getArm(self, i):
        if isinstance(i, (int,long)):
            return self._arms[i]
        else:
            raise ValueError("I need an int")
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
                raise ValueError("Bandit class required.")
            if not isinstance(games[i]['noTrials'], int):
                raise ValueError("Integer value needed.")
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

