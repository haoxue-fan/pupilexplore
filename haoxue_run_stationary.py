#!/usr/bin/env python
# -*- coding: utf-8 -*-

# now importing bandit experiment functions
from haoxue_bandit import *


## ----------------------------------------------------------------------------
## Execute the experiment
## ----------------------------------------------------------------------------

runtime = ExperimentRuntime(module_directory(ExperimentRuntime.run), "experiment_config_stationary.yaml")

runtime.start()