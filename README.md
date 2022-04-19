# Instructions 

This repository contains experimental software for experiments reported in an article "Uncertainty in learning, choice and visual fixation" by Hrvoje Stojic, Jacob Orquin, Peter Dayan, Raymond Dolan and Maarten Speekenbrink. Preprint can be found on [PsyArXiv](https://psyarxiv.com/zuge2). DOI: 10.31234/osf.io/zuge2

Experiment consists of a bandit task where we also monitor participants' overt attention through eye tracking. Implementation is based on Python, relying heavily on [PsychoPy](https://www.psychopy.org/) library. Code was developed for EyeLink 1000 eye tracker by [SR Research](https://www.sr-research.com/), but it should be easily ported to other eye trackers. Everything was tested on a Linux operating system. Everyone is welcome to use the code, but please cite the paper above.

The most relevant are the Python scripts:  
- `bandit.py` script contains all the code related to the experiment  
- `run_stationary.py` and `run_restless.py` are main scripts that execute the experiment, either the stationary or the restless (or nonstationary) version of the multi-armed bandit task  
- `experiment_config_stationary.yaml` and `experiment_config_restless.yaml` contain the parameters of two versions of the experiment; these parameters change, for example, how long certain phases of the trial last, or what colors are used for certain stimuli  
- `iohub_config.yaml` contains specifications of the devices, like the eye-tracker or the monitor used in the experiment  

There are several folders as well:  
- in the `data` folder we save behavioral data for each participants in the form of `csv` file  
- `stimuli` folder contains specifications of the multi-armed bandit task stimuli, in the form of `csv` files    
- `instructions` folder contains textual instructions that we show to the participants used in the experiment; they are formatted in special way so that different pages are recognized  
- `img` folder contains images of the symbols that participants learn about; `practice` folder also contains images, but a separate set that is used during the practice trials only - these images come with a separate license, see LICENSE file for details   


## Preparing the equipment

Setup:  
1. EyeLink camera
2. EyeLink computer to which the camera is connected
3. Host computer where experimetal software is running which controls the EyeLink 

Software has been tested with Linux operating system on the host computer (Ubuntu 17.10).


### Install EyeLink SDK and PsychoPy

Download Psychopy version 1.85.3 from the following [link](https://github.com/psychopy/psychopy/releases), according to the operating system used on the computer where you would like to run the experiment. It is very important to install exactly this version, as some functions in the older or newer versions might be different and the software might not work. For example, if you have Windows operating system on your lab computer you would select `StandalonePsychoPy-1.85.3b-win32.exe`.   

On Windows installing Psychopy application should be enough, while on Linux EyeLink SDK has to be installed first and then Psychopy (Linux version does not automatically install the required EyeLink libraries). Using Python virtual environment is recommended.


### Establishing connection between the EyeLink and the host computer

EyeLink 1000 Eyetracker

    Plug in the power supply to the eye tracker
    Remove the lenscap
    Turn on the EyeLink computer
    Choose “Eyelink” from the startup menu
    (From the DOS prompt, you can type elcl) 

Connect the host computer to the tracker computer with a crossover cable. Then in the host computer (for Ubuntu OS)

    Open network manager 
    Create a new Ethernet connection (name can be anything)
    Open "IPv4 Settings" tab, change method to "Manual" and click add
    Add Address "100.1.1.2" and Netmask "24"
    Click "Save" and try connecting to the newly created connection
    Open terminal and try pinging the tracker computer with the following command: `ping 100.1.1.2`
    The output should look similar to this one:
        PING 10.0.0.2 (10.0.0.2) 56(84) bytes of data.
        64 bytes from 10.0.0.2: icmp_seq=1 ttl=128 time=0.457 ms

See for some helpful instructions on connecting two computers in Ubuntu [here](https://askubuntu.com/questions/22835/how-to-network-two-ubuntu-computers-using-ethernet-without-a-router).


### Setup of the physical space and equipment layout

**Tips**  
- Always use the chinrest! It should be fixed and the same position should be used with all participants, move the chair/table instead to accomodate the person's height.  
- Desk should be wide enough so that the tracker can be about 50cm from the chinrest.  
- Chair should be immovable, not the rotating one, so that people dont move in their chinrest.  
- Don't disturb the participants after setting the calibration and starting with the recording.   
- Equal illumination and sound isolation would be preferable. 
- Camera should be placed below the monitor, eyelashes are not getting in the way of the camera that way.  
- The acuity on the camera can be modulated to get the better picture, one should see the eye-lashes clearly.  

**Calibration and validation**
- Should be always run.  
- If its a longer experiment do it multiple times in the breakes.  
- In validation, if any offset is more than 2 its a bad calibration.    
- After validation is accepted we can proceed with the experiment.    

**Pupil tracking**  
- Participants with eye-glasses, ideally contact lenses as well, should be excluded.  
- Participants with eyelids partially occluding the eyes should be excluded as well.    
- If participants have to fixate on various locations on the screen pupils will not be measured correctly.  
- Luckily there is a correction method published by Hayes and Petrov (2016) "Mapping and correcting the influence of gaze position on pupil size measurements":  
    - Exact physical layout needs to be used (near, medium or far from the article).  
    - Tracking area in the centroid mode.  
    - Pupil threshold should be lower, 60-80, and kept the same throughout the experiment.  
    - If layout has to be changed, any change should be recorded.  


## Using the experimental software

1. This allows you to run Psychopy and program your own experiments, but it also allows for running the experiments in these folders. To start the experiment there are two choices. You can either start Psychopy and then from within the Psychopy application navigate to the `software` folder and open either `run_stationary.py` or `run_restless.py`, or on UNIX operating systems you can open terminal, navigate to one of the two folders and run the command `python run_stationary.py` or `python run_restless.py`. This should start the experiment.  
2. It is VERY important to set up the `iohub_config.yaml` file according to the setup in the lab. In the `Display` section you should add your display specifications, there are three examples of the displays we have been using while developing and testing. You can simply adapt the current display (lines 104 to 124), by changing the crucial elements: `physical_dimensions` sets the dimensions of your display, `default_eye_distance` specifies the distance of your display from the headrest and `pixel_dimensions` specifies the resolution of your display in pixels. This will make sure that the eye-tracker records correct gaze locations; rest of the variables are not that relevant. If you are using EyeLink 1000 you can keep the `eyetracker.hw.sr_research.eyelink.EyeTracker`, check the [iohub documentation](http://www.isolver-solutions.com/iohubdocs/iohub/api_and_manual/iohub_process/config_files.html) how to adapt it for your eye tracker setup and the meaning of individual parameters. 


### After the experiment has started

When the experiment program is started:  
1. A window will appear with some general information, this is for the experimenter to make sure a correct script is launched. Press OK to continue.  
2. Another window will appear where two pieces of information HAVE to be entered: 
    - `code` has to be a unique integer number for EACH run (it's best to start from 1 and whenever the experiment is launched increment the number for 1)  
    - `seed` has to be an integer number; it can be the same number from one session to another, BUT note that this will mean that all the randomizations will be exactly the same in the sessions with the same seed number (e.g. images will be assigned to the same locations, rewards will be the same) - usually one would want to use a unique number in each session here as well, to ensure proper randomization  
3. A dialogue will appear with four basic questions about the participant: age, gender, whether the participant is wearing eye glasses or contact lenses. Experimenter should click OK to continue after completing the questions.  
4. After this the experiment will start for the participant. The participant first goes through general instructions and then continues to the practice trials. Before the practice trials the experiment will start the eye tracker calibration if in eye tracking mode (this can be set in `experiment_config_stationary.yaml` or `experiment_config_restless.yaml` file). After the practice trials there is another round of calibration, before the real task begins.   

Other useful notes:  
- In some phases of the experiment it is possible to abort the experiment by pressing left CONTROL and Q keys simultaneously. Otherwise, the experiment can be aborted by shutting down the PsychoPy application or the python process in the terminal.  
- Sometimes eye tracker loses calibration during the experiment. If this occurs, experimenter can interrupt the experiment in the inter trial interval and do the calibration again, by pressing left CONTROL and P keys simultaneously.
