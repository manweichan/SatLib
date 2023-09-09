# SatLib

## Overview
SatLib is a python library for constellation based calculations. Much of it is based on [Poliastro](https://docs.poliastro.space/en/stable/) and extends it for use with satellite constellations. Fundamental functionality such as propagating constellations, determining constellation access to ground stations, and determining when satellites in the constellation have windows of opportunity to perform inter-satellite links are included in the library. The main classes are contained in satbox.py and instructions to run an example notebook that showcase some of the capabilities are included below. Note: This library is still under development.

# Running the example notebook

This tutorial assumes working knowledge of the terminal with [git installed](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). There are many [resources](https://www.makeuseof.com/tag/beginners-guide-mac-terminal/) online to learn from if you need. You should also have a working version of python, or you can download it [here](https://www.python.org/downloads/). This tutorial was tested using python 3.7.3

Draft documentation is under `docs/_build/html/index.html`, which you can open in your browser.

## Initialize Git
Begin by moving to a directory that you want to work in and initialize git if needed `git init`

## Create a virtual environment
I suggest creating a [virtual environment](https://docs.python.org/3/library/venv.html)
Example: `python3 -m venv satelliteExample`, which you type in the terminal

## Activate your environment: 
`source satelliteExample/bin/activate`

## Clone the library: 
`git clone https://github.com/manweichan/SatLib.git`

## Change to the SatLib directory: 
`cd SatLib`

## Install the relevant packages using:
`pip install -r requirements.txt`

#### Alternatively, install each library individually
`pip install notebook`

`pip install numpy`

`pip install poliastro`

`pip install poliastro czml3`

`pip install seaborn`

`pip install dill`

## Start Jupyter Notebook
Once the libraries are installed, open up a jupyter notebook with the command:
`jupyter notebook`

## Open notebook
Open satbox_ex.ipynb

## Run the notebook!

## Other notebooks to look at include
###  satbox_czml_ex.ipynb - Example notebook of how to create czml files for Cesium
###  RGT_maneuver_ex.ipynb - Exmple notebook of how to reconfigure satellites into repeat ground track orbits over a target site
###  constellation_performance.ipynb - Example notebook of how to extract constellation performance from metrics: Age of information, system response time, and total pass time