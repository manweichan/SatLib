# SatLib

Running the example notebook

This tutorial assumes working knowledge of the terminal with [git installed](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). There are many [resources](https://www.makeuseof.com/tag/beginners-guide-mac-terminal/) online to learn from if you need. You should also have a working version of python, or you can download it [here](https://www.python.org/downloads/). This tutorial was written using python 3.7.3

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
Open satellite_constellation_ex.ipynb

9. Run the notebook!