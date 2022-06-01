import os
import satbox as sb
import orbitalMechanics as om
import utils as utils
from poliastro.czml.extract_czml import CZMLExtractor
import CZMLExtractor_MJD as CZMLExtractor_MJD
import numpy as np
from poliastro import constants
from poliastro.earth import Orbit
from poliastro.earth.sensors import min_and_max_ground_range, ground_range_diff_at_azimuth
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody.propagation import propagate
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import func_twobody
from poliastro.util import norm
import astropy
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.plotting import OrbitPlotter3D, OrbitPlotter2D
from poliastro.twobody.events import(NodeCrossEvent,)
import seaborn as sns
from astropy.coordinates import EarthLocation, GCRS, ITRS, CartesianRepresentation, SkyCoord
import comms as com
from copy import deepcopy
import dill
import Interval_Finder as IF 
import sys
import json

#User inputs
userInputs = sys.argv[1]
input_dict = json.loads(userInputs)

# Visualizing walker constellation and isl
i = input_dict['i'] *u.deg
t = input_dict['t'] 
p = input_dict['p']
f = input_dict['f']
alt = input_dict['alt'] *u.km
dist_threshold = input_dict['dist_threshold']
elev_threshold = input_dict['elev_threshold']
GS_pos = input_dict['GS_pos']
#calendar dates

#propogating the constellation    
walker = sb.Constellation.from_walker(i, t, p, f, alt)
t2propagate = 5 * u.day
tStep = 60 * u.s
walkerSim = sb.SimConstellation(walker, t2propagate, tStep, verbose = True)
walkerSim.propagate()

#Viualizing ISL links
relative_position_data_ISL = IF.get_relative_position_data_ISL(walkerSim,dist_threshold)
satellites = IF.find_feasible_links_ISL(t, relative_position_data_ISL)
L_avail_ISL = IF.get_availability_ISL(satellites, relative_position_data_ISL)
L_poly_ISL = IF.get_polyline_ISL(satellites, relative_position_data_ISL)

#Visualizing ground stations and gs-to-sat links using elevation_angle_threshold_gs
relative_position_data_GS = IF.get_relative_position_data_GS(walkerSim, GS_pos, elev_threshold)
objects = IF.find_feasible_links_GS(relative_position_data_GS)
L_avail_GS = IF.get_availability_GS(objects, relative_position_data_GS)
L_poly_GS = IF.get_polyline_GS(objects,relative_position_data_GS)

# Generating czml file
file = walker.generate_czml_file(prop_duration=1, sample_points=100, satellites=satellites,cobjects=objects, L_avail_ISL=L_avail_ISL,cL_poly_ISL=L_poly_ISL, L_avail_GS=L_avail_GS, L_poly_GS=L_poly_GS, GS_pos=GS_pos, GS=True, show_polyline_ISL=True, show_polyline_GS=True)
print(file)