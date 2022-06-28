import satbox as sb
import astropy.units as u
from astropy.time import Time
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
alt1 = input_dict['alt'] *u.km
alt2 = input_dict['alt'] * 1000
dist_threshold = input_dict['dist_threshold']
elev_threshold = input_dict['elev_threshold']
conicSensorAngle = input_dict['conicSensorAngle']
time = input_dict['time']
prop_dur = input_dict['prop_dur']
GS_pos = input_dict['GS_pos']

#convert time input to astropy units
epoch = Time(time, format='isot', scale='utc')

#propogating the constellation    
walker = sb.Constellation.from_walker(i, t, p, f, alt1)
t2propagate = prop_dur*u.day
tStep = 60 * u.s
walkerSim = sb.SimConstellation(walker, t2propagate, tStep, verbose=True)
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
file = walker.generate_czml_file(prop_duration=prop_dur, sample_points=prop_dur*144, satellites=satellites,cobjects=objects, L_avail_ISL=L_avail_ISL,cL_poly_ISL=L_poly_ISL, L_avail_GS=L_avail_GS, L_poly_GS=L_poly_GS, GS_pos=GS_pos, alt=alt2, conicSensorAngle=conicSensorAngle, GS=True, show_polyline_ISL=True, show_polyline_GS=True, show_conicSensor=True)
print(file)