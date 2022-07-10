import satbox as sb
import astropy.units as u
from astropy.time import Time
import Interval_Finder as IF 
import sys
import json

#User Inputs
userInputs = sys.argv[1]
input_dict = json.loads(userInputs)

# Retreiving User Inputs
i = input_dict['i']*u.deg
t = input_dict['t'] 
p = input_dict['p']
f = input_dict['f']
alt1 = input_dict['alt']*u.km
alt2 = input_dict['alt']*1000
time = input_dict['time']
prop_dur = input_dict['prop_dur']
dist_threshold = input_dict['dist_threshold']
elev_threshold = input_dict['elev_threshold']
conicSensorAngle = input_dict['conicSensorAngle']
GS_pos = input_dict['GS_pos']

#Propogating Satellite Constellation
epoch = Time(time, format='isot', scale='utc')
walker = sb.Constellation.from_walker(i, t, p, f, alt1, epoch=epoch)
t2propagate = prop_dur*u.day
tStep = 60*u.s
walkerSim = sb.SimConstellation(walker, t2propagate, tStep, verbose=True)
walkerSim.propagate()

#Preparing Necessary Information
if dist_threshold is not None:
    show_polyline_ISL = True
    relative_position_data_ISL = IF.get_relative_position_data_ISL(walkerSim,dist_threshold)
    satellites = IF.find_feasible_links_ISL(t, relative_position_data_ISL)
    L_avail_ISL = IF.get_availability_ISL(satellites, relative_position_data_ISL)
    L_poly_ISL = IF.get_polyline_ISL(satellites, relative_position_data_ISL)
else:
    show_polyline_ISL = False
    satellites = None
    L_avail_ISL = None
    L_poly_ISL = None
if GS_pos is not None:
    GS = True
    if elev_threshold is not None:
        show_polyline_GS = True
        relative_position_data_GS = IF.get_relative_position_data_GS(walkerSim, GS_pos, elev_threshold)
        objects = IF.find_feasible_links_GS(relative_position_data_GS)
        L_avail_GS = IF.get_availability_GS(objects, relative_position_data_GS)
        L_poly_GS = IF.get_polyline_GS(objects,relative_position_data_GS)
    else:
        show_polyline_GS = False
        objects = None
        L_avail_GS = None
        L_poly_GS = None
        for gs in GS_pos:
            gs[0] = gs[0] *u.deg
            gs[1] = gs[1] *u.deg
else:
    GS = False
    show_polyline_GS = False
    objects = None
    L_avail_GS = None
    L_poly_GS = None
if conicSensorAngle is not None:
    show_conicSensor = True
else:
    show_conicSensor = False

# Generating czml file
file = walker.generate_czml_file(prop_duration=prop_dur, sample_points=prop_dur*144, satellites=satellites, objects=objects, L_avail_ISL=L_avail_ISL, L_poly_ISL=L_poly_ISL, L_avail_GS=L_avail_GS, L_poly_GS=L_poly_GS, GS_pos=GS_pos, alt=alt2, conicSensorAngle=conicSensorAngle, GS=GS, show_polyline_ISL=show_polyline_ISL, show_polyline_GS=show_polyline_GS, show_conicSensor=show_conicSensor)
print(file)