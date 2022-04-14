import satPlotTools as spt
import orbitalMechanics as om
import constellationClasses as cc
import comms as com
import utils as utils
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
import astropy.units as u
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.plotting import OrbitPlotter3D, OrbitPlotter2D



def get_relative_position_data(constellation,d):
    """
    Returns relative position data (array?)

    Parameters
    ----------
    constellation (obj): the constellation object
    d (int) : threshold (kilometers)
    """

    #Get relative position data between satellites in the constellation
    relative_position_data = constellation.get_relative_velocity_analysis()
    #Get feasibility of intersatellite links/getting the true/false array for each satellite-to-satellite pair
    distanceThreshold = d * u.km
    utils.get_isl_feasibility(relative_position_data,
                        distanceConstraint=distanceThreshold)
    return relative_position_data



def find_feasible_links(constellation, num_sats, relative_position_data):
    """
        Returns a list of strings with the pairs of satellites that can communicate during the set time interval
        
        Parameters
        ----------
        constellation (obj) : the constellation object we are looking at 
        num_sats (int) : the number of satellites in the constellation
        relative_position_data (array?) : the array of relative_position_data
            
    """
    #Takes the number of satellites and gives a list of every possible pair of satellites
    L=[]
    for i in range(num_sats):
        L.append(str(i))

    L1=[]
    for i in range(len(L)):
        j=i+1
        while j != len(L):
            L1.append(L[i]+'-'+L[j])
            j+=1
    
    
    L2 = L1.copy()

    #Eliminating the pairs from the list that are never able to communicate
    #iterate through the pairs satellites
    for i in L2: 
       
        #True false mask of ISL opportunities
        mask = relative_position_data['satData'][i]['islFeasible']
       
        #If this pair has no True time intervals, eliminate it from the list
        if any(mask) == False:
            L1.remove(i)

    return L1



def get_availability(constellation, satellites,relative_position_data):
    
    
    """
        Returns a list of strings with the time intervals for communication avaialability
        
        Parameters
        ----------
        constellation (obj): the constellation we are looking at 
        satellites (str) : the id of the two satellites in questions
        relative_position_data (array?) :the array of relative_position_data
            
    """

    #True false mask of ISL opportunities
    #Gets satellite time interval data between sateelites 
    mask = relative_position_data['satData'][satellites]['islFeasible'] 
    timeArray = relative_position_data['satData'][satellites]['times']

    #Apply true false mask to time array
    intervalsISL = utils.get_true_intervals(mask, timeArray)
    #If there are no True intervals, let user know
    if any(mask)==False:
        return 'No Intervals Found: Intersatellite Comunication Link is not Feasible'
    #Reformatting the data into a format czml can use
    else:
        L=[]
        for idx, interval in enumerate(intervalsISL):
            start_time = interval[0]
            stop_time = interval[1]
    
            #Format time to isot format
            L.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')

        return L



def get_multi_availability(constellation, satellites,relative_position_data):
    
    
    """
        Ruturns a list of availiabilities for the list of pairs of satellites
        
        Parameters
        ----------
        constellation (obj): the constellation object we are looking at 
        satellites (list of strings) : the pairs of satellites get availabilty for
        relative_position_data (array?) : the array of relative_position_data
            
    """
    #True false mask of ISL opportunities
    #Gets satellite data between sateelites 
    #Iterates through the pairs of satellites
    L_avail=[]
    for i in satellites:
        mask = relative_position_data['satData'][i]['islFeasible'] 
        timeArray = relative_position_data['satData'][i]['times']

        #Apply true false mask to time array
        intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
        #If there are no True intervals, we let user know
        if any(mask)==False:
            return 'No Intervals Found: Intersatellite Comunication Link is not Feasible'
        else:
            L=[]
            for idx, interval in enumerate(intervalsISL):
                start_time = interval[0]
                stop_time = interval[1]
    
                #Format time to isot format
                L.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')

        L_avail.append(L)
    
    return L_avail



def get_polyline_intervals(constellation, satellites, relative_position_data):
        
    
    """
        Returns a list of dictionaries with the polyline intervals.
        
        Parameters
        ----------
        constellation (obj): the constellation we are looking at 
        satellites (str) : the id of the two satellites in questions
        relative_position_data (array?) :the array of relative_position_data
    """

    #True false mask of ISL opportunities
    #Gets satellite data between sateelites
    mask = relative_position_data['satData'][satellites]['islFeasible'] 
    timeArray = relative_position_data['satData'][satellites]['times']

    #If there are no True intervals, we let user know
    if any(mask)==False:
        return 'Intersatellite Comunication Link is not Feasible'

    #Getting and reformatting false intervals
    false_intervalsISL = utils.get_false_intervals(mask, timeArray)
    L_false=[]
    for idx, interval in enumerate(false_intervalsISL):
        start_time = interval[0]
        stop_time = interval[1]

        #Format time to isot format
        L_false.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')
    
    #Getting and reformatting true intervals
    true_intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
    L_true=[]
    for idx, interval in enumerate(true_intervalsISL):
        start_time = interval[0]
        stop_time = interval[1]

        #Format time to isot format
        L_true.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')
    
    #Creating the list of dictionaries
    L_final=[]
    if mask[0]==True:
        for i in range(len(L_true)):
            try:
                D1={}
                D1['interval']=L_true[i]
                D1['boolean']='true'
                L_final.append(D1)
                D2={}
                D2['interval']=L_false[i]
                D2['boolean']='false'
                L_final.append(D2)
            except:
                pass
    if mask[0]==False:
        for i in range(len(L_false)):
            try:
                D1={}
                D1['interval']=L_false[i]
                D1['boolean']='false'
                L_final.append(D1)
                D2={}
                D2['interval']=L_true[i]
                D2['boolean']='true'
                L_final.append(D2)
            except:
                pass
    return L_final



def get_multi_poly(constellation, satellites, relative_position_data):
        
    
    """
        Returns of list of the polyline intervals for the list of pairs of satellites.
        
        Parameters
        ----------
        constellation (obj): the constellation we are looking at 
        satellites (list of strings): the pairs of satellites to find polyline intervals for
        relative_position_data (array?) : the array of relative_position_data
    """

    #True false mask of ISL opportunities
    #Gets satellite data between sateelites
    #Iterate throught the pairs of satellites 
    L_poly=[]
    for i in satellites:
        mask = relative_position_data['satData'][i]['islFeasible'] 
        timeArray = relative_position_data['satData'][i]['times']

        #If there are no True intervals, we let user know
        if any(mask)==False:
            return 'Intersatellite Comunication Link is not Feasible'

        #Get false intervals
        false_intervalsISL = utils.get_false_intervals(mask, timeArray)
        L_false=[]
        for idx, interval in enumerate(false_intervalsISL):
            start_time = interval[0]
            stop_time = interval[1]
    
            L_false.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')
    
        #Get true intervals
        true_intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
        L_true=[]
        for idx, interval in enumerate(true_intervalsISL):
            start_time = interval[0]
            stop_time = interval[1]
    
            L_true.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')
    
        #Creating dictionary
        L_final=[]
        if mask[0]==True:
            for i in range(len(L_true)):
                try:
                    D1={}
                    D1['interval']=L_true[i]
                    D1['boolean']='true'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_false[i]
                    D2['boolean']='false'
                    L_final.append(D2)
                except:
                    pass
        if mask[0]==False:
            for i in range(len(L_false)):
                try:
                    D1={}
                    D1['interval']=L_false[i]
                    D1['boolean']='false'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_true[i]
                    D2['boolean']='true'
                    L_final.append(D2)
                except:
                    pass
        L_poly.append(L_final)

    return L_poly




def get_relative_position_data_gs(constell,GS,d):
    """
    Returns a list of dict. keys are time intervals and values are booleans. each dict is for a different satellite-GS combination

    Parameters
    ----------
    constellation (obj): The constellation object
    GS (list): list of the groundstation obj
    time_intervals (list): list of propogation time intervals
    d (int): threshold in km 
    """
    #Getting all satellites objects
    sats = constell.constellation.get_sats()
    #Creating dictinary to store sat position data
    sat_dict = {}
    for i in range(len(sats)):
        sat_dict[str(i)] = sats[i].rvECEF.cartesian
    #Creating dictionary to store gs position data
    gs_dict = {}
    for i in range(len(GS)):
        gs_dict["GS"+str(i)] = GS[i].get_ECEF().cartesian
    
    #Creating dictionary to store sat/gs position data with true/false masks
    final_dict = {}
    #import ipdb;ipdb.set_trace()
    for i in gs_dict.keys():
        for j in sat_dict.keys():
            diff = sat_dict[j] - gs_dict[i]
            posNorm = diff.norm()
            trueIdx = posNorm < (d*u.km)
            final_dict[i + "-" + j] = trueIdx

    #Adding times to dictionary
    final_dict["times"] = constell.constellation.planes[0].sats[0].rvECEF.obstime.isot

    return final_dict



def find_feasible_links_gs(constellation, num_sats, num_gs, relative_position_data):
    """
        Returns a list of strings with the pairs of satellites that can communicate during the set time interval
        
        Parameters
        ----------
        constellation (obj) : the constellation object we are looking at 
        num_sats (int) : the number of satellites in the constellation
        num_gs (int) : the number of ground stations
        relative_position_data (array?) : the array of relative_position_data
            
    """
    #Takes the number of satellites and groundstations and give all possible pairs of objects
    L=[]
    for i in range(num_sats):
        L.append(str(i))

    L1=[]
    for i in range(num_gs):
        L1.append("GS" + str(i))
    
    L2 = []
    for i in L1:
        for j in L:
            L2.append(i + "-" + j)

    L_final = L2.copy()

    #Eliminating the pairs from the list that are never able to communicate
    #iterate through the pairs satellites
    for i in L2: 

        #True false mask of ISL opportunities
        mask = relative_position_data[i]
       
        #If this pair has no True time intervals, eliminate it from the list
        if any(mask) == False:
            L_final.remove(i)

    return L_final



def get_multi_availability_gs(constellation, objects, relative_position_data):
    
    
    """
        Returns a list of strings with the time intervals for communication avaialability
        
        Parameters
        ----------
        constellation (obj): the constellation we are looking at 
        objects (list) : list of object pairs
        relative_position_data (array?) :the array of relative_position_data
            
    """
    #True false mask of ISL opportunities
    #Gets satellite time interval data between objects
    #Iterates through the pairs of objects
    L_avail=[]
    for i in objects:
        mask = relative_position_data[i] 
        timeArray = relative_position_data['times']

    #Apply true false mask to time array
        intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
    #If there are no True intervals, let user know
        if any(mask)==False:
            return 'No Intervals Found: Intersatellite Comunication Link is not Feasible'
    #Reformatting the data into a format czml can use
        else:
            L=[]
            for idx, interval in enumerate(intervalsISL):
                start_time = interval[0]
                stop_time = interval[1]
    
        
                L.append(start_time + 'Z/' + stop_time + 'Z')

        L_avail.append(L)
    
    return L_avail



def get_multi_poly_gs(constellation, objects, relative_position_data):
        
    
    """
        Returns of list of the polyline intervals for the list of pairs of objects.
        
        Parameters
        ----------
        constellation (obj): the constellation we are looking at 
        objects (list) : list of object pairs
        relative_position_data (array?) : the array of relative_position_data
    """

    #True false mask of ISL opportunities
    #Gets position data
    #Iterate throught the pairs of satellites 
    L_poly=[]
    for i in objects:
        mask = relative_position_data[i] 
        timeArray = relative_position_data['times']

         #If there are no True intervals, we let user know
        if any(mask)==False:
            return 'Intersatellite Comunication Link is not Feasible'

        #Get false intervals
        false_intervalsISL = utils.get_false_intervals(mask, timeArray)
        L_false=[]
        for idx, interval in enumerate(false_intervalsISL):
            start_time = interval[0]
            stop_time = interval[1]
    
            L_false.append(start_time + 'Z/' + stop_time + 'Z')

        #Get true intervals
        true_intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
        L_true=[]
        for idx, interval in enumerate(true_intervalsISL):
            start_time = interval[0]
            stop_time = interval[1]
    
            L_true.append(start_time + 'Z/' + stop_time + 'Z')
    
        #Creating dictionary
        L_final=[]
        if mask[0]==True:
            for i in range(len(L_true)):
                try:
                    D1={}
                    D1['interval']=L_true[i]
                    D1['boolean']='true'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_false[i]
                    D2['boolean']='false'
                    L_final.append(D2)
                except:
                    pass
        if mask[0]==False:
            for i in range(len(L_false)):
                try:
                    D1={}
                    D1['interval']=L_false[i]
                    D1['boolean']='false'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_true[i]
                    D2['boolean']='true'
                    L_final.append(D2)
                except:
                    pass
        L_poly.append(L_final)

    return L_poly



