import operator
import copy
import collections

# import cartopy.crs as ccrs
import numpy as np
import poliastro
from astropy.coordinates import (
    GCRS,
    ITRS,
    CartesianRepresentation,
    EarthLocation)
from astropy import time
import astropy.units as u
from poliastro.twobody.propagation import propagate
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import func_twobody
from poliastro.bodies import Earth
from poliastro.czml.extract_czml import CZMLExtractor

import matplotlib.pyplot as plt

import satbox as sb

def find_non_dominated_time_deltaV(flatArray):
    """
    Find non-dominated trade space instances when looking at time to pass and delta V value
    of manuever to achieve the pass

    ToDo: Need to look back in code to figure out what this is doing in more detail
    """
    keyfun = operator.attrgetter('time2Pass.value')
    flatSort = copy.deepcopy(flatArray)
    flatSort.sort(key=keyfun) # Sort list by key value (usually time2Pass.value)
    next_idx = 0
    paretoList = copy.deepcopy(flatSort) #List of efficient points that will continually be cut
    paretoSats = [] #List of satellites on pareto front
    while next_idx < len(paretoList) and next_idx < 100:
        best = paretoList[next_idx] #Best satellite, lowest value by key
        paretoList = [f for f in paretoList if f.deltaV < best.deltaV] #Cut out all longer times that take more deltaV
        paretoSats.append(best) #Add best satellite to list
    return paretoSats

def find_non_dominated(flatArray, attribute1, attribute2):
    """
    find non-dominated trade space instances between two attributes, attribute 1 and attribute 2
    
    ToDo: Need to look back in code to figure out how this is working

    """
    keyfun = operator.attrgetter(attribute1)
    flatSort = copy.deepcopy(flatArray)
    flatSort.sort(key=keyfun) # Sort list by key value (usually time2Pass.value)
    next_idx = 0
    paretoList = copy.deepcopy(flatSort) #List of efficient points that will continually be cut
    paretoSats = [] #List of satellites on pareto front
    while next_idx < len(paretoList) and next_idx < 100:
        best = paretoList[next_idx] #Best satellite, lowest value by key
        paretoList = [f for f in paretoList if getattr(f, attribute2) < gettattr(best, attribute2)] #Cut out all longer times that take more deltaV
        paretoSats.append(best) #Add best satellite to list
    return paretoSats
    
def flatten(x): # Flattens a nested list
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def getLonLat(orb, timeInts, pts, J2 = True):
    """
    Function gets latitudes and longitudes for ground track plotting
    
    Inputs
    orb [poliastro orbit object]
    timeInts [astropy object] : time intervals to propagate where 0 refers to the epoch of the orb input variable
    pts [integer] : Number of plot points
    J2 : If True, orbits propagate with J2 perturbation
    
    Outputs
    lon: Longitude (astropy angle units)
    lat: Latitude (astropy angle units)
    height: height (astropy distance units)
    
    """
    if J2:
        def f(t0, state, k):
            du_kep = func_twobody(t0, state, k)
            ax, ay, az = J2_perturbation(
                t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
            )
            du_ad = np.array([0, 0, 0, ax, ay, az])

            return du_kep + du_ad

        coords = propagate(
            orb,
            time.TimeDelta(timeInts),
            method = cowell,
            f=f,
            )
                       
    elif not J2:
        coords = propagate(orb,
         timeDeltas,
         rtol=1e-11,)
    else:
        print("Check J2 input value")
    time_rangePlot = orb.epoch + timeInts
    gcrsXYZ = GCRS(coords, obstime=time_rangePlot,
                   representation_type=CartesianRepresentation)
    itrs_xyz = gcrsXYZ.transform_to(ITRS(obstime=time_rangePlot))
    loc = EarthLocation.from_geocentric(itrs_xyz.x,itrs_xyz.y,itrs_xyz.z)

    lat = loc.lat
    lon = loc.lon
    height = loc.height
    
    return lon, lat, height


# def plotGroundTrack(lon, lat, style):
#     """
#     Plots ground tracks on map using cartopy

#     Inputs
#     lon (deg): Longitude
#     lat (deg): Latitude
#     style (string): linestyle from matplotlib

#     Outputs
#     fig: matplotlib fig object
#     ax: matplotlib ax object
#     """
#     fig = plt.figure(figsize=(15, 25))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.stock_img()
#     ax.plot(lon, lat, 'k', transform=ccrs.Geodetic())
#     ax.plot(lon[0], lat[0], 'r^', transform=ccrs.Geodetic())
#     return fig, ax


def ground_range_spherical(lat1, lon1, lat2, lon2):
    """
    Get ground range between two points on a sphere
    Vallado 4th edition eqn 11-2 pg 854
    
    Corresponding latitudes to Vallado's equations
    subscript 1 is tgt
    subscript 2 is lch
    
    Inputs must be in radians
    """
    delLon = lon1 - lon2
    ground_range = np.arccos(np.sin(lat1) * np.sin(lat2) + 
              np.cos(lat1) * np.cos(lat2) * np.cos(delLon))
    return ground_range

def get_isl_feasibility(dataDict, distanceConstraint = None, slewRateConstraint = None, 
                        dopplerConstraintMax=None, dopplerConstraintMin=None):
    """
    Determines ISL feasibility given the input constraints

    Adds a field to dataDict 'islFeasible' that is a boolean mask of when ISLs are available
    between the two satellites

    Args
        dataDict : Dictionary with keys 'pDiff', 'pDiffNorm', 'pDiffDot', 'slewRate', and 'dopplerShift', 'numSats'
                corresponding to position between satellites, norm of position between satellites
                relative velocity between satellites, slew rates, doppler shift between satellites,
                and number of satellites. Dictionary can be build using the get_relative_velocity_analysis 
                Constellation method.
        distanceConstraint [astropy distance quantity] : Set distance required to close link (usually link budget constrained)
        slewRateConstraint [astropy 1/(time) quantity] : Default unis is rad/s. Slew rate constraint (determined by pointing/ADCS capability)
        dopplerConstraintMax [unitless] : Doppler factor determines relative change in frequency between source and observer
        dopplerConstraintMin [unitless] : Doppler factor determines relative change in frequency between source and observer



    """
    satData = dataDict['satData']
    satKeys = satData.keys()

    numSats = dataDict['numSats'] #number of satellites in the constellation

    for key in satKeys:
        pairData = satData[key]
        timeSteps = len(pairData['timeDeltas'])
        los = pairData['LOS'] #Line of sight mask
        if not any(los): #If no LOS, declare ISL infeasible
            pairData['islFeasible'] = los 
            continue 

        if distanceConstraint:
            distanceMask = pairData['relPosition']['relPosNorm'] < distanceConstraint #find all times when position less than constraint
        else:
            distanceMask = np.full(timeSteps, True) #else set mask to all true

        if slewRateConstraint:
            slewMask = pairData['relVel']['slewRate'] < slewRateConstraint
        else:
            slewMask = np.full(timeSteps, True) #else set mask to all true

        if dopplerConstraintMin and dopplerConstraintMax:
            dopplerMask = (pairData['relVel']['dopplerShift'] < dopplerConstraintMax and 
                pairData['relVel']['dopplerShift'] > dopplerConstraintMin)
        else:
            dopplerMask =  np.full(timeSteps, True) #else set mask to all true

        pairData['islDistanceMask'] = distanceMask
        pairData['islSlewMask'] = slewMask
        pairData['dopplerMask'] = dopplerMask
        pairData['islFeasible'] = np.logical_and.reduce((los, distanceMask, slewMask, dopplerMask))

def calc_temp_resolution(constellation, gs, altDrift = 650*u.km, constraint_type = 'nadir', constraint_angle = 25*u.deg,
                         t2propagate = 5*u.day, tStep = 15*u.s, verbose=True):
    """
    Calculate the temporal resolution of a walker constellation and ground station

    Parameters
    ----------
    constellation: ~satbox.Constellation class
        Constellation to run analysis on
    gs: ~satbox.groundLoc class
        Ground location to target for observing
    altDrift: ~astropy.unit.Quantity
        Altitude of reconfigurable drift orbit
    constraint_type: ~string | "nadir" or "elevation"
            constrain angles with either a "nadir" (usually analogous to sensor FOV) or "elevation" (minimum elevation angle from ground station) constraint
    constraint_angle: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation
    t2propagate: ~astropy.unit.Quantity
        Amount of time to Propagate starting from satellite.epoch
    tStep: ~astropy.unit.Quantity
        Time step used in the propagation
    verbose: Boolean
        Prints out debug statements if True

    Returns
    -------
    output: Dict
        dataOut - temporal resolution data (Dict)
        driftTimes - drift time data (Dict)
        sched - schedule data (Dict)
        walkerSim - propagated walker constellation (satbox.SimConstellation)
        accessObj - access object (satbox.DataAccessConstellation)
    """
    r_drift = poliastro.constants.R_earth + altDrift

    schedDict = constellation.gen_GOM_2_RGT_scheds(r_drift, gs)
    sats2Maneuver, driftTimes, sched = constellation.get_lowest_drift_time_per_plane(schedDict) #Assumes one satellite per plane will get there
    
    walkerSim = sb.SimConstellation(constellation, t2propagate, tStep, verbose = verbose)
    
    walkerSim.propagate(select_sched_sats = sats2Maneuver, verbose=verbose)

    #Create an access object
    accessObject = sb.DataAccessConstellation(walkerSim, gs)

    accessObject.calc_access(constraint_type, constraint_angle)
    
    #Get sat IDSs
    satsStr = sats2Maneuver.values()
    satsStrs = [sat for sat in satsStr]

    #Satellites that will be doing the sensing
    senseSats = [int(string[0].split()[-1]) for string in satsStrs]
    
    dataOut = accessObject.get_temp_resolution(None, sats=senseSats)

    output = {
                'dataOut':dataOut,
                'driftTimes':driftTimes,
                'sched':sched,
                # 'walkerSim':walkerSim,
                'accessObj':accessObject,
    }
    
    return output

def calc_temp_resolution_ascend_descend(constellation, gs, altDrift = 650*u.km, constraint_type = 'nadir', constraint_angle = 25*u.deg,
                         t2propagate = 5*u.day, tStep = 15*u.s, verbose=True):
    """
    Calculate the temporal resolution of a walker constellation and ground station

    Parameters
    ----------
    constellation: ~satbox.Constellation class
        Constellation to run analysis on
    gs: ~satbox.groundLoc class
        Ground location to target for observing
    altDrift: ~astropy.unit.Quantity
        Altitude of reconfigurable drift orbit
    constraint_type: ~string | "nadir" or "elevation"
            constrain angles with either a "nadir" (usually analogous to sensor FOV) or "elevation" (minimum elevation angle from ground station) constraint
    constraint_angle: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation
    t2propagate: ~astropy.unit.Quantity
        Amount of time to Propagate starting from satellite.epoch
    tStep: ~astropy.unit.Quantity
        Time step used in the propagation
    verbose: Boolean
        Prints out debug statements if True

    Returns
    -------
    output: Dict
        dataOut - temporal resolution data (Dict)
        driftTimes - drift time data (Dict)
        sched - schedule data (Dict)
        walkerSim - propagated walker constellation (satbox.SimConstellation)
        accessObj - access object (satbox.DataAccessConstellation)
    """
    r_drift = poliastro.constants.R_earth + altDrift

    schedDict = constellation.gen_GOM_2_RGT_scheds(r_drift, gs, verbose)
    sats2Maneuver, driftTimes, sched = constellation.get_ascending_descending_per_plane(schedDict) #Assumes one satellite per plane will get there
    walkerSim = sb.SimConstellation(constellation, t2propagate, tStep, verbose = verbose)
    
    walkerSim.propagate(select_sched_sats = sats2Maneuver, verbose=verbose)

    #Create an access object
    accessObject = sb.DataAccessConstellation(walkerSim, gs)

    accessObject.calc_access(constraint_type, constraint_angle)
    
    #Get sat IDSs
    satsStr = sats2Maneuver.values()
    satsStrs = [sat for sat in satsStr]
    satsStrsFlat = [item for sublist in satsStrs for item in sublist]

    #Satellites that will be doing the sensing
    senseSats = [int(string.split()[-1]) for string in satsStrsFlat]
    dataOut = accessObject.get_temp_resolution(None, sats=senseSats)

    output = {
                'dataOut':dataOut,
                'driftTimes':driftTimes,
                'sched':sched,
                # 'walkerSim':walkerSim,
                'accessObj':accessObject,
    }
    
    return output

def get_start_stop_intervals(mask, refArray):
    """
    Given a mask of booleans, return a list of start/stop intervals
    where starts refer to the beginnings of true entries

    Args
        mask (array of bools) : Array of booleans
        refArray (array) : Array that mask is to be applied to (usually an array of times)

    Returns
        startStopIntervals (list) : list of start/stop intervals
    """
    accDiff = np.diff(mask*1) #multiply by one to turn booleans into int
    maskEnd = np.squeeze(np.where(accDiff==-1))
    maskStart = np.squeeze(np.where(accDiff==1))
    if mask[0] == 1 and mask[-1] == 1 and all(mask): #Begin and end on true and all true
        startIdx = 0
        endIdx = -1
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 1 and mask[-1] == 1: #Begin and end on an mask
        startIdx = np.append(np.array([0]), maskStart)
        endIdx = np.append(maskEnd, len(refArray)-1)
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 1 and mask[-1] == 0: #Begin on mask and end in no mask
        startIdx = np.append(np.array([0]), maskStart)
        endIdx = maskEnd
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 0 and mask[-1] == 1:#begin in false, end in true
        startIdx = maskStart
        if maskEnd.size == 0:
            endIdx = len(refArray) - 1
        else:
            endIdx = np.append(maskEnd, len(refArray)-1)
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 0 and mask[-1] == 0 and any(mask)==False:
        startStopIntervals = [(None, None)]
        print('No True Intervals Found')
        return startStopIntervals
    elif mask[0] == 0 and mask[-1] == 0:
        startIdx = maskStart
        endIdx = maskEnd
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    else:
        print("check function")
        
    if not isinstance(startTimes, list):
        startTimes = [startTimes]

    if not isinstance(endTimes, list):
        endTimes = [endTimes]
    startStopIntervals = np.column_stack((startTimes[0], endTimes[0]))

    return startStopIntervals

def get_false_intervals(mask, refArray):
    """
    Given a mask of booleans, return a list of start/stop intervals
    where starts refer to the beginnings of true entries

    Args
        mask (array of bools) : Array of booleans
        refArray (array) : Array that mask is to be applied to (usually an array of times)

    Returns
        startStopIntervals (list) : list of start/stop intervals
    """
    accDiff = np.diff(mask*1) #multiply by one to turn booleans into int
    maskEnd = np.squeeze(np.where(accDiff==1))
    maskStart = np.squeeze(np.where(accDiff==-1))
    if mask[0] == 0 and mask[-1] == 0 and any(mask)==False:
        startIdx = 0
        endIdx = -1
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 0 and mask[-1] == 0: #Begin and end on an mask
        startIdx = np.append(np.array([0]), maskStart)
        endIdx = np.append(maskEnd, len(refArray)-1)
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 0 and mask[-1] == 1: #Begin on mask and end in no mask
        startIdx = np.append(np.array([0]), maskStart)
        endIdx = maskEnd
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 1 and mask[-1] == 0:#begin in false, end in true
        startIdx = maskStart
        if maskEnd.size == 0:
            endIdx = len(refArray) - 1
        else:
            endIdx = np.append(maskEnd, len(refArray)-1)
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    elif mask[0] == 1 and mask[-1] == 1 and all(mask):
        startStopIntervals = [(None, None)]
        print('No False Intervals Found')
        return startStopIntervals
    elif mask[0] == 1 and mask[-1] == 1:
        startIdx = maskStart
        endIdx = maskEnd
        startTimes = refArray[startIdx]
        endTimes = refArray[endIdx]
    else:
        print("check function")
        
    if not isinstance(startTimes, list):
        startTimes = [startTimes]

    if not isinstance(endTimes, list):
        endTimes = [endTimes]
    startStopIntervals = np.column_stack((startTimes[0], endTimes[0]))

    return startStopIntervals

def get_potential_isl_keys(satID, keys, excludeList = None):
    """
    Get keys that incorporate the satID in the name
    Args:
        satID (int): Satellite ID of satellite in question
        keys (Dict keys): Keys of all satellite pairs
        excludeList (list) : list of strings of satellites to exclude
    """
    islOppsAll = []
    for key in keys:
        if str(satID) in key:
            islOppsAll.append(key)
    #Exclude mirrors i.e. keep '7-1' but discard '1-7'
    if excludeList:
        islOppsKeys = [x for x in islOppsAll if x.split('-')[0]==str(satID)
                      and x.split('-')[1] not in excludeList]
    else:
        islOppsKeys = [x for x in islOppsAll if x.split('-')[0]==str(satID)]
    return islOppsKeys

def get_divisors(t):
    """
    Gets divisors for integer t. Useful to determine number of planes in 
    a walker constellation with total satellites, t.
    """
    if t%2 == 0: #even
        endRange = int(t/2 + 1)
    elif t%2 == 1:
        endRange = int(np.ceil(t/2))
    divisors = []
    for i in range(1, endRange):
        if t%i==0:
            divisors.append(i)
    return divisors

def get_walker_params(t):
    """
    Gets walker parameters t/p/f given a total number of satellites
    
    Args:
        t (int) : total number of satellites
        
    Returns:
        walkerParams (list) : list of walker constellation parameters
    """
    walkerParams = []
    planes = get_divisors(t)
    planes.append(t) #each satellite with own plane
    
    for p in planes:
        fRange = range(0, p)
        for f in fRange:
            paramTuple = (t, p, f)
            walkerParams.append(paramTuple)
    
    return walkerParams

def get_norm(vec):
    """
    Gets the norm of a vector
    """
    elementSum = 0
    for ele in vec:
        ele2 = ele*ele 
        elementSum += ele2
    norm = np.sqrt(elementSum)
    return norm

def get_unit_vec(vec):
    """
    Gets the unit vector of a vector
    """
    vecNorm = get_norm(vec)
    unitVec = vec / vecNorm
    return unitVec