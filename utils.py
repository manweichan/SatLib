import sys
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

def calc_temp_resolution(constellation, gs, altChange = 100*u.km, constraint_type = 'nadir', constraint_angle = 25*u.deg,
                         t2propagate = 5*u.day, tStep = 15*u.s, verbose=True):
    """
    Calculate the temporal resolution of a walker constellation and ground station

    Parameters
    ----------
    constellation: ~satbox.Constellation class
        Constellation to run analysis on
    gs: ~satbox.groundLoc class
        Ground location to target for observing
    altChange: ~astropy.unit.Quantity
        Altitude change to get to drift orbit
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
    # r_drift = poliastro.constants.R_earth + altDrift

    schedDict = constellation.gen_GOM_2_RGT_scheds(altChange, gs)
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

def calc_temp_resolution_ascend_descend(constellation, gs, altChange = 100*u.km, constraint_type = 'nadir', constraint_angle = 25*u.deg,
                         t2propagate = 5*u.day, tStep = 15*u.s, verbose=True):
    """
    Calculate the temporal resolution of a walker constellation and ground target
    Temporal resolution defined as revisit time.

    Parameters
    ----------
    constellation: ~satbox.Constellation class
        Constellation to run analysis on
    gs: ~satbox.groundLoc class
        Ground location to target for observing
    altChange: ~astropy.unit.Quantity
        Altitude change to get to drift orbit
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
    # r_drift = poliastro.constants.R_earth + altDrift

    schedDict = constellation.gen_GOM_2_RGT_scheds(altChange, gs, tStep)
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

def get_start_stop_intervals(mask, refArray, verbose=False):
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
        if verbose:
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

class TimeVaryingGraph(object):
    def __init__(self, contacts, relaySats):
        """
        Parameters
        ----------
        relOutputData: ~dict
            Dictionary with keys 'contacts' and 'time'
            -----
            'contacts' is a dictionary of boolean arrays of when contacts are
            available between nodes. Node keys are of the form 'i-j' where
            i, and j are integers
            -----
            'time' is a dictionary of astropy times associated with each pair
            of nodes
            -----
        relay_sats: ~list
            List of sat IDs for the satellites that can act as relays
        
        """
        self.contacts = contacts.get('contacts')
        self.contactTimes = contacts.get('time')
        relaySatsStr = [str(s) for s in relaySats]
        self.relaySats = relaySatsStr
    
    def get_relay_sats(self):
        "Returns the nodes of the graph."
        return self.relaySats
    
    def get_outgoing_edges(self, node, currentTime):
        """
        Returns the neighbors of a node.
        
        node: ~str
            node to get the neighbors of
        currentTime: ~astropy.time
            time that node has information, so only contacts after this time
            are considered
        
        """
        
        connections = []
        for key in self.contacts.keys(): #loop over each pair of nodes
            sats = key.split('-')
            satSource = sats[0]
            if satSource != str(node):
                continue
            satDestination = sats[1]
            
            currentTimeMask = currentTime < self.contactTimes.get(key)
            
            relevantContacts = self.contacts.get(key)[currentTimeMask]
            if any(relevantContacts):
                connections.append(satDestination)
        return connections
    
    def value(self, node1, node2, currentTime):
        "Returns the value of an edge between two nodes."
        key = f'{node1}-{node2}'
        
        #Only care about future contacts
        currentTimeMask = currentTime < self.contactTimes.get(key)
        
        #Time array and contacts array truncated so only future times available
        relevantTime = self.contactTimes.get(key)[currentTimeMask]
        relevantContacts = self.contacts.get(key)[currentTimeMask]
        
        firstContactIdx = np.argmax(relevantContacts)
        timeOfFirstContact = relevantTime[firstContactIdx]
        
        time2FirstContact = timeOfFirstContact - currentTime        
        
        return time2FirstContact

def time_varying_dijkstra_algorithm(graph, start_node, start_time, verbose=False): #Notes for reference from Jain paper "Routing in a delay tolerant network"
    unvisited_nodes = list(graph.get_relay_sats()) # This is Q in Jain
    
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {} # This is L in Jain
     
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
    
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = 100 * u.yr #Set max value to a century from now
    for node in unvisited_nodes:
        shortest_path[node] = start_time + max_value  #Maximum time in years added to start_time
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = start_time
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node, shortest_path[current_min_node])
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor, shortest_path[current_min_node]) # This graph.value is the one we have to change.
            # if verbose:
                # print(f'Edge {current_min_node}-{neighbor}: {tentative_value}')
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
        
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path

def print_dijkstra_result(previous_nodes, shortest_path, start_node, target_node):
    """
    Print dijkstra results

    Parameters
    ----------
    previous_nodes: ~dict
        Output of time_varying_dijkstra_algorithm()
    shortest_path: ~dict
        Output of time_varying_dijkstra_algorithm()
    start_node: ~str
        Starting node
    target_node: ~str
        node to end route
    """
    path = []
    node = target_node
    
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
    
    # Add the start node manually
    path.append(start_node)
    
    pathStr = [str(p) for p in path]
    
    print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    print(" -> ".join(reversed(pathStr)))
    pathStr.reverse()

    return pathStr

def prep_dijkstra(constellation, groundStations, groundTarget,
                         recon=True,
                         altChange=100*u.km,
                         constraint_type_gs='elevation', 
                         constraint_angle_gs=25*u.deg,
                         constraint_type_sense='nadir',
                         constraint_angle_sense=20*u.deg,
                         t2propagate=3*u.day,
                         tStep=15*u.s,
                         verbose=False):
    """
    Propagates satellites and creates schedules in preparation for Dijkstra routing

    Parameters
    ----------
    constellation: ~satbox.Constellation class
        Constellation to run analysis on
    groundStations: ~list
        List of satbox.groundLoc stations for data downlink
    groundTarget: ~satbox.groundLoc
        Target location for Earth observation
    recon: ~bool
        If true, reconfigures orbits
    isl: ~bool
        If true, uses ISLs to determine data routing
    altChange: ~astropy.unit.Quantity
        Altitude change to get to drift orbit
    constraint_type_gs: ~string | "nadir" or "elevation"
            constraint angles type for ground station pass
    constraint_angle_gs: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation for ground station
    constraint_type_sense: ~string | "nadir" or "elevation"
            constraint angles type for remote sensing pass
    constraint_angle_sense: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation for ground station
    t2propagate: ~astropy.unit.Quantity
        Amount of time to Propagate starting from satellite.epoch
    tStep: ~astropy.unit.Quantity
        Time step used in the propagation
    verbose: Boolean
        Prints out debug statements if True

    Returns
    -------
    outputDict: ~dict
        Dictionary with keys
        groundStationNodes   - List of ground station IDx
        sats2Maneuver        - List of satellites to be reconfigured
        relOutput            - Relative position and velocity data between satellites in constellation
        accessObjectGS       - Access objects for access with ground stations
        accessObjectTarget   - Access objects for access with ground target
    """
    ## Get groundStationNodes
    groundStationNodes = [str(g.groundID) for g in groundStations]

    ##########  Generate Recon Schedule  ##########
    # r_drift = poliastro.constants.R_earth + altDrift
    if verbose:
        print("Step 1 of 5: Generating Schedule")
    schedDict = constellation.gen_GOM_2_RGT_scheds(altChange, groundTarget)

    ##########  Propagating  ##########
    if verbose:
        print("Step 2 of 5: Propagating Satellites")

    sats2Maneuver = constellation.get_ascending_descending_per_plane(schedDict)[0]
    if not recon:
        select_sched_sats = None
        skip_all_sched = True
    else:
        select_sched_sats = sats2Maneuver
        skip_all_sched = False
    walkerSim = sb.SimConstellation(constellation, t2propagate, tStep, verbose = False)
    walkerSim.propagate(select_sched_sats = select_sched_sats, verbose=False)

    ##########  Relative Position Data  ##########
    if verbose:
        print("Step 3 of 5: Calculating Relative Data")
    relOutput = walkerSim.get_relative_velocity_analysis()

    ##########  Access  ##########
    if verbose:
        print("Step 4 of 5: Calculating Access to groundStations")
    accessObjectGS = sb.DataAccessConstellation(walkerSim, groundStations)
    accessObjectGS.calc_access(constraint_type_gs, constraint_angle_gs)

    accessObjectTarget = sb.DataAccessConstellation(walkerSim, groundTarget)
    accessObjectTarget.calc_access(constraint_type_sense, constraint_angle_sense)

    outputDict = {
                    'groundStationNodes': groundStationNodes,
                    'sats2Maneuver':     sats2Maneuver,
                    'relOutput': relOutput,
                    'accessObjectGS': accessObjectGS,
                    'accessObjectTarget': accessObjectTarget,
    }

    return outputDict

def run_dijkstra_routing(prep_dijkstra_output,
                         isl=True,
                         distanceThreshold=1000*u.km,
                         slewThreshold=3*np.pi/180/u.s, 
                         islTimeThreshold=2.5*u.min,
                         downlinkTimeThreshold=30*u.s,
                         verbose=False):
    """
    Run the dijkstra routing given the data prepared in prep_dijkstra

    Parameters
    ----------
    prep_dijkstra_output: dict
        Output of prep_dijkstra()
    isl: ~bool
        If true, uses ISLs to determine data routing
    distanceThreshold: ~astropy.unit.Quantity
        Distance at which one can close an intersatellite link between satellites
    slewThreshold: ~astropy.unit.Quantity (1/u.s) in radians
        Intersatellite links cannot close links at slew rates greater than this
    islTimeTreshold: ~astropy.unit.Quantity
        ISL contacts must be longer than this to transmit the message in full
    downlinkTimeThreshold: ~astropy.unit.Quantity
        Ground contact must be longer than this to transmit the images in full
    verbose: Boolean
        Prints out debug statements if True

    Returns
    -------
    outputDict: ~dict
        Dictionary with keys
        downlinks_all      - Time of downlinks. Structure[sat][pass#][groundStation]
        paths_all          - The path through the nodes for all downlinks Structure[sat][pass#]
        previous_nodes_all - The previous node that leads to fastest route. Structure[sat][pass#][node]
        shortest_path_all  - The shortest path to each of the nodes. Structure[source][pass#][destination_node]
        passTimes          - pass times for satellite of groundTarget. Structure[sat][intervals/length]
        contacts           - True/False arrays of when contacts are available (useful for plotting)

    """

    #Extract values from pref_dijkstra_output
    groundStationNodes = prep_dijkstra_output.get('groundStationNodes')
    sats2Maneuver = prep_dijkstra_output.get('sats2Maneuver')
    relOutput = prep_dijkstra_output.get('relOutput')
    accessObjectGS = prep_dijkstra_output.get('accessObjectGS')
    accessObjectTarget = prep_dijkstra_output.get('accessObjectTarget')

    ########## Dijkstra ##########
    if verbose:
        print("Step 5 of 5: Run Dijkstra")
    contacts = {}
    contacts['contacts'] = {}
    contacts['time'] = {}
    satData = relOutput.get('satData')

    if isl: #Only calculate sat2sat contacts if ISL available
        for key in satData:
            satPairData = satData.get(key)
            LOS = satPairData.get('LOS')
            positionData = satPairData.get('relPosition').get('relPosNorm')
            positionThresholdMask = positionData < distanceThreshold
            slewRate = satPairData.get('relVel').get('slewRate')
            slewRateMask = slewRate < slewThreshold
            
            LOSPosMask = np.logical_and(LOS, positionThresholdMask)
            contactMask = np.logical_and(LOSPosMask, slewRateMask)
            
            times = satPairData.get('times')
            ssIntervals = get_start_stop_intervals(contactMask, times) #start_stop_intervals

            if any(ssIntervals[0]): #Make sure there are intervals
                ssIntsVals = [(s[1]-s[0]).to(u.min) for s in ssIntervals]

                #Convert to numpy array
                ssIntsNp = [s.to(u.min).value for s in ssIntsVals] * u.min

                #Find where it is less than time threshold
                ints2CutIdx = ssIntsNp < islTimeThreshold
                ints2Cut = ssIntervals[ints2CutIdx]

                for interval in ints2Cut: #Loop through intervals to turn false
                    t0Cut = interval[0]
                    t1Cut = interval[1]

                    id0Cut = np.where(times == t0Cut)[0][0]
                    id1Cut = np.where(times == t1Cut)[0][0]

                    contactMask[id0Cut+1:id1Cut+1] = False

            contacts['contacts'][key] = contactMask
            contacts['time'][key] = times
        
    #Calculate access between ground stations and satellites
    for access in accessObjectGS.allAccessData:
        key = f'{access.groundLocID}-{access.satID}' #Ground as source
        key2 = f'{access.satID}-{access.groundLocID}' #Ground as sink

        cutMask = copy.deepcopy(access.accessMask)

        #Cut out access times that are less than time required to transfer data
        intervalLengths = access.accessIntervalLengths

        if any(intervalLengths): #Makes sure there are access intervals

            accessIntervals = access.accessIntervals

            #Convert to numpy array
            intsLenNp = [s.to(u.min).value for s in intervalLengths] * u.min

            #Find where it is less than time threshold
            ints2CutIdx = intsLenNp < downlinkTimeThreshold
            ints2Cut = accessIntervals[ints2CutIdx]

            for interval in ints2Cut: #Loop through intervals to turn false
                t0Cut = interval[0]
                t1Cut = interval[1]

                id0Cut = np.where(access.time == t0Cut)[0][0]
                id1Cut = np.where(access.time == t1Cut)[0][0]
                cutMask[id0Cut+1:id1Cut+1] = False
        contacts.get('contacts')[key] = cutMask#access.accessMask
        contacts.get('contacts')[key2] = cutMask#access.accessMask
        contacts.get('time')[key] = access.time
        contacts.get('time')[key2] = access.time

    # get nodes
    nodesGS = []
    for key in contacts.get('contacts').keys():
        splitKey = key.split('-')
        source = splitKey[0]
        destination = splitKey[1]
        
        if source not in nodesGS:
            nodesGS.append(source)
        if destination not in nodesGS:
            nodesGS.append(destination)

    # Find times of passes for the sensing satellites
    # Extract sats to maneuver

    maneuverSatIDs = []
    for planeKey in sats2Maneuver:
        plane = sats2Maneuver[planeKey]
        for satKey in plane:
            satID = satKey.split()[1]
            maneuverSatIDs.append(satID)

    #Get pass times for sensing satellites
    passTimes = {}
    for obj in accessObjectTarget.allAccessData:
        satIDStr = str(obj.satID)
        if satIDStr in maneuverSatIDs:
            passTimes[satIDStr] = {}
            passTimes[satIDStr]['intervals'] = obj.accessIntervals
            passTimes[satIDStr]['length'] = obj.accessIntervalLengths
    
    walkerGraph = TimeVaryingGraph(contacts, nodesGS)

    previous_nodes_all = {}
    shortest_path_all = {}

    for sat in maneuverSatIDs:
        if verbose:
            print(f'Dijkstra for Sat {sat}')
        passNum = 0
        satKey = f'sat {sat}'
        previous_nodes_all[satKey] = {}
        shortest_path_all[satKey] = {}
        
        if not isl: #No satellite nodes except for sensing satellite
            nodesNoISL = copy.deepcopy(groundStationNodes)
            nodesNoISL.append(str(sat)) #Just add satellite 
            allContactKeys = list(contacts['contacts'].keys())
            keys2keep = []
            for k in allContactKeys:
                keySplit = k.split('-')
                source = keySplit[0]
                destination = keySplit[1]
                if source in nodesNoISL and destination in nodesNoISL:
                    keys2keep.append(k)
            newContacts = { my_key: contacts['contacts'][my_key] for my_key in keys2keep}
            newTimes = { my_key: contacts['time'][my_key] for my_key in keys2keep}
            newContactGraph = {}
            newContactGraph['contacts'] = newContacts
            newContactGraph['time'] = newTimes
            walkerGraph = TimeVaryingGraph(newContactGraph, nodesNoISL)

        for intervals in passTimes[sat]['intervals']:
            if not any(intervals): #skip if no passes
                continue
            passKey = f'pass {passNum}'
            startTime = intervals[1] #End of pass
            previous_nodes, shortest_path = time_varying_dijkstra_algorithm(graph=walkerGraph, start_node=sat, start_time=startTime, verbose=True)
            previous_nodes_all[satKey][passKey] = previous_nodes
            shortest_path_all[satKey][passKey] = shortest_path
            passNum += 1

    downlinks_all = {}
    paths_all = {}
    for sat in maneuverSatIDs:
        satKey = f'sat {sat}'
        downlinks_all[satKey] = {}
        paths_all[satKey] = {}
        shortest_path_sat = shortest_path_all.get(satKey)
        for passNum in shortest_path_sat:
            passKey = passNum
            passData = shortest_path_sat.get(passNum)
            
            downlinkOptions = {my_key: passData[my_key] for my_key in groundStationNodes}
            quickestDownlinkKey = min(downlinkOptions, key=downlinkOptions.get)
            downlinks_all[satKey][passKey] = {f'{quickestDownlinkKey}': downlinkOptions.get(quickestDownlinkKey)}
            
            previousNodesPass = previous_nodes_all.get(satKey).get(passKey)
            
            if any(previousNodesPass) and quickestDownlinkKey in previousNodesPass.keys(): #Check for empty passes
                path = print_dijkstra_result(previousNodesPass, passData, start_node=sat, target_node=quickestDownlinkKey)
                paths_all[satKey][passKey] = path

    outputDict = {
                    'downlinks_all': downlinks_all,
                    'paths_all':     paths_all,
                    'shortest_path_all': shortest_path_all,
                    'previous_nodes_all': previous_nodes_all,
                    'passTimes': passTimes,
                    'contacts': contacts,
    }

    return outputDict

def get_fastest_downlink(constellation, groundStations, groundTarget,
                         recon=True, isl=True,
                         altChange=100*u.km, 
                         constraint_type_gs='elevation', 
                         constraint_angle_gs=25*u.deg, 
                         constraint_type_sense='nadir',
                         constraint_angle_sense=20*u.deg,
                         t2propagate=3*u.day,
                         tStep=15*u.s, 
                         distanceThreshold=1000*u.km,
                         slewThreshold=3*np.pi/180/u.s, 
                         islTimeThreshold=2.5*u.min,
                         downlinkTimeThreshold=30*u.s,
                         verbose=False):
    """
    Gets the route for fastest downlink for a reconfigurable constellation 
    to a set of ground stations

    Parameters
    ----------
    constellation: ~satbox.Constellation class
        Constellation to run analysis on
    groundStations: ~list
        List of satbox.groundLoc stations for data downlink
    groundTarget: ~satbox.groundLoc
        Target location for Earth observation
    recon: ~bool
        If true, reconfigures orbits
    isl: ~bool
        If true, uses ISLs to determine data routing
    altChange: ~astropy.unit.Quantity
        Altitude change to get to drift orbit
    constraint_type_gs: ~string | "nadir" or "elevation"
            constraint angles type for ground station pass
    constraint_angle_gs: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation for ground station
    constraint_type_sense: ~string | "nadir" or "elevation"
            constraint angles type for remote sensing pass
    constraint_angle_sense: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation for ground station
    t2propagate: ~astropy.unit.Quantity
        Amount of time to Propagate starting from satellite.epoch
    tStep: ~astropy.unit.Quantity
        Time step used in the propagation
    distanceThreshold: ~astropy.unit.Quantity
        Distance at which one can close an intersatellite link between satellites
    slewThreshold: ~astropy.unit.Quantity (1/u.s) in radians
        Intersatellite links cannot close links at slew rates greater than this
    islTimeTreshold: ~astropy.unit.Quantity
        ISL contacts must be longer than this to transmit the message in full
    downlinkTimeThreshold: ~astropy.unit.Quantity
        Ground contact must be longer than this to transmit the images in full
    verbose: Boolean
        Prints out debug statements if True

    Returns
    -------
    outputDict: ~dict
        Dictionary with keys
        downlinks_all      - Time of downlinks. Structure[sat][pass#][groundStation]
        paths_all          - The path through the nodes for all downlinks Structure[sat][pass#]
        previous_nodes_all - The previous node that leads to fastest route. Structure[sat][pass#][node]
        shortest_path_all  - The shortest path to each of the nodes. Structure[source][pass#][destination_node]
        passTimes          - pass times for satellite of groundTarget. Structure[sat][intervals/length]
        contacts           - True/False arrays of when contacts are available (useful for plotting)
    """

    prepOutput = prep_dijkstra(constellation, groundStations, groundTarget, recon=True,
                         altChange=altChange,
                         constraint_type_gs=constraint_type_gs, 
                         constraint_angle_gs=constraint_angle_gs,
                         constraint_type_sense=constraint_type_sense,
                         constraint_angle_sense=constraint_angle_sense,
                         t2propagate=t2propagate,
                         tStep=tStep,
                         verbose=verbose)
    dijkstraOutput = run_dijkstra_routing(prepOutput, isl=isl,
                         distanceThreshold=distanceThreshold,
                         slewThreshold=slewThreshold, 
                         islTimeThreshold=islTimeThreshold,
                         downlinkTimeThreshold=downlinkTimeThreshold,
                         verbose=verbose)
    return dijkstraOutput
    

def calc_metrics(dijkstraData, T=3*u.day):
    """
    Calculates Age of Information
    
    Parameters
    ----------
    dijkstraData: dict
        Output of utils.get_fastest_downlink
    T: ~astropy.unit.Quantity
        Time of entire simulation (to calculate average)
        
    Returns
    -------
    AoI: ~astropy.unit.Quantity
        average age of information
    srt: ~astropy.unit.Quantity
        system response time
    passTimeSum: ~astropy.unit.Quantity
        total sum of pass time over target
    """
    downlinks_all = dijkstraData.get('downlinks_all')
    passTimes = dijkstraData.get('passTimes')
    timeStream = dijkstraData.get('contacts').get('time')

    tStreamKeys = list(timeStream.keys())
    t0 = timeStream[tStreamKeys[0]][0] #Get start of sim
    tf = timeStream[tStreamKeys[0]][-1] #Get end time of sim

    downlinks_all_flat = {}
    #Flatten downlinks data dictionary
    for satKey in downlinks_all:
        satDict = downlinks_all.get(satKey)
        for passKey in satDict:
            passDict = satDict.get(passKey)
            for gsKey in passDict:
                dlTime = passDict.get(gsKey)
                newKey = satKey + '|' + passKey# + '|' + gsKey
                downlinks_all_flat[newKey] = dlTime

    downlinks_all_sorted_raw = dict(sorted(downlinks_all_flat.items(), key=lambda item: item[1]))
    #Remove edge cases
    downlinks_all_sorted = {}
    for key in downlinks_all_sorted_raw:
        if downlinks_all_sorted_raw[key] <= tf:
            downlinks_all_sorted[key] = downlinks_all_sorted_raw[key]
        else:
            continue    

    passTimeDict = {}
    passTimeSum = 0
    #Flatten passTimes
    for key in downlinks_all_sorted:
        satPassGS = key.split('|')
        sat = satPassGS[0].split()[1]
        passNum = satPassGS[1].split()[1]

        passTime = passTimes[sat].get('intervals')[int(passNum)][1]

        passTimeLength = passTimes[sat].get('length')[int(passNum)]

        passTimeSum += passTimeLength

        finalKey = f'sat {sat}|pass {passNum}'

        passTimeDict[finalKey] = passTime

    # Calculate age of information
    keys = list(downlinks_all_sorted.keys())
    startIdx = 1

    frac = 0

    # Calculate AoI

    for idx in range(1, len(keys)):
        key_i = keys[idx]
        key_im1 = keys[idx-1]

        t21 = (downlinks_all_sorted[key_i] - passTimeDict[key_im1]).sec**2
        t11 = (downlinks_all_sorted[key_im1] - passTimeDict[key_im1]).sec**2

        frac += (t21 - t11)/2 *u.s*u.s


    AoI = (frac/T).decompose().to(u.min)
    srt = (downlinks_all_sorted[keys[0]] - t0).sec #In seconds
    srtMin = srt/60 * u.min

    outputMetrics = {
                        'AoI': AoI,
                        'srt': srtMin,
                        'passTimeSum': passTimeSum,
    }

    return outputMetrics

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