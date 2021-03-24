from orbitalMechanics import *
from satClasses import *

import numpy as np
from numpy.linalg import norm
from poliastro import constants
from poliastro.earth import Orbit
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
import astropy
import astropy.units as u
from astropy import time

import matplotlib.pyplot as plt

def getNuIntersect(orb1, orb2):
    """Given two orbits, determine true anomaly (argument of latitude)
    of each circular orbit for the position of closest approach of the orbits
    themselves (not the individual satellites). This will help
    determine which point in the orbit is desired for rendezvous for ISL.

    Input
    orb1 (Satellite object poliastro)
    orb2 (Satellite object poliastro)

    Output
    nus: Set of 2 true anomalies for each respective orbit for optimal ISL distance closing

    """
    L1 = np.cross(orb1.r, orb1.v)  # Angular momentum of first orbit
    L2 = np.cross(orb2.r, orb2.v)
    L1hat = L1/norm(L1)
    L2hat = L2/norm(L2)
    intersectLine = np.cross(L1hat, L2hat)

    # Check for co-planar orbits
    Lcheck = np.isclose(L1hat, L2hat)  # Check if angular momentum is the same
    if np.all(Lcheck):  # Coplanar orbits
        print("Orbits are co-planar")
        nus1 = None
        nus2 = None
        return nus1, nus2

    # Check to see if intersect line vector points to first close pass after RAAN
    if intersectLine[-1] > 0:  # Intersection line points to first crossing after RAAN
        pass
    # Flip intersection lilne to point to first crossing after RAAN
    elif intersectLine[-1] < 0:
        intersectLine = -intersectLine

    # Get Raan direction GCRF
    khat = np.array([0, 0, 1])  # Z direction
    raanVec1 = np.cross(khat, L1hat)
    raanVec2 = np.cross(khat, L2hat)

    # Angle formula: cos(theta) = (a dot b) / (norm(a) * norm(b))
    # Gets angle from True anomaly at RAAN (0 deg) to intersect point
    ab1 = np.dot(raanVec1, intersectLine) / \
        (norm(raanVec1) * norm(intersectLine))
    ab2 = np.dot(raanVec2, intersectLine) / \
        (norm(raanVec2) * norm(intersectLine))
    nu1 = np.arccos(ab1)
    nu2 = np.arccos(ab2)

    # Check for equatorial orbits
    L1Eq = np.isclose(orb1.inc, 0 * u.deg)
    L2Eq = np.isclose(orb2.inc, 0 * u.deg)
    if L1Eq:
        nu1 = orb2.raan
    elif L2Eq:
        nu2 = orb1.raan
    else:
        pass
    # return true anomaly of second crossing as well
    nus1raw = [nu1, nu1 + np.pi * u.rad]
    nus2raw = [nu2, nu2 + np.pi * u.rad]

    # Make sure all values under 360 deg
    nus1 = [np.mod(nu, 360*u.deg) for nu in nus1raw]
    nus2 = [np.mod(nu, 360*u.deg) for nu in nus2raw]

    return nus1, nus2



# May have to redo this function later in order to do a state search
def scheduleISL(intOrb, tarOrb, numIntOrbs, numTarOrbs):
    """
    Function outputs a schedule of burns in order to attain an ISL given two
    orbits.
    
    Inputs
    intOrb (Poliastro object): Orbit of interceptor aka the one making the maneuver
    tarOrb (Poliastro orbit): Orbit of target satellite
    numIntOrbs (integer): Number of orbits interceptor makes before rendezvous with target. Usually same integer as numTarOrbs
    numTarOrbs (integer): Number of orbits target makes before rendezvous with target. Usually same as numIntOrbs
    
    Outputs
    sched (Schedule object): List of scheduled burns
    """
    nus1, nus2 = getNuIntersect(intOrb, tarOrb) #Get true anomaly of orbit intersects with respect to each orbit
    
    sched  = []

    if nus1 is not None:
        for idx, val in enumerate(nus1):
            "Cycle through each intersect points (should be 2)"
            # Extract true anomaly for each orbit of the corresponding intersection point
            intNu1 = nus1[idx] #True anomaly of intercept for orbit 1
            intNu2 = nus2[idx]

            # Propagate intOrb to true anomaly
            t2int_orb1 = intOrb.time_to_anomaly(intNu1)
            orb1_init2rvd = intOrb.propagate(t2int_orb1)
            
            tAtInitMan = intOrb.epoch + t2int_orb1 # Keep track off time. Time at first maneuver.

            #Propagate target orbit by same amount
            orb2_init2rvd = tarOrb.propagate(t2int_orb1)
        
            # Determine phase angle for target to get to intercept point. Defined as rvd point to current true anomaly
            phaseAng_tar2rvd = intNu2 - orb2_init2rvd.nu

            if phaseAng_tar2rvd < -180 * u.deg: #Make sure all points are between -180 and 180
                phaseAng_tar2rvd = phaseAng_tar2rvd + 2 * np.pi * u.rad
            elif phaseAng_tar2rvd > 180 * u.deg:
                phaseAng_tar2rvd = phaseAng_tar2rvd - 2 * np.pi * u.rad
            else:
                pass

            ## Get phasing maneuver. 

            # delVs apply to interceptor
            # import ipdb; ipdb.set_trace()
            # tPhase, delVPhase, delV1, delV2, aPhase = t_phase_coplanar(orb2_init2rvd.a.to(u.m).value, 
            #                               phaseAng_tar2rvd.to(u.rad).value, numTarOrbs,
            #                                                    numIntOrbs)
            tPhase, delVPhase, delV1, delV2, aPhase, flag = t_phase_coplanar(orb2_init2rvd.a, 
                                  phaseAng_tar2rvd.to(u.rad), numTarOrbs, numIntOrbs)
            tPhaseS = tPhase #convert to seconds (astropy)
            # delV1_ms = delV1 #convert to m/s (astropy)
            # delV2_ms = delV1
            orb2_rvd = orb2_init2rvd.propagate(tPhaseS) #target orbit at rendezvous
            
            #Keep track of time for 2nd maneuver
            tAtFinMan = tAtInitMan + tPhaseS
            
            v_dir1 = orb1_init2rvd.v/norm(orb1_init2rvd.v) #Velocity direction of interceptor orbit at node
            delVVec1 = v_dir1 * delV1 #DeltaV vector GCRS frame
            deltaV_man1 = Maneuver.impulse(delVVec1)
            orb1_phase_i = orb1_init2rvd.apply_maneuver(deltaV_man1) #Applied first burn
            orb1_phase_f = orb1_phase_i.propagate(tPhaseS) #Propagate through phasing orbit

            v_dir2 = orb1_phase_f.v/norm(orb1_phase_f.v) #Get direction of velocity at end of phasing
            delVVec2 = v_dir2 * delV2
            deltaV_man2 = Maneuver.impulse(v_dir2 * delV2) #Delta V  of recircularizing orbit
            orb1_rvd = orb1_phase_f.apply_maneuver(deltaV_man2) #Orbit at rendezvous
            
            burnSched_i = BurnSchedule(tAtInitMan, delVVec1) #First burn
            burnSched_f = BurnSchedule(tAtFinMan, delVVec2) #Second burn
            sched.append([burnSched_i, burnSched_f])
    return sched

def get_pass_times_anomalies(orb, gs, dates,
                             refVernalEquinox = astropy.time.Time("2021-03-20T0:00:00", format = 'isot', scale = 'utc')):
    """
    Given an orbit and ground site, gets the pass time and true anomaly 
    required to complete a pass. This is a rough estimate of the time and good
    for quick but not perfectly accurate scheduling. 
    Loosely based on Legge's thesis section 3.1.2

    Inputs:
    orb (Satellite object from satClasses): orbit of satellite
    gs (GroundStation orbject from satClasses): Ground site for overpass
    dates (astropy time object): Days for which pass times are provided
    refVernalEquinox (astropy time object): Date of vernal equinox. Default is for 2021

    Outputs:
    times (array of astropy time objects): Times of overhead passes
    anoms (deg): True anomalies required to complete pass. First is for ascending pass, second is for descending pass
    """

    ## TODO: Doesn't yet account for RAAN drift due to J2 perturbation

    ## Extract relevant orbit and ground station parameters
    i = orb.inc
    lon = gs.lon
    lat = gs.lat
    raan = orb.raan
    delLam = np.arcsin(np.tan(lat) / np.tan(i)) #Longitudinal offset
    theta_GMST_a = raan + delLam - lon #ascending sidereal angle of pass
    theta_GMST_d = raan - delLam - lon - np.pi * u.rad #deescending sidereal angle of pass
    
    
    delDDates = [day - refVernalEquinox for day in dates] #Gets difference in time from vernal equinox
    delDDateDecimalYrList = [delDates.to_value('year') for delDates in delDDates] #Gets decimal year value of date difference
    delDDateDecimalYr = np.array(delDDateDecimalYrList)
    
    #Get solar time values for ascending and descending pass
    theta_GMT_a_raw = theta_GMST_a - 2*np.pi * delDDateDecimalYr * u.rad + np.pi * u.rad
    theta_GMT_d_raw = theta_GMST_d - 2*np.pi * delDDateDecimalYr * u.rad + np.pi * u.rad

    theta_GMT_a = np.mod(theta_GMT_a_raw, 360 * u.deg)
    theta_GMT_d = np.mod(theta_GMT_d_raw, 360 * u.deg)

    angleToHrs_a = astropy.coordinates.Angle(theta_GMT_a).hour
    angleToHrs_d = astropy.coordinates.Angle(theta_GMT_d).hour

    tPass_a = [day + angleToHrs_a[idx] * u.hr for idx, day in enumerate(dates)]
    tPass_d = [day + angleToHrs_d[idx] * u.hr for idx, day in enumerate(dates)]
    times = [tPass_a, tPass_d]
    raans, anoms = desiredRAAN_FromPassTime(tPass_a[0], gs, i)

    return times, anoms

def desiredRAAN_FromPassTime(tPass, gs, i):
    """
    Gets the desired orbit specifications from a desired pass time and groundstation
    Based on equations in section 3.1.2 in Legge's thesis (2014)
    
    Inputs
    tPass (astropy time object): Desired time of pass. Local UTC time preferred
    gs (GroundStation object): Ground station/location of pass
    i (astropy u.rad): Inclination of orbit
    
    Outputs
    raans [List]: 2 element list where 1st element corresponding to RAAN in the ascending case
                    and the 2nd element correspond to RAAN in the descending case
    Anoms [List]: 2 element list where the elements corresponds to true Anomalies (circular orbit)
                    of the ascending case and descending case respectively
    """

    tPass.location = gs.loc #Make sure location is tied to time object
    theta_GMST = tPass.sidereal_time('mean', 'greenwich') #Greenwich mean sidereal time
    # import ipdb; ipdb.set_trace()

    ## Check if astropy class. Make astropy class if not
    if not isinstance(gs.lat, astropy.units.quantity.Quantity):
        gs.lat = gs.lat * u.deg
    if not isinstance(gs.lon, astropy.units.quantity.Quantity):
        gs.lon = gs.lon * u.deg
    if not isinstance(i, astropy.units.quantity.Quantity):
        i = i * u.rad
    
    dLam = np.arcsin(np.tan(gs.lat.to(u.rad)) / np.tan(i.to(u.rad)))

    #From Legge eqn 3.10 pg.69
    raan_ascending = theta_GMST - dLam + np.deg2rad(gs.lon)
    raan_descending = theta_GMST + dLam + np.deg2rad(gs.lon) + np.pi * u.rad
    
    
    #Get mean anomaly (assuming circular earth): https://en.wikipedia.org/wiki/Great-circle_distance
    n1_ascending = np.array([np.cos(raan_ascending),np.sin(raan_ascending),0]) #RAAN for ascending case in ECI norm
    n1_descending = np.array([np.cos(raan_descending),np.sin(raan_descending),0]) #RAAN for descending case in ECI norm
    
    n2Raw = gs.loc.get_gcrs(tPass).data.without_differentials() / gs.loc.get_gcrs(tPass).data.norm() #norm of ground station vector in ECI
    n2 = n2Raw.xyz
    
    n1_ascendingXn2 = np.cross(n1_ascending, n2) #Cross product
    n1_descendingXn2 = np.cross(n1_descending, n2)

    n1_ascendingDn2 = np.dot(n1_ascending, n2) #Dot product
    n1_descendingDn2 = np.dot(n1_descending, n2)

    n1a_X_n2_norm = np.linalg.norm(n1_ascendingXn2)
    n1d_X_n2_norm = np.linalg.norm(n1_descendingXn2)
    if gs.lat > 0 * u.deg and gs.lat < 90 * u.deg: #Northern Hemisphere case
        ca_a = np.arctan2(n1a_X_n2_norm, n1_ascendingDn2) #Central angle ascending
        ca_d = np.arctan2(n1d_X_n2_norm, n1_descendingDn2) #Central angle descending
    elif gs.lat < 0 * u.deg and gs.lat > -90 * u.deg: #Southern Hemisphere case
        ca_a = 2 * np.pi * u.rad - np.arctan2(n1a_X_n2_norm, n1_ascendingDn2) 
        ca_d = 2 * np.pi * u.rad - np.arctan2(n1d_X_n2_norm, n1_descendingDn2)
    elif gs.lat == 0 * u.deg: #Equatorial case
        ca_a = 0 * u.rad
        ca_d = np.pi * u.rad
    elif gs.lat == 90 * u.deg or gs.lat == -90 * u.deg: #polar cases
        ca_a = np.pi * u.rad
        ca_d = 3 * np.pi / 2 * u.rad
    else:
        print("non valid latitude")
    raans = [raan_ascending, raan_descending]
    Anoms = [ca_a, ca_d]
    
    return raans, Anoms

def getDesiredPassOrbits(constellation, passTimes, tInit, alts, incs, anoms):
    """
    For a constellation and ground station, returns a set of desired Satellite objects that will
    pass the ground station at the requested pass times

    Inputs:
    Constellation (List): Constellation list where satellites are organized in the by planes then satellites. Ex indices: Walker[planeIdx][satIdx]
    passTimes (List): Desired pass times for each satellite in the constellation. Similar structure as Constellation
    tInit (astropy time object): Initialization time. This is the time that objects are back propagated to in order to find the initial difference in mean anomaly
    alts (List): Altitudes for the corresponding satellites. Similar structure as Constellation
    incs (List): Inclinations for the corresopnding satellites. Similar structure as Constellation
    anoms (List): Anomalies (Keplar orbital property) for each satellite during the pass. Retrieved through a function such as get_pass_times_anomalies

    Outputs:
    passOrbits_a (List): List of orbits that will pass the ground station at the desired passTime on the ascending portion of orbit
    passOrbits_d (List): List of orbits that will pass the ground station at the desired passTime on the descending portion of orbit
    """
    passOrbits_a = []  # Index is passOrbits_a[plane][sat][day]
    passOrbits_d = []
    for idxPlane, plane in enumerate(passTimes):
        planeSats_a = []
        planeSats_d = []
        for idxSat, sat in enumerate(plane):
            satTimes_a = []
            satTimes_d = []

            sat0 = constellation.planes[idxPlane].sats[idxSat] #Get current satellite
            sat0Epoch = sat0.epoch
            sat0Nu = sat0.nu
            for idxTime, times_a in enumerate(sat[0]):
                orbit_back_a = Satellite.circular(Earth, 
                                                  alt=alts[idxPlane][idxSat],
                                                  inc=incs[idxPlane][idxSat],
                                                  raan=constellation.planes[idxPlane].sats[idxSat].raan,
                                                  arglat=anoms[idxPlane][idxSat][0],
                                                  epoch=times_a)

                # Back propagate orbits as well
                orbit_back0_a = orbit_back_a.propagate(tInit)
                nu = orbit_back0_a.nu

                # Find nu difference
                phaseAng = sat0Nu - nu

                # Find time to pass
                t2Pass = times_a - sat0Epoch

                backDict = {
                    "orbPass": orbit_back_a, #Desired orbit
                    "orb0": orbit_back0_a, #Back propagated orbits
                    "nuBack": nu,
                    "phaseAng": phaseAng,
                    "t2Pass": t2Pass
                }
                satTimes_a.append(backDict)

            for idxTime, times_d in enumerate(sat[1]):
                orbit_back_d = Satellite.circular(Earth,
                                                  alt=alts[idxPlane][idxSat],
                                                  inc=incs[idxPlane][idxSat], 
                                                  raan=constellation.planes[idxPlane].sats[idxSat].raan,
                                                  arglat=anoms[idxPlane][idxSat][1],
                                                  epoch=times_d)

                # Back propagate orbits as well
                orbit_back0_d = orbit_back_d.propagate(tInit)
                nu = orbit_back0_d.nu

                # Find nu difference
                phaseAng = sat0Nu - nu

                # Find time to pass
                t2Pass = times_d - sat0Epoch

                backDict = {
                    "orbPass": orbit_back_d,
                    "orb0": orbit_back0_d, #Back propagated orbits
                    "nuBack": nu,
                    "phaseAng": phaseAng,
                    "t2Pass": t2Pass
                }
                satTimes_d.append(backDict)

            planeSats_a.append(satTimes_a)
            planeSats_d.append(satTimes_d)
        passOrbits_a.append(planeSats_a)
        passOrbits_d.append(planeSats_d)
    return passOrbits_a, passOrbits_d


def findNonDominated(timesPassFlat, delVFlat):
    """
    Finds the non-dominated points in a set of points. The utopia point is the minimum in this case (0,0)
    First time making this function, it takes in times and deltaV, so we want the results that are the 
    fastest and cost the least deltaV

    Inputs
    timesPassFlat (np.array): flat array ; can use np.array.flatten()
    delVFlat (np.arrray): flat array ; can use np.array.flatten()

    Outputs
    timesEff (np.array): array of the most efficient values from timesPassFlat
    delVEff (np.array): array of the most efficient valuees from delVFlat
    """
    idSort = np.argsort(timesPassFlat)

    timesSorted = timesPassFlat[idSort]
    delVSorted = delVFlat[idSort]

    timesInLoop = np.copy(timesSorted)
    delVInLoop = np.copy(delVSorted)

    is_efficient = np.arange(timesSorted.shape[0])
    n_points = timesSorted.shape[0]
    next_pt_idx = 0
    while next_pt_idx < len(timesInLoop):
        # Find delta Vs that took less fuel, but more time
        #     import ipdb; ipdb.set_trace()
        # Already starting with 'fastest' time because times werre sorted
        nondominated_point_mask = delVInLoop < delVInLoop[next_pt_idx]
        nondominated_point_mask[:next_pt_idx+1] = True
        is_efficient = is_efficient[nondominated_point_mask]
        timesInLoop = timesInLoop[nondominated_point_mask]
        delVInLoop = delVInLoop[nondominated_point_mask]
        next_pt_idx += 1
    timesEff = timesSorted[is_efficient]
    delVEff = delVSorted[is_efficient]
    return timesEff, delVEff

def findParetoIds(delVEff, timesEff, t_a, t_d, delv_a, delv_d):
    """
    Given a set of efficient pareto values (timesEff), return the indices that they correspond to in the larger arrays: t_a, t_d

    Inputs
    timeseEff (np.array): array of efficient values from larger set
    t_a (List): List of all times (ascending orbits) in original data structure 
    t_d (List): List of all times (descending orbits) in original data structure

    Outputs
    paretoIdx (List): List of indices corresponding to the larger array (t_a/t_d)
    tags (List of strings): List of which array (t_a or t_d, to 'a' and 'd' respectively) that was matched
    """
    paretoIdx = []
    tags = []
    for paretoID, teff in enumerate(timesEff):
        idx = np.argwhere(t_a == teff)
        tag = 'a'
        if idx.size == 0:
            idx = np.argwhere(t_d == teff)
            tag = 'd'
        tags.append(tag)
        if tag == 'a':
            for index in idx:
                delVCheck = delv_a[tuple(index)]
                if delVCheck == delVEff[paretoID]:
                    idxFinal = index
                    break
        elif tag == 'd':
            for index in idx:
                delVCheck = delv_d[tuple(index)]
                if delVCheck == delVEff[paretoID]:
                    idxFinal = index
                    break
        else:
            raise Exception("tag not defined correctly")

        paretoIdx.append(idxFinal)
    return paretoIdx, tags

def getPassSats(constellation, gs, tInit, daysAhead, plot = False, savePlot = False, figName = None, pltLgd = False,
                numSats = False):
    """
    Given a constellation, ground station, initial time, and how many days ahead to plan, the function returns
    which ids and which orbits are the optimal in terms of deltaV and timing

    Inputs
    Constellation (List): Constellation list where satellites are organized in the by planes then satellites. Ex indices: Walker[planeIdx][satIdx]
    gs (groundStation object): Ground station object to predict passes
    tInit (astropy time object): Time to initialize planner
    daysAhead (int): Amount of days ahead to for the scheduler to plan for
    plot (Bool): Plots if True
    savePlot (Bool): Saves if False
    figName (Str): String for figure name if savePlot = True
    pltLgd (Bool): Plots legend if True
    numSats (int): Integer number 'n'. Ouputs the 'n' fastest ground passes in ids.

    Outputs
    idsOut (Dict): Dictionary of indices corresponding to which satellite in the constellation are optimally positioned to make a maneuver. First idx is plane, second idx is satellite in plane, 3rd idx is pass corresopnding to day after initialization
    tags (List of strings): List of which array (t_a or t_d, to 'a' and 'd' respectively) that correspond to if the desired pass corresponds to the ascending or descending pass
    delVOut (Dict): Dictionary of delV burns required
    tOut (Dict): Dictionary of expected pass times
    orbOut (Dict): Dictionary of desired 'final' orbits after maneuver
    """
    # Get times of interest
    tInitMJDRaw = tInit.mjd
    tInitMJD = int(tInitMJDRaw)
    
    dayArray = np.arange(0, daysAhead + 1)
    days2InvestigateMJD = list(tInitMJD + dayArray) #Which days to plan over
    days2Investigate = [time.Time(dayMJD, format='mjd', scale='utc')
                        for dayMJD in days2InvestigateMJD]

    passTimes = []  # First layer down depicts satellites in a plane. Second layer depicts individual satellites in a plane. 3rd layer 1st idx is for ascending pass, 2nd idx is for descending pass, 4th idx is pass corresopnding to day after initialization
    anoms = []
    aSats = getSatValues_const(constellation, 'a') #Semi Major axes
    alts = [[sat.a - poliastro.constants.R_earth for sat in plane.sats] for plane in constellation.planes]
    incs = getSatValues_const(constellation, 'inc')
    for plane in constellation.planes:
        passTimesPlane = []
        anomaliesPlane = []
        incPlane = []
        for sat in plane.sats:
            passTime, anom = get_pass_times_anomalies(
                sat, gs, days2Investigate)
            
            #Append to each list representing each plane
            passTimesPlane.append(passTime)
            anomaliesPlane.append(anom)
            
        passTimes.append(passTimesPlane)
        anoms.append(anomaliesPlane)

    passOrbits_a, passOrbits_d = getDesiredPassOrbits(constellation, 
                                                      passTimes, 
                                                      tInit,
                                                      alts, 
                                                      incs, 
                                                      anoms)

    # Extract nus and times
    phaseAngs_a_raw = [[[satID["phaseAng"] for satID in sat]
                        for sat in plane] for plane in passOrbits_a]
    phaseAngs_d_raw = [[[satID["phaseAng"] for satID in sat]
                        for sat in plane] for plane in passOrbits_d]
#     phaseAngs_a_raw = getSatValues_constPlan(passOrbits_a, 'phaseAng')
#     phaseAngs_d_raw = getSatValues_constPlan(passOrbits_d, 'phaseAng')

    phaseAngs_a = [[[np.mod(ang, 360 * u.deg) - 180 * u.deg for ang in sat]
                    for sat in plane] for plane in phaseAngs_a_raw]
    phaseAngs_d = [[[np.mod(ang, 360 * u.deg) - 180 * u.deg for ang in sat]
                    for sat in plane] for plane in phaseAngs_d_raw]

    t2obs_a = [[[satID["t2Pass"] for satID in sat]
                for sat in plane] for plane in passOrbits_a]
    t2obs_d = [[[satID["t2Pass"] for satID in sat]
                for sat in plane] for plane in passOrbits_d]

    timesSec_a = [[[timeVal.to_value('s') for timeVal in sat]
                   for sat in plane] for plane in t2obs_a]
    timesSec_d = [[[timeVal.to_value('s') for timeVal in sat]
                   for sat in plane] for plane in t2obs_d]

    timesArr_a = np.array(timesSec_a)
    timesArr_d = np.array(timesSec_d)

    period = [[sat.period.value for sat in plane.sats] for plane in constellation.planes]
    periodArr = np.array(period)

    semiMajors = [[sat.a.to(u.m).value for sat in plane.sats] for plane in constellation.planes]
    semiMajorArr = np.array(semiMajors)

    if timesArr_a.ndim != periodArr.ndim:
        periodArr = np.repeat(periodArr[:, :, np.newaxis], np.shape(timesArr_a)[-1], axis=2)
    orbRaw_a = timesArr_a / periodArr
    orbKInt_a = orbRaw_a.astype(int)

    orbRaw_d = timesArr_d / periodArr
    orbKInt_d = orbRaw_d.astype(int)

    phaseAngsArr_a = [[[ang.to(u.rad).value for ang in sat]
                       for sat in plane] for plane in phaseAngs_a]
    phaseAngsArr_d = [[[ang.to(u.rad).value for ang in sat]
                       for sat in plane] for plane in phaseAngs_d]

    t_a, delv_a, delv1_a, delv2_a, aphase_a, flag_a = t_phase_coplanar(
        semiMajorArr, phaseAngsArr_a, orbKInt_a, orbKInt_a)
    t_d, delv_d, delv1_d, delv2_d, aphase_d, flag_d = t_phase_coplanar(
        semiMajorArr, phaseAngsArr_d, orbKInt_d, orbKInt_d)


    # Get feasible times/deltaVs
    timesFeasible_a = timesArr_a[flag_a]
    timesFeasible_d = timesArr_d[flag_d]
    delVFeasible_a = delv_a[flag_a]
    delVFeasible_d = delv_d[flag_d]

    # Flatten data arrays to put into pareto analysis
    timesPassFlat_a = timesFeasible_a.flatten()
    timesPassFlat_d = timesFeasible_d.flatten()
    delVFlat_a = delVFeasible_a.flatten()
    delVFlat_d = delVFeasible_d.flatten()

    timesPassFlat = np.append(timesPassFlat_a, timesPassFlat_d)
    delVFlat = np.append(delVFlat_a, delVFlat_d)

    
    timesEff, delVEff = findNonDominated(timesPassFlat, delVFlat)
    ids, tags = findParetoIds(delVEff, timesEff, timesArr_a, timesArr_d, delv_a, delv_d) #Gets ids and tags, which are for either ascending or descending


    if numSats: #Find ids for lowest times
        idChosen = ids[0:numSats]
    else:
        idChosen = None
    idsOut = {
                'idsAll' : ids, #In order of times
                'idsChosen' : idChosen
    }

    ## Get feasible delta Vs
    delv1_a_out = [delv1_a[tuple(x)] for x,t in zip(ids, tags) if t == 'a']
    delv2_a_out = [delv2_a[tuple(x)] for x,t in zip(ids, tags) if t == 'a']

    delv1_d_out = [delv1_d[tuple(x)] for x,t in zip(ids, tags) if t == 'd']
    delv2_d_out = [delv2_d[tuple(x)] for x,t in zip(ids, tags) if t == 'd']

    delv_a_out = [delv_a[tuple(x)] for x,t in zip(ids, tags) if t == 'a']
    delv_d_out = [delv_d[tuple(x)] for x,t in zip(ids, tags) if t == 'd']

    delVOut = {
                'delV1_a' : delv1_a_out,
                'delV2_a' : delv2_a_out,
                'delV1_d' : delv1_d_out,
                'delV2_d' : delv2_d_out,
                'delVTot_a' : delv_a_out,
                'delVTot_d' : delv_d_out
    }

    ## Get feasible times
    t_a_out = [timesArr_a[tuple(x)] for x,t in zip(ids, tags) if t == 'a']
    t_d_out = [timesArr_d[tuple(x)] for x,t in zip(ids, tags) if t == 'd']
    tAll_out = t_a_out + t_d_out
    tOut = {
            'tPass_a' : t_a_out,
            'tPass_d' : t_d_out,
            'tAll'    : tAll_out
    }

    ## Get feasible orbits
    orb_a_out =  [passOrbits_a[x[0]][x[1]][x[2]] for x,t in zip(ids, tags) if t == 'a']#[passOrbits_a[tuple(x)] for x,t in zip(ids, tags) if t == 'a']
    orb_d_out =  [passOrbits_d[x[0]][x[1]][x[2]] for x,t in zip(ids, tags) if t == 'd']#[passOrbits_d[tuple(x)] for x,t in zip(ids, tags) if t == 'd']
    # orb_a_out_test = [passOrbits_a[x[0]][x[1]][x[2]] if t == 'a' else None for x,t in zip(ids, tags)]#[passOrbits_a[tuple(x)] for x,t in zip(ids, tags) if t == 'a']
    orb_a_ids = [idx for idx,t in zip(ids, tags) if t == 'a']
    orb_d_ids = [idx for idx,t in zip(ids, tags) if t == 'd']
    orbOut = {
            'orb_a' : orb_a_out,
            'orb_a_id' : orb_a_ids,
            'orb_d' : orb_d_out,
            'orb_d_id' : orb_d_ids,
            # 'test' : orb_a_out_test
    }

    if plot:
        plt.figure()
        ax = plt.gca()
        for planeIdx, plane in enumerate(t_a):
            for satIdx, sat in enumerate(plane):
                color = next(ax._get_lines.prop_cycler)['color']
                planeStr = str(planeIdx + 1)
                satStr = str(satIdx + 1)
                mask_a = flag_a[planeIdx][satIdx]
                mask_d = flag_d[planeIdx][satIdx]
                if pltLgd:
                    myLabel = f'Plane: {planeStr} | Sat: {satStr} '
                else:
                    myLabel = None
                plt.plot(timesArr_a[planeIdx][satIdx][mask_a]/60/60, delv_a[planeIdx][satIdx][mask_a], 'o', color=color,
                         label=myLabel)
                plt.plot(timesArr_d[planeIdx][satIdx][mask_d]/60/60,
                         delv_d[planeIdx][satIdx][mask_d], 'x', color=color)
        #         plt.plot(timesArr_a[planeIdx][satIdx]/60/60, delv_a[planeIdx][satIdx], 'o', color = color,
        #                  label = f'Plane: {planeStr} | Sat: {satStr} ')
        #         plt.plot(timesArr_d[planeIdx][satIdx]/60/60, delv_d[planeIdx][satIdx], 'x', color = color)
        plt.plot(timesEff/60/60, delVEff, 'r-', label='Pareto Front')
        plt.plot(0,0,'g^',label='Utopia Point')
        plt.xlabel('Time (Hours)', fontsize = 14)
        plt.ylabel('Delta V (m/s)', fontsize = 14)
        plt.title(
            'Propellant and Time cost to first pass \n \'x\' for descending passes | \'o\' for ascending passes',
            fontsize = 16)
        # ax.set_xlim(left=0)
        # ax.set_ylim(bottom=0)

        plt.legend(loc='upper right')
        if savePlot:
            plt.savefig(figName, facecolor="w", dpi=300, bbox_inches='tight')

    return idsOut, tags, delVOut, tOut, orbOut

#### Utility Functions ####
def getSatValues_const(constellation, attribute):
    """
    Gets a specific attribute from each satellite in a constellation (Ex: generated by constellationFuncs.genWalker)

    Inputs:
    Constellation (List): Constellation list where satellites are organized in the by planes then satellites. Ex indices: Walker[planeIdx][satIdx]
    Attribute (str): String indicating which attribute to extract

    Outputs:
    List of attributes, data structure of plane -> sat preserved
    """
    return [[getattr(sat,attribute) for sat in plane.sats] for plane in constellation.planes]
def getSatValues_const_func(constellation, attribute, func):
    """
    Gets a specific value from each satellite in a constellation, subject to a function
    Constellation can be generrated by constellationFuncs.genWalker

    Inputs:
    Constellation (List): Constellation list where satellites are organized in the by planes then satellites. Ex indices: Walker[planeIdx][satIdx]
    Attribute (str): String indicating which attribute to extract
    Func (func): Function to be applied

    Ex to get values 10 km higher than from semi-major axis 'a' in walker constellation:

    getSatValues_const_func(walker, lambda a : a + 10 * u.km, 'a')


    Outputs:
    List of values, data structure of plane -> sat preserved
    """
    return [[func(getattr(sat,attribute)) for sat in plane.sats] for plane in constellation.planes]

def getSatValues_constPlan(constellation, attribute):
    """
    Gets a specific attribute from each satellite in a constellation in the planner

    Inputs:
    Constellation (List): Constellation list where satellites are organized in the by planes then satellites. Ex indices: Walker[planeIdx][satIdx]
    Attribute (str): String indicating which attribute to extract

    Outputs:
    List of attributes, data structure of plane -> sat preserved
    """
    return [[[getattr(time,attribute) for time in sat] for sat in plane.sats] for plane in constellation.planes]
