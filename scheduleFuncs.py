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
    for idx, val in enumerate(nus1):
        "Cycle through each intersect points (should be 2)"
        # Extract true anomaly for each orbit of the corresponding intersection point
        intNu1 = nus1[idx] #True anomaly of intercept for orbit 1
        intNu2 = nus2[idx]

        # Propagate orb1 to true anomaly
        t2int_orb1 = orb1.time_to_anomaly(intNu1)
        orb1_init2rvd = orb1.propagate(t2int_orb1)
        
        tAtInitMan = orb1.epoch + t2int_orb1 # Keep track off time. Time at first maneuver.

        #Propagate target orbit by same amount
        orb2_init2rvd = orb2.propagate(t2int_orb1)
    
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
        tPhase, delVPhase, delV1, delV2, aPhase = t_phase_coplanar(orb2_init2rvd.a.to(u.m).value, 
                                      phaseAng_tar2rvd.to(u.rad).value, numTarOrbs,
                                                                   numIntOrbs)
        tPhaseS = tPhase * u.s #convert to seconds (astropy)
        delV1_ms = delV1 * u.m / u.s #convert to m/s (astropy)
        delV2_ms = delV1 * u.m / u.s
        orb2_rvd = orb2_init2rvd.propagate(tPhaseS) #target orbit at rendezvous
        
        #Keep track of time for 2nd maneuver
        tAtFinMan = tAtInitMan + tPhaseS
        
        v_dir1 = orb1_init2rvd.v/norm(orb1_init2rvd.v) #Velocity direction of interceptor orbit at node
        delVVec1 = v_dir1 * delV1_ms #DeltaV vector GCRS frame
        deltaV_man1 = Maneuver.impulse(delVVec1)
        orb1_phase_i = orb1_init2rvd.apply_maneuver(deltaV_man1) #Applied first burn
        orb1_phase_f = orb1_phase_i.propagate(tPhaseS) #Propagate through phasing orbit

        v_dir2 = orb1_phase_f.v/norm(orb1_phase_f.v) #Get direction of velocity at end of phasing
        delVVec2 = v_dir2 * delV2_ms
        deltaV_man2 = Maneuver.impulse(v_dir2 * delV2_ms) #Delta V  of recircularizing orbit
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

