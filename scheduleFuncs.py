from orbitalMechanics import *

import numpy as np
from numpy.linalg import norm
from poliastro import constants
from poliastro.earth import Orbit
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
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