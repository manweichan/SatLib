import numpy as np
import poliastro
from poliastro import constants
from astropy import units as u
####################### Earth Environment #######################


def atmDensity_below500km(f10, Ap, alt_meters):
    """
    Calculates atmospheric density
    Useful for satellites below 500 km
    From http://www.sws.bom.gov.au/Category/Educational/Space%20Weather/Space%20Weather%20Effects/SatelliteOrbitalDecayCalculations.pdf
    Inputs (f10, Ap, alt_meters)
    f10: Solar radio flux index (https://www.swpc.noaa.gov/phenomena/f107-cm-radio-emissions)
    Ap: Geomagnetic A index (https://www.spaceweatherlive.com/en/help/the-ap-index. 0-400 according to Aussie paper sws.bom.gov)
    alt_meters: altitude in meters
    """
    alt = alt_meters * 1e-3  # Convert to km
    SH = (900 + 2.5 * (f10 - 70) + 1.5 * Ap) / (27 - .012 * (alt - 200))
    density = 6e-10 * np.exp(-(alt - 175)/SH)
    return density


def dragForce(cd, rho, a, v):
    """
    Calculates drag force
    Inputs (cd, rho, a, v)
    cd:  Coefficient of drag
    rho: Density of fluid
    a:   Cross sectional area
    v:   Speed of object relative to fluid
    """
    dForce = cd * a * rho * v**2/2
    return dForce


def magField(r, m=7.98e15):
    """
    Calculates magnetic field strength at a certain radius from the center of the Earth (1st order)
    Inputs (r, m)
    r: Radius of earth center (dipole center) to spacecaraft (m)
    m: magnetic moment of the earth times magnetic constant (tesla * m^3)
    """

    B = 2*m/r**3  # Magnetic field
    return B

####################### Delta V calculations (Maneuvers) #######################

def delV_circInc(v, i):
    """
    Calculates deltaV required for a circular orbit inclination change
    Inputs:
    v: Orbital velocity
    i: Inclination change

    Outputs:
    delV: deltaV required to make maneuver with same units as input 'v'
    """
    delV = 2 * v * np.sin(i/2.)
    return delV


def circ2elip_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth.value):
    """
    Delta V required to go from circular orbit to elliptical
    circ2elip_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: semi major axis of circular orbit (m)
    a2: semi major axis of larger ellipse (m) 
    muPlanet: Gravitation parameter (Earth default)
    """
    v1 = np.sqrt(muPlanet/a1) * (np.sqrt(2*a2/(a1 + a2)) - 1)
    return np.abs(v1)


def elip2circ_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth.value):
    """
    Delta V required to go from elliptical orbit to circular orbit
    elip2circ_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: radii of departure circular orbit (m)
    a2: radii of arrival of final circular orbit (Assuming Hohmann) (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    v2 = np.sqrt(muPlanet/a2) * (1 - np.sqrt(2*a1/(a1 + a2)))
    return np.abs(v2)


def delV_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth.value):
    """
    Delta V required to conduct a Hohmann Transfer
    delV_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: radii of departure initial circular orbit (m)
    a2: radii of arrival of final circular orbit (Assuming Hohmann) (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    v1 = circ2elip_Hohmann(a1, a2)
    v2 = elip2circ_Hohmann(a1, a2)
    delV = v1 + v2
    return delV


def t_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth.value):
    """
    Time required to conduct a Hohmann Transfer
    t_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: semi major axis of circular orbit (m)
    a2: semi major axis of desired circular orbit (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    t = np.pi * np.sqrt((a1 + a2)**3 / (8 * muPlanet))
    return t


def delV_HohmannPlaneChange(a1, a2, theta, muPlanet=poliastro.constants.GM_earth.value):
    """
    Calculates the delta-V value to conduct a combined plane change and Hohmann
    delV_HohmannPlaneChange(a1, a2, theta)
    a1: semi major axis of circular orbit (m)
    a2: semi major axis of desired circular orbit (m)
    theta: degrees of inclination change (rad)
    muPlanet: Gravitation parameter (Earth default)
    """
    a = (a1 + a2) / 2  # semi major axis of transfer ellipse

    # Determine which orbit to conduct an inclination change at (larger radius)
    if a1 > a2:
        rIncChange = a1
        rInPlane = a2
        # Maneuver to go from transfer elllipse to circular orbit (no plane change)
        delV_NoPlaneChange = elip2circ_Hohmann(rIncChange, rInPlane)

    else:
        rIncChange = a2
        rInPlane = a1
        # Maneuver to go from circular orbit to elliptical transfer orbit (no plane change)
        delV_NoPlaneChange = circ2elip_Hohmann(rInPlane, rIncChange)
        # #Combined plane change and Delta V maneuver
        # vi = np.sqrt(muPlanet * (2/rIncChange - 1/a)) #velocity of transfer ellipse
        # vf = vi = circVel_fromRad(rIncChange) #final circular velocity

    # Combined plane change and Delta V maneuver
    v1 = circVel_fromRad(rIncChange)  # Initial velocity
    # velocity of transfer ellipse
    v2 = np.sqrt(muPlanet * (2/rIncChange - 1/a))
    delV_PlaneChange = np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(theta))
    delV_Total = delV_NoPlaneChange + delV_PlaneChange
    return delV_Total

#Plan is to make a graph (x: altitude, y: deltaV)
def highAltIncManeuver(rStart, rIncChange, rEnd, deli):
    """
    Calculates Delta V for a maneuver where you conduct a Hohmann transfer to a higher altitude
    in order to complete an inclination change that may save deltaV at a lower velocity. The maneuver
    ends with a hohmann transfer back down to the desired end orbit.
    
    Inputs
    rStart: Starting circular orbit radius (m)
    rIncChange: Radius that you make the inclination change (m)
    rEnd: Final radius orbit (m)
    deli: Inclination change (i)

    Outputs
    delVtot: Total delta V for all maneuvers (m/s)
    delVHoh1: Delta V of first Hohmann transfer (m/s)
    delVInc: Delta V for inclination change (m/s)
    delHoh2: DeltaV for second Hohmann transfer (m/s)
    """

    ## First Hohmann transfer
    delVHoh1 = delV_Hohmann(rStart, rIncChange)
    #Change altitude at new altitude
    vAtAlt = circVel_fromRad(rIncChange)
    delVInc = delV_circInc(vAtAlt, deli)
    ## Second Hohmann back to final orbit
    delHoh2 = delV_Hohmann(rIncChange, rEnd)
    delVtot = delVHoh1 + delVInc + delHoh2
    return delVtot, delVHoh1, delVInc, delHoh2

## Vallado coplanar phasing (6.6.1)
def t_phase_coplanar(a_tgt, theta, k_tgt, k_int, mu = poliastro.constants.GM_earth.value):
    """
    Get time to phase, deltaV and semi-major axis of phasing orbit in a coplanar phasing maneuver (same altitude)
    From Vallado section 6.6.1 Circular Coplanar Phasing. Algorithm 44

    Inputs:
    a_tgt (m) : semi-major axis of target orbit
    theta (rad): phase angle measured from target to the interceptor. Positive in direction of target motion.
    k_tgt : integer that corresponds to number of target satellite revolutions before rendezvous
    k_int : integer that corresponds to number of interceptor satellite revolutions before rendezvous
    mu (m^3 / s^2) : Gravitational constant of central body. Default is Earth

    Outputs:
    t_phase (s) : time to complete phasing maneuver
    deltaV (m/s) : Delta V required to complete maneuver
    delV1 (m/s) : Delta V of first burn (positive in direction of orbital velocity)
    delV2 (m/s) : Delta V of second burn which recircularies (positive in direction of orbital velocity)
    a_phase (m) : Semi-major axis of phasing orbit
    """
    w_tgt = np.sqrt(mu/a_tgt**3)
    t_phase = (2 * np.pi * k_tgt + theta) / w_tgt
    a_phase = (mu * (t_phase / (2 * np.pi * k_int))**2)**(1/3)
    deltaV = 2 * np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt)) #For both burns. One to go into ellipse, one to recircularize
    if theta >=0: #If theta > 0, interceptor in front of target, first burn is to increase semi-major axis to slow down (positive direction for burn in direction of velocity)
        delV1 = np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))
        delV2 = -np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))
    elif theta < 0: #If theta < 0, interceptor behind target, first burn decreases semi-major axis to catch up (negative indicates burn opposite direction of orbital velocity)
        delV1 = -np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))
        delV2 = np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))

    # IF NEEDED, output variables in a dictionary
    # outputDict = {
    #     "t_phase": t_phase,
    #     "deltaV" : deltaV,
    #     "delV1"  : delV1,
    #     "delV2"  : delV2,
    #     "a_phase": a_phase
    # }
    return t_phase, deltaV, delV1, delV2, a_phase

####################### Keplarian Orbital Mechanics #######################


def precRate_RAAN(a, e, w, i, J2=1.08262668e-3, rPlanet=constants.R_earth.to(u.m).value):
    """
    https://en.wikipedia.org/wiki/Nodal_precession
    Calculates right ascension nodal precession rate of a satellite around a body
    inputs(a, e, w, i, J2 = 1.08262668e-3, rPlanet = 6378.1e3)
    a: semi major axis (meters)
    e: eccentricity
    w: Mean motion/angular velocity of satellite motion (2 pi / T) where T is satellite period. 
    i: inclination (radians)
    J2: second dynamic form factor of body
    rPlanet: body's equitorial radius

    Ouputs:
    wp: precession rate 
    Note For a satellite in a prograde orbit, precession is westward, i.e. node and satellite
    move in opposite directions

    """
    wp = ((-3/2) * rPlanet**2 * J2 * w * np.cos(i)) / (a * (1 - e**2))**2
    return wp


def delPrecRate_RAAN(a, e, w, i, J2=1.08262668e-3, rPlanet=constants.R_earth.to(u.m).value):
    """
    Calculates differential right ascension nodal precession rate of a satellite around a body
    with respect to a change in inclination
    inputs(a, e, w, i, J2 = 1.08262668e-3, rPlanet = 6378.1e3)
    a: semi major axis (meters)
    e: eccentricity
    w: Mean motion/angular velocity of satellite motion (2 pi / T) where T is satellite period. 
    J2: second dynamic form factor of body
    rPlanet: body's equitorial radius

    Ouputs:
    wp: precession rate 
    Note For a satellite in a prograde orbit, precession is westward, i.e. node and satellite
    move in opposite directions

    """
    delWP = ((-3/2) * rPlanet**2 * J2 * w * np.sin(i)) / (a * (1 - e**2))**2
    return delWP


def delPrecRateDelA_Raan(a, e, w, i, J2=1.08262668e-3, rPlanet=constants.R_earth.to(u.m).value):
    """
    Calculates differential right ascension nodal precession rate of a satellite around a body
    with respect to a change in semi major axis
    inputs(a, e, w, i, J2 = 1.08262668e-3, rPlanet = 6378.1e3)
    a: semi major axis (meters)
    e: eccentricity
    w: Mean motion/angular velocity of satellite motion (2 pi / T) where T is satellite period. 
    J2: second dynamic form factor of body
    rPlanet: body's equitorial radius

    Ouputs:
    wp: precession rate
    """
    delWP_a = (3 * rPlanet ** 2 * J2 * w * np.cos(i)) / (a**3 * (1-e**2)**2)
    return delWP_a


def orbitalPeriod_fromAlt(alt, rPlanet=constants.R_earth.to(u.m).value, muPlanet=poliastro.constants.GM_earth.value):
    """
    Get orbital period from altitude input for circular orbit
    Outputs orbital Period in seconds
    inputs (alt, rPlanet = 6378.1e3, muPlanet = 3.986e14)
    alt: altitude in meters
    rPlanet: radius of planet. Earth is default no input needed unless it's another body
    muPlanet: gravitation parameter of planet. Earth is default no input needed unless it's another body
    """
    r = rPlanet + alt
    t = 2*np.pi*np.sqrt(r**3/muPlanet)
    return t


def circVel_fromAlt(alt, rPlanet=constants.R_earth.to(u.m).value, muPlanet=poliastro.constants.GM_earth.value):
    """
    Inputs (altitude, radius of planet-Earth is default, Gravitation parameter of planet-Earth is default)
    alt unit is in meters
    """
    r = alt + rPlanet
    v = np.sqrt(muPlanet/r)
    return v


def circVel_fromRad(r, muPlanet=poliastro.constants.GM_earth.value):
    """
    Inputs (orbit radius, Gravitation parameter of planet-Earth is default)
    alt unit is in meters
    """
    v = np.sqrt(muPlanet/r)
    return v


def orbitalPeriod_fromA(a, muPlanet=poliastro.constants.GM_earth.value):
    """
    Get orbital period from radius of orbit input
    Outputs orbital Period in seconds
    inputs (r, muPlanet = 3.986e14)
    a: semiMajor axis
    muPlanet: gravitation parameter of planet. Earth is default no input needed unless it's another body
    """
    t = 2*np.pi*np.sqrt(a**3/muPlanet)
    return t


def vis_viva(r, a, muPlanet=poliastro.constants.GM_earth.value):
    """
    Get velocity from vis-viva equation
    Outputs velocity of orbiting object
    inputs (r, a, muPlanet = 3.986e14)
    r: distance between bodies (m)
    a: semi major axis (m)
    muPlanet: gravitation parameter of planet. Earth is default no input needed unless it's another body
    """
    v = np.sqrt(muPlanet * (2/r - 1/a))
    return v

def getSSOE_fromAI(a, i, dRAAN = 1.991063853e-7, re = constants.R_earth.to(u.m).value, mu = constants.GM_earth.value, J2 = constants.J2_earth.value):
    """
    Gets the eccentricity of a Sun Synchronous Orbit (SSO) given altitude and inclination
    From Vallado (2013) eqn 11-16

    Inputs
    a : semi-major axis of orbit (m)
    i : inclination (radians)
    dRAAN: Desired rate of change of Right angle of the ascending node (RAAN). Default is Earth (rad/s)
    re : radius of the planet. Default is Earth (m)
    mu : Gravitation parameter. Default is Earth (m^3 / s^2)
    J2 : J2 gravitational term. Default is Earth

    Outputs
    e : eccentricity
    """
    num = -3 * re**2 * J2 * np.sqrt(mu) * np.cos(i)
    den = 2 * a**(7/2)  * dRAAN
    innerSqrt = np.sqrt(num/den)
    e = np.sqrt(1 - innerSqrt)
    return e

def getSSOI_fromAE(a, e, dRAAN = 1.991063853e-7, re = constants.R_earth.to(u.m).value, mu = constants.GM_earth.value, J2 = constants.J2_earth.value):
    """
    Gets the altitude of a Sun Synchronous Orbit (SSO) given altitude and eccentricity
    From Vallado (2013) eqn 11-15

    Inputs
    a : semi-major axis of orbit (m)
    e : eccentricity
    dRAAN: Desired rate of change of Right angle of the ascending node (RAAN). Default is Earth (rad/s)
    re : radius of the planet. Default is Earth (m)
    mu : Gravitation parameter. Default is Earth (m^3 / s^2)
    J2 : J2 gravitational term. Default is Earth

    Outputs
    i : inclination (radians)
    """
    num = -2 * a**(7/2)  * dRAAN * (1 - e**2)**2
    den = 3 * re**2 * J2 * np.sqrt(mu)
    # breakpoint()
    cosi = num / den
    i = np.arccos(cosi)
    return i

def getSSOA_fromEI(e, i, dRAAN = 1.991063853e-7, re = constants.R_earth.to(u.m).value, mu = constants.GM_earth.value, J2 = constants.J2_earth.value):
    """
    Gets the altitude of a Sun Synchronous Orbit (SSO) given eccentricity and inclination
    From Vallado (2013) eqn 11-14

    Inputs
    e : eccentricity
    i : inclination (radians)
    dRAAN: Desired rate of change of Right angle of the ascending node (RAAN). Default is Earth (rad/s)
    re : radius of the planet. Default is Earth (m)
    mu : Gravitation parameter. Default is Earth (m^3 / s^2)
    J2 : J2 gravitational term. Default is Earth

    Outputs
    a : semi-major axis of orbit (m)
    """
    num = -3 * re**2 * J2 * np.sqrt(mu) * np.cos(i)
    den = 2 * dRAAN * (1 - e**2) ** 2
    a72 = num / den
    a = a72 ** (2/7)
    return a

def getRGTOrbit(k_r, k_d, e, i, detectThresh=10, maxIters=100):
    """
    Gets Repeat Ground Track (RGT) orbit semi-major axis for a specified eccentricity, inclination,
    and repeat track parameters.

    Inputs
    k_r (integer): Number of revolutions until repeat ground track
    k_d (integer): Number of days to repeat ground track
    e : eccentricity
    i (radians) : inclination
    detectThresh (m) : Difference between current and previous semi-major axis guess that will stop search loop
    maxIters (integer) : Maximum number of iterations before loop ends if no solution is found
    
    Outputs
    a_new (m) : semi-major axis of RGT
    alt (m) : altitude of RGT
    """

    # Constants
    periodEarth = poliastro.constants.rotational_period_earth
    w_earth = 2 * np.pi / periodEarth.to(u.s).value
    mu = poliastro.constants.GM_earth.value
    j2 = poliastro.constants.J2_earth.value
    r_earth = poliastro.constants.R_earth.to(u.m).value
    
    k_revPerDay = k_r / k_d
    n = k_revPerDay * w_earth
    a_new = (mu * (1/n0)**2)**(1/3)
    delLam = 2 * np.pi * k_d / k_r #Change in successive node arrivals (radians)
    for idx in range(maxIters):
        a = a_new
        p = a * (1 - e**2)
        OmegaDot = - 3 * n * j2 * np.cos(i) / 2 * (r_earth / p)**2 #Change in RAAN due to perturbations
        delLamPeriod = 2 * np.pi * OmegaDot / n #Change in lambda by period
        delLon = delLam + delLamPeriod
        n = 2 * np.pi * w_earth / delLon
        a_new = (mu * (1/n)**2)**(1/3)
        diff = np.abs(a - a_new)
        if diff < detectThresh:
            break
    alt = a_new - r_earth #altitude
    return a_new, alt

####################### Eclipse Calculation #######################


def worstCaseEclipse(alt, rPlanet=constants.R_earth.to(u.m).value, muPlanet=poliastro.constants.GM_earth.value):
    """alt unit is in meters"""
    rOrbit = alt + rPlanet  # Radius of orbit (m)
    alph = np.arccos(rPlanet/rOrbit)  # Outer central angle
    beta = np.pi - 2*alph  # Central angle of eclipsed orbit

    fracOrbitEcl = beta/(2*np.pi)  # Fraction of orbit in eclipse
    period = 2*np.pi*np.sqrt(rOrbit**3/muPlanet)
    timeInEclipse = period * fracOrbitEcl  # If period is input, get time in eclipse

    print("Central Angle of Eclipsed Orbit : ", beta*180/np.pi, "deg")
    print("Fraction of Orbit in Eclipse    : ", fracOrbitEcl)
    print("Period                          :  {:.0f} s, {:.1f} min, {:.1f} hrs".format(
        period, period/60, period/60/60))
    print("Time in Eclipse                 :  {:.0f} s, {:.1f} min, {:.1f} hrs".format(
        timeInEclipse, timeInEclipse/60, timeInEclipse/60/60))
