import numpy as np
import poliastro
from poliastro import constants
import astropy
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

def a_ecc_hohmann(a1, a2):
    """
    Find semi-major axis and eccentricity of the Hohmann Transfer Orbit

    Parameters
    ----------
    a1: ~astropy.unit.Quantity
        semi major axis of circular orbit (m)
    a2: ~astropy.unit.Quantity
        semi major axis of final orbit (m) 

    Returns
    -------
    ecc: 
        Eccentricity of the Hohmann transfer orbit
    muPlanet: Gravitation parameter (Earth default)
    """
    if not isinstance(a1, astropy.units.quantity.Quantity):
       a1 = a1 * u.m
    if not isinstance(a2, astropy.units.quantity.Quantity):
       a2 = a2 * u.m

    a = (a1 + a2)/2

    #Find periapsis and apoapsis
    if a1 > a2:
        apoapsis = a1
        periapsis = a2 
    elif a1 < a2:
        apoapsis = a2 
        periapsis = a1
    else:
        print("Hohmann orbits must be different")
        return

    eccNum = apoapsis - periapsis
    eccDen = apoapsis + periapsis
    ecc = eccNum / eccDen
    return a, ecc

def circ2elip_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth):
    """
    Delta V required to go from circular orbit to elliptical
    circ2elip_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: semi major axis of circular orbit (m)
    a2: semi major axis of final orbit (m) 
    muPlanet: Gravitation parameter (Earth default)
    """
    if not isinstance(a1, astropy.units.quantity.Quantity):
       a1 = a1 * u.m
    if not isinstance(a2, astropy.units.quantity.Quantity):
       a2 = a2 * u.m
    v1 = np.sqrt(muPlanet/a1) * (np.sqrt(2*a2/(a1 + a2)) - 1)
    return np.abs(v1)


def elip2circ_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth):
    """
    Delta V required to go from elliptical orbit to circular orbit
    elip2circ_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: radii of departure circular orbit (m)
    a2: radii of arrival of final circular orbit (Assuming Hohmann) (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    if not isinstance(a1, astropy.units.quantity.Quantity):
       a1 = a1 * u.m
    if not isinstance(a2, astropy.units.quantity.Quantity):
       a2 = a2 * u.m
    v2 = np.sqrt(muPlanet/a2) * (1 - np.sqrt(2*a1/(a1 + a2)))
    return np.abs(v2)


def delV_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth):
    """
    Delta V required to conduct a Hohmann Transfer
    delV_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: radii of departure initial circular orbit (m)
    a2: radii of arrival of final circular orbit (Assuming Hohmann) (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    if not isinstance(a1, astropy.units.quantity.Quantity):
       a1 = a1 * u.m
    if not isinstance(a2, astropy.units.quantity.Quantity):
       a2 = a2 * u.m
    v1 = circ2elip_Hohmann(a1, a2)
    v2 = elip2circ_Hohmann(a1, a2)
    delV = v1 + v2
    return delV


def t_Hohmann(a1, a2, muPlanet=poliastro.constants.GM_earth):
    """
    Time required to conduct a Hohmann Transfer
    t_Hohmann(a1, a2, muPlanet = 3.986e14)
    a1: semi major axis of circular orbit (m)
    a2: semi major axis of desired circular orbit (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    if not isinstance(a1, astropy.units.quantity.Quantity):
       a1 = a1 * u.m
    if not isinstance(a2, astropy.units.quantity.Quantity):
       a2 = a2 * u.m
    t = np.pi * np.sqrt((a1 + a2)**3 / (8 * muPlanet))
    return t


def delV_HohmannPlaneChange(a1, a2, theta, muPlanet=poliastro.constants.GM_earth):
    """
    Calculates the delta-V value to conduct a combined plane change and Hohmann
    delV_HohmannPlaneChange(a1, a2, theta)
    a1: semi major axis of circular orbit (m)
    a2: semi major axis of desired circular orbit (m)
    theta: degrees of inclination change (rad)
    muPlanet: Gravitation parameter (Earth default)
    """

    if not isinstance(a1, astropy.units.quantity.Quantity):
       a1 = a1 * u.m
    if not isinstance(a2, astropy.units.quantity.Quantity):
       a2 = a2 * u.m
    if not isinstance(theta, astropy.units.quantity.Quantity):
       theta = theta * u.rad
       
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
def t_phase_coplanar(a_tgt, theta, k_tgt, k_int, mu = poliastro.constants.GM_earth):
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
    passFlag (bool) : True if orbit is valid i.e. rp > rPlanet so satellite won't crash into Earth
    """
    if not isinstance(a_tgt, u.quantity.Quantity):
        a_tgt = a_tgt * u.m
    if not isinstance(theta, u.quantity.Quantity):
        theta = theta * u.rad

    ## Check to see if dimensions match
    if theta.ndim != a_tgt.ndim:
        a_tgt = np.repeat(a_tgt[:, :, np.newaxis], np.shape(theta)[-1], axis=2)

    w_tgt = np.sqrt(mu/a_tgt**3)
    t_phase = (2 * np.pi * k_tgt + theta.to(u.rad).value) / w_tgt
    a_phase = (mu * (t_phase / (2 * np.pi * k_int))**2)**(1/3)


    rp = 2 * a_phase - a_tgt #radius of perigee
    passFlag = rp > poliastro.constants.R_earth #Check which orbits are non-feasible due to Earth's radius
    deltaV = 2 * np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt)) #For both burns. One to go into ellipse, one to recircularize

    #Get individual burn values
    delV1Raw = deltaV / 2

    #If theta > 0, interceptor in front of target, first burn is to increase semi-major axis to slow down (positive direction for burn in direction of velocity)
    thetaMask = theta >= 0 
    thetaMaskPosNeg = thetaMask * 2 - 1 #Create mask to determine first burn is in direction of velocity if theta > 0, negative if not

    delV1 = delV1Raw * thetaMaskPosNeg #Apply mask to delV1Raw
    delV2 = -delV1 # Second burn is negative of first burn

    # ## Save this code in case the above didn't work (this works for individual runs but not with np arrays)
    # if theta >=0: #If theta > 0, interceptor in front of target, first burn is to increase semi-major axis to slow down (positive direction for burn in direction of velocity)
    #     delV1 = np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))
    #     delV2 = -np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))
    # elif theta < 0: #If theta < 0, interceptor behind target, first burn decreases semi-major axis to catch up (negative indicates burn opposite direction of orbital velocity)
    #     delV1 = -np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))
    #     delV2 = np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt))

    # IF NEEDED, output variables in a dictionary
    # outputDict = {
    #     "t_phase": t_phase,
    #     "deltaV" : deltaV,
    #     "delV1"  : delV1,
    #     "delV2"  : delV2,
    #     "a_phase": a_phase
    # }
    return t_phase.to(u.s), deltaV.to(u.m/u.s), delV1.to(u.m/u.s), delV2.to(u.m/u.s), a_phase.to(u.km), passFlag


def aPhase_fromFixedTime(t_phase, alt, mu = poliastro.constants.GM_earth, 
                         rPlanet = poliastro.constants.R_earth):
    """
    Get semi-major axis of phasing orbit given time to phase and
    target orbit altitude. 
    THIS FUNCITION ASSUMES both satellites start at the same altitude
    
    Inputs
    t_phase (s): time to complete phasing maneuver
    alt (m): altitude of target orbit, should be same as interceptor orbit
    mu (m^3 / s^2): gravitational parameter (default is Earth)
    rPlanet (m): radius of planet (default is Earth)
    """
    if not isinstance(t_phase, astropy.units.quantity.Quantity):
        t_phase = t_phase * u.s
    if not isinstance(alt, astropy.units.quantity.Quantity):
        alt = alt * u.m
    if not isinstance(mu, astropy.units.quantity.Quantity):
        mu = mu * u.m * u.m * u.m / (u.s * u.s)
    if not isinstance(rPlanet, astropy.units.quantity.Quantity):
        rPlanet = rPlanet * u.m
    
    t_tgt = orbitalPeriod_fromAlt(alt)
    kint = np.floor((t_phase / t_tgt).to(u.one)) #Scale intercept orbits to match target orbit period

    toSquare = t_phase / (kint * 2 * np.pi)
    to13 = mu * toSquare**2 #term to the 1/3
    a_phase = (to13)**(1/3)
    
    a_tgt = alt + rPlanet
    deltaV = 2 * np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt)) #For both burns. One to go into ellipse, one to recircularize
    return a_phase.to(u.km), deltaV.to(u.m / u.s), kint

def coplanar_phase_different_orbs(v_i, a_int, a_tgt, mu=constants.GM_earth):
    """
    This function returns the maneuvers necessary to phase an intercept
    between two satellites in different circular orbits.
    
    Based on Vallado 4th ed Algorithm 45. Needed to add some checks that
    weren't explicitly stated in the book. For example, sign convention
    is confusing and had to make sure they are right in this algorithm.
    
    Parameters
    ----------
    v_i: ~astropy.unit.Quantity
        Initial phase difference between chaser and target
    a_int: ~astropy.unit.Quantity
        semi-major axis of chaser (intercepter)
    a_tgt: ~astropy.unit.Quantity
        semi-major axis of target 
    mu: ~astropy.unit.Quantity
        Gravitation parameter of body
        
    Returns
    ---------
    t_trans: ~astropy.unit.Quantity
        Hohmann transfer time
    delVTot: ~astropy.unit.Quantity
        Delta V of Hohmann Transfer
    a_trans: ~astropy.unit.Quantity
        semi-major axis of transfer orbit
    t_wait: ~astropy.unit.Quantity
    
    """
    assert isinstance(v_i, astropy.units.quantity.Quantity), ('v_i' 
                                         ' must be astropy.units.quantity.Quantity')
    assert isinstance(a_int, astropy.units.quantity.Quantity), ('a_int' 
                                         ' must be astropy.units.quantity.Quantity')
    assert isinstance(a_tgt, astropy.units.quantity.Quantity), ('a_tgt' 
                                         ' must be astropy.units.quantity.Quantity')
    assert isinstance(mu, astropy.units.quantity.Quantity), ('mu' 
                                         ' must be astropy.units.quantity.Quantity')
    
    pi = np.pi
    
    w_tgt = np.sqrt(mu / a_tgt**3)
    w_int = np.sqrt(mu / a_int**3)
    
    a_trans = (a_int + a_tgt)/2
    
    t_trans = pi * np.sqrt(a_trans**3 / mu)
    
    alpha = w_tgt * t_trans #lead angle
    alpha = alpha % (2*pi) #modulo to ensure angle from 0 to 2pi
    
    v = alpha - pi

    if w_tgt < w_int:
        vDiff = v - v_i.to(u.rad).value
    elif w_tgt > w_int:
        vDiff = v_i.to(u.rad).value - v
    else:
        vDiff = 0
        print("In the same orbit")
    k = 0
    
    t_wait = (vDiff + 2 * pi * k) / (abs(w_int - w_tgt))
    while t_wait < 0:
        k += 1
        t_wait = (vDiff + 2 * pi * k) / (abs(w_int - w_tgt))
    
    delV1 = circ2elip_Hohmann(a_int, a_tgt)
    delV2 = elip2circ_Hohmann(a_int, a_tgt)

    delVTot =  delV1 + delV2
    return t_trans.to(u.s), delVTot.to(u.km/u.s), a_trans.to(u.km), t_wait.to(u.min)
    
####################### Keplarian Orbital Mechanics #######################


# def precRate_RAAN(a, e, w, i, J2=constants.J2_earth, rPlanet=constants.R_earth.to(u.m).value):
#     """
#     https://en.wikipedia.org/wiki/Nodal_precession
#     Calculates right ascension nodal precession rate of a satellite around a body
#     inputs(a, e, w, i, J2 = 1.08262668e-3, rPlanet = 6378.1e3)
#     a: semi major axis (meters)
#     e: eccentricity
#     w: Mean motion/angular velocity of satellite motion (2 pi / T) where T is satellite period. 
#     i: inclination (radians)
#     J2: second dynamic form factor of body
#     rPlanet: body's equitorial radius

#     Ouputs:
#     wp: precession rate 
#     Note For a satellite in a prograde orbit, precession is westward, i.e. node and satellite
#     move in opposite directions

#     """
#     wp = ((-3/2) * rPlanet**2 * J2 * w * np.cos(i)) / (a * (1 - e**2))**2
#     return wp

def precRate_RAAN(a, e, i,J2=constants.J2_earth, rPlanet=constants.R_earth.to(u.m), muPlanet=poliastro.constants.GM_earth):
    """
    https://en.wikipedia.org/wiki/Nodal_precession
    Calculates right ascension nodal precession rate of a satellite around a body given J2 perturbation

    Parameters
    ----------
    a: ~astropy.unit.Quantity
        semi-major axis of the orbit
    e: float
        eccentricity of the orbit
    i: ~astropy.unit.Quantity
        inclination of the orbit
    J2: float
        second dynamic form factor of body. Default poliastro constant for earth
    rPlanet: ~astropy.unit.Quantity
        body's equatorial radius. Default poliastro constant for earth
    muPlanet: ~astropy.unit.Quantity
        body's gravitational constant. Default poliastro constant for earth

    Returns
    -------
    dRaan: astropy.unit.Quantity
        Precession rate of RAAN in rad/s

    Note For a satellite in a prograde orbit, precession is westward, i.e. node and satellite
    move in opposite directions

    """
    n = np.sqrt(muPlanet/a**3)
    p = (a * (1 - e**2))
    dRaan = -(3 * n * rPlanet**2 * J2 * np.cos(i)) / (2 * p**2)
    return dRaan

def delPrecRate_RAAN(a, e, w, i, J2=constants.J2_earth, rPlanet=constants.R_earth.to(u.m).value):
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


def delPrecRateDelA_Raan(a, e, w, i, J2=constants.J2_earth, rPlanet=constants.R_earth.to(u.m).value):
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


def precRate_omega(a, e, i, J2=constants.J2_earth, rPlanet=constants.R_earth.to(u.m), muPlanet=poliastro.constants.GM_earth):
    """
    Calculates the precession of the argument of perigee (omega) due to J2

    Parameters
    ----------
    a: ~astropy.unit.Quantity
        semi-major axis of the orbit
    e: float
        eccentricity of the orbit
    i: ~astropy.unit.Quantity
        inclination of the orbit
    J2: float
        second dynamic form factor of body. Default poliastro constant for earth
    rPlanet: ~astropy.unit.Quantity
        body's equatorial radius. Default poliastro constant for earth
    muPlanet: ~astropy.unit.Quantity
        body's gravitational constant. Default poliastro constant for earth

    Returns
    -------
    delOmega: ~astropy.unit.Quantity
        precession rate of the argument of perigee in rad/s
    """
    n = np.sqrt(muPlanet/a**3)
    p = (a * (1 - e**2))
    delOmega = -(3 * n * rPlanet**2 * J2 * np.cos(i)) / (2 * p**2)
    return delOmega

def precRate_anom(a, e, i, J2=constants.J2_earth, rPlanet=constants.R_earth.to(u.m), muPlanet=poliastro.constants.GM_earth):
    """
    Calculates the precession of the mean anomaly due to J2 perturbations

    Parameters
    ----------
    a: ~astropy.unit.Quantity
        semi-major axis of the orbit
    e: float
        eccentricity of the orbit
    i: ~astropy.unit.Quantity
        inclination of the orbit
    J2: float
        second dynamic form factor of body. Default poliastro constant for earth
    rPlanet: ~astropy.unit.Quantity
        body's equatorial radius. Default poliastro constant for earth
    muPlanet: ~astropy.unit.Quantity
        body's gravitational constant. Default poliastro constant for earth

    Returns
    -------
    delAnom: ~astropy.unit.Quantity
        precession rate of the mean anomaly in rad/s
    """
    n = np.sqrt(muPlanet/a**3)
    p = (a * (1 - e**2))
    delAnom = -(3 * n * rPlanet**2 * J2 * np.sqrt(1 - e**2) * (3 * np.sin(i)**2 - 2)) / (4 * p**2)
    return delAnom

def orbitalPeriod_fromAlt(alt, rPlanet=constants.R_earth, muPlanet=poliastro.constants.GM_earth):
    """
    Get orbital period from altitude input for circular orbit
    Outputs orbital Period in seconds
    Inputs (alt, rPlanet = 6378.1e3, muPlanet = 3.986e14)
    alt (m): altitude. If not an astropy unit, defaults to m
    rPlanet (m): radius of planet. Earth is default no input needed unless it's another body
    muPlanet (m^3 / s^2): gravitation parameter of planet. Earth is default no input needed unless it's another body

    Outputs
    t : Orbital period
    """
    if not isinstance(alt, u.quantity.Quantity):
        alt = alt * u.m
    r = rPlanet + alt
    t = 2*np.pi*np.sqrt(r**3/muPlanet)
    return t


def circVel_fromAlt(alt, rPlanet=constants.R_earth.to(u.m).value, muPlanet=poliastro.constants.GM_earth):
    """
    Inputs (altitude, radius of planet-Earth is default, Gravitation parameter of planet-Earth is default)
    alt unit is in meters
    """
    if not isinstance(alt, u.quantity.Quantity):
        alt = alt * u.m
    r = alt + rPlanet
    v = np.sqrt(muPlanet/r)
    return v


def circVel_fromRad(r, muPlanet=poliastro.constants.GM_earth):
    """
    Inputs (orbit radius, Gravitation parameter of planet-Earth is default)
    alt unit is in meters
    """
    if not isinstance(r, u.quantity.Quantity):
        r = r * u.m
    v = np.sqrt(muPlanet/r)
    return v


def orbitalPeriod_fromA(a, muPlanet=poliastro.constants.GM_earth):
    """
    Get orbital period from radius of orbit input
    Outputs orbital Period in seconds
    inputs (r, muPlanet = 3.986e14)
    a (m): semiMajor axis. if no astropy units are assigned, will default to meters
    muPlanet: gravitation parameter of planet. Earth is default no input needed unless it's another body

    Output
    t : Orbital period
    """
    if not isinstance(a, u.quantity.Quantity):
        a = a * u.m
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

    if not isinstance(i, astropy.units.quantity.Quantity):
        i_astro = i * u.rad
        i_clean = i_astro.to(u.rad).value
    else:
        i_clean = i

    # Constants
    periodEarth = poliastro.constants.rotational_period_earth
    w_earth = 2 * np.pi / periodEarth.to(u.s).value
    mu = poliastro.constants.GM_earth.value
    j2 = poliastro.constants.J2_earth.value
    r_earth = poliastro.constants.R_earth.to(u.m).value
    
    k_revPerDay = k_r / k_d
    n = k_revPerDay * w_earth
    a_new = (mu * (1/n)**2)**(1/3)
    delLam = 2 * np.pi * k_d / k_r #Change in successive node arrivals (radians)
    for idx in range(maxIters):
        a = a_new
        p = a * (1 - e**2)
        OmegaDot = - 3 * n * j2 * np.cos(i_clean) / 2 * (r_earth / p)**2 #Change in RAAN due to perturbations
        delLamPeriod = 2 * np.pi * OmegaDot / n #Change in lambda by period
        delLon = delLam + delLamPeriod
        n = 2 * np.pi * w_earth / delLon
        a_new = (mu * (1/n)**2)**(1/3)
        diff = np.abs(a - a_new)
        if diff < detectThresh:
            break
    alt = a_new - r_earth #altitude

    alt_out = alt * u.m
    a_new_out = a_new * u.m
    return a_new_out, alt_out

# Nodal period of satellite
def get_nodal_period(a, i, J2=constants.J2_earth, Re=constants.R_earth, mu=constants.GM_earth):
    """
    Function from equation 2 of Lee, Eun 
    (Ground Track Acquisition and Maintenance Maneuver Modeling)
    
    Also Vallado 4th ed 11-24
    
    Parameters
    ----------
    a: ~astropy.unit.Quantity
        Semi-major axis of orbit
    i: ~astropy.unit.Quantity
        Inclination of orbit
        
    Returns
    -------
    Pn: ~astropy.unit.Quantity
        Nodal period of satellite
    """
    
    n = np.sqrt(mu / a**3)
    
    term1 = 2 * np.pi / n
    
    term2 = 1 - 3 / 2 * J2 * (Re / a)**2 * (4 * np.cos(i)**2 - 1)
    
    Pn = term1 * term2
    return Pn

def nodal_period_displacement(pn0, aRGT, eRGT, iRGT, aSat):
    """
    Gets ground track displacement per nodal period
    From Aorpimai (Repeat-Groundtrack Orbit Acquisition and Maintenance 
    for Earth-Observation Satellites) eqn 20
    
    Parameters
    ----------
    pn0: ~astropy.unit.Quantity
        nodal period of desired repeat-groundtrack orbit
    aRGT: ~astropy.unit.Quantity
        semi-major axis of rgt orbit
    eRGT: ~astropy.unit.Quantity
        eccentriciy of rgt orbit
    iRGT: ~astropy.unit.Quantity
        inclination axis of rgt orbit
    aSat: ~astropy.unit.Quantity
        semi-major axis of satellite orbit (trying to attain RGT orbit)
        
    Returns
    -------
    deltaL: ~astropy.unit.Quantity
        ground track displacement per nodal period (radians)
    """
    
    # Get earth rotation rate
    siderealDaySec = 86164.0905 * u.s #Sidereal day in seconds
    we = (2*np.pi)/siderealDaySec #Rotation rate of the earth
    
    deltaA = abs(aSat - aRGT)
    
    omegaDot = precRate_RAAN(aRGT, eRGT, iRGT)
    
    deltaL = 3/2 * pn0 * (we - omegaDot) * deltaA / aRGT
    
    return deltaL

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
