import numpy as np

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
    alt = alt_meters * 1e-3 #Convert to km
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

    B = 2*m/r**3 #Magnetic field
    return B

####################### Delta V calculations #######################

#####Inlination change#####
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

def circ2elip_Hohmann(r1, r2, muPlanet = 3.986e14):
    """
    Delta V required to go from circular orbit to elliptical
    circ2elip_Hohmann(r1, r2, muPlanet = 3.986e14)
    r1: semi major axis of circular orbit (m)
    r2: semi major axis of larger ellipse (m) 
    muPlanet: Gravitation parameter (Earth default)
    """
    v1 = np.sqrt(muPlanet/r1) * (np.sqrt(2*r2/(r1 + r2)) - 1)
    return np.abs(v1)

def elip2circ_Hohmann(r1, r2, muPlanet = 3.986e14):
    """
    Delta V required to go from elliptical orbit to circular orbit
    elip2circ_Hohmann(r1, r2, muPlanet = 3.986e14)
    r1: semi major axis of circular orbit (m)
    r2: semi major axis of larger ellipse (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    v2 = np.sqrt(muPlanet/r2) * (1 - np.sqrt(2*r1/(r1 + r2)))
    return np.abs(v2)

def delV_Hohmann(r1, r2, muPlanet = 3.986e14):
    """
    Delta V required to conduct a Hohmann Transfer
    delV_Hohmann(r1, r2, muPlanet = 3.986e14)
    r1: semi major axis of initial circular orbit (m)
    r2: semi major axis of desired circular orbit (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    v1 = circ2elip_Hohmann(r1, r2)
    v2 = elip2circ_Hohmann(r1, r2)
    delV = v1 + v2
    return delV

def t_Hohmann(r1, r2, muPlanet = 3.986e14):
    """
    Time required to conduct a Hohmann Transfer
    t_Hohmann(r1, r2, muPlanet = 3.986e14)
    r1: semi major axis of circular orbit (m)
    r2: semi major axis of desired circular orbit (m)
    muPlanet: Gravitation parameter (Earth default)
    """
    t = np.pi * np.sqrt((r1 + r2)**3 / (8 * muPlanet))
    return t

def delV_HohmannPlaneChange(r1, r2, theta, muPlanet = 3.986e14):
    """
    Calculates the delta-V value to conduct a combined plane change and Hohmann
    delV_HohmannPlaneChange(r1, r2, theta)
    r1: semi major axis of circular orbit (m)
    r2: semi major axis of desired circular orbit (m)
    theta: degrees of inclination change (rad)
    muPlanet: Gravitation parameter (Earth default)
    """
    a = (r1 + r2) / 2 #semi major axis of transfer ellipse

    if r1 > r2: #Determine which orbit to conduct an inclination change at (larger radius)
        rIncChange = r1
        rInPlane = r2
        #Maneuver to go from transfer elllipse to circular orbit (no plane change)
        delV_NoPlaneChange = elip2circ_Hohmann(rIncChange, rInPlane)

    else:
        rIncChange = r2
        rInPlane = r1
        #Maneuver to go from circular orbit to elliptical transfer orbit (no plane change)
        delV_NoPlaneChange = circ2elip_Hohmann(rInPlane, rIncChange)
        # #Combined plane change and Delta V maneuver
        # vi = np.sqrt(muPlanet * (2/rIncChange - 1/a)) #velocity of transfer ellipse
        # vf = vi = circVel_fromRad(rIncChange) #final circular velocity

    #Combined plane change and Delta V maneuver
    v1 = circVel_fromRad(rIncChange) #Initial velocity
    v2 = np.sqrt(muPlanet * (2/rIncChange - 1/a)) #velocity of transfer ellipse
    delV_PlaneChange = np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(theta))
    delV_Total = delV_NoPlaneChange + delV_PlaneChange
    return delV_Total

####################### Keplarian Orbital Mechanics #######################

def precRate_RAAN(a, e, w, i, J2 = 1.08262668e-3, rPlanet = 6378.1e3):
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
    wp = ( (-3/2) * rPlanet**2 * J2 * w * np.cos(i) ) / (a * (1 - e**2))**2
    return wp

def delPrecRate_RAAN(a, e, w, i, J2 = 1.08262668e-3, rPlanet = 6378.1e3):
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
    delWP = ( (-3/2) * rPlanet**2 * J2 * w * np.sin(i) ) / (a * (1 - e**2))**2
    return delWP

def delPrecRateDelA_Raan(a, e, w, i, J2 = 1.08262668e-3, rPlanet = 6378.1e3):
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

def orbitalPeriod_fromAlt(alt, rPlanet = 6378.1e3, muPlanet = 3.986e14):
    """
    Get orbital period from altitude input
    Outputs orbital Period in seconds
    inputs (alt, rPlanet = 6378.1e3, muPlanet = 3.986e14)
    alt: altitude in meters
    rPlanet: radius of planet. Earth is default no input needed unless it's another body
    muPlanet: gravitation parameter of planet. Earth is default no input needed unless it's another body
    """
    r = rPlanet + alt
    t = 2*np.pi*np.sqrt(r**3/muPlanet)
    return t

def circVel_fromAlt(alt, rPlanet = 6378.1e3, muPlanet = 3.986e14):
    """
    Inputs (altitude, radius of planet-Earth is default, Gravitation parameter of planet-Earth is default)
    alt unit is in meters
    """
    r = alt + rPlanet
    v = np.sqrt(muPlanet/r)
    return v

def circVel_fromRad(r, muPlanet = 3.986e14):
    """
    Inputs (orbit radius, Gravitation parameter of planet-Earth is default)
    alt unit is in meters
    """
    v = np.sqrt(muPlanet/r)
    return v

def orbitalPeriod_fromRad(r, muPlanet = 3.986e14):
    """
    Get orbital period from radius of orbit input
    Outputs orbital Period in seconds
    inputs (r, muPlanet = 3.986e14)
    alt: altitude in meters
    muPlanet: gravitation parameter of planet. Earth is default no input needed unless it's another body
    """
    t = 2*np.pi*np.sqrt(r**3/muPlanet)
    return t

def vis_viva(r, a, muPlanet = 3.986e14):
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

####################### Eclipse Calculation #######################

def worstCaseEclipse(alt, rPlanet = 6378.1e3, muPlanet = 3.986e14):
    """alt unit is in meters"""
    rOrbit = alt + rPlanet #Radius of orbit (m)
    alph = np.arccos(rPlanet/rOrbit) #Outer central angle
    beta = np.pi - 2*alph #Central angle of eclipsed orbit

    fracOrbitEcl = beta/(2*np.pi) #Fraction of orbit in eclipse
    period = 2*np.pi*np.sqrt(rOrbit**3/muPlanet)
    timeInEclipse = period * fracOrbitEcl #If period is input, get time in eclipse


    print("Central Angle of Eclipsed Orbit : ", beta*180/np.pi, "deg")
    print("Fraction of Orbit in Eclipse    : ", fracOrbitEcl)
    print("Period                          :  {:.0f} s, {:.1f} min, {:.1f} hrs".format(period,period/60,period/60/60))
    print("Time in Eclipse                 :  {:.0f} s, {:.1f} min, {:.1f} hrs".format(timeInEclipse,timeInEclipse/60,timeInEclipse/60/60))