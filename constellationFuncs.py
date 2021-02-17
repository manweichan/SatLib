import astropy
import astropy.units as u
from satClasses import *

def genWalker(i, t, p, f, alt):
    """
    Generate a walker constellation
    Outputs a set of satellite orbits 
    
    Inputs:
    i (rad)                  : inclination
    t (int)                  : total number of satellites
    p (int)                  : total number of planes
    f (int between 0 and p-1): determines relative spacing between satellites in adjacent planes
    alt (km)                 : altitude of orbit
    
    Outputs:
    sats (list)              : List of satellite objects (SatClasses). Satellites organized by plane
    """
    
    #Check for astropy classes
    if not isinstance(i, astropy.units.quantity.Quantity):
        i = i * u.rad
    if not isinstance(alt, astropy.units.quantity.Quantity):
        alt = alt * u.km
    
    #Check f is bettween 0 and p-1
    assert f >= 0 and f <= p-1, "f must be between 0 and p-1"
    
    s = t/p #number of satellites per plane
    pu = 360 * u.deg / t #Pattern Unit to define other variables
    
    interPlaneSpacing = pu * p
    nodeSpacing = pu * s
    phaseDiff = pu * f
    
    sats = []
    
    for plane in range(0,p): #Loop through each plane
        planeSats = []
        raan = plane * nodeSpacing
        for sat in range(0,int(s)): #Loop through each satellite in a plane
            omega0 = plane * phaseDiff
            omega = omega0 + sat * interPlaneSpacing
            orbLoop = Satellite.circular(Earth, alt = alt,
                 inc = i, raan = raan, arglat = omega)
            planeSats.append(orbLoop)
        sats.append(planeSats)
        
    return sats