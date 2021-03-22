import astropy
import astropy.units as u
# from satClasses import *
import satClasses as sc
import matplotlib.pyplot as plt
import propEqns as pe
import orbitalMechanics as om
import poliastro
import numpy as np

def genWalker(i, t, p, f, alt, epoch = False):
    """
    Generate a walker constellation
    Outputs a set of satellite orbits 
    
    Inputs:
    i (rad)                  : inclination
    t (int)                  : total number of satellites
    p (int)                  : total number of planes
    f (int between 0 and p-1): determines relative spacing between satellites in adjacent planes
    alt (km)                 : altitude of orbit
    epoch (astropy time)     : epoch to initiate satellites. Default of False defines satellites at J2000
    
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
            if epoch:
                orbLoop = sc.Satellite.circular(Earth, alt = alt,
                     inc = i, raan = raan, arglat = omega, epoch = epoch)
            else:
                orbLoop = sc.Satellite.circular(Earth, alt = alt,
                     inc = i, raan = raan, arglat = omega)
            planeSats.append(orbLoop)
        sats.append(planeSats)
        
    return sats

def getSatValues(constellation, value):
    return [[getattr(sat,value) for sat in plane] for plane in constellation]

def secondStagePlaneDispersion(dryMass, wetMass, payloadMass, satelliteMass, satsPerPlane, loiterAlt, finalAlt, loiterInc, finalInc, isp,
                                rPlanet = poliastro.constants.R_earth, plot = False, verbose = False):
    """
    Determines how many planes one can achieve using second stage/space tug burns

    ** Assuming inputs with astropy units, astropy will handle unit conversions, otherwise inputs must be in the units specified below **
    Inputs
    dryMass (kg) : second stage dry mass
    wetMass (kg) : second stage wet mass
    payloadMass (kg) : second stage payload mass capacity
    satelliteMass (kg) : mass of one satellite in constellation
    satsPerPlane (int) : number of satellites in a plane
    loiterAlt (m) : altitude of phasing orbit
    finalAlt (m) : altitude of final orbit
    loiterInc (rad) : inclination of phasing orbit
    finalInc (rad) : inclination of final orbit
    isp (s) : isp of rocket fuel used
    rPlanet (m) : radius of planet (default is Earth)
    plot (bool) : set to True for plotting
    verbose (bool) : set to True for print statements

    Outputs
    deployedPlanes : Number of planes deployed
    """
    #Check for astropy classes
    if not isinstance(dryMass, astropy.units.quantity.Quantity):
        dryMass = dryMass * u.kg
    if not isinstance(wetMass, astropy.units.quantity.Quantity):
        wetMass = wetMass * u.kg
    if not isinstance(payloadMass, astropy.units.quantity.Quantity):
        payloadMass = payloadMass * u.kg
    if not isinstance(satelliteMass, astropy.units.quantity.Quantity):
        satelliteMass = satelliteMass * u.kg
    if not isinstance(loiterAlt, astropy.units.quantity.Quantity):
       loiterAlt = loiterAlt * u.m
    if not isinstance(finalAlt, astropy.units.quantity.Quantity):
       finalAlt = finalAlt * u.m
    if not isinstance(loiterInc, astropy.units.quantity.Quantity):
       loiterInc = loiterInc * u.rad
    if not isinstance(finalInc, astropy.units.quantity.Quantity):
       finalInc = finalInc * u.rad
    if not isinstance(isp, astropy.units.quantity.Quantity):
       isp = isp * u.s
    if not isinstance(rPlanet, astropy.units.quantity.Quantity):
       rPlanet = rPlanet * u.m

    ## Constant definition
    g0 = 9.80665 * u.m / u.s / u.s

    initFuelMass = wetMass - dryMass
    initMass = wetMass + payloadMass

    possiblePlanes = np.floor(payloadMass/(satsPerPlane * satelliteMass))
    massPerDrop = satsPerPlane * satelliteMass #Offloaded mass in payload
    incChanges = loiterInc - finalInc #Inclination changes of maneuver

    rLoiter = loiterAlt + rPlanet
    rFinal = finalAlt + rPlanet
    delV12_1 = om.delV_HohmannPlaneChange(rFinal, rLoiter, incChanges)
    delV12_2 = om.delV_HohmannPlaneChange(rLoiter, rFinal, incChanges)
    delV12 = delV12_1 + delV12_2 #Total delta V required to go to loiter orbit and return

    fuelTracker = initFuelMass
    maneuverTracker = 0
    massTracker = initMass - massPerDrop #Assume insert at drop orbit
    payloadTracker = payloadMass
    massPlot = []
    fuelPlot = []
    payloadPlot = []
    maneuverPlot = []
    propPlot = []
    while fuelTracker > 0 and maneuverTracker < possiblePlanes and massTracker > dryMass:
        mProp = massTracker * (1 - np.exp(-delV12/(isp * g0)))
        fuelTracker = fuelTracker - mProp #Track fuel storage
        if fuelTracker < 0: #Ran out of fuel
            if verbose:
                print("Out of fuel")
            break
        massTracker = massTracker - mProp - massPerDrop #Track total mass
        payloadTracker -= massPerDrop
        maneuverTracker += 1
        # import ipdb; ipdb.set_trace()
        # fuelPlot = np.append(fuelPlot, fuelTracker)
        # massPlot = np.append(massPlot, massTracker)
        # payloadPlot= np.append(payloadPlot, payloadTracker)
        # maneuverPlot = np.append(maneuverPlot, maneuverTracker)
        # propPlot = np.append(propPlot, mProp)
        fuelPlot.append(fuelPlot)
        massPlot.append(massPlot)
        payloadPlot.append(payloadPlot)
        maneuverPlot.append(maneuverPlot)
        propPlot.append(propPlot)
        if verbose:
            print("Mass Used Propellant: ", mProp)
            print("Mass Fuel:            ", fuelTracker)
            print("Mass Total:           ", massTracker)
            print("Mass Payload:         ", payloadTracker)
            print("Planes Delployed:     ", maneuverTracker + 1) #+1 because first deployment is free (start in desired orobit)
    deployedPlanes = maneuverTracker + 1
    return deployedPlanes
    if plot:
        plt.figure()
        plt.plot(maneuverPlot[1:-1], massPlot[1:-1], 'o', label = "Total Mass")
        plt.plot(maneuverPlot[1:-1], fuelPlot[1:-1], 'o', label = "Total Fuel")
        plt.plot(maneuverPlot[1:-1], payloadPlot[1:-1], 'o', label = "Payload Mass")
        plt.plot(maneuverPlot[1:-1], propPlot[1:-1], 'o', label = "Propulsion Used")
        plt.legend()
        plt.grid()
        plt.title("Fregat Deployment Specs")
        plt.xlabel("Maneuver")
        plt.ylabel("Mass (kg)")



