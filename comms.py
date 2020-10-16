##Comms functions

import numpy as np

def satFootprintConeAng_fromAltMEA(alt, MEA, rPlanet = 6378.1e3):
	"""
	Calculates the nadir coverage area from the satellite nadir vector given an altitude and minimum elevation angle (MEA)
	satFootprintConeAng_fromAltMEA(alt, MEA, rPlanet = 6378.1e3)
	alt: altitude (m)
	MEA: minimum elevation angle of ground station for connectivity (radians)
	rPlanet: radius of planet (m, Earth default)

	Outputs half angle of nadir cone in radians
	"""
	phi = np.arcsin(rPlanet * np.sin(np.deg2rad(90 + np.rad2deg(MEA)))/(rPlanet + alt))
	return phi

def MEA_fromIncLatAlt(i, lat, alt, rPlanet = 6378.1e3):
	"""
	Calculates the Minimum Elevation Angle (MEA) required to reach a satellite at a specified orbit
	Outputs Minimum Elevation Angle in radians
	MEA_fromIncLatAlt(i, lat, alt, rPlanet = 6378.1e3)
	i: inclination (rad)
	lat: latitude (rad)
	alt: altitude (m)
	rPlanet: radius of planet (m, Earth default)
	"""

	psi = lat - i #central ground angle
	d = np.sqrt(rPlanet**2 + (rPlanet + alt)**2 - 2*re*(rPlanet+alt)*np.cos(psi)) #Slant range
	MEA_rad = -(np.arcsin((((re + alt) * np.sin(psi))/d)) - np.deg2rad(90)) #Insert a negative to account for arcsin being limited to 90 degrees (need values up to 180 degrees)
	return MEA_rad

def ECA_fromNadirEle(nu, ele, angDef = 'r'):
	"""
	Gets Earth central angle from the nadir angle and elevation angle
	Inputs:
	nu: Nadir angle measured at the satellite and defined from subsatellite point to target (radians)
	ele: Elevation angle at the target point from horizontal up to spacecraft (radians)
	angDef: Notes if inputs are in radians 'r' or degrees 'd'. Default is radians 'r'

	Outpus:
	lam: Earth central angle measured from center of earth from subsatellite point to target
	"""
	try:
		type(angDef) == str
	except:
		print("angDef not a string. Value should be 'r', 'R', 'Rad', 'rad', 'd', 'D', 'Deg', 'deg'")

	angInput = angDef.lower()
	if angInput == 'r' or angInput == 'rad':
		rightAng = np.pi/2
	elif angInput == 'd' or angInput == 'deg':
		rightAng = 90

	lam = rightAng - nu - ele
	return lam

def footprintLength(nu, ele, alt, bw, unit = 'km', re = 6371e3):
    """
    Gets the length of the satellite footprint. SME SMAD eqn 10-1a
    Inputs:
    nu: Nadir angle measured at the satellite and defined from subsatellite point to target of footprint Toe (radians)
    ele: Elevation angle at toe of footprint (radians)
    alt: altitude of satellite (m)
    bw: beamwidth (radians)
    unit: unit of output. Input can be 'deg', 'km', or 'nmi' for degrees, kilometers, nautical miles respectively
    re: radius of the central body. Earth is default at 6371 km 
    Outputs:
    fpLength: footprint length in unit specified by 'unit' input
    """
    if unit == 'km':
    	K_l = 111.319543
    elif unit == 'd':
    	K_l = 1
    elif unit =='nmi':
    	K_l = 60.1077447
    else:
    	print("input for 'uni' must be 'deg', 'km', or 'nmi'")

    lam_fo = ECA_fromNadirEle(nu, ele)  
    nu_heel = nu - bw   
    sinrho = re / (re + alt)
    e_heel = np.arccos(np.sin(nu_heel) / sinrho)
    lam_fi = ECA_fromNadirEle(nu_heel, e_heel)
    fpLength = K_l * (np.rad2deg(lam_fo) - np.rad2deg(lam_fi))
    return fpLength


