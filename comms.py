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

def footprintLength(nu, ele, alt, bw, unit = 'km', re = 6378.1e3):
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


def slantRange_fromAltECA(alt, lam, re = 6378.1e3):
	"""
	Gets relevant slant range angles if given altitude and earth central angle (ECA). 
	Section 8.3.1 from SME SMAD, starting from eqn 8-26

	Inputs
	alt : altitude in (m)
	lam : Earth central angle of ground station (target). Usually latitude (rad)
	re  : radius of planet. Default is earth. (m)

	Outputs
	nu  : nadir angle measured from spacecraft subsatellite point to ground target (rad)
	ele : elevation angle measured at the target between spacecraft and local horizontal (rad)
	D   : distance to the target from the ground station (m)
	"""
	sinrho = re / (re + alt)
	tanNuNum = (sinrho * np.sin(lam)) #numerator
	tanNuDen = (1 - sinrho * np.cos(lam))
	nu = np.arctan2(tanNuNum, tanNuDen)
	ele = np.pi/2 - nu - lam
	D = re * (np.sin(lam) / np.sin(nu)) 
	return nu, ele, D

def slantRange_fromAltNu(alt, nu, re = 6378.1e3):
	"""
	Gets relevant slant range angles if given altitude and nadir angle (nu). 
	Section 8.3.1 from SME SMAD, starting from eqn 8-26

	Inputs
	alt : altitude in (m)
	nu  : nadir angle measured from spacecraft subsatellite point to ground target (rad)
	re  : radius of planet. Default is earth. (m)

	Outputs
	lam : Earth central angle of ground station (target). Usually latitude (rad)
	ele : elevation angle measured at the target between spacecraft and local horizontal (rad)
	D   : distance to the target from the ground station (m)
	"""
	sinrho = re / (re + alt)

	cosEle = np.sin(nu) / sinrho
	ele = np.arccos(cosEle)

	lam = np.pi/2 - nu - ele

	D = re * (np.sin(lam) / np.sin(nu)) 

	return lam, ele, D

def slantRange_fromAltEle(alt, ele, re = 6378.1e3):
	"""
	Gets relevant slant range angles if given altitude and elevation angle (nu). 
	Section 8.3.1 from SME SMAD, starting from eqn 8-26

	Inputs
	alt : altitude in (m)
	ele : elevation angle measured at the target between spacecraft and local horizontal (rad)
	re  : radius of planet. Default is earth. (m)

	Outputs
	lam : Earth central angle of ground station (target). Usually latitude (rad)
	nu  : nadir angle measured from spacecraft subsatellite point to ground target (rad)
	D   : distance to the target from the ground station (m)
	"""
	sinrho = re / (re + alt)

	sinnu = np.cos(ele) * sinrho
	nu = np.arcsin(sinnu)

	lam = np.pi/2 - ele - nu

	D = re * (np.sin(lam) / np.sin(nu)) 

	return lam, nu, D
