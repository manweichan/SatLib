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
