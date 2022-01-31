import cartopy.crs as ccrs
import numpy as np
import poliastro
from astropy.coordinates import (
    GCRS,
    ITRS,
    CartesianRepresentation,
    EarthLocation)
from astropy import time
import astropy.units as u
from poliastro.twobody.propagation import propagate
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation
from poliastro.bodies import Earth
from poliastro.czml.extract_czml import CZMLExtractor

import matplotlib.pyplot as plt

def getLonLat(orb, timeInts, pts, J2 = True):
    """
    Function gets latitudes and longitudes for ground track plotting
    
    Inputs
    orb [poliastro orbit object]
    timeInts [astropy object] : time intervals to propagate where 0 refers to the epoch of the orb input variable
    pts [integer] : Number of plot points
    J2 : If True, orbits propagate with J2 perturbation
    
    Outputs
    lon: Longitude (astropy angle units)
    lat: Latitude (astropy angle units)
    height: height (astropy distance units)
    
    """
    if J2:
        def f(t0, state, k):
            du_kep = func_twobody(t0, state, k)
            ax, ay, az = J2_perturbation(
                t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
            )
            du_ad = np.array([0, 0, 0, ax, ay, az])

            return du_kep + du_ad

        coords = propagate(
            orb,
            time.TimeDelta(timeInts),
            method = cowell,
            f=f,
            )
                       
    elif not J2:
        coords = propagate(orb,
         timeDeltas,
         rtol=1e-11,)
    else:
        print("Check J2 input value")
    time_rangePlot = orb.epoch + timeInts
    gcrsXYZ = GCRS(coords, obstime=time_rangePlot,
                   representation_type=CartesianRepresentation)
    itrs_xyz = gcrsXYZ.transform_to(ITRS(obstime=time_rangePlot))
    loc = EarthLocation.from_geocentric(itrs_xyz.x,itrs_xyz.y,itrs_xyz.z)

    lat = loc.lat
    lon = loc.lon
    height = loc.height
    
    return lon, lat, height


def plotGroundTrack(lon, lat, style):
    """
    Plots ground tracks on map using cartopy

    Inputs
    lon (deg): Longitude
    lat (deg): Latitude
    style (string): linestyle from matplotlib

    Outputs
    fig: matplotlib fig object
    ax: matplotlib ax object
    """
    fig = plt.figure(figsize=(15, 25))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.plot(lon, lat, 'k', transform=ccrs.Geodetic())
    ax.plot(lon[0], lat[0], 'r^', transform=ccrs.Geodetic())
    return fig, ax


def writeCZMLFile_constellation(constellation, dirPath, periods = 1, sample_points = 80, singleSat = False,
	groundtrack = False, orbitPath = True):
	"""
	Writes a CZML file for plotting in Cesium

	Input
	constellation (List): list of Satellite objects (SatClasses) grouped by planes
	dirPath (str): directory path to save file
	periods (float): Number of periods to propagate visualization for
	sample_points (int): Number of sample points for visualization
	singleSat (Bool): indicate if generating file for only a single satellite as opposed to a constelllation
	groundtrack (Bool): True to show groundtrack
	orbitPath (Bool): True to show orbit path

	Program writes to dirPath
	"""
	if singleSat: #Single satellite constellation
		seedSat = constellation
	else:
		seedSat = constellation[0][0]
	start_epoch = seedSat.epoch #iss.epoch
	end_epoch = start_epoch + seedSat.period * periods
	earth_uv = "https://earthobservatory.nasa.gov/ContentFeature/BlueMarble/Images/land_shallow_topo_2048.jpg"
	extractor = CZMLExtractor(start_epoch, end_epoch, sample_points,
                          attractor=Earth, pr_map=earth_uv)

	if singleSat:
		extractor.add_orbit(constellation, groundtrack_show=groundtrack,
	                    groundtrack_trail_time=0, path_show=orbitPath)
	else:
		for plane in constellation: #Loop through each plane
		    for sat in plane: #Loop through each satellite in a plane
		        extractor.add_orbit(sat, groundtrack_show=groundtrack,
		                    groundtrack_trail_time=0, path_show=orbitPath)

	testL = [str(x) for x in extractor.packets]
	joinedStr = ','.join(testL)
	toPrint = '[' + joinedStr + ']'
	f = open(dirPath, "w")
	f.write(toPrint)
	f.close()
