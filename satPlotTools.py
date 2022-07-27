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
# from cycler import cycler
from itertools import cycle

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


def plotGroundTrack(lon, lat, style_color = 'k', 
    style_line = '-', style_init = 'r^', style_width=1, label = None):
    """
    Plots ground tracks on map using cartopy

    Parameters
    ----------
    lon (deg): Longitude
    lat (deg): Latitude
    style_groundtrack (string): matplotlib linestyle for ground track
    style_init (string): matplotlib linestyle for starting satellite point

    Returns
    ----------
    fig: matplotlib fig object
    ax: matplotlib ax object
    """
    fig = plt.figure(figsize=(15, 25))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.plot(lon, lat, 
        color=style_color, 
        linestyle = style_line,
        linewidth = style_width, 
        transform=ccrs.Geodetic(),
        label = label)
    ax.plot(lon[0], lat[0], style_init, transform=ccrs.Geodetic())
    return fig, ax


def addGroundTrack(lon, lat, ax, style_color = 'k', 
    style_line = '-', style_init = 'r^', style_width=1, label=None):
    """
    Plots ground tracks on map using cartopy

    Parameters
    ----------
    lon: deg 
        Longitude
    lat: deg 
        Latitude
    ax: matplotlib 
        matplotlib ax object
    style_color: string 
        matplotlib color
    style_line: string 
        matplotlib line style
    style_init: string 
        matplotlib style for initial data point
    style_width: int 
        matplotlib linestyle for starting satellite point


    Returns
    ----------
    ax: matplotlib ax object
    """
    ax.plot(lon, lat, 
        color=style_color, 
        linestyle = style_line,
        linewidth = style_width, 
        transform=ccrs.Geodetic(),
        label=label)
    ax.plot(lon[0], lat[0], style_init, transform=ccrs.Geodetic())


    return ax

def addGroundLoc(lon, lat, ax, style_color = 'k', 
    style_marker = 'o', style_width=1, label = None):
    """
    Plots ground tracks on map using cartopy

    Parameters
    ----------
    lon: deg 
        Longitude
    lat: deg 
        Latitude
    ax: matplotlib 
        matplotlib ax object
    style_color: string 
        matplotlib color for marker
    style_marker: string 
        matplotlib data marker
    style_width: int 
        matplotlib linestyle for starting satellite point


    Returns
    ----------
    ax: matplotlib ax object
    """
    ax.plot(lon, lat, 
        color=style_color, 
        marker = style_marker,
        linewidth = style_width, 
        transform=ccrs.Geodetic(),
        label=label)
    # ax.plot(lon[0], lat[0], style_init, transform=ccrs.Geodetic())
    if label:
        ax.legend()

    return ax

def plotSimConstellationGroundTrack(simConstellation, satellites = [], legend=False):
    """
    Plot ground tracks of a SimConstellation

    Parameters
    ----------
    simConstellation: ~satbox.SimConstellation
        Constellation to plot
    satellites: list
        List of individual satellites to plot. Empty list plots all satellites


    """
    assert simConstellation.propagated==1, "Run propagate() on class first"
    propConstellation = simConstellation.constellation

    #Get number of satellites
    nSats = len(simConstellation.initConstellation.get_sats())

    color = iter(plt.cm.jet(np.linspace(0,1,nSats)))
    
    firstRun = 1
    # if len(satellites) == 0:
    for plane in propConstellation.planes:
        for sat in plane.sats:
            c = next(color)
            plotLabel = True
            for segment in sat.coordSegmentsLLA:
                # import ipdb; ipdb.set_trace()
                lon = segment.lon
                lat = segment.lat
                if sat.initSat.satID in satellites or len(satellites) == 0:
                    linestyle_cycler = cycle(['-','--',':','-.'])
                    linewidth_cycler = cycle([1,2,3,4])
                    if firstRun == 1:
                        if plotLabel:
                            label=f'Sat {sat.satID}'
                            plotLabel = False
                        else:
                            label=None
                        fig, ax = plotGroundTrack(lon, lat, 
                            style_color=c, 
                            style_line=next(linestyle_cycler),
                            style_width=next(linewidth_cycler),
                            label=label)
                        firstRun = 0
                    else:
                        if plotLabel:
                            label=f'Sat {sat.satID}'
                            plotLabel = False
                        else:
                            label=None
                        ax = addGroundTrack(lon, lat, ax, 
                            style_color=c, 
                            style_line=next(linestyle_cycler),
                            style_width=next(linewidth_cycler),
                            label=label)
    if legend:
        ax.legend()
    return fig, ax

def addSimConstellationGroundTrack(simConstellation, ax, satellites = [], legend=False):
    """
    Plot ground tracks of a SimConstellation

    Parameters
    ----------
    simConstellation: ~satbox.SimConstellation
        Constellation to plot
    satellites: list
        List of individual satellites to plot. Empty list plots all satellites


    """
    assert simConstellation.propagated==1, "Run propagate() on class first"
    propConstellation = simConstellation.constellation

    #Get number of satellites
    nSats = len(simConstellation.initConstellation.get_sats())

    color = iter(plt.cm.jet(np.linspace(0,1,nSats)))
    
    for plane in propConstellation.planes:
        for sat in plane.sats:
            c = next(color)
            plotLabel = True
            for segment in sat.coordSegmentsLLA:
                lon = segment.lon
                lat = segment.lat
                if sat.initSat.satID in satellites or len(satellites) == 0:
                    if plotLabel:
                        label=f'Sat {sat.satID}'
                        plotLabel = False
                    else:
                        label=None
                    linestyle_cycler = cycle(['-','--',':','-.'])
                    linewidth_cycler = cycle([1,2,3,4])
                    ax = addGroundTrack(lon, lat, ax, 
                        style_color=c, 
                        style_line=next(linestyle_cycler),
                        style_width=next(linewidth_cycler),
                        label=label)
    if legend:
        ax.legend()
    return ax

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

def plot_dijkstra(contacts, path, to_plot=-1):
    """
    Plot dijkstra contacts

    contacts: ~dict
        Two fields "contacts" and "time" which are also dictionaries of boolean arrays of contacts between nodes
    path: ~list
        List of nodes for dijkstra routing
    to_plot: ~int
        Index of last element you want to plot (if you want to cut graph shorter)
    """
    fig,axs = plt.subplots(len(path)-1)
    axs[0].set_title("Nodes: " + " -> ".join(path))
    for n, ax in enumerate(axs):
        source = path[n]
        destination = path[n+1]
        
        key = f'sat {source}-{destination}'
        
        los = contacts['contacts'][key]
        time = contacts['time'][key]
        timePlot = [t.datetime for t in time]
        
        ax.plot(timePlot[0:to_plot], los[0:to_plot], label=key)
        ax.set(ylabel=f'{key}')
        ax.set_yticks([0,1])
    fig.autofmt_xdate()
    return fig, ax
