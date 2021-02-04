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
        coords = propagate(orb,
             time.TimeDelta(timeInts),
             method = cowell,
             rtol=1e-11,
             ad=J2_perturbation,
             J2 = Earth.J2.value,
             R = Earth.R.to(u.km).value)
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
#     plt.show()
    return fig, ax