import numpy as np
import astropy
from astropy import units as u
from astropy import time
from astropy.time import Time
from astropy.coordinates import EarthLocation
import poliastro
from poliastro import constants
from poliastro.earth import Orbit
from poliastro.bodies import Earth
from poliastro.frames.equatorial import GCRS

class Satellite(Orbit):
    def __init__(self, state, epoch, dataMem = None, schedule = None):
        super().__init__(state, epoch) #Inherit class for Poliastro orbits (allows for creation of orbits)
        
        if dataMem is None: #Set up ability to hold data
            self.dataMem = []
        else:
            self.dataMem = dataMem
            
        if schedule is None: #Set up ability to hold a schedule
            self.schedule = []
        else:
            self.schedule = schedule
    
    def add_data(self, data): #Add data object to Satellite
        self.dataMem.append(data)
    
    def remove_data(self, data): #Remove data object from Satellite (i.e. when downlinked)
        if data in self.dataMem:
            self.dataMem.remove(data)
            
    def add_schedule(self, sch_commands): #Adds a schedule to the satellite
        self.schedule.append(sch_commands)
        

class Data():
    def __init__(self, tStamp, dSize):
        """
        Data properties
        tStamp: time stamp data was taken
        dSize: Size of data (bytes)
        """
        self.tStamp = tStamp
        self.dSize = dSize
        
class Schedule():
    def __init__(self, burns = None, xLinks = None, dLinks = None,
                 images = None):
        """
        Defines schedule of satellite
        burns: time, deltaV, and direction (ECI frame) of burns
        xLinks: time and direction (ECI frame) for crosslink
        dLinks: time and direction (ECI frame) for downlink
        image: time and direction for imaging opportunity
        """
        if burns is None: #Set up schedule of burn
            self.burns = []
        else:
            self.burns = burns
        
        if xLinks is None: #Set up schedule of xLinks
            self.xLinks = []
        else:
            self.xLinks = xLinks
            
        if dLinks is None: #Set up schedule of xLinks
            self.dLinks = []
        else:
            self.dLinks = dLinks
            
        if images is None: #Set up schedule of xLinks
            self.images = []
        else:
            self.images = images  
            
    def add_burn(self, burn): #Adds burn
        if burn not in self.burns:
            self.burns.append(burn)
            
    def add_xLink(self, xLink): #Adds crosslink
        if xLink not in self.xLinks:
            self.xLinks.append(xLink)
            
    def add_dLink(self, dLink): #Adds crosslink
        if dLink not in self.dLinks:
            self.dLinks.append(dLink)
            
    def add_image(self, image): #Adds crosslink
        if image not in self.images:
            self.images.append(image)

    def clear_schedule(self): #Clears schedule
        self.burns = []
        self.xLinks = []
        self.dLinks = []
        self.images = []
        
class BurnSchedule():
    def __init__(self, time, deltaV):
        """
        Defines maneuver class for schedule
        time: time of maneuver ####TODO#### Figure out if we want to do this by epoch
        deltaV (m/s): deltaV of maneuver in ECI frame
        direction (ECI frame): direction to apply maneuver
        """
        self.time = time
        self.deltaV = deltaV
        
class xLinksSchedule():
    def __init__(self, time, direction):
        """
        Defnes crosslinks for schedules
        time: time of xLink ####TODO#### Figure out if we want to do this by epoch
        direction (ECI Frame): pointing direction of xLink
        """
        self.time = time
        self.direction = direction
        
class dLinksSchedule():
    def __init__(self, time, direction):
        """
        Defnes downlinks for schedules
        time: time of xLink ####TODO#### Figure out if we want to do this by epoch
        direction (ECI Frame): pointing direction of xLink
        """
        self.time = time
        self.direction = direction
        
class imageSchedule():
    def __init__(self, time, direction):
        """
        Defnes image taking schedule
        time: time of xLink ####TODO#### Figure out if we want to do this by epoch
        direction (ECI Frame): pointing direction of xLink
        """
        self.time = time
        self.direction = direction
        
class groundStation():
    def __init__(self, lon, lat, h, data = None):
        """
        Define a ground station. Requires astopy.coordinates.EarthLocation
        lat (deg): latitude 
        lon (deg): longitude
        h (m): height above ellipsoid
        loc (astropy object) : location in getdetic coordinates
        data (data obj): Data transferred to ground station
        """
#         super().__init__()
        self.lon = lon
        self.lat = lat
        self.h = h
        self.loc = EarthLocation.from_geodetic(lon, lat, h, ellipsoid='WGS84')
        
        if data is None: #keep track of downloaded data
            self.data = []
        else:
            self.data = data
            
    def propagate_to_ECI(self, obstime): #Gets GCRS coordinates (ECI)
        return self.loc.get_gcrs(obstime) 
    
    def get_ECEF(self):
        return self.loc.get_itrs()
    
    #Note ITRF is body fixed (ECEF)
        