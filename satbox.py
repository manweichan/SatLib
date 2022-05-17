import os
import numpy as np
from poliastro import constants
from poliastro.earth import Orbit
from poliastro.earth.sensors import min_and_max_ground_range, ground_range_diff_at_azimuth
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody.propagation import propagate, func_twobody
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation
from poliastro.util import norm
# from poliastro.frames.equatorial import GCRS
from poliastro.czml.extract_czml import CZMLExtractor
import matplotlib.pyplot as plt
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.plotting import OrbitPlotter3D, OrbitPlotter2D
from poliastro.twobody.events import(
    NodeCrossEvent,
)
# import cartopy.crs as ccrs

import seaborn as sns
import astropy.units as u
import astropy
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, GCRS, ITRS, CartesianRepresentation, SkyCoord
from CZMLExtractor_MJD import CZMLExtractor_MJD
# import OpticalLinkBudget.OLBtools as olb
import utils as utils
import orbitalMechanics as om
import comms as com
from copy import deepcopy

import dill

# Constellation Class


class Constellation():
    """
    Defines the Constellation class that holds a set of orbital planes
    """

    def __init__(self, planes = []):
        if isinstance(planes, list):
            self.planes = planes
        else:
            self.planes = []
            self.planes.append(planes)

    def add_plane(self, plane):
        """
        Add a plane to the Constellation
        """
        self.planes.append(plane)

    @classmethod
    def from_list(cls, planes):
        """
        Create constellation object from a list of plane objects
        """
        for idx, plane in enumerate(planes):
            if idx == 0:
                const = cls(plane)
            else:
                const.add_plane(plane)
        return const

    @classmethod
    def from_walker(cls, i, t, p, f, alt, epoch=False, raan_offset=0 * u.deg):
        """
        Generate a walker constellation
        Outputs a set of satellite orbits

        Args:
            i (rad)                  : inclination
            t (int)                  : total number of satellites
            p (int)                  : total number of planes
            f (int between 0 and p-1): determines relative spacing between satellites in adjacent planes
            alt (km)                 : altitude of orbit
            epoch (astropy time)     : epoch to initiate satellites. Default of False defines satellites at J2000
            raan_offset (astropy deg): offset the raan of the first satellite

        Returns:
            constClass (object)      : Constellation class from satClasses
        """

        # Check for astropy classes
        if not isinstance(i, astropy.units.quantity.Quantity):
            i = i * u.rad
            print("WARNING: Inclination treated as radians")
        if not isinstance(alt, astropy.units.quantity.Quantity):
            alt = alt * u.km
            print("WARNING: Altitude treated as kilometers")

        if not isinstance(raan_offset, astropy.units.quantity.Quantity):
            raan_offset = raan_offset * u.deg
            print("WARNING: raan_offset treated as degrees")

        # Check f is bettween 0 and p-1
        assert f >= 0 and f <= p-1, "f must be between 0 and p-1"

        s = t/p  # number of satellites per plane
        pu = 360 * u.deg / t  # Pattern Unit to define other variables

        interPlaneSpacing = pu * p
        nodeSpacing = pu * s
        phaseDiff = pu * f

        allPlanes = []
        planeIDCounter = 0
        satIDCounter = 0
        for plane in range(0, p):  # Loop through each plane
            planeSats = []
            raan = plane * nodeSpacing + raan_offset
            for sat in range(0, int(s)):  # Loop through each satellite in a plane
                omega0 = plane * phaseDiff
                omega = omega0 + sat * interPlaneSpacing
                if epoch:
                    orbLoop = Satellite.circular(Earth, alt=alt,
                         inc=i, raan=raan, arglat=omega, epoch=epoch)
                else:
                    orbLoop = Satellite.circular(Earth, alt=alt,
                         inc=i, raan=raan, arglat=omega)
                orbLoop.satID = satIDCounter
                orbLoop.planeID = planeIDCounter
                satIDCounter += 1
                planeSats.append(orbLoop)
            planeToAppend = Plane.from_list(planeSats)
            planeToAppend.planeID = planeIDCounter
            planeIDCounter += 1
            allPlanes.append(planeToAppend)
        constClass = cls.from_list(allPlanes)
        return constClass

    def get_access(self, groundLoc, timeDeltas=None, fastRun=True, verbose=False):
        """
        Calculate access for an entire constellation and list of ground locations

        Args:
            groundLoc (GroundLoc object or list of GroundLoc objects): ground location object from constellationClasses
            timeDeltas (astropy TimeDelta object): Time intervals to get position/velocity data
            fastRun (Bool) : Takes satellite height average to calculate max/min ground range. Assumes circular orbit and neglects small changes in altitude over an orbit
        """
        allAccessData = []
        if isinstance(groundLoc, list):
            print('calculating access...')
            for gIdx, groundLocation in enumerate(groundLoc):
                if verbose:
                    print(f'location {gIdx+1} out of {len(groundLoc)}')
                accessList = self.__get_access(
                    groundLocation, timeDeltas, fastRun)
                allAccessData.extend(accessList)
        elif isinstance(groundLoc, GroundLoc):
            accessList = self.__get_access(
                groundLoc, timeDeltas, fastRun, verbose=verbose)
            # allAccessData.append(accessList)
            allAccessData = accessList
        dataObjOut = DataAccessConstellation(allAccessData)
        return dataObjOut

    def __get_access(self, groundLoc, timeDeltas, fastRun, verbose=False):
        """
        Private method to calculate access for individual sat/groundLoc pairs
        for cleaner code

        Args:
            groundLoc (GroundLoc object): ground location object from constellationClasses
            timeDeltas (astropy TimeDelta object): Time intervals to get position/velocity data
            fastRun (Bool) : Takes satellite height average to calculate max/min ground range. Assumes circular orbit and neglects small changes in altitude over an orbit
        """
        accessList = []
        for planeIdx, plane in enumerate(self.planes):
            if not plane:  # Continue if empty
                continue
            if verbose:
                print(f'plane {planeIdx + 1} out of {len(self.planes)}')
            planeSats = []
            for satIdx, sat in enumerate(plane.sats):
                if verbose:
                    print(f'sat {satIdx + 1} out of {len(plane.sats)}')
                if not hasattr(sat, 'rvECI') and timeDeltas == None:
                    print("WARNING: Satellite has no rvECI attribute:\n"
                    "EITHER input timeDeltas (astropy quantity) as timeDeltas argument OR\n"
                    "run sat.get_rv_from_propagate(timeDeltas) before inputting satellite object into this function\n"
                    "breaking")
                    return None
                elif not hasattr(sat, 'rvECI'):
                    print(f"Propagating Plane {planeIdx}\nSatellite {satIdx}")
                    sat.get_rv_from_propagate(timeDeltas)
                    tofs = timeDeltas
                else:
                    tofs = sat.rvTimeDeltas

                # print('satellite object input ok')
                access = sat.get_access_sat(groundLoc, timeDeltas, fastRun)
                accessList.append(access)
        # accessData = DataAccessConstellation(accessList)
        return accessList


    def propagate(self, time):
        """
        Propagates satellites in the constellation to a certain time.

        Args:
            time(astropy time object): Time to propagate to
        """
        planes2const=[]
        for plane in self.planes:
            if not plane:  # Continue if empty
                continue

            planeSats=[]
            for satIdx, sat in enumerate(plane.sats):
                #Save properties that are erased during propagation
                satID = sat.satID
                planeID = sat.planeID
                note = sat.note 
                task = sat.task
                manSched = sat.manSched

                satProp=sat.propagate(time)

                satProp.satID = satID
                satProp.planeID = planeID
                satProp.note = note 
                satProp.task = task
                satProp.manSched = manSched

                planeSats.append(satProp)
            plane2append=Plane.from_list(planeSats)
            planes2const.append(plane2append)
        return Constellation.from_list(planes2const)

    def add_comms_payload(self, commsPL):
        """
        Adds comms payload to each satellite in the constellation.

        Args:
            commsPL (CommsPayload Object): Communications payload
        """
        planes2const=[]
        for plane in self.planes:
            if not plane:  # Continue if empty
                continue

            planeSats=[]
            for satIdx, sat in enumerate(plane.sats):
                satComms=sat.add_comms_payload(commsPL)
                planeSats.append(satComms)
            plane2append=Plane.from_list(planeSats)
            planes2const.append(plane2append)
        return Constellation.from_list(planes2const)

    def add_sensor_payload(self, sensorPL):
        """
        Adds sensor payload to each satellite in the constellation.

        Args:
            sensorPL (RemoteSensor Object): Remote Sensor payload
        """
        planes2const=[]
        for plane in self.planes:
            if not plane:  # Continue if empty
                continue

            planeSats=[]
            for satIdx, sat in enumerate(plane.sats):
                satSens=sat.add_remote_sensor(sensorPL)
                planeSats.append(satSens)
            plane2append=Plane.from_list(planeSats)
            planes2const.append(plane2append)
        return Constellation.from_list(planes2const)

    def remove_comms_payload(self):
        """
        Removes comms payload in each satellite in the constellation.
        """
        planes2const=[]
        for plane in self.planes:
            if not plane:  # Continue if empty
                continue

            planeSats=[]
            for satIdx, sat in enumerate(plane.sats):
                satComms=sat.reset_payload()
                planeSats.append(satComms)
            plane2append=Plane.from_list(planeSats)
            planes2const.append(plane2append)
        return Constellation.from_list(planes2const)

    def get_rv_from_propagate(self, timeDeltas, method="J2"):
        """
        Propagates satellites and returns position (R) and Velocity (V) values
        at the specific timeDeltas input. Defaults to propagation using J2 perturbation

        Applies method of the same name in Satellite class to each satellite
        in this constellation

        Args:
            timeDeltas (astropy TimeDelta object): Time intervals to get position/velocity data
        """
        planes2const=[]

        for plane in self.planes:
            if not plane:  # Continue if empty
                constrained_layout

            planeSats=[]
            for satIdx, sat in enumerate(plane.sats):
                sat.sat_get_rv_from_propagate(timeDeltas, method=method)
                planeSats.append(sat)
            planes2append=Plane.from_list(planeSats)
            planes2const.append(planes2append)
        return Constellation.from_list(planes2const)

    def get_sats(self):
        """
        Gets a list of all the satellite objects in the constellation
        """
        satList = []
        planes = self.planes
        for plane in planes:  # Continue if empty
            if not plane:
                continue
            for sat in plane.sats:
                satList.append(sat)
        return satList

    def combine_constellations(self, constellation):
        """
        Combines two constellations by appending the first constellation with the planes
        of a second constellation

        Usually run reassign_sat_ids afterwards to reassign satellite Ids in the new constellation
        """
        constellationID = 0
        planes2append = constellation.planes
        newConst = deepcopy(self)

        for plane in newConst.planes:  # assign constellation ID to original constellation
            for sat in plane.sats:
                sat.constellationID = constellationID

        constellationID += 1
        for plane in planes2append:
            for sat in plane.sats:
                sat.constellationID= constellationID  # Add new constellationID to second constellation
            newConst.add_plane(plane)

        return newConst

    def reassign_sat_ids(self):
        """
        Reassign satellite IDs, normally down after combining constellations
        """
        satIDNew = 0
        for plane in self.planes:
            for sat in plane.sats:
                sat.satID = satIDNew
                satIDNew += 1



    def plot(self):
        """
        Plot constellation using Poliastro interface
        """
        # Doesn't seem to work in jupyter notebooks. Must return plot object and show in jupyter notebook
        sats = self.get_sats()
        op = OrbitPlotter3D()
        for sat in sats:
            op.plot(sat, label=f"ID {sat.satID:.0f}")
        # op.show()
        return op

    def plot_sats(self, satIDs):
        """
        Plot individual satellites using Poliastro interface

        Args:
            satIDs [list] : list of integers referring to satIDs to plot
        """
        sats = self.get_sats()
        op = OrbitPlotter3D()
        for sat in sats:
            if sat.satID in satIDs:
                op.plot(sat, label=f"ID {sat.satID:.0f}")
        return op

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def gen_GOM_2_RGT_scheds(self, r_drift, gs, tStep=15*u.s):
        """
        Generates schedules to take each satellite in the constellation to the
        desired RGT track. This function doesn't decide which satellites are
        given the RGT tasking.
        
        Parameters
        ----------
        self: satbox.Constellation
            Constellation in global observation mode
        r_drift: ~astropy.unit.Quantity
            Radius of drift orbit (assuming circular)
        gs: satbox.GroundLoc
            Ground location that RGT should pass
        tSTep: ~astropy.unit.Quantity
            time step used when propagating satellite into drift orbit
            
        Returns
        -------
        schedDict: Dict of satbox.ManeuverSchedules
            Schedule that will take self to desired RGT orbit. Key is [planeID][satID]
        """

        sats = self.get_sats()

        schedDict = {}

        for sat in sats:
            sched = sat.gen_GOM_2_RGT_sched(r_drift, gs, tStep)
            satID = sat.satID
            planeID = sat.planeID

            if f"Plane {planeID}" in schedDict:
                schedDict[f"Plane {planeID}"][f"Sat {satID}"] = sched
            else:
                schedDict[f"Plane {planeID}"] = {} #Initialize
                schedDict[f"Plane {planeID}"][f"Sat {satID}"] = sched

        return schedDict

    @staticmethod
    def get_lowest_drift_time_per_plane(schedDict):
        """
        Gets the satellite in each plane with the lowest drift time to RGT

        Parameters
        ----------
        schedDict: Dict of satbox.ManeuverSchedules
            Schedule that will take satellites in constellation to desired RGT orbits. Key is [planeID][satID]

        Returns
        -------
        sats2Maneuver: Dict
            Dict of satellites to maneuver
        driftTimes: Dict
            Dict of drift times
        """

        sats2Maneuver = {}
        driftTimes = {}
        scheds = {}

        for planeKey in schedDict.keys():
            plane = schedDict[planeKey]
            sats2Maneuver[planeKey] = []
            driftHolder = None
            for satKey in plane.keys():
                tDrift = plane[satKey].driftTime 
                if driftHolder is None:
                    driftHolder = tDrift 
                    saveSat = satKey
                    sched = plane[satKey]
                elif driftHolder > tDrift:
                    driftHolder = tDrift
                    saveSat = satKey
                    sched = plane[satKey]
                elif driftHolder <= tDrift:
                    continue
                else:
                    print("something is wrong with tDrift comparison")

            sats2Maneuver[planeKey].append(saveSat)
            driftTimes[f'{planeKey} {saveSat}'] = driftHolder
            scheds[f'{planeKey} {saveSat}'] = sched

        return sats2Maneuver, driftTimes, scheds



    def generate_czml_file(self, prop_duration, sample_points, 
                            fname=None, satellites=None, objects=None, L_avail_ISL=None,L_poly_ISL=None, L_avail_GS=None, L_poly_GS=None, GS_pos=None, GS=False, scene3d=True, specificSats=False, show_polyline_ISL=False, show_polyline_GS=False, create_file=False):
        """
        Generates CZML file for the constellation for plotting

        Args:
            fname (string): File name (including path to file) for saved czml file. Currently plots in a directory czmlFiles
            prop_duration (astropy time): Time to propagate in simulation
            sample_points (int): Number of sample points
            scene3d (bool): Set to false for 2D plot
            specificSats (list of ints) : List of satIDs to plot
            show_polyline-isl (boo) : Whether or not to show intersatellite communication links. Default is set to False
            show_polyline_gs (boo) : Whether or not to show gs-sat communication links. Default is set to False
            satellites (list) : list of satellite pairs that can can communicate (or you want to communicate)
            objects (list) : list of sat-gs pairs that can can communicate (or you want to communicate)
            L_avail (list) : list of availability intervals
            L_poly (list) : list of polyline intervals
            L_avail_gs (list) : list of availability intervals
            L_poly_gs (list) : list of polyline intervals
            G_pos (list of list): postions of each satellite
            create_file (boo) : whether  or not to create a czml file or just return text
        """
        seedSat = self.get_sats()[0]
        start_epoch= seedSat.epoch  # iss.epoch

        end_epoch = start_epoch + prop_duration

        earth_uv = "https://earthobservatory.nasa.gov/ContentFeature/BlueMarble/Images/land_shallow_topo_2048.jpg"

        extractor = CZMLExtractor_MJD(start_epoch, end_epoch, sample_points,
                                      attractor=Earth, pr_map=earth_uv, scene3D=scene3d)

        if specificSats:
            for plane in self.planes:  # Loop through each plane
                for sat in plane.sats:  # Loop through each satellite in a plane
                    if sat.satID in specificSats:
                        extractor.add_orbit(sat, groundtrack_show=False,
                                groundtrack_trail_time=0, path_show=True)
                        # breakpoint()
        else:
            for plane in self.planes:  # Loop through each plane
                for sat in plane.sats:  # Loop through each satellite in a plane
                    extractor.add_orbit(sat, groundtrack_show=False,
                                groundtrack_trail_time=0, path_show=True)

        #adding groundstations
        if GS:
            for i in GS_pos:
                extractor.add_ground_station(pos=i)


        #adding isl
        if show_polyline_ISL:
            if (satellites==None) or (L_avail_ISL==None) or (L_poly_ISL==None):
                return "You have chosen to visualize the intersatellite communication links between satellites. Please make sure the following paramenters have been inputted: satellites, L_avail, and L_poly"    
        
            # sat-sat links
            r1=[]
            r2=[]
            for i in satellites:
                x = i.split('-')
                r1.append(x[0])
                r2.append(x[1])

        
            for i in range(len(satellites)):
                extractor.add_communication(id_name=r1[i]+'to'+r2[i],
                                        id_description=None,
                                        reference1=r1[i],
                                        reference2=r2[i],
                                        true_time_intervals=L_avail_ISL[i],
                                        time_intervals=L_poly_ISL[i],
                                        )
        #adding gs-sat links
        if show_polyline_GS:
            if (objects==None) or (L_avail_GS==None) or (L_poly_GS==None):
                return "You have chosen to visualize the gs-sat links between satellites. Please make sure the following paramenters have been inputted: objects, L_avail_gs, and L_poly_gs"
            r3=[]
            r4=[]
            for i in objects:
                x = i.split('-')
                r3.append(x[0])
                r4.append(x[1])
        
            for i in range(len(objects)):
                extractor.add_communication(id_name=r3[i]+'to'+r4[i],
                                        id_description=None,
                                        reference1=r3[i],
                                        reference2=r4[i],
                                        true_time_intervals=L_avail_GS[i],
                                        time_intervals=L_poly_GS[i],
                                        )
                
        if create_file:     
            doc = [str(x) for x in extractor.packets]
            toPrint = ','.join(doc)
            toPrint = '[' + toPrint + ']'

            #To create file with name fname.czml
            dir = os.getcwd()
            fileDir = os.path.join(dir, fname + ".czml")
            f = open(fileDir, "w")
            f.write(toPrint)
            f.close()
            
        else:
            doc = [str(x) for x in extractor.packets]
            toPrint = ','.join(doc)
            toPrint = '[' + toPrint + ']'

            return toPrint


        #Manwei's code
        '''
        testL = [str(x) for x in extractor.packets]
        toPrint = ','.join(testL)
        toPrint = '[' + toPrint + ']'
        cwd = os.getcwd()

        czmlDir = os.path.join(cwd, "czmlFiles")

        # Check if directory is available
        # czmlDirExist = os.path.isdir(czmlDir)
        os.makedirs(czmlDir, exist_ok=True)

        fileDir = os.path.join(cwd, czmlDir, fname + ".czml")
        f = open(fileDir, "w")
        f.write(toPrint)
        f.close()

    def dill_write(self, fname_prefix):
        """
        Writes current constellation to a pickle file using the dill library

        Args:
            fname_prefix (string) : file name string, excluding the .pkl
        """
        fname = fname_prefix + '.pkl'
        with open(fname, 'wb') as f:
            dill.dump(self, f)
        '''


class SimConstellation():
    """
    Defines the SimConstellation class. This object represents the constellation
    as it is propagated in the simulation
    """

    def __init__(self, constellation, t2propagate, tStep, verbose = False):
        """
        Initialize Satellite in simulator

        Parameters
        ----------
        constellation: ~satbox.Constellation
            Satellite initializing object
        t2propagate: ~astropy.unit.Quantity
            Amount of time to Propagate starting from satellite.epoch
        tStep: ~astropy.unit.Quantity
            Time step used in the propagation
        verbose: Bool
            Prints out debug statements if True
        """
        assert isinstance(constellation, Constellation), ('constellation argument must' 
                                                  ' be a Constellation object')
        assert isinstance(t2propagate, astropy.units.quantity.Quantity), ('t2propagate'
                                                 'must be an astropy.units.quantity.Quantity')
        assert isinstance(tStep, astropy.units.quantity.Quantity), ('tStep'
                                                 'must be an astropy.units.quantity.Quantity')
        self.initConstellation = constellation
        self.t2propagate = t2propagate
        self.tStep = tStep
        self.verbose = verbose
        self.planes = []

        self.constellation = Constellation()

        self.propagated = 0 #Check to see if constellation has been propagated

    def propagate(self, method="J2", select_sched_sats=None, verbose=False):
        """
        Propagate satellites in a constellation Simulator

        Parameters
        ----------
        method: str ("J2")
            J2 to propagate using J2 perturbations
        select_sched_sats: dict
            Dictionary of satellites to propagate with burn schedule. Key is plane, value is satellite. Form {'Plane 3': 'Sat12'}
        """
        planes2const = []
        for plane in self.initConstellation.planes:
            if not plane: #continue if empty
                continue
            planeKey = f'Plane {plane.planeID}'
            planeSats = []
            for sat in plane.sats:
                satPropInit = SimSatellite(sat, self.t2propagate, self.tStep, verbose=self.verbose)

                skip_sched = False #Default you will not skip schedule
                #Select satellites if applicable
                if select_sched_sats is not None:
                    satStr = f'Sat {sat.satID}'
                    if satStr in select_sched_sats[planeKey]:
                        skip_sched = False
                        if verbose:
                            print(f"Not skipping schedule for {planeKey} {satStr}")
                    else:
                        skip_sched = True

                satPropInit.propagate(skip_sched=skip_sched)
                planeSats.append(satPropInit)
            plane2append = Plane.from_list(planeSats)
            planes2const.append(plane2append)
        self.constellation = self.constellation.from_list(planes2const)
        self.propagated = 1 #Indicate constellation has been propagated

    @classmethod
    def from_list(cls, planes):
        """
        Create constellation object from a list of plane objects
        """
        for idx, plane in enumerate(planes):
            if idx == 0:
                const = cls(plane)
            else:
                const.add_plane(plane)
        return const

    def get_propagated_sats(self):
        """
        Get satellites 
        """
        if self.propagated == 0:
            print("Run SimConstellation.propagate() first to propagate satellites")
            return

        constellation = self.constellation  
        sats = constellation.get_sats()
        return sats

    def get_relative_velocity_analysis(self, verbose=False):
        """
        Gets relative velocities between satellites in the constellation

        Needs to run get_rv_from_propagate first to get position/velocity
        values first

        Args:
            verbose: prints loop status updates

        Returns:
            First layer key are the satellites being compared i.e. '4-10'
            means that satellite 4 is compared to satellite 10. Second layer
            key are the specific data types described below

            LOS (Bool): Describes if there is a line of sight between the satellites

            pDiff : Relative position (xyz)

            pDiffNorm : magnitude of relative positions

            pDiffDot : dot product of subsequent relative position entries (helps determine if there is a 180 direct crossing)

            flag180 : Flag to determine if there was a 180 degree 'direct crossing'

            velDiffNorm : relative velocities

            slewRate : slew rates required to hold pointing between satellites (rad/s)

            dopplerShift : Effective doppler shifts due to relative velocities
        """

        # Check if first satellite has rvECI Attribute
        if not hasattr(self.constellation.planes[0].sats[0], 'coordECI'):
            print("Run self.propagate() first")
            return
        c=3e8 * u.m / u.s

        sats=self.constellation.get_sats()
        numSats=len(sats)

        outputData={}

        outputData['numSats']=numSats
        outputData['satData']={}
        for satRef in sats:
            if verbose:
                print(f'Reference sat {satRef.satID} out of {numSats}')
            for sat in sats:

                if satRef.satID == sat.satID:
                    continue
                if verbose:
                    print(f'Refererence compared to {sat.satID}')
                # Reference orbit RV values
                satRef_r=satRef.coordECI.without_differentials()
                satRef_v=satRef.coordECI.differentials

                # Comparison orbit RV values
                sat_r=sat.coordECI.without_differentials()
                sat_v=sat.coordECI.differentials

                # Determine LOS availability (Vallado pg 306 5.3)
                adotb=sat_r.dot(satRef_r)
                aNorm=sat_r.norm()
                bNorm=satRef_r.norm()
                theta=np.arccos(adotb/(aNorm * bNorm))

                theta1=np.arccos(constants.R_earth / aNorm)
                theta2=np.arccos(constants.R_earth / bNorm)

                LOSidx=(theta1 + theta2) > theta





                # Relative positions
                pDiff=satRef_r - sat_r
                pDiffNorm=pDiff.norm()

                pDiffDot=pDiff[:-1].dot(pDiff[1:])
                if min(pDiffDot) < 0:  # Checks for 180 deg crossinig
                    flag180=1
                else:
                    flag180=0

                velDiff=satRef_v["s"] - sat_v["s"]
                velDiffNorm=velDiff.norm()

                # Slew Equations
                # Do it using the slew equation (From Trevor Dahl report)
                rCV=pDiff.cross(velDiff)  # r cross v
                slewRateOrb=rCV / pDiffNorm**2
                slewRateOrbNorm=slewRateOrb.norm()


                # Doppler shift
                pDiffU=pDiff/pDiffNorm  # unit vector direction of relative position
                # Get velocity of destination satellite (Reference orbit)
                rdDot=satRef_v["s"].to_cartesian()
                numTerm=rdDot.dot(pDiffU)
                rsDot=sat_v["s"].to_cartesian()
                denTerm=rsDot.dot(pDiffU)
                num=c - numTerm
                den=c - denTerm
                fd_fs=num/den

                # Perform max/min analysis

                maxPos=max(pDiffNorm)
                minPos=min(pDiffNorm)

                maxVel=max(velDiffNorm)
                minVel=min(velDiffNorm)

                slewMax=max(slewRateOrbNorm)
                slewMin=min(slewRateOrbNorm)

                dopplerMax=max(fd_fs)
                dopplerMin=min(fd_fs)

                # Check if adjacent sats (i.e. SatIDs are consecutive)
                idDiff=satRef.satID - sat.satID
                idDiffAbs=abs(idDiff)
                if idDiffAbs == 1 or idDiffAbs == numSats - 1:
                    adjacentFlag=1  # Flag means satellites are adjacent
                else:
                    adjacentFlag=0

                posDict={
                            'relPosVec': pDiff,
                            'relPosNorm': pDiffNorm,
                            'relPosMax': maxPos,
                            'relPosMin': minPos,
                            'delRelPos': pDiffDot,
                }

                velDict={
                            'relVel': velDiffNorm,
                            'slewRate': slewRateOrbNorm,
                            'dopplerShift': fd_fs,
                            'velMax': maxVel,
                            'velMin': minVel,
                            'slewMax': slewMax,
                            'slewMin': slewMin,
                            'dopplerMin': dopplerMin,
                            'dopplerMax': dopplerMax,
                }

                dictEntry={
                            'LOS': LOSidx,
                            'relPosition': posDict,
                            'flag180': flag180,
                            'relVel': velDict,
                            'adjacent': adjacentFlag,
                            'timeDeltas': sat.timeDeltas,
                            'times': sat.timesAll,
                }

                dictKey=str(satRef.satID) + '-' + str(sat.satID)
                outputData['satData'][dictKey]=dictEntry
        return outputData


# Plane class
class Plane():
    """
    Defines an orbital plane object that holds a set of satellite objects
    """
    def __init__(self, sats, a=None, e=None, i=None, planeID=None, passDetails=None):
        if isinstance(sats, list):
            self.sats = sats
        else:
            self.sats = []
            self.sats.append(sats)
        self.passDetails = passDetails
        self.a = a
        self.ecc = e
        self.inc = i
        self.planeID = planeID

    @ classmethod
    def from_list(cls, satList, planeID=None):
        """
        Create plane from a list of satellite objects
        """
        if not satList:  # Empty list
            plane = []
            print("Empty list entered")
        else:
            if isinstance(satList, Satellite):
                plane = cls(satList, planeID = planeID)
                # plane.add_sat(sat)
            elif isinstance(satList, list):
                for idx, sat in enumerate(satList):
                    if idx == 0:
                        plane = cls(sat, planeID = planeID)
                    else:
                        plane.add_sat(sat)
            else:
                print("satList type not recognized")
        return plane

    def set_plane_parameters(self):
        """
        Sets the plane parameters a, e, i, from the average orbital parameters
        of the satellites in the plane
        """
        semiMajorAxes = [sat.a for sat in self.sats]
        eccentricities = [sat.ecc for sat in self.sats]
        inclinations = [sat.inc for sat in self.sats]
        a_ave = sum(semiMajorAxes) / len(semiMajorAxes)
        e_ave = sum(eccentricities) / len(eccentricities)
        i_ave = sum(inclinations) / len(inclinations)
        self.a = a_ave
        self.ecc = e_ave
        self.inc = i_ave

    def add_sat(self, sat):
        """
        Add satellite object to the plane
        """
        self.sats.append(sat)

    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

# Satellite class
class Satellite(Orbit):
    """
    Defines the Satellite class. Inherits from Poliastro Orbit class.
    The Satellite class is an initializer class.
    """

    def __init__(self, state, epoch, satID=None, manSched=None,
                 commsPayload=None, remoteSensor=None,
                 planeID=None, note=None,
                 task=None):
        # Inherit class for Poliastro orbits (allows for creation of orbits)
        super().__init__(state, epoch)
        self.satID = satID
        self.planeID = planeID
        self.note = note
        self.task = task
        self.alt = self.a - constants.R_earth
        self.maneuverSchedule = manSched

        if commsPayload is None:  # Set up ability to hold an optical payload
            self.commsPayload = []
        else:
            self.commsPayload = commsPayload

        if remoteSensor is None:  # Set up ability to hold an optical payload
            self.remoteSensor = []
        else:
            self.remoteSensor = remoteSensor

    def add_data(self, data):  # Add data object to Satellite
        self.dataMem.append(data)

    def remove_data(self, data):  # Remove data object from Satellite (i.e. when downlinked)
        if data in self.dataMem:
            self.dataMem.remove(data)

    def add_man_schedule(self, manSchedule):  # Adds a schedule to the satellite
        self.maneuverSchedule = manSchedule

    def add_comms_payload(self, commsPL):
        self.commsPayload.append(commsPL)
        return self

    def add_remote_sensor(self, remoteSensor):
        self.remoteSensor.append(remoteSensor)
        return self

    def reset_comms_payload(self):
        self.commsPayload = []

    def reset_remote_sensor(self):
        self.remoteSensor = []

    def reset_man_schedule(self):
        self.maneuverSchedule = None

    def gen_sched_rgt_acquisition(self, groundLoc, k_r=15, k_d=1):
        """
        Generates a burn schedule that takes a satellite in a drift orbit into
        a desired RGT orbit over the ground location of interest. 

        **Schedule is generated using current satellite (self) epoch**

        Parameters
        ----------
        groundLoc: satbox.GroundLoc  
            This is the ground location that you want to get a pass from
        k_r: int
            Number of revolutions until repeat ground track
        k_d: int
            Number of days to to repeat ground track

        Returns
        -------
        sched: satbox.ManeuverSchedule
            Hohmann transfer schedule to attain RGT orbit
        """
        assert isinstance(groundLoc, GroundLoc), ('groundLoc argument must' 
                                                  ' be a satbox.GroundLoc object')


        #Find desired RGT orbits for specified ground location
        rgtOrbits = self.get_rgt(groundLoc, days=2, k_r=k_r, k_d=k_d)

        #Calculate algorithm for both ascending and descending pass
        rgtDesired = rgtOrbits[0] #Ascending pass
        rgtDesiredD = rgtOrbits[1] #Descending pass

        #Propagate the RGT to the next node crossing
        _, lon_eq_rgt = self.__get_lon_next_node_crossing(rgtDesired)
        _, lon_eq_rgtD = self.__get_lon_next_node_crossing(rgtDesiredD)

        ## Get other longitude crossings
        lonSplit = 360*u.deg / k_r
        crossPointArray = np.linspace(0 * u.deg, 360 * u.deg - lonSplit, k_r)
        crossPointsRaw = lon_eq_rgt + crossPointArray
        crossPointsRawD = lon_eq_rgtD + crossPointArray

        #Mod 360
        crossPoints360 = crossPointsRaw % (360 * u.deg)
        crossPoints360D = crossPointsRawD % (360 * u.deg)
        #Wrap angles > 180 around
        wrapAngles = [-180*u.deg + x%(180*u.deg) if x > 180*u.deg else x for x in crossPoints360]
        wrapAnglesD = [-180*u.deg + x%(180*u.deg) if x > 180*u.deg else x for x in crossPoints360D]


        ## Calculate distance from GOM equatorial crossing to ground track equatorial crossing
        driftSatEq, lon_eq_driftSat = self.__get_lon_next_node_crossing(self)

        if self.a > rgtDesired.a: #Westward relative drift if drift is higher than RGT
            eqDists = [lon_eq_driftSat - cross for cross in wrapAngles]
        elif self.a < rgtDesired.a: #Eastward relative drift if drift is lower than RGT
            eqDists = [cross - lon_eq_driftSat for cross in wrapAngles]
        else:
            print("Drift orbit is in same orbit as rgt orbit")
            return

        if self.a > rgtDesiredD.a: #Westward relative drift if drift is higher than RGT
            eqDistsD = [lon_eq_driftSat - cross for cross in wrapAnglesD]
        elif self.a < rgtDesiredD.a: #Eastward relative drift if drift is lower than RGT
            eqDistsD = [cross - lon_eq_driftSat for cross in wrapAnglesD]
        else:
            print("Drift orbit is in same orbit as rgt orbit")
            return

        #mod Distances by 360 to make sortable (Westward distance to go)
        eqDistsLeft = [d%(360*u.deg) for d in eqDists]
        eqDistsLeftD = [d%(360*u.deg) for d in eqDistsD]

        #This is the distance to go to acquire ground track
        eqDistsLeftSorted = sorted(eqDistsLeft)
        eqDistsLeftSortedD = sorted(eqDistsLeftD)

        minDist = eqDistsLeftSorted[0]
        minDistD = eqDistsLeftSortedD[0]

        ## Define Hohmann transfer parameters
        hoh_a, hoh_ecc = om.a_ecc_hohmann(self.a, rgtDesired.a)
        hoh_aD, hoh_eccD = om.a_ecc_hohmann(self.a, rgtDesiredD.a)
        pnHohmann = om.get_nodal_period(hoh_a, self.inc)
        pnHohmannD = om.get_nodal_period(hoh_aD, self.inc)

        # Get nodal periods
        pnRGT = om.get_nodal_period(rgtDesired.a, rgtDesired.inc)
        pnDrift = om.get_nodal_period(self.a, self.inc)

        t_hoh = om.t_Hohmann(self.a, rgtDesired.a)

        #Descending pass
        pnRGTD = om.get_nodal_period(rgtDesiredD.a, rgtDesiredD.inc)
        t_hohD = om.t_Hohmann(self.a, rgtDesiredD.a)






        #Calculate equatorial nodal displacement
        deltaL_hoh = om.nodal_period_displacement(pnRGT, rgtDesired.a, rgtDesired.ecc, rgtDesired.inc, hoh_a)
        deltaL_hohD = om.nodal_period_displacement(pnRGTD, rgtDesiredD.a, rgtDesiredD.ecc, rgtDesiredD.inc, hoh_aD)
        deltaL_drift = om.nodal_period_displacement(pnRGT, rgtDesired.a, rgtDesired.ecc, rgtDesired.inc, self.a)
        deltaL_driftD = om.nodal_period_displacement(pnRGTD, rgtDesiredD.a, rgtDesiredD.ecc, rgtDesiredD.inc, self.a)
        
        # Get drift rate of drift orbit and hohmann orbit
        deltaLDriftDot = deltaL_drift / pnDrift
        deltaLDriftDotD = deltaL_driftD / pnDrift
        deltaL_hohDot = deltaL_hoh / pnHohmann
        deltaL_hohDotD = deltaL_hohD / pnHohmannD


        deltaL_hoh_transfer = (t_hoh * deltaL_hohDot) * u.rad
        deltaL_hoh_transferD = (t_hohD * deltaL_hohDotD) * u.rad

        # Subtract distance that will be covered by Hohmann Transfer

        #Choose next selection if hohmann transfer drift is longer than minimum Distance
        if minDist < deltaL_hoh_transfer:
            minDist = eqDistsLeftSorted[1]
        if minDistD < deltaL_hoh_transferD:
            minDistD = eqDistsLeftSortedD[1]

        dist2drift = minDist - deltaL_hoh_transfer
        dist2driftD = minDistD - deltaL_hoh_transferD

        # find drift time by dividing dist2drift by deltaLDriftDot
        t2drift = dist2drift / (deltaLDriftDot * u.rad)
        t2driftD = dist2driftD / (deltaLDriftDotD * u.rad)

        # Propagate equatorial crossing by t2drift
        driftSatAtHohmann = driftSatEq.propagate(t2drift, method=cowell, f=self.j2_f) ## TODO: Area where we can speed up code
        driftSatAtHohmannD = driftSatEq.propagate(t2driftD, method=cowell, f=self.j2_f) ## TODO: Area where we can speed up code

        sched = ManeuverSchedule()
        schedD = ManeuverSchedule()
        sched.gen_hohmann_schedule(driftSatAtHohmann, rgtDesired.a)
        schedD.gen_hohmann_schedule(driftSatAtHohmannD, rgtDesiredD.a)

        sched.passType = 'a'
        if t2drift > t2driftD: #Choose descending pass if quicker
            sched = schedD
            sched.passType = 'd'
            sched.desiredOrbit = rgtDesiredD
            sched.driftTime = t2driftD
        else:
            sched.desiredOrbit = rgtDesired
            sched.driftTime = t2drift

        sched.desiredOrbits = rgtOrbits #Pass desired rgt orbit for debugging
        sched.eqCrossings = [wrapAngles, wrapAnglesD]
        return sched

    @staticmethod
    def __get_lon_next_node_crossing(satellite):
        """
        Gets the longitude of the next node crossing

        Parameters
        ----------
        satellite: ~satbox.satellite
            Satellite object in equstion

        Returns
        -------
        satEq: ~satbox.Satellite
            Satellite object at equator
        satLon: ~astropy.unit.Quantity
            longitude of satellite at crossing

        """
        #Propagate the RGT to the next node crossing
        t2propagate = 1.5 * satellite.period.to(u.s).value
        tofs = TimeDelta(np.arange(0, t2propagate, 1)*u.s)

        node_event = NodeCrossEvent(terminal=True)
        events = [node_event]
        rr,vv = cowell(
                Earth.k,
                satellite.r,
                satellite.v,
                tofs,
                f=satellite.j2_f,
                events=events,
        )

        #Propagate satellite to equatorial position
        satEq = satellite.propagate(node_event.last_t, method=cowell, f=satellite.j2_f)

        satECI = GCRS(rr[-1][0], rr[-1][1], rr[-1][2], representation_type="cartesian", obstime = satellite.epoch+node_event.last_t)

        satECISky = SkyCoord(satECI)
        satECEF = satECISky.transform_to(ITRS)
        satEL = EarthLocation.from_geocentric(satECEF.x, satECEF.y, satECEF.z)
        ## Convert to LLA

        ## 1st Node crossing of desired RGT track
        lla = satEL.to_geodetic() #to LLA

        satLon = lla.lon
        return satEq, satLon

    @staticmethod
    def j2_f(t0, state, k):
        du_kep = func_twobody(t0, state, k)
        ax, ay, az = J2_perturbation(
            t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
        du_ad = np.array([0, 0, 0, ax, ay, az])
        return du_kep + du_ad

    def get_rgt(self, groundLoc, days=3 , tInitSim=None, task=None, k_r=15, k_d=1,
                                 refVernalEquinox=astropy.time.Time("2022-03-22T0:00:00", format = 'isot', scale = 'utc')):
        """
        Given an orbit and ground site, gets the desired repeat ground track orbit 
        Loosely based on Legge's thesis section 3.1.2

        Parameters
        ----------
        groundLoc: satbox.GroundLoc  
            This is the ground location that you want to get a pass from
        days int: int 
            Amount of days ahead for the scheduler to plan for
        tInitSim: ~astropy.time.Time
            Time to initialize planner
        task: string 
            The assigned task for the desired satellite. Options: 'Image', 'ISL', 'Downlink'
        k_r: int
            Number of revolutions until repeat ground track
        k_d: int
            Number of days to to repeat ground track
        refVernalEquinox: ~astropy.time.Time 
            Date of vernal equinox. Default is for 2021

        Returns
        ----------
        rgtOrbits: (array of satbox.Satellite objects): 
            Orbit of satellite at ground pass (potential position aka ghost position)
        """
        assert isinstance(groundLoc, GroundLoc), ('groundLoc argument must' 
                                                  ' be a satbox.GroundLoc object')
        assert isinstance(days, int), ('days argument must' 
                                                  ' be an integer')

        if tInitSim is None:
            tInit = self.epoch
            satInit = self  
        else:
            assert isinstance(tInitSim, astropy.units.quantity.Quantity), ('tInitSim '
                                                 'must be an astropy.units.quantity.Quantity')
            tInit = tInitSim
            satInit = self.propagate(tInitSim)

        
        tInitMJDRaw = tInit.mjd
        tInitMJD = int(tInitMJDRaw)

        dayArray = np.arange(1, days + 2)
        days2InvestigateMJD = list(tInitMJD + dayArray) #Which days to plan over
        days = [Time(dayMJD, format='mjd', scale='utc')
                            for dayMJD in days2InvestigateMJD]
        
        # #Get geocentric coordinates of ground station
        gsGeocentric = groundLoc.loc.to_geocentric()

        # #Convert to angles see this website: https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
        gsLonGeocentric = np.arctan2(gsGeocentric[1], gsGeocentric[0])
        r = np.sqrt(gsGeocentric[0]**2 + gsGeocentric[1]**2 + gsGeocentric[2]**2)
        p = np.sqrt(gsGeocentric[0]**2 + gsGeocentric[1]**2)
        gsLatGeocentric = np.arctan2(gsGeocentric[2], p)

        # Calculate reduced latitude (https://en.wikipedia.org/wiki/Latitude#Parametric_latitude_(or_reduced_latitude))
        f = 1/298.257223563 #Flattening factor of WGS-84
        beta = np.arctan((1-f)*np.tan(groundLoc.lat))

        # #Using geocentric latitude
        # asinGeocentric = np.tan(gsLatGeocentric) / np.tan(satInit.inc)
        # delLamGeocentric = np.arcsin(asinGeocentric)
        
        # #Using reduced latitude
        # asinReduced = np.tan(beta) / np.tan(satInit.inc)
        # delLamReduced = np.arcsin(asinReduced)

        # #Using geodetic latitude
        # iGeodetic = np.arctan2(np.tan(satInit.inc), 1-f)
        # asinGeodetic = np.tan(groundLoc.lat) / np.tan(iGeodetic)
        # delLamGeodetic = np.arcsin(asinGeodetic)

        ## Extract relevant orbit and ground station parameters
        i = satInit.inc
        lon = groundLoc.lon
        # lat = groundLoc.lat
        lat = gsLatGeocentric
        raan = satInit.raan

        #Angle check
        asin = np.tan(lat) / np.tan(i)
        if asin > 1:
            print(f'Arcsin angle {asin} rounding to 1')
            asin = 1  * u.one
        elif asin < -1:
            print(f'Arcsin angle {asin} rounding to -1')
            asin = -1 * u.one
        delLam = np.arcsin(asin) #Longitudinal offset
        # import ipdb;ipdb.set_trace()

        theta_GMST = raan + delLam - lon #ascending sidereal angle of pass

        ## Create quicker calculation of descending raan
        raanA = raan
        raanD = theta_GMST + delLam + lon + np.pi * u.rad # pegs both raans to this GMST time


        delDDates = [day - refVernalEquinox for day in days] #Gets difference in time from vernal equinox
        delDDateDecimalYrList = [delDates.to_value('year') for delDates in delDDates] #Gets decimal year value of date difference
        delDDateDecimalYr = np.array(delDDateDecimalYrList)
    
        #Get solar time values for ascending and descending pass
        theta_GMT_raw = theta_GMST - 2*np.pi * delDDateDecimalYr * u.rad + np.pi * u.rad

        theta_GMT = np.mod(theta_GMT_raw, 360 * u.deg)

        angleToHrs = astropy.coordinates.Angle(theta_GMT).hour

        tPass = []
        for d_idx, day in enumerate(days):
            timePass = day + angleToHrs[d_idx] * u.hr
            if timePass > self.epoch: #Make sure time is in the future
                tPass.append(timePass)
        # timesRaw = [tPass_a, tPass_d]

        rgtOrbits = []

        rgt_r, rgt_alt = om.getRGTOrbit(k_r, k_d, satInit.ecc, satInit.inc)

        for timePass in tPass:
            raans_not_used, anoms = self.__desired_raan_from_pass_time(timePass, groundLoc) ##Only need one time to find anomaly since all passes should be the same geometrically

            ghostSatFutureA = Satellite.circular(Earth, alt = rgt_alt,
                 inc = satInit.inc, raan = raanA, arglat = anoms[0], epoch = timePass)
            ghostSatFutureD = Satellite.circular(Earth, alt = rgt_alt,
                 inc = satInit.inc, raan = raanD, arglat = anoms[1], epoch = timePass)

            # ghostSatFutureA.satID = self.satID #tag with satID
            # ghostSatFutureD.satID = self.satID #tag with satID

            ghostSatFutureA.note = 'a' #Tag with ascending or descending
            ghostSatFutureD.note = 'd' #Tag with ascending or descending

            # ##Tag satellite with maneuver number
            # ghostSatFuture.manID = idx

            ## Tag ghost satellites with sat and plane IDs
            ghostSatFutureA.satID = self.satID
            ghostSatFutureA.planeID = self.planeID
            ghostSatFutureD.satID = self.satID
            ghostSatFutureD.planeID = self.planeID

            rgtOrbits.append(ghostSatFutureA)
            rgtOrbits.append(ghostSatFutureD)

        return rgtOrbits

    # def get_rgt(self, groundLoc, days=7 , tInitSim=None, task=None, k_r=15, k_d=1,
    #                              refVernalEquinox=astropy.time.Time("2021-03-20T0:00:00", format = 'isot', scale = 'utc')):
    #     """
    #     Given an orbit and ground site, gets the desired repeat ground track orbit 
    #     Loosely based on Legge's thesis section 3.1.2

    #     Parameters
    #     ----------
    #     groundLoc: satbox.GroundLoc  
    #         This is the ground location that you want to get a pass from
    #     days int: int 
    #         Amount of days ahead for the scheduler to plan for
    #     tInitSim: ~astropy.time.Time
    #         Time to initialize planner
    #     task: string 
    #         The assigned task for the desired satellite. Options: 'Image', 'ISL', 'Downlink'
    #     k_r: int
    #         Number of revolutions until repeat ground track
    #     k_d: int
    #         Number of days to to repeat ground track
    #     refVernalEquinox: ~astropy.time.Time 
    #         Date of vernal equinox. Default is for 2021

    #     Returns
    #     ----------
    #     rgtOrbits: (array of satbox.Satellite objects): 
    #         Orbit of satellite at ground pass (potential position aka ghost position)

    #     Todo:
    #         Account for RAAN drift due to J2 perturbation
    #     """

    #     assert isinstance(groundLoc, GroundLoc), ('groundLoc argument must' 
    #                                               ' be a satbox.GroundLoc object')
    #     assert isinstance(days, int), ('days argument must' 
    #                                               ' be an integer')

    #     if tInitSim is None:
    #         tInit = self.epoch
    #         satInit = self  
    #     else:
    #         assert isinstance(tInitSim, astropy.units.quantity.Quantity), ('tInitSim '
    #                                              'must be an astropy.units.quantity.Quantity')
    #         tInit = tInitSim
    #         satInit = self.propagate(tInitSim)

        
    #     tInitMJDRaw = tInit.mjd
    #     tInitMJD = int(tInitMJDRaw)

    #     dayArray = np.arange(0, days + 1)
    #     days2InvestigateMJD = list(tInitMJD + dayArray) #Which days to plan over
    #     days = [Time(dayMJD, format='mjd', scale='utc')
    #                         for dayMJD in days2InvestigateMJD]
        
    #     # #Get geocentric coordinates of ground station
    #     # gsGeocentric = groundLoc.loc.to_geocentric()

    #     # #Convert to angles see this website: https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
    #     # gsLonGeocentric = np.arctan2(gsGeocentric[1], gsGeocentric[0])
    #     # r = np.sqrt(gsGeocentric[0]**2 + gsGeocentric[1]**2 + gsGeocentric[2]**2)
    #     # p = np.sqrt(gsGeocentric[0]**2 + gsGeocentric[1]**2)
    #     # gsLatGeocentric = np.arctan2(gsGeocentric[2], p)

    #     ## Extract relevant orbit and ground station parameters
    #     i = satInit.inc
    #     lon = groundLoc.lon
    #     lat = groundLoc.lat
    #     # lat = gsLatGeocentric
    #     raan = satInit.raan

    #     #Angle check
    #     asin = np.tan(lat) / np.tan(i)
    #     if asin > 1:
    #         print(f'Arcsin angle {asin} rounding to 1')
    #         asin = 1  * u.one
    #     elif asin < -1:
    #         print(f'Arcsin angle {asin} rounding to -1')
    #         asin = -1 * u.one
    #     delLam = np.arcsin(asin) #Longitudinal offset
    #     theta_GMST_a = raan + delLam - lon #ascending sidereal angle of pass
    #     theta_GMST_d = raan - delLam - lon - np.pi * u.rad #descending sidereal angle of pass

    #     ## Create quicker calculation of descending raan
    #     raanA = raan
    #     raanD = theta_GMST_a + delLam + lon + np.pi * u.rad # pegs both raans to this GMST time

    #     delDDates = [day - refVernalEquinox for day in days] #Gets difference in time from vernal equinox
    #     delDDateDecimalYrList = [delDates.to_value('year') for delDates in delDDates] #Gets decimal year value of date difference
    #     delDDateDecimalYr = np.array(delDDateDecimalYrList)
    
    #     #Get solar time values for ascending and descending pass
    #     theta_GMT_a_raw = theta_GMST_a - 2*np.pi * delDDateDecimalYr * u.rad + np.pi * u.rad
    #     theta_GMT_d_raw = theta_GMST_d - 2*np.pi * delDDateDecimalYr * u.rad + np.pi * u.rad

    #     theta_GMT_a = np.mod(theta_GMT_a_raw, 360 * u.deg)
    #     theta_GMT_d = np.mod(theta_GMT_d_raw, 360 * u.deg)

    #     angleToHrs_a = astropy.coordinates.Angle(theta_GMT_a).hour
    #     angleToHrs_d = astropy.coordinates.Angle(theta_GMT_d).hour
        
    #     tPass_a = []
    #     tPass_d = []
    #     for d_idx, day in enumerate(days):
    #         timePass_a = day + angleToHrs_a[d_idx] * u.hr
    #         timePass_d = day + angleToHrs_d[d_idx] * u.hr
    #         if timePass_a > self.epoch: #Make sure time is in the future
    #             tPass_a.append(timePass_a)
    #         if timePass_d > self.epoch:
    #             tPass_d.append(timePass_d)
    #     timesRaw = [tPass_a, tPass_d]

    #     note_a = f"Potential ascending pass times for Satellite: {self.satID}"
    #     note_d = f"Potential descending pass times for Satellite: {self.satID}"
    #     scheduleItms_a = [PointingObject(tPass) for tPass in tPass_a]
    #     for schItm in scheduleItms_a:
    #         schItm.note = note_a
    #         schItm.passType = 'a'
        
    #     scheduleItms_d = [PointingObject(tPass) for tPass in tPass_d]
    #     for schItm in scheduleItms_d:
    #         schItm.note = note_d
    #         schItm.passType = 'd'
        
    #     scheduleItms = scheduleItms_a + scheduleItms_d
        
    #     rgtOrbits = []

    #     rgt_r, rgt_alt = om.getRGTOrbit(k_r, k_d, satInit.ecc, satInit.inc)

    #     for idx, sch in enumerate(scheduleItms):
    #         raans, anoms = self.__desired_raan_from_pass_time(sch.time, groundLoc) ##Only need one time to find anomaly since all passes should be the same geometrically
    #         import ipdb; ipdb.set_trace()
    #         if sch.passType == 'a': #Ascending argument of latitude
    #             omega = anoms[0]
    #             raan = raans[0]
    #         elif sch.passType == 'd': #Descending argument of latitude
    #             omega = anoms[1]
    #             raan = raans[1]
            
    #         #Get desired satellite that will make pass of ground location
    #         ghostSatFuture = Satellite.circular(Earth, alt = rgt_alt,
    #              inc = satInit.inc, raan = raan, arglat = omega, epoch = sch.time)
    #         ghostSatFuture.satID = self.satID #tag with satID
    #         ghostSatFuture.note = sch.passType #Tag with ascending or descending
            
    #         ##Tag satellite with maneuver number
    #         ghostSatFuture.manID = idx

    #         ## Tag ghost satellites with sat and plane IDs
    #         ghostSatFuture.satID = self.satID
    #         ghostSatFuture.satID = self.planeID

    #         #Include RGT Constant as an attribute
    #         # ghostSatFuture.rgtConstant = k_r * raan + k_d * omega 
    #         # ghostSatFuture.equatorCrossings = wrapAngles
    #         rgtOrbits.append(ghostSatFuture)
    

    #     return rgtOrbits

    def gen_GOM_2_RGT_sched(self, r_drift, gs, tStep=15*u.s):
        """
        Generates a schedule to take a satellite in GOM to RGT over the desired
        ground location (gs)
        
        Parameters
        ----------
        self: satbox.Satellite
            Satellite in global observation mode
        r_drift: ~astropy.unit.Quantity
            Radius of drift orbit (assuming circular)
        gs: satbox.GroundLoc
            Ground location that RGT should pass
            
        Returns
        -------
        sched: satbox.ManeuverSchedule
            Schedule that will take self to desired RGT orbit
        """
        
        # Create drift satellite
        sched = ManeuverSchedule()
        sched.gen_hohmann_schedule(self, r_drift)
        satDrift = self
        satDrift.add_man_schedule(sched)
        
        #Get Hohmann Time
        hohmannStartTime = sched.schedule[0].time
        hohmannStopTime = sched.schedule[1].time
        hohmannTime = hohmannStopTime - hohmannStartTime
        
        hohmannPropagateTime = hohmannTime + tStep #add tStep buffer
        
        #Propagate to drift orbit
        satDriftSim = SimSatellite(satDrift, hohmannPropagateTime.sec * u.s, tStep, verbose=True)
        satDriftSim.propagate()
        
        driftSat = satDriftSim.satSegments[2]
        rgtAqSched = driftSat.gen_sched_rgt_acquisition(gs)
        driftSat.add_man_schedule(rgtAqSched)
        
        for s in rgtAqSched.schedule:
            sched.add_maneuver(s)

        sched.desiredOrbit = rgtAqSched.desiredOrbit
        sched.desiredOrbits = rgtAqSched.desiredOrbits
        sched.eqCrossings = rgtAqSched.eqCrossings
        sched.passType = rgtAqSched.passType
        sched.driftTime = rgtAqSched.driftTime

        return sched
        
    def __desired_raan_from_pass_time(self, tPass, groundLoc):
        """        Gets the desired orbit specifications from a desired pass time and groundstation
        Based on equations in section 3.1.2 in Legge's thesis (2014)

        Parameters
        ----------
            tPass: ~astropy.time.Time 
                Desired time of pass. Local UTC time preferred
            GroundLoc: satbox.GroundLoc 
                GroundLoc class. This is the ground location that you want to get a pass from

        Returns
        ----------
            raans [List]: 2 element list where 1st element corresponding to RAAN in the ascending case
                            and the 2nd element correspond to RAAN in the descending case

            Anoms [List]: 2 element list where the elements corresponds to true Anomalies (circular orbit)
                            of the ascending case and descending case respectively"""
    
         ## Check if astropy class. Make astropy class if not
        if not isinstance(groundLoc.lat, astropy.units.quantity.Quantity):
            groundLoc.lat = groundLoc.lat * u.deg
        if not isinstance(groundLoc.lon, astropy.units.quantity.Quantity):
            groundLoc.lon = groundLoc.lon * u.deg
#         if not isinstance(i, astropy.units.quantity.Quantity):
#             i = i * u.rad
        tPass.location = groundLoc.loc #Make sure location is tied to time object
        theta_GMST = tPass.sidereal_time('mean', 'greenwich') #Greenwich mean sidereal time
        
        # #Get geocentric coordinates of ground station
        gsGeocentric = groundLoc.loc.to_geocentric()

        # #Convert to angles see this website: https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
        # gsLonGeocentric = np.arctan2(gsGeocentric[1], gsGeocentric[0])
        r = np.sqrt(gsGeocentric[0]**2 + gsGeocentric[1]**2 + gsGeocentric[2]**2)
        p = np.sqrt(gsGeocentric[0]**2 + gsGeocentric[1]**2)
        gsLatGeocentric = np.arctan2(gsGeocentric[2], p)

        i = self.inc

        ## Angle check
        # asin = np.tan(groundLoc.lat.to(u.rad)) / np.tan(i.to(u.rad))
        #Calculate in geocentric coordinates
        asin = np.tan(gsLatGeocentric) / np.tan(i)
        if asin > 1:
            print(f'Arcsin angle {asin} rounding to 1')
            asin = 1  * u.one
        elif asin < -1:
            print(f'Arcsin angle {asin} rounding to -1')
            asin = -1 * u.one
        dLam = np.arcsin(asin)

        #From Legge eqn 3.10 pg.69
        raan_ascending = theta_GMST - dLam + np.deg2rad(groundLoc.lon)
        raan_descending = theta_GMST + dLam + np.deg2rad(groundLoc.lon) + np.pi * u.rad

        #Get mean anomaly (assuming circular earth): https://en.wikipedia.org/wiki/Great-circle_distance
        n1_ascending = np.array([np.cos(raan_ascending),np.sin(raan_ascending),0]) #RAAN for ascending case in ECI norm
        n1_descending = np.array([np.cos(raan_descending),np.sin(raan_descending),0]) #RAAN for descending case in ECI norm

        n2Raw = groundLoc.loc.get_gcrs(tPass).data.without_differentials() / groundLoc.loc.get_gcrs(tPass).data.norm() #norm of ground station vector in ECI
        n2 = n2Raw.xyz

        n1_ascendingXn2 = np.cross(n1_ascending, n2) #Cross product
        n1_descendingXn2 = np.cross(n1_descending, n2)

        n1_ascendingDn2 = np.dot(n1_ascending, n2) #Dot product
        n1_descendingDn2 = np.dot(n1_descending, n2)

        n1a_X_n2_norm = np.linalg.norm(n1_ascendingXn2)
        n1d_X_n2_norm = np.linalg.norm(n1_descendingXn2)
        if groundLoc.lat > 0 * u.deg and groundLoc.lat < 90 * u.deg: #Northern Hemisphere case
            ca_a = np.arctan2(n1a_X_n2_norm, n1_ascendingDn2) #Central angle ascending
            ca_d = np.arctan2(n1d_X_n2_norm, n1_descendingDn2) #Central angle descending
        elif groundLoc.lat < 0 * u.deg and groundLoc.lat > -90 * u.deg: #Southern Hemisphere case
            ca_a = 2 * np.pi * u.rad - np.arctan2(n1a_X_n2_norm, n1_ascendingDn2) 
            ca_d = 2 * np.pi * u.rad - np.arctan2(n1d_X_n2_norm, n1_descendingDn2)
        elif groundLoc.lat == 0 * u.deg: #Equatorial case
            ca_a = 0 * u.rad
            ca_d = np.pi * u.rad
        elif groundLoc.lat == 90 * u.deg or groundLoc.lat == -90 * u.deg: #polar cases
            ca_a = np.pi * u.rad
            ca_d = 3 * np.pi / 2 * u.rad
        else:
            print("non valid latitude")
            
        raans = [raan_ascending, raan_descending]
        Anoms = [ca_a, ca_d]

        return raans, Anoms


    def __eq__(self, other):
        """Check for equality with another object"""
        return self.__dict__ == other.__dict__

# SimSatellite class
class SimSatellite():
    """
    Defines the SimSatellite class. This object represents the satellite
    as it is propagated in the simulation
    """

    def __init__(self, satellite, t2propagate, tStep, manSched = None, verbose = False):
        """
        Initialize Satellite in simulator

        Parameters
        ----------
        satellite: ~satbox.Satellite
            Satellite initializing object
        t2propagate: ~astropy.unit.Quantity
            Amount of time to Propagate starting from satellite.epoch
        tStep: ~astropy.unit.Quantity
            Time step used in the propagation
        manSched: satbox.ManeuverSchedule
        verbose: Bool
            Prints out debug statements if True
        """

        assert isinstance(satellite, Satellite), ('satellite argument must' 
                                                  ' be a Satellite object')
        assert isinstance(t2propagate, astropy.units.quantity.Quantity), ('t2propagate'
                                                 'must be an astropy.units.quantity.Quantity')
        assert isinstance(tStep, astropy.units.quantity.Quantity), ('tStep'
                                                 'must be an astropy.units.quantity.Quantity')
        if (manSched is not None):
            assert isinstance(manSched, ManeuverSchedule), ('manSched'
                                                 'must be an ManeuverSchedule object')
        self.initSat = satellite
        self.t2propagate = t2propagate
        self.tStep = tStep
        if satellite.maneuverSchedule is not None:
            self.maneuverSchedule = satellite.maneuverSchedule
            if verbose:
                print("maneuver schedule taken from satellite object")
                if manSched is not None:
                    print("Overwriting initialization schedule with schedule from satellite object")
        elif manSched is not None:
            self.maneuverSchedule = manSched
            #Set satellite schedule to maneuver schedule
            satellite.maneuverSchedule = manSched
            if verbose:
                print("maneuver schedule taken initialization")
        else:
            self.maneuverSchedule = None


        #TimeDelta objects used to propagate satellite
        self.timeDeltas = TimeDelta(np.arange(0, 
                                              t2propagate.to(u.s).value,
                                              tStep.to(u.s).value) * u.s)

        #Preserve original attributes
        self.satID = satellite.satID
        self.planeID = satellite.planeID
        self.note = satellite.note 
        self.task = satellite.task
        self.propagated = 0 #check if propagated

        #Times in UTC (default)
        self.times = satellite.epoch + self.timeDeltas

        #List to hold all satellite segments
        self.satSegments = []
        self.timeSegments = []
        self.cartesianRepSegments = []
        self.coordSegmentsECI = []
        self.coordSegmentsECEF = []
        self.coordSegmentsLLA = []

    def propagate(self, method="J2", skip_sched=False):
        """
        Run simulator for satellite Simulator

        Parameters
        ----------
        method: str ("J2")
            J2 to propagate using J2 perturbations
        skip_sched: bool
            boolean to skip burn schedule when propagating
        """

        if method == "J2":
            def f(t0, state, k): #Define J2 perturbation
                du_kep = func_twobody(t0, state, k)
                ax, ay, az = J2_perturbation(
                    t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
                )
                du_ad = np.array([0, 0, 0, ax, ay, az])

                return du_kep + du_ad

        currentSat = self.initSat #Initialize satellite

        #Save properties that are erased during propagation
        satID = self.initSat.satID
        planeID = self.initSat.planeID
        note = self.initSat.note 
        task = self.initSat.task
        manSched = self.initSat.maneuverSchedule

        self.satSegments.append(currentSat)


        #If not maneuver schedule
        if (self.maneuverSchedule is None) or skip_sched==True: 
            if method == "J2":
                coords = propagate(
                    self.initSat,
                    self.timeDeltas,
                    method=cowell,
                    f=f,
                    )
                coordsAll = coords
            else:
                coords = propagate(
                    self.initSat,
                    self.timeDeltas,
                    )
                coordsAll = coords
            satECI = GCRS(coords.x, coords.y, coords.z, representation_type="cartesian", obstime = self.times)
            satECISky = SkyCoord(satECI)
            satECEF = satECISky.transform_to(ITRS)
            satEL = EarthLocation.from_geocentric(satECEF.x, satECEF.y, satECEF.z)
            ## Convert to LLA
            lla_sat = satEL.to_geodetic() #to LLA

            timesAll = self.times
            self.timeSegments.append(self.times)
            self.cartesianRepSegments.append(coords)
            self.coordSegmentsECI.append(satECISky)
            self.coordSegmentsECEF.append(satECEF)
            self.coordSegmentsLLA.append(lla_sat)

        elif (self.maneuverSchedule is not None):
            schedule = self.maneuverSchedule.schedule

            #Sort maneuver schedule by time
            schedule.sort(key=lambda x: x.time)


            for manIdx, man in enumerate(schedule):
                assert man.time >= currentSat.epoch, "maneuver time before satellite epoch"

                #Poliastro maneuver object    
                poliMan = Maneuver.impulse(man.deltaVVec)

                segmentTimeLen = man.time - currentSat.epoch

                if segmentTimeLen.to(u.s).value == 0: #Burn initialized at same time as sim start
                    sat_i = currentSat 
                    sat_f = sat_i.apply_maneuver(poliMan)

                else:
                    tDeltas = TimeDelta(np.arange(0,
                                                  segmentTimeLen.to(u.s).value,
                                                  self.tStep.to(u.s).value ) * u.s)
                    if method == "J2":
                        coords = propagate(
                            currentSat,
                            tDeltas,
                            method=cowell,
                            f=f,
                            )
                        sat_i = currentSat.propagate(segmentTimeLen,
                                                     method=cowell,
                                                     f=f)
                    else:
                        coords = propagate(
                            currentSat,
                            tDeltas)
                        sat_i = currentSat.propagate(segmentTimeLen)

                    timesSegment = currentSat.epoch + tDeltas
                    satECISeg = GCRS(coords.x, 
                                          coords.y, 
                                          coords.z, 
                                          representation_type="cartesian", 
                                          obstime = timesSegment)
                    satECISkySeg = SkyCoord(satECISeg)
                    satECEFSeg = satECISkySeg.transform_to(ITRS)
                    ## Turn coordinates into an EarthLocation object
                    satELSeg = EarthLocation.from_geocentric(satECEFSeg.x, satECEFSeg.y, satECEFSeg.z)
                    ## Convert to LLA
                    lla_satSeg = satELSeg.to_geodetic() #to LLA

                    self.timeSegments.append(timesSegment)
                    self.cartesianRepSegments.append(coords)
                    self.coordSegmentsECI.append(satECISkySeg)
                    self.coordSegmentsECEF.append(satECEFSeg)
                    self.coordSegmentsLLA.append(lla_satSeg)
                sat_f = sat_i.apply_maneuver(poliMan)

                currentSat = sat_f

                currentSat.satID = satID
                currentSat.planeID = planeID
                currentSat.note = note 
                currentSat.task = task
                currentSat.maneuverSchedule = manSched

                self.satSegments.append(currentSat)

            #Propagate after the maneuver
            timeEnd = self.initSat.epoch + self.t2propagate
            timeLeft = timeEnd - currentSat.epoch
            assert timeLeft >= 0, "timeLeft is < 0, can't create time Deltas"
            if timeLeft.to(u.s).value != 0: #No time left
                tDeltas = TimeDelta(np.arange(0,
                                          timeLeft.to(u.s).value,
                                          self.tStep.to(u.s).value ) * u.s)
                if method == "J2":
                    coords = propagate(
                        currentSat,
                        tDeltas,
                        method=cowell,
                        f=f,
                        )
                    sat_i = currentSat.propagate(timeLeft,
                                                 method=cowell,
                                                 f=f)
                else:
                    coords = propagate(
                        currentSat,
                        tDeltas)
                    sat_i = currentSat.propagate(timeLeft)
                timesSegment = currentSat.epoch + tDeltas
                satECISeg = GCRS(coords.x, 
                                      coords.y, 
                                      coords.z, 
                                      representation_type="cartesian", 
                                      obstime = timesSegment)
                satECISkySeg = SkyCoord(satECISeg)
                satECEFSeg = satECISkySeg.transform_to(ITRS)
                ## Turn coordinates into an EarthLocation object
                satELSeg = EarthLocation.from_geocentric(satECEFSeg.x, satECEFSeg.y, satECEFSeg.z)
                ## Convert to LLA
                lla_satSeg = satELSeg.to_geodetic() #to LLA

                #TODO: May have to eliminate some of these conversions and only convert when needed
                self.timeSegments.append(timesSegment)
                self.cartesianRepSegments.append(coords)
                self.coordSegmentsECI.append(satECISkySeg)
                self.coordSegmentsECEF.append(satECEFSeg)
                self.coordSegmentsLLA.append(lla_satSeg)
            if len(self.cartesianRepSegments) > 1:
                coordsAll= astropy.coordinates.concatenate_representations([*self.cartesianRepSegments])
            else:
                coordsAll = self.cartesianRepSegments[0]
            timesAll = np.concatenate([*self.timeSegments], axis=None)
            satECI = GCRS(coordsAll.x, 
                              coordsAll.y, 
                              coordsAll.z, 
                              representation_type="cartesian", 
                              obstime = timesAll)
        satECISky = SkyCoord(satECI)
        satECEF = satECISky.transform_to(ITRS)

        ## Turn coordinates into an EarthLocation object
        satEL = EarthLocation.from_geocentric(satECEF.x, satECEF.y, satECEF.z)

        ## Convert to LLA
        lla_sat = satEL.to_geodetic() #to LLA
        self.coordECI = coordsAll #ECI coordinates
        self.LLA = lla_sat #Lat long alt of satellite
        self.rvECEF = satECEF #ECEF coordinates
        self.timesAll = timesAll #all times

        self.propagated = 1


# Ground location class
class GroundLoc():
    def __init__(self, lon, lat, h, groundID=None, name=None, identifier=None):
        """
        Define a ground location. Requires astopy.coordinates.EarthLocation

        Args:
            lon (deg): longitude
            lat (deg): latitude
            h (m): height above ellipsoid
            name: is the name of a place (i.e. Boston)
            identifier (string): identifies purpose of GroundLoc
                Current options
                * 'target' -  Location to image
                * 'downlink' - Location to downlink data

        Methods:
            propagate_to_ECI(self, obstime)
            get_ECEF(self)
        """
        if not isinstance(lon, astropy.units.quantity.Quantity):
            lon = lon * u.deg
        if not isinstance(lat, astropy.units.quantity.Quantity):
            lat = lat * u.deg
        if not isinstance(h, astropy.units.quantity.Quantity):
            h = h * u.m
        self.lon = lon
        self.lat = lat
        self.h = h
        self.loc = EarthLocation.from_geodetic(lon, lat, height=h, ellipsoid='WGS84')
        self.groundID = groundID
        self.identifier = identifier

    def propagate_to_ECI(self, obstime):  # Gets GCRS coordinates (ECI)
        return self.loc.get_gcrs(obstime)
    def get_ECEF(self):
        return self.loc.get_itrs()

# Ground station class
class GroundStation(GroundLoc):
    def __init__(self, lon, lat, h, data, commsPayload=None, groundID=None, name=None):
        GroundLoc.__init__(self, lon, lat, h, groundID, name)
        self.data = data
        self.commsPayload = commsPayload

# ManeuverObject Class
class ManeuverObject():
    """
    Not to be confused with the poliastro class 'Maneuver'
    """
    def __init__(self, time, deltaV, satID=None, note=None):
        """
        Parameters
        ----------
        time: ~astropy.time.Time
            Time of maneuver
        deltaV: ~astropy.unit.Quantity - 3D vector
            Change of velocity in ECI frame
        satID: int
            satellite ID of satellite to make maneuver. For housekeeping.
        note: string
            Any special notes to include
        """
        assert isinstance(time, astropy.time.core.Time), ('time must be an'
                                                         'astropy.time.core.Time object')
        assert len(deltaV) == 3, 'deltaV must be a 3 vector'
        assert isinstance(deltaV, astropy.units.quantity.Quantity), ('deltaV' 
                                         ' must be astropy.units.quantity.Quantity')

        self.time = time

        #delta v vector
        self.deltaVVec = deltaV
        self.satID = satID
        self.note = note

        #Total delta v
        self.deltaVTot = utils.get_norm(deltaV)

class ManeuverSchedule():
    """
    Maneuver schedule for satellite
    """
    def __init__(self):
        self.schedule = []
        self.desiredOrbit = None
        self.eqCrossings = None
        self.passType = None

    def add_maneuver(self, manObj):
        """
        Add maneuver object to schedule

        Parameters
        ----------
        manObj: satbox.ManeuverObject
            ManeuverObject to add to schedule
        """
        self.schedule.append(manObj)

    def gen_hohmann_schedule(self, orb, r_f):
        """
        Generate a schedule for a Hohmann transfer
        
        Parameters
        ----------
        orb: ~satbox.Satellite
            Satellite initializing object
        r_f: ~astropy.unit.Quantity
            Desired final orbital radius at end of Hohmann transfer
        
        """
        #Check if it's a forward or backward propulsive hohmann
        if orb.a <= r_f:
            prop_dir = 1 #propel in velocity direction to increase velocity
        elif orb.a > r_f:
            prop_dir = -1 #Slow down satellite to reduce altitude
        else:
            assert "orbital radius is neither greater than or less than final radius"
        
        #Total deltaV to move from circular to elliptical orbit
        delv1Tot = om.circ2elip_Hohmann(orb.a, r_f)

        #Get unit vector
        delv1_dir = utils.get_unit_vec(orb.v) * prop_dir
        
        #Get deltaV in inertial frame
        delv1 = delv1Tot * delv1_dir

        #Total delta V to move from elliptical to final circular orbit
        delv2Tot = om.elip2circ_Hohmann(orb.a, r_f)
        delv2_dir = -delv1_dir #Reverse deltaV direction (assuming perfect Hohmann)
        delv2 = delv2Tot * delv2_dir

        #Get time of hohmann transfer
        t_hohmann = om.t_Hohmann(orb.a, r_f)
        
        #Add to burn scheduler

        # import ipdb; ipdb.set_trace()
        man1_hoh = ManeuverObject(orb.epoch, delv1.to(u.km/u.s))
        man2_hoh = ManeuverObject(orb.epoch + t_hohmann, delv2.to(u.km/u.s))
        # import ipdb; ipdb.set_trace()

        self.add_maneuver(man1_hoh)
        self.add_maneuver(man2_hoh)

    def gen_intersect_sched(self, orb_i, orb_tgt, method="J2"):
        """
        Calculate the schedule needed to intersect a desired orbit in the same plane. 
        For this particular research case, orb_i is usually the drift (intermediate)
        orbit that the satellite hangs out in before conducting a Hohmann 
        transfer to enter the RGT orbit.
        
        Parameters
        ----------
        orb_i: ~satbox.Satellite
            Satellite object in it's current state
        orb_tgt: ~satbox.Satellite
            Desired orbit in any state (epoch)
        method: string
            Method of propagation. Currently only "J2" supported
        
        """
        #Propagate RGT orbit to same epoch as orb_i
        if method=="J2":
            def f(t0, u_, k):
                du_kep = func_twobody(t0, u_, k)
                ax, ay, az = J2_perturbation(
                    t0, u_, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
                )
                du_ad = np.array([0, 0, 0, ax, ay, az])
                return du_kep + du_ad
            orb_tgt_i = orb_tgt.propagate(orb_i.epoch, method=cowell, f=f)
        else:
            orb_tgt_i = orb_tgt.propagate(orb_i.epoch)
        
        #Get initial phase angle
        #Defined from the target to the chaser. 
        #Positive direction in target direction of motion
        v_i = orb_i.arglat - orb_tgt_i.arglat
        if v_i < (-180*u.deg): #Wrap angle around
            v_i = v_i % (360 * u.deg)
        elif v_i > (180*u.deg):
            v_i = v_i - (360 * u.deg)
        
        a_int = orb_i.a
        a_tgt = orb_tgt_i.a
        t_trans, delVTot, a_trans, t_wait = om.coplanar_phase_different_orbs(v_i, 
                                                                          a_int, 
                                                                          a_tgt)
        #Propagate orb_i to time of first burn
        if method=="J2":
            orb_i_1st_burn = orb_i.propagate(t_wait, method=cowell, f=f)
        else:
            orb_i_1st_burn = orb_i.propagate(t_wait)

        ## Debug statements ##
        print(f"T_wait: {t_wait}")
        orb_tgt_180 = orb_tgt.propagate(orb_i.epoch + t_wait + t_trans, method=cowell, f=f)
        anomalyDiff = orb_i_1st_burn.arglat.to(u.deg) - orb_tgt_180.arglat.to(u.deg)
        print(f"Anomaly Diff: {anomalyDiff}")
        print("T_transfer: ", t_trans)
        print("a_trans: ", a_trans)
        ## End Debug statements ##

        self.gen_hohmann_schedule(orb_i_1st_burn, orb_tgt_i.a)

    def get_delV_total(self):
        """
        Get total V of all the maneuvers held in the schedule
        """
        schedule = self.schedule
        if len(schedule) == 0:
            print("No items in schedule")
            return

        delVTot = sum(sch.deltaVTot for sch in schedule)
        return delVTot

class PointingObject():
    def __init__(self, time, note=None, passType=None,direction=None, action=None):
        self.time = time 
        self.note = note
        self.direction = direction
        self.action = action
        self.passType = passType

# Data class
class Data():
    def __init__(self, tStamp, dSize):
        self.tStamp = tStamp
        self.dSize = dSize

# CommsPayload class
class CommsPayload():
    """
    Define a communications payload.

    Args:
        freq : frequency (easier for rf payloads)
        wavelength : wavelength (easier for optical payloads)
        p_tx (W) : Transmit power
        g_tx (dB) : Transmit gain
        sysLoss_tx (dB) : Transmit system loss
        l_tx (dB) : Other transmit loss
        pointingErr (dB) : Pointing error
        g_rx (dB) : Receive gain
        tSys (K) : System temperature on receive side
        sysLoss_rx (dB) : Recieve system loss
        beamWidth (rad) : FWHM beam width for optical payload (Gaussian analysis)
        aperture (m) : Optical recieve aperture diameter
        l_line (dB) : Line loss on receive side
    """
    def __init__(self, freq=None, wavelength=None, p_tx=None,
                 g_tx=None, sysLoss_tx=0, l_tx=0,
                 pointingErr=0, g_rx=None, t_sys=None,
                 sysLoss_rx=0,  beamWidth=None, aperture=None,
                 l_line=0):
        self.freq = freq
        self.wavelength = wavelength
        self.p_tx = p_tx
        self.g_tx = g_tx
        self.sysLoss_tx = sysLoss_tx
        self.l_tx = l_tx
        self.pointingErr = pointingErr
        self.g_rx = g_rx
        self.t_sys = t_sys
        self.sysLoss_rx = sysLoss_rx
        self.beamWidth = beamWidth
        self.aperture = aperture
        self.l_line = l_line

        c = 3e8 * u.m / u.s
        if self.freq == None:
            self.freq = c / self.wavelength
        elif self.wavelength == None:
            self.wavelength = c / self.freq

    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)

    def get_EIRP(self):
        # TRY WITH ASTROPY UNITS
        P_tx = self.p_tx
        P_tx_dB = 10 * np.log10(P_tx)

        l_tx_dB = self.l_tx
        g_tx_dB = self.g_tx

        EIRP_dB = P_tx_dB - l_tx_dB + g_tx_dB
        return EIRP_dB

    def set_sys_temp(self, T_ant, T_feeder, T_receiver, L_feeder):
        """
        Calculate system temperature.
        Reference: Section 5.5.5 of Maral "Satellite Communication Systems" pg 186

        Args:
            T_ant (K) : Temperature of antenna
            T_feeder (K) : Temperature of feeder
            T_receiver (K) : Temperature of receiver
            L_feeder (dB) : Feeder loss
        """
        feederTerm= 10**(L_feeder/10)  # Convert from dB
        term1 = T_ant / feederTerm
        term2 = T_feeder * (1 - 1/feederTerm)
        term3 = T_receiver
        Tsys = term1 + term2 + term3
        self.t_sys = Tsys

    def get_GonT(self):
        # TRY WITH ASTROPY UNITS
        T_dB = 10 * np.log10(self.t_sys)
        GonT_dB = self.g_rx - T_dB - self.l_line
        return GonT_dB

class RemoteSensor():
    """
    Class to describe a remote sensor

    fov: full field of view of sensor cone
    """
    def __init__(self, fov, wavelength):
        self.fov = fov
        self.wavelength = wavelength

# class MissionOption():
    """
    Holder for mission options
    """
    def __init__(self, listOfTransfers, maneuverGoal):
        self.listOfTransfers= listOfTransfers
        self.maneuverGoal = maneuverGoal

    def get_mission_specs(self):
        self.maneuverCosts = [m.deltaV for m in self.listOfTransfers]
        self.totalCost = sum(self.maneuverCosts)

        # Find downlink satellite
        dlSat = [s for s in self.listOfTransfers if s.mySatFin.task == 'Downlink']
        if dlSat:
            self.dlTime = dlSat[0].time2Pass
        else:
            self.dlTime = 'Not scheduled yet'

    def get_weighted_dist_from_utopia(self, w_delv, w_t):
        """
        Get weighted distance from utopia point

        Args:
            w_delv: delV weight
            w_t: time weight
        """
        squared = self.dlTime.to(u.hr).value**2 * w_t + self.totalCost.to(u.m/u.s).value**2 * w_delv
        dist = np.sqrt(squared)
        self.utopDistWeighted = dist


class DataAccessSat():
    """
    Data object that holds information on access between satellites and ground locations

    INTENDED to be run with only satellite objects
    """
    def __init__(self, simSat, groundLoc):
        """
        Parameters
        ----------
        simSat: ~satbox.SimSatellite
            SimSatellite object that has been propagated
        groundLoc: ~satbox.GroundLoc 
            Ground Location object
        """
        if simSat.propagated == 0:
            print("run simSat.propagate() first")
            return

        self.sat = simSat
        self.groundLoc = groundLoc

        # Extract IDs for easy calling
        self.satID = simSat.satID
        self.groundLocID = groundLoc.groundID
        self.groundIdentifier = groundLoc.identifier

        # Holders for other variables
        self.accessIntervals = None
        self.accessMask = None
        self.accessIntervalLengths = None
        self.accessElevations = None

    def calc_access(self, constraint_type, constraint_angle):
        """
        Calculate access between a satellite and a ground station given a 
        constraint type and constraint angle

        Warning: Nadir constraint angle is the half-angle associated with the FOV, so a 
        camera with 90 deg of FOV, would be equivalent to a constraint_angle of 45 deg

        Parameters:
        -----------
        constraint_type: ~string | "nadir" or "elevation"
            constrain angles with either a "nadir" (usually analogous to sensor FOV) or "elevation" (minimum elevation angle from ground station) constraint
        constraint_angle: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation
        """

        satECEF = self.sat.rvECEF.cartesian

        gsECEF = self.groundLoc.get_ECEF().cartesian


        ## Find earth central angle between ground station and satellite
        dotProduct = satECEF.dot(gsECEF)
        satECEFNorm = satECEF.norm()
        gsECEFNorm = gsECEF.norm()

        #cosine formula
        cosECA = dotProduct / (satECEFNorm * gsECEFNorm)

        ECA = np.arccos(cosECA) #Earth central angle

        #Get satellite altitudes
        alts = satECEFNorm - constants.R_earth

        nu, ele, _ = com.slantRange_fromAltECA(alts, ECA)

        #See what angles satisfy the constraints
        if constraint_type == 'elevation':
            accessMask = ele > constraint_angle
        elif constraint_type == 'nadir':
            _, eleFromNu, _ = com.slantRange_fromAltNu(alts, constraint_angle)
            accessMask = ele > eleFromNu
        else:
            assert False, "constraint_type not recognized"

        # import ipdb;ipdb.set_trace()
        accessIntervals = utils.get_start_stop_intervals(accessMask, self.sat.timesAll)


        # Get access times given access Intervals
        intervalLengths = []
        if (accessIntervals[0][0] is not None) and (accessIntervals[0][1] is not None):
            for interval in accessIntervals:
                startTime = interval[0]
                stopTime = interval[1]

                intervalLength = stopTime - startTime
                intervalLengths.append(intervalLength.to(u.s))


        self.accessIntervals = accessIntervals
        self.accessMask = accessMask
        self.accessIntervalLengths = intervalLengths
        self.accessElevations = ele
        
    def plot_tombstone(self):
        """
        plots tombstone plot


        """
        if not hasattr(self, 'accessIntervals'):
            print("data not processed yet\n"
                  "running self.process_data()")
            return

        allTimes = self.sat.timesAll
        if isinstance(allTimes, np.ndarray):
            timePlot = [t.datetime for t in allTimes]
        else: 
            timePlot = self.sat.timesAll.datetime

        fig, ax = plt.subplots()
        # fig.set_figwidth(15)
        fig.suptitle(f'Access for Satellite {self.satID}\n and GS {self.groundLocID}')
        ax.plot(timePlot, self.accessMask)
        ax.set_xlabel('Date Time')
        ax.set_ylabel('1 if access')
        ax.set_yticks([0,1])
        fig.autofmt_xdate()
        fig.show()


    # def process_data_ground_range(self, method="vector", re=constants.R_earth, fastRun=True):
    #     """
    #     Process access data using ground range method. Will be deprecated soon.
    #     """

    #     satCoords = self.sat.rvECI
    #     tofs = self.sat.rvTimeDeltas
    #     absTime= self.sat.epoch + tofs  # absolute time from time deltas

    #     satECI = GCRS(satCoords.x, satCoords.y, satCoords.z, representation_type="cartesian", obstime = absTime)
    #     satECISky = SkyCoord(satECI)
    #     satECEF = satECISky.transform_to(ITRS)
    #     # Turn coordinates into an EarthLocation object
    #     satEL = EarthLocation.from_geocentric(satECEF.x, satECEF.y, satECEF.z)

    #     # Convert to LLA
    #     lla_sat= satEL.to_geodetic()  # to LLA

    #     # Calculate ground range (great circle arc) between the satellite nadir
    #     # point and the ground location
    #     groundRanges = utils.ground_range_spherical(lla_sat.lat, lla_sat.lon, self.groundLoc.lat, self.groundLoc.lon)

    #     # Calculate the max ground range given the satellite sensor FOV
    #     if fastRun:
    #         sat_h_ave = lla_sat.height.mean()
    #         lam_min_all, lam_max_all = min_and_max_ground_range(sat_h_ave, self.sat.remoteSensor[0].fov, 0*u.deg, re)

    #         access_mask = groundRanges < abs(lam_max_all)
    #     else:
    #         lam_min_all = []
    #         lam_max_all = []
    #         for height in lla_sat.height:
    #             lam_min, lam_max = min_and_max_ground_range(height, self.sat.remoteSensor[0].fov, 0*u.deg, re)
    #             lam_min_all.append(lam_min)
    #             lam_max_all.append(lam_max)

    #     accessIntervals = utils.get_start_stop_intervals(access_mask, absTime)
    #     self.accessIntervals = accessIntervals
    #     self.accessMask = access_mask
    #     self.lam_max = lam_max_all
    #     self.groundRanges = groundRanges

    # def plot_access(self, absolute_time=True):
    #     """
    #     plots access

    #     Args:
    #         absolute_time (Bool) : If true, plots time in UTC, else plots in relative time (i.e. from sim start)
    #     """
    #     if not hasattr(self, 'accessIntervals'):
    #         print("data not processed yet\n"
    #               "running self.process_data()")
    #         self.process_data()

    #     if absolute_time:
    #         timePlot = self.sat.rvTimes.datetime
    #     else:
    #         timePlot = self.sat.rvTimeDeltas.to_value('sec')

    #     fig, ax = plt.subplots()
    #     fig.suptitle(f'Access for Satellite {self.satID}\n and GS {self.groundLocID}')
    #     ax.plot(timePlot, self.accessMask)
    #     ax.set_xlabel('Date Time')
    #     ax.set_ylabel('1 if access')
    #     fig.autofmt_xdate()
    #     fig.show()

class DataAccessConstellation():
    """
    access data for a constellation objects

    INTENDED to be run with constellation objects
    """
    def __init__(self, simConstellation, groundLoc):
        """
        Parameters
        ----------
        simConstellation: ~satbox.SimConstellation
            SimConstellation object that has been propagated
        groundLoc: ~satbox.GroundLoc | can be list
            Ground Location object, or list of GroundLocation objects
        """
        if simConstellation.propagated == 0:
            print("run simConstellation.propagate() first")
            return

        self.constellation = simConstellation
        self.groundLoc = groundLoc

        #Extract IDs for easy calling
        # self.groundLocID = groundLoc.groundID
        # self.groundIdentifier = groundLoc.identifier
    

    def calc_access(self, constraint_type, constraint_angle):
        """
        Calculate access between each satellite in a constellation and a ground station given a 
        constraint type and constraint angle

        Warning: Nadir constraint angle is the half-angle associated with the FOV, so a 
        camera with 90 deg of FOV, would be equivalent to a constraint_angle of 45 deg

        Parameters:
        -----------
        constraint_type: ~string | "nadir" or "elevation"
            constrain angles with either a "nadir" (usually analagous to sensor FOV) or "elevation" (minimum elevation angle from ground station) constraint
        constraint_angle: ~astropy.unit.Quantity
            angle used as the access threshold to determine access calculation
        """

        sats = self.constellation.get_propagated_sats()

        allAccessData = []

        for sat in sats:
            if isinstance(self.groundLoc, list):
                for groundLoc in self.groundLoc:
                    dataAccess = DataAccessSat(sat, groundLoc)
                    dataAccess.calc_access(constraint_type, constraint_angle)
                    allAccessData.append(dataAccess)
            else:
                dataAccess = DataAccessSat(sat, self.groundLoc)
                dataAccess.calc_access(constraint_type, constraint_angle)
                allAccessData.append(dataAccess)

        self.allAccessData = allAccessData

    def plot_total_access(self, gLocs, plot_style = 'b-'):
        """
        plots total coverage (satellite agnostic) for a particular
        groundLocation (gLocs) or a list of gLocs

        Args:
            gLocs [list] : list of groundLocation IDs to plot
        """

        if not isinstance(gLocs, list):
            gLocs= [gLocs]  # Turn into list

        # extract accessMasks into a list
        accessMasks = [data.accessMask for data in self.allAccessData if data.groundLocID in gLocs]
        totalAccess = [any(t) for t in zip(*accessMasks)]

        if not any(totalAccess):
            percCoverage = 0
            print('No Coverage')
            return
        else:
            percCoverage = sum(totalAccess) / len(totalAccess) * 100

        #Choose time scale to be first satellite
        allTimes = self.allAccessData[0].sat.timesAll
        if isinstance(allTimes, np.ndarray):
            timePlot = [t.datetime for t in allTimes]
        else: 
            timePlot = allTimes.datetime

        fig, ax = plt.subplots()
        ax.plot(timePlot, totalAccess, plot_style)
        ax.set_xlabel('Date Time')
        ax.set_yticks([])
        fig.autofmt_xdate()
        fig.suptitle(f'Total access for Ground Location(s): {gLocs}\n'
                        f'Coverage Percentage: {percCoverage:.1f} %')
        fig.supylabel('Access')
        return ax, fig

    def plot_all(self, legend=False):
        """
        Plots all access combinations of satellites and ground stations

        Args:
            absolute_time (Bool) : If true, plots time in UTC, else plots in relative time (i.e. from sim start)
            legend (Bool): Plot legend if true
        """

        numPlots = len(self.allAccessData)
        fig = plt.figure()
        gs= fig.add_gridspec(numPlots, hspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        fig.suptitle('Tombstone plots for satellite access')
        fig.supylabel('Access')

        for accessIdx, access in enumerate(self.allAccessData):
            allTimes = self.allAccessData[accessIdx].sat.timesAll
            if isinstance(allTimes, np.ndarray):
                timePlot = [t.datetime for t in allTimes]
            else: 
                timePlot = allTimes.datetime
            ax = axs[accessIdx]
            lab = f'Sat: {access.satID} | GS: {access.groundLocID}'
            ax.plot(timePlot, access.accessMask, label=lab)

            ax.set_xlabel('Date Time')
            ax.set_yticks([])
            ylab= ax.set_ylabel(lab)  # makes y label horizontal
            ylab.set_rotation(0)
            if legend:
                ax.legend()

            fig.autofmt_xdate()
            # fig.grid()

            # fig.show()
        plt.tight_layout()

        return axs, fig

    def plot_some(self, sats, gLocs, legend=False):
        """
        plots a selection of sats and ground locations
        Args:
            sats [list] : list of satellite IDs to plot
            gLocs [list] : list of groundLocation IDs to plot
            legend (Bool): Plot legend if true
        """

        if not isinstance(sats, list):
            sats = [sats]
        if not isinstance(gLocs, list):
            gLocs = [gLocs]

        numSats = len(sats)
        numGs = len(gLocs)
        numTot = numSats * numGs

        fig = plt.figure()
        gs= fig.add_gridspec(numTot, hspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        fig.suptitle('Tombstone plots for satellite access')
        fig.supylabel('Access')

        accessCounter = 0
        for accessIdx, access in enumerate(self.allAccessData):
            if access.satID not in sats or access.groundLocID not in gLocs:
                pass
            else:
                allTimes = self.allAccessData[accessIdx].sat.timesAll
                if isinstance(allTimes, np.ndarray):
                    timePlot = [t.datetime for t in allTimes]
                else: 
                    timePlot = allTimes.datetime

                if numTot == 1:
                    ax = axs
                else:
                    ax = axs[accessCounter]

                lab = f'Sat: {access.satID} | GS: {access.groundLocID}'
                ax.plot(timePlot, access.accessMask, label=lab)

                ax.set_xlabel('Date Time')
                ax.set_yticks([])
                ylab = ax.set_ylabel(lab)
                ylab.set_rotation(0)  # makes y label horizontal
                if legend:
                    ax.legend()

                fig.autofmt_xdate()
                accessCounter += 1
        plt.tight_layout()

        return axs, fig

    def get_temp_resolution(self, gLocs, sats='all', length_threshold=0*u.s):
        """
        Gets the temporal resolution (observation cadence) of select sats
        
        Parameters
        ----------
        sats: [list]
            list of satellite IDs to plot. If 'all', entire constellation taken into account
        gLocs: [list]
            list of groundLocation IDs to plot
        length_threshold: astropy.unit.Quantity
            threshold where any interval below this value is not considered in analysis

        Returns
        -------
        dataOut: dict
            Dictionary of times [startSorted and endSorted] and time values [startResolutions and endResolutions] between starts and ends of passes
        """
        if not isinstance(sats, list):
            sats = [sats]
        if not isinstance(gLocs, list):
            gLocs = [gLocs]

        allIntervals = []

        if 'all' in sats:
            for accessIdx, access in enumerate(self.allAccessData):


            
                #Find applicable intervals

                newArray = [q.to(u.s).value for q in access.accessIntervalLengths]
                newArrayQuantity = np.array(newArray) * u.s

                newArrayLengths = [q.to(u.s).value for q in access.accessIntervalLengths]

                goodIntervalIdx = newArrayQuantity > length_threshold
                goodIntervals = access.accessIntervals[goodIntervalIdx]
                goodIntervalLengths = newArrayQuantity[goodIntervalIdx]

                allIntervals.extend(goodIntervals)

                # intDiffs = np.diff(access.accessIntervals, axis=0)    
        else:
            for accessIdx, access in enumerate(self.allAccessData):
                if access.satID in sats:
                    #Find applicable intervals

                    newArray = [q.to(u.s).value for q in access.accessIntervalLengths]
                    newArrayQuantity = np.array(newArray) * u.s

                    newArrayLengths = [q.to(u.s).value for q in access.accessIntervalLengths]

                    goodIntervalIdx = newArrayQuantity > length_threshold
                    goodIntervals = access.accessIntervals[goodIntervalIdx]
                    goodIntervalLengths = newArrayQuantity[goodIntervalIdx]

                    allIntervals.extend(goodIntervals)

        startTimes = np.array([k[0] for k in allIntervals])
        endTimes = np.array([k[1] for k in allIntervals])

        endIdxSort = np.argsort(endTimes)
        startIdxSort = np.argsort(startTimes)

        startSorted = startTimes[startIdxSort]
        endSorted = endTimes[endIdxSort]

        startResolutions = np.diff(startSorted)
        endResolutions = np.diff(endSorted)

        dataOut = {
                    'startSorted': startSorted,
                    'endSorted' : endSorted,
                    'startResolutions' : startResolutions,
                    'endResolutions' : endResolutions
        }

        return dataOut




