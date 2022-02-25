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
# import cartopy.crs as ccrs

import seaborn as sns
import astropy.units as u
import astropy
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, GCRS, ITRS, CartesianRepresentation, SkyCoord
# import OpticalLinkBudget.OLBtools as olb
import utils as utils
import orbitalMechanics as om
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
        # assert self.planes[0].sats[0].rvECI, "Run self.get_rv_from_propagate first to get rv values"
        # import ipdb;ipdb.set_trace()
        if not hasattr(self.planes[0].sats[0], 'rvECI'):
            print("Run self.get_rv_from_propagate first to get rv values")
            return
        c=3e8 * u.m / u.s

        sats=self.get_sats()
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
                satRef_r=satRef.rvECI.without_differentials()
                satRef_v=satRef.rvECI.differentials

                # Comparison orbit RV values
                sat_r=sat.rvECI.without_differentials()
                sat_v=sat.rvECI.differentials

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
                            'timeDeltas': sat.rvTimeDeltas,
                            'times': sat.rvTimes,
                }

                dictKey=str(satRef.satID) + '-' + str(sat.satID)
                outputData['satData'][dictKey]=dictEntry
        return outputData

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

    # def plot2D(self, timeInts, pts):
    #     """
    #     Plots 2D ground tracks

    #     Args:
    #         timeInts [astropy object] : time intervals to propagate where 0 refers to the epoch of the orb input variable
    #         pts [integer] : Number of plot points


    #     """
    #     sats = self.get_sats()

    #     fig = plt.figure()
    #     ax = plt.axes(projection=ccrs.PlateCarree())
    #     ax.stock_img()

    #     plottedConstellationIds = [] #Constellation IDs that have been plotted

    #     cmap = ['k', 'b', 'g', 'p']
    #     cmapIdx = -1
    #     for sat in sats:
    #         lon, lat, h = utils.getLonLat(sat, timeInts, pts)
    #         if hasattr(sat, 'constellationID'):#sat.constellationID is not None:

    #             if sat.constellationID not in plottedConstellationIds:
    #                 plottedConstellationIds.append(sat.constellationID)
    #                 cmapIdx += 1
    #                 # print(cmapIdx)
    #                 # breakpoint()
    #             ax.plot(lon, lat, cmap[cmapIdx], transform=ccrs.Geodetic())
    #             ax.plot(lon[0], lat[0], 'r^', transform=ccrs.Geodetic())
    #         else:
    #             ax.plot(lon, lat, 'k', transform=ccrs.Geodetic())
    #             ax.plot(lon[0], lat[0], 'r^', transform=ccrs.Geodetic())
    #     return fig, ax


    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def generate_czml_file(self, fname, prop_duration, sample_points, scene3d=True,
                            specificSats=False):
        """
        Generates CZML file for the constellation for plotting

        Args:
            fname (string): File name (including path to file) for saved czml file. Currently plots in a directory czmlFiles
            prop_duration (astropy time): Time to propagate in simulation
            sample_points (int): Number of sample points
            scene3d (bool): Set to false for 2D plot
            specificSats (list of ints) : List of satIDs to plot
        """
        seedSat = self.get_sats()[0]
        start_epoch= seedSat.epoch  # iss.epoch

        end_epoch = start_epoch + prop_duration

        earth_uv = "https://earthobservatory.nasa.gov/ContentFeature/BlueMarble/Images/land_shallow_topo_2048.jpg"

        extractor = CZMLExtractor(start_epoch, end_epoch, sample_points,
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

    def propagate(self, method="J2"):
        """
        Propagate satellites in a constellation Simulator

        Parameters
        ----------
        method: str ("J2")
            J2 to propagate using J2 perturbations
        """
        planes2const = []
        for plane in self.initConstellation.planes:
            if not plane: #continue if empty
                continue

            planeSats = []
            for sat in plane.sats:
                satPropInit = SimSatellite(sat, self.t2propagate, self.tStep, verbose=self.verbose)
                satPropInit.propagate()
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

        #Times in UTC (default)
        self.times = satellite.epoch + self.timeDeltas

        #List to hold all satellite segments
        self.satSegments = []
        self.timeSegments = []
        self.cartesianRepSegments = []
        self.coordSegmentsECI = []
        self.coordSegmentsECEF = []
        self.coordSegmentsLLA = []

    def propagate(self, method="J2"):
        """
        Run simulator for satellite Simulator

        Parameters
        ----------
        method: str ("J2")
            J2 to propagate using J2 perturbations
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
        if (self.maneuverSchedule is None): 
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
    def __init__(self, Sat, groundLoc, fastRun):
        """
        Args:
            Sat (Satellite Object): Satellite object that has been propagated
            groundLoc (rroundLoc Object): Ground Location object
            fastRun (Boolean): True for a faster run that uses average altitude of satellite instead of using altitude of each time step
                                Assumes a circular orbit

        """
        self.sat = Sat
        self.groundLoc = groundLoc
        self.fastRun = fastRun

        # Extract IDs for easy calling
        self.satID = Sat.satID
        self.groundLocID = groundLoc.groundID
        self.groundIdentifier = groundLoc.identifier

    def process_data(self, re=constants.R_earth):
        satCoords = self.sat.rvECI
        tofs = self.sat.rvTimeDeltas
        absTime= self.sat.epoch + tofs  # absolute time from time deltas

        # breakpoint()
        satECI = GCRS(satCoords.x, satCoords.y, satCoords.z, representation_type="cartesian", obstime = absTime)
        satECISky = SkyCoord(satECI)
        satECEF = satECISky.transform_to(ITRS)
        # Turn coordinates into an EarthLocation object
        satEL = EarthLocation.from_geocentric(satECEF.x, satECEF.y, satECEF.z)

        # Turn coordinates into ecef frame
        # satECEF2 = satEL.get_itrs(self.sat.epoch + tofs) #ECEF frame

        # Convert to LLA
        lla_sat= satEL.to_geodetic()  # to LLA



        # Calculate ground range (great circle arc) between the satellite nadir
        # point and the ground location
        groundRanges = utils.ground_range_spherical(lla_sat.lat, lla_sat.lon, self.groundLoc.lat, self.groundLoc.lon)
        # plt.figure()
        # plt.plot(tofs.value, lla_sat.lat, '.', label='latitude')
        # plt.plot(tofs.value, lla_sat.lon, '.', label='longitude')
        # plt.ylabel('longitude')
        # plt.legend()

        # Calculate the max ground range given the satellite sensor FOV
        if self.fastRun:
            sat_h_ave = lla_sat.height.mean()
            lam_min_all, lam_max_all = min_and_max_ground_range(sat_h_ave, self.sat.remoteSensor[0].fov, 0*u.deg, re)

            access_mask = groundRanges < abs(lam_max_all)
        else:
            lam_min_all = []
            lam_max_all = []
            for height in lla_sat.height:
                lam_min, lam_max = min_and_max_ground_range(height, self.sat.remoteSensor[0].fov, 0*u.deg, re)
                lam_min_all.append(lam_min)
                lam_max_all.append(lam_max)
        # plt.figure()
        # plt.hlines(lam_max_all.value, tofs.value[0], tofs.value[-1])
        # plt.plot(tofs.value, groundRanges, label='ground range')

        accessIntervals = utils.get_start_stop_intervals(access_mask, absTime)
        self.accessIntervals = accessIntervals
        self.accessMask = access_mask
        self.lam_max = lam_max_all
        self.groundRanges = groundRanges

    def plot_access(self, absolute_time=True):
        """
        plots access

        Args:
            absolute_time (Bool) : If true, plots time in UTC, else plots in relative time (i.e. from sim start)
        """
        if not hasattr(self, 'accessIntervals'):
            print("data not processed yet\n"
                  "running self.process_data()")
            self.process_data()

        if absolute_time:
            timePlot = self.sat.rvTimes.datetime
        else:
            timePlot = self.sat.rvTimeDeltas.to_value('sec')

        fig, ax = plt.subplots()
        fig.suptitle(f'Access for Satellite {self.satID}\n and GS {self.groundLocID}')
        ax.plot(timePlot, self.accessMask)
        ax.set_xlabel('Date Time')
        ax.set_ylabel('1 if access')
        fig.autofmt_xdate()
        fig.show()

class DataAccessConstellation():
    """
    access data for a constellation objects

    INTENDED to be run with constellation objects
    """
    def __init__(self, accessList):
        self.accessList = accessList

    def plot_total_access(self, gLocs, absolute_time=True):
        """
        plots total coverage (satellite agnostic) for a particular
        groundLocation (gLocs) or a list of gLocs

        Args:
            gLocs [list] : list of groundLocation IDs to plot
            absolute_time (Bool) : If true, plots time in UTC, else plots in relative time (i.e. from sim start)
        """

        if not isinstance(gLocs, list):
            gLocs= [gLocs]  # Turn into list

        # extract accessMasks into a list
        accessMasks = [data.accessMask for data in self.accessList if data.groundLoc in gLocs]
        totalAccess = [any(t) for t in zip(*accessMasks)]

        if not any(totalAccess):
            percCoverage = 0
            print('No Coverage')
            return
        else:
            percCoverage = sum(totalAccess) / len(totalAccess) * 100

        if absolute_time:  # Choose time scale to be the first satellite
            timePlot = self.accessList[0].sat.rvTimes.datetime
        else:
            timePlot = self.accessList[0].to_value('sec')
        fig, ax = plt.subplots()
        ax.plot(timePlot, totalAccess)
        ax.set_xlabel('Date Time')
        ax.set_yticks([])
        fig.autofmt_xdate()
        fig.suptitle(f'Total access for Ground Location(s): {gLocs}\n'
                        f'Coverage Percentage: {percCoverage:.1f} %')
        fig.supylabel('Access')
        return ax, fig

    def plot_all(self, absolute_time=True, legend=False):
        """
        Plots all access combinations of satellites and ground stations

        Args:
            absolute_time (Bool) : If true, plots time in UTC, else plots in relative time (i.e. from sim start)
            legend (Bool): Plot legend if true
        """

        numPlots = len(self.accessList)
        fig = plt.figure()
        gs= fig.add_gridspec(numPlots, hspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        fig.suptitle('Tombstone plots for satellite access')
        # fig, axs = plt.subplot(numPlots, sharex=True, sharey=True)
        # axs[0].set_ylabel('1 if access')
        fig.supylabel('Access')

        for accessIdx, access in enumerate(self.accessList):
            if absolute_time:
                timePlot = access.sat.rvTimes.datetime
            else:
                timePlot = access.sat.rvTimeDeltas.to_value('sec')
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

    def plot_some(self, sats, gLocs, absolute_time=True, legend=False):
        """
        plots a selection of sats and ground locations
        Args:
            sats [list] : list of satellite IDs to plot
            gLocs [list] : list of groundLocation IDs to plot
            absolute_time (Bool) : If true, plots time in UTC, else plots in relative time (i.e. from sim start)
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
        for accessIdx, access in enumerate(self.accessList):
            if access.satID not in sats or access.groundLocID not in gLocs:
                pass
            else:
                if absolute_time:
                    timePlot = access.sat.rvTimes.datetime
                else:
                    timePlot = access.sat.rvTimeDeltas.to_value('sec')
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
