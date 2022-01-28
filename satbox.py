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
from copy import deepcopy

import dill

# Constellation Class


class Constellation():
    """
    Defines the Constellation class that holds a set of orbital planes
    """

    def __init__(self, planes):
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

    # def plan_reconfigure(self, GroundLoc, GroundStation, tInit, days, figName=None, selectionMethod=1, selectionParams=None, plot = False, savePlot = False):
        """
        Plane reconfiguration of the constellation to visit and image a GroundLoc and downlink to a GroundStation

        Args:
            GroundLoc (GroundLoc class): Ground location to image
            GroundStation (GroundStation/GroundLoc class): Ground station to downlink data to
            tInit (astropy time object): Time to initialize planner
            days (int): Amount of days ahead to for the scheduler to plan for | TO DO: Determine if this is actually necessary in code
            figName (str): figname to save plots to
            selectionMethod (int): Determines how satellites are selected
            selectionParams (list of 2 element lists): Need 4 different weights, 1 to select imaging satellites, 2 to select ISL satellites, 3 to select downlink satellites, 4 to select missiions.
                    Elements 5,6,7,8 are the number of selected satellites to move into the next round of the selection cycle.
                    Example: [[1,1],[1,3],[1,5],[1,8] 5, 5, 5, 6]
            plot (Bool): Outputs plots if true
            savePlot (Bool): Save plot outputs

        Returns:
            List: Suggested missions
        """
        if selectionMethod == 1:
            imageWeights = selectionParams[0]
            ISLWeights = selectionParams[1]
            DLWeights = selectionParams[2]
            missionWeights = selectionParams[3]
            nSatImage = selectionParams[4]
            nSatISL = selectionParams[5]
            nSatDL = selectionParams[6]
            nSatMission = selectionParams[7]

            # Maybe get rid of ghostSatsInit and ghostSatsPass as well
            imageManeuvers, paretoSats, ghostSatsInit, ghostSatsPass, rmc = self.get_pass_maneuver(GroundLoc,
                                                                                                        tInit,
                                                                                                        days,
                                                                                                        task='Image',
                                                                                                        plot=plot,
                                                                                                        savePlot=True,
                                                                                                        figName=f'figures/{figName}ImagePass.png')

            imageManeuvers_flat = utils.flatten(imageManeuvers)

            [obj.get_weighted_dist_from_utopia(
                imageWeights[0], imageWeights[1]) for obj in imageManeuvers_flat]  # Gets utopia distance
            imageManeuvers_flat.sort(key=lambda x: x.utopDistWeighted)
            selectWeightedManeuversImage = imageManeuvers_flat[0:nSatImage]

            if plot:
                tMaxPlot = (days + 1) * u.day
                xlimits = [-5, tMaxPlot.to(u.hr).value]
                ylimits = [-5, 500]
                plt.figure(constrained_layout=True)
                plt.xlabel('Time (hrs)', fontsize=14)
                plt.ylabel('Delta V (m/s)', fontsize=14)
                plt.title(
                    'Potential Maneuver Options\n \'o\' for ascending, \'x\' for descending passes', fontsize=16)
                lgdCheck = []
                for man in imageManeuvers_flat:
                #     clrIdx = int(man.planeID)
                    # passType = man.note
                    # if passType =='a':
                    #   style = 'ko'
                    # elif passType == 'd':
                    #   style = 'kx'
                    plt.plot(man.time2Pass.to(u.hr), man.deltaV, 'ko')  # ,
                        # label = label)
                selectTimesW = [p.time2Pass.to(
                    u.hr).value for p in selectWeightedManeuversImage]
                selectDelVW = [
                    p.deltaV.value for p in selectWeightedManeuversImage]
                # plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
                # plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
                plt.plot(selectTimesW, selectDelVW, 'b^',
                         label='Selected Sats Weighted', markersize=10)

                plt.xlim(xlimits)
                plt.ylim(ylimits)
                plt.plot(0, 0, 'g*', label='Utopia Point', markersize=20)
                plt.legend()
                if savePlot:
                    from datetime import datetime
                    now = datetime.now()
                    timestamp = now.isoformat()
                    fname = "figures/pool/" + \
                        str(timestamp) + "imagingMans"+".png"
                    plt.savefig(fname, facecolor="w", dpi=300)

            # Create new constellation objects
            origSats = self.get_sats()  # Original satellites
            numOriginalPlanes = len(self.planes)

            selectSats = [s.mySat for s in selectWeightedManeuversImage]

            # Create constellation of potential relay satellites
            potentialRelaySats = [
                sat for sat in origSats if sat not in selectSats]
            relayPlanes = []
            for planeIdx in range(0, numOriginalPlanes):
                sats = [s for s in potentialRelaySats if s.planeID == planeIdx]
                plane2append = Plane.from_list(sats)
                relayPlanes.append(plane2append)
            potentialRelaySatsConstellation = Constellation.from_list(
                relayPlanes)

            # Create constellation of ghost satellites that will make the image pass
            ghostImagingSats = [
                s.mySatFin for s in selectWeightedManeuversImage]
            desiredPassSatsPlane = []
            for planeIdx in range(0, numOriginalPlanes):
                sats = [s for s in ghostImagingSats if s.planeID == planeIdx]
                plane2append = Plane.from_list(sats)
                desiredPassSatsPlane.append(plane2append)
            ghostImagingSatsConstellation = Constellation.from_list(
                desiredPassSatsPlane)

            # Might not need a lot of these outputs. TO DO eliminate needless outputs
            maneuverObjs, transfersISL, timeOfISL, paretoSatsISL, missionOptions = potentialRelaySatsConstellation.get_ISL_maneuver(ghostImagingSatsConstellation,
                                                                                                                            perturbation='none',
                                                                                                                            plotFlag=plot)

            # Weighted Choice
            [obj.get_weighted_dist_from_utopia(
                ISLWeights[0], ISLWeights[1]) for obj in transfersISL]  # Gets utopia distance
            transfersISL.sort(key=lambda x: x.utopDistWeighted)
            selectWeightedManeuversISL = transfersISL[0:nSatISL]

            if plot:
                tMaxPlot = (days + 1) * u.day
                xlimits = [-5, tMaxPlot.to(u.hr).value]
                plt.figure(constrained_layout=True)
                plt.xlabel('Time (hrs)', fontsize=14)
                plt.ylabel('Delta V (m/s)', fontsize=14)
                plt.title(
                    'Potential ISL Options\n \'o\' for ascending, \'x\' for descending passes', fontsize=16)
                lgdCheck = []
                for man in transfersISL:
                #     clrIdx = int(man.planeID)
                    # passType = man.note
                    # if passType =='a':
                    #   style = 'ko'
                    # elif passType == 'd':
                    #   style = 'kx'
                    plt.plot(man.time2Pass.to(u.hr), man.deltaV, 'ko')  # ,
                        # label = label)
                selectTimesW = [p.time2Pass.to(
                    u.hr).value for p in selectWeightedManeuversISL]
                selectDelVW = [
                    p.deltaV.value for p in selectWeightedManeuversISL]
                # plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
                # plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
                plt.plot(selectTimesW, selectDelVW, 'b^',
                         label='Selected Sats Weighted', markersize=10)

                plt.xlim(xlimits)
                plt.ylim(ylimits)
                plt.plot(0, 0, 'g*', label='Utopia Point', markersize=20)
                plt.legend()

                if savePlot:
                    fname = "figures/pool/" + \
                        str(timestamp) + "ISLMans" + ".png"
                    plt.savefig(fname, facecolor="w", dpi=300)

            # Might need to uncomment this
            # for s in selectWeightedManeuversISL:
            #     s.mySat.task = 'ISL'

            for s in selectWeightedManeuversISL:
                # Adds target satellite(ground image) to previous interactions with the downlink satellite
                s.mySatFin.add_previousSatInteraction(s.target)

            # Create constellation of ghost ISL satellites
            # Final orbital config
            ghostISLSatsWeighted = [
                s.mySatFin for s in selectWeightedManeuversISL]
            selectPlanes_f = []
            for planeIdx in range(0, numOriginalPlanes):
                sats_f = [
                    s for s in ghostISLSatsWeighted if s.planeID == planeIdx]
                plane2append_f = Plane.from_list(sats_f, planeID=planeIdx)
                selectPlanes_f.append(plane2append_f)
            selectSatsISL_f = Constellation.from_list(
                selectPlanes_f)  # Satellite orbits after ISL maneuver

            # Find downlink capable satellites
            downlinkManeuvers, pSats_GP, ghostSatsInit_GP, ghostSatsPass_GP, runningMissionCost = selectSatsISL_f.get_pass_maneuver(GroundStation,
                                                                                                                                tInit,
                                                                                                                                days,
                                                                                                                                useSatEpoch=True,
                                                                                                                                task='Downlink',
                                                                                                                                plot=True,
                                                                                                                                savePlot=plot,
                                                                                                                                figName=f'figures/{figName}groundPassPlanner.png')

            downlinkManeuvers_flat = utils.flatten(downlinkManeuvers)

            [obj.get_weighted_dist_from_utopia(
                DLWeights[0], DLWeights[1]) for obj in downlinkManeuvers_flat]  # Gets utopia distance
            downlinkManeuvers_flat.sort(key=lambda x: x.utopDistWeighted)
            selectWeightedDownlinkManeuvers = downlinkManeuvers_flat[0:nSatMission]

            if plot:
                tMaxPlot = (days + 1) * u.day
                xlimits = [-5, tMaxPlot.to(u.hr).value]
                plt.figure(constrained_layout=True)
                plt.xlabel('Time (hrs)', fontsize=14)
                plt.ylabel('Delta V (m/s)', fontsize=14)
                plt.title(
                    'Potential Downlink Options\n \'o\' for ascending, \'x\' for descending passes', fontsize=16)
                lgdCheck = []
                for man in downlinkManeuvers_flat:
                #     clrIdx = int(man.planeID)
                    # passType = man.note
                    # if passType =='a':
                    #   style = 'ko'
                    # elif passType == 'd':
                    #   style = 'kx'
                    plt.plot(man.time2Pass.to(u.hr), man.deltaV, 'ko')  # ,
                        # label = label)
                selectTimesW = [p.time2Pass.to(
                    u.hr).value for p in selectWeightedDownlinkManeuvers]
                selectDelVW = [
                    p.deltaV.value for p in selectWeightedDownlinkManeuvers]
                # plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
                # plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
                plt.plot(selectTimesW, selectDelVW, 'b^',
                         label='Selected Sats Weighted', markersize=10)

                plt.xlim(xlimits)
                plt.ylim(ylimits)
                plt.plot(0, 0, 'g*', label='Utopia Point', markersize=20)
                plt.legend()
                if savePlot:
                    fname = "figures/pool/" + \
                        str(timestamp) + "DLMans" + ".png"
                    plt.savefig(fname, facecolor="w", dpi=300)

            # Find best overall mission
            missionCosts = utils.flatten(runningMissionCost)
            [obj.get_weighted_dist_from_utopia(
                missionWeights[0], missionWeights[1]) for obj in missionCosts]  # Gets utopia distance
            missionCosts.sort(key=lambda x: x.utopDistWeighted)
            selectMissionOptions = missionCosts[0:nSatMission]

            if plot:
                tMaxPlot = (days + 1) * u.day
                xlimits = [-5, tMaxPlot.to(u.hr).value]
                plt.figure(constrained_layout=True)
                plt.xlabel('Time (hrs)', fontsize=14)
                plt.ylabel('Delta V (m/s)', fontsize=14)
                plt.title(
                    'Potential Mission Options\n \'o\' for ascending, \'x\' for descending passes', fontsize=16)
                lgdCheck = []
                for man in missionCosts:
                #     clrIdx = int(man.planeID)
                    # passType = man.note
                    # if passType =='a':
                    #   style = 'ko'
                    # elif passType == 'd':
                    #   style = 'kx'
                    plt.plot(man.dlTime.to(u.hr), man.totalCost, 'ko')  # ,
                        # label = label)
                selectTimesW = [p.dlTime.to(
                    u.hr).value for p in selectMissionOptions]
                selectDelVW = [p.totalCost.value for p in selectMissionOptions]
                # plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
                # plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
                plt.plot(selectTimesW, selectDelVW, 'b^',
                         label='Selected Sats Weighted', markersize=10)

                plt.xlim(xlimits)
                plt.ylim(ylimits)
                plt.plot(0, 0, 'g*', label='Utopia Point', markersize=20)
                plt.legend()
                if savePlot:
                    fname = "figures/pool/" + \
                        str(timestamp) + "missionMans" + ".png"
                    plt.savefig(fname, facecolor="w", dpi=300)

            return selectMissionOptions

    # def get_pass_maneuver(self, GroundLoc, tInit, days, useSatEpoch = False, task = None, plot = False, savePlot = False, figName = None):
        """
        Gets set a maneuvers that will pass a specific ground location on
        select days

        Args:
            GroundLoc (Obj): Ground location object to be passed over
            tInit (astropy time object): Time to initialize planner
            days (int): Amount of days ahead to for the scheduler to plan for
            useSatEpoch (Bool): If true, uses individual satellite current epoch as tInit. Useful for downklink portion of planner
            task (str): The assigned task for the desired satellite. Options: 'Image', 'ISL', 'Downlink'
            plot (Bool): Plots maneuvers if True
            savePlot(Bool): Saves plot if True
            figName (str): file name for figure


        Returns:

            maneuverPoolCut (list of Maneuver objects): Physically viable maneuvers (selected because some maneuvers would cause satellites to crash into Earth),

            maneuverPoolAll (list of Maneuver objects): All potential maneuvers,

            paretoSats (list of pareto front satellites): Satellite objects on the pareto front,

            ghostSatsInitAll (array of Satellite objects): Orbit of ghost satellite if it were propagated back to initial time,

            ghostSatsPassAll(array of Satellite objects): Orbit of satellite at ground pass (potential position aka ghost position)

        """
        if savePlot:
            assert figName, "Need to name your figure using figName input"

        maneuverPoolAll = []
        maneuverPoolCut = []
        ghostSatsPassAll = []
        ghostSatsInitAll = []
        missionCosts = []
        for plane in self.planes:
            if not plane:  # Check if plane is empty
                continue
            maneuverPlaneCut, maneuverPlaneAll, ghostSatsPass, ghostSatsInit, runningMissionCost = plane.get_pass_details(
                GroundLoc, tInit, days, useSatEpoch, task)
            maneuverPoolAll.append(maneuverPlaneAll)
            maneuverPoolCut.append(maneuverPlaneCut)
            ghostSatsPassAll.append(ghostSatsPass)
            ghostSatsInitAll.append(ghostSatsInit)
            missionCosts.append(runningMissionCost)

        # Flatten list
        semiFlatCut = [item for sublist in maneuverPoolCut for item in sublist]
        flatCut = [item for sublist in semiFlatCut for item in sublist]

        paretoSats = utils.find_non_dominated_time_deltaV(flatCut)
        ax = None
        if plot:

            colors = sns.color_palette('tab10', len(self.planes))
            sns.set_style('whitegrid')
            style = ['o', 'x']
            # fig = plt.figure(constrained_layout=True)
            fig, ax = plt.subplots(constrained_layout=True)
            plt.xlabel('Time (hrs)', fontsize=14)
            plt.ylabel('Delta V (m/s)', fontsize=14)
            plt.title(
                'Potential Maneuver Options\n \'o\' for ascending, \'x\' for descending passes', fontsize=16)

            lgdCheck = []
            for man in flatCut:

                clrIdx = int(man.planeID)
                passType = man.note
                if passType == 'a':
                    style = 'o'
                elif passType == 'd':
                    style = 'x'
                else:
                    style = 'o'

                if clrIdx in lgdCheck:
                    label = None
                else:
                    label = f'{clrIdx}'
                    lgdCheck.append(clrIdx)
                plt.plot(man.time2Pass.to(u.hr), man.deltaV, style, color=colors[clrIdx],
                        label=label)
            paretoTimes = [p.time2Pass.to(u.hr).value for p in paretoSats]
            paretoDelV = [p.deltaV.value for p in paretoSats]
            plt.plot(paretoTimes, paretoDelV, 'r-', label='Pareto Front')

            plt.plot(0, 0, 'g*', label='Utopia Point', markersize=20)
            plt.legend(title='Plane Number')
            # plt.grid()
            if savePlot:
                plt.savefig(figName, facecolor="w",
                            dpi=300, bbox_inches='tight')
            # plt.tight_layout()
            plt.show()

        return maneuverPoolCut, paretoSats, ghostSatsInitAll, ghostSatsPassAll, missionCosts

    # def get_ISL_maneuver(self, Constellation, perturbation='J2', plotFlag=False, savePlot = False, figName = None):
        """
        Calculate potential ISL maneuvers between all satellites in current constellation with
        all satellites of another satellite

        Args:
            self (Constellation object): Constellation making the maneuvers
            Constellation (Constellation object): Constellation to intercept
            perturbation (string): 'J2' for J2 perturbation, else 'none'
            plotFlag (boolean): plot if True
            savePlot(Bool): Saves plot if True
            figName (str): file name for figure

        Returns:
            maneuverObjs [list] : list of maneuver objects to create ISL opportunity

            deltaVs [list] : list of total deltaVs

            timeOfISL [list] : list of times that ISL opportunity will happen

            paretoSats [list] : list of satellites on the pareto front
        """
        if savePlot:
            assert figName, "Need to name your figure using figName input"

        selfSats = self.get_sats()
        targetSats = Constellation.get_sats()
        tInit = selfSats[0].epoch  # Get time at beginnnig of simulation

        maneuverObjs = []
        transfers = []
        timeOfISL = []
        islOrbs = []
        missionOptions = []
        for satInt in selfSats:
            for satTar in targetSats:
                output = satInt.schedule_coplanar_intercept(
                    satTar, perturbation=perturbation)
                maneuverObjs.append(output['maneuverObjects'])
                transfers.append(output['transfer'])
                timeOfISL.append(output['islTime'])
                missionDict = {
                                'imagingSatGhost': satTar.previousTransfer,
                                'ISLSatGhost': output['transfer']
                }
                missionOptions.append(missionDict)

        paretoSats = utils.find_non_dominated_time_deltaV(transfers)

        if plotFlag:

            colors = sns.color_palette('tab10', len(self.planes))
            sns.set_style('whitegrid')
            # style = ['o', 'x']
            plt.figure(constrained_layout=True)
            plt.xlabel('Time (hrs)', fontsize=14)
            plt.ylabel('Delta V (m/s)', fontsize=14)
            plt.title(
                'Potential Maneuver Options\n for ISL data transfer', fontsize=16)

            lgdCheck = []
            paretoTimes = []
            for man in transfers:
                clrIdx = int(man.planeID)

                if clrIdx in lgdCheck:
                    label = None
                else:
                    label = f'{clrIdx}'
                    lgdCheck.append(clrIdx)
                timeAtPass = man.time - tInit  # man.time gets time at ISL, tInit is the current time
                plt.plot(timeAtPass.to(u.hr), man.deltaV, 'o', color=colors[clrIdx],
                        label=label)
            paretoTimes = [(p.time-tInit).to(u.hr).value for p in paretoSats]
            paretoDelV = [p.deltaV.value for p in paretoSats]
            plt.plot(paretoTimes, paretoDelV, 'r-', label='Pareto Front')

            plt.plot(0, 0, 'g*', label='Utopia Point', markersize=20)
            plt.legend(title='Plane Number')
            # plt.grid()
            if savePlot:
                plt.savefig(figName, facecolor="w",
                            dpi=300, bbox_inches='tight')
            plt.show()

        return maneuverObjs, transfers, timeOfISL, paretoSats, missionOptions

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
                satProp=sat.propagate(time)
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
        Gets a list of all the satetllite objects in the constellation
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

    # def get_pass_details(self, GroundLoc, tInit, days, useSatEpoch, task):
        """
        Gets set a maneuvers that will pass a specific ground location on
        select days

        Args:
            GroundLoc (Obj): Ground location object to be passed over
            tInit (astropy time object): Time to initialize planner
            days (int): Amount of days ahead to for the scheduler to plan for
            useSatEpoch (Bool): If true, uses individual satellite current epoch as tInit. Useful for downklink portion of planner
            task (string): The assigned task for the desired satellite. Options: 'Image', 'ISL', 'Downlink'

        Returns:
            maneuverListAll (list of Maneuver objects): All potential maneuvers

            maneuverListCut (list of Maneuver objects): Physically viable maneuvers (selected because some maneuvers would cause satellites to crash into Earth)
        """
        maneuverListAll = []
        maneuverListCut = []
        ghostSatsPassAll = []
        ghostSatsInitAll = []
        missionCostsAll = []
        for sat in self.sats:
            ghostSatsPass, ghostSatsInit, potentialManeuversAll, potentialManeuversCut, runningMissionCost = sat.get_desired_pass_orbs(GroundLoc, tInit, days, useSatEpoch, task=task)
            for man in potentialManeuversAll:
                man.planeID = self.planeID
            for man in potentialManeuversCut:
                man.planeID = self.planeID
            for sat in ghostSatsPass:
                sat.planeID = self.planeID
            for sat in ghostSatsInit:
                sat.planeID = self.planeID
            maneuverListAll.append(potentialManeuversAll)
            maneuverListCut.append(potentialManeuversCut)
            ghostSatsPassAll.append(ghostSatsPass)
            ghostSatsInitAll.append(ghostSatsInit)
            missionCostsAll.append(runningMissionCost)
        return maneuverListCut, maneuverListAll, ghostSatsPassAll, ghostSatsInitAll, missionCostsAll
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

# Satellite class
class Satellite(Orbit):
    """
    Defines the Satellite class. Inherits from Poliastro Orbit class.
    The Satellite class is an initializer class.
    """

    def __init__(self, state, epoch, satID=None, schedule=None,
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

        if schedule is None:  # Set up ability to hold a schedule
            self.schedule = []
        else:
            self.schedule = schedule

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

    def add_schedule(self, sch_commands):  # Adds a schedule to the satellite
        self.schedule.append(sch_commands)

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

    def reset_schedule(self):
        self.schedule = []

    def __eq__(self, other):
        """Check for equality with another object"""
        return self.__dict__ == other.__dict__

# SimSatellite class
class SimSatellite():
    """
    Defines the SimSatellite class. This object represents the satellite
    as it is propagated in the simulation
    """

    def __init__(self, satellite, t2propagate, tStep, manSched = None):
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
        self.maneuverSchedule = manSched

        #TimeDelta objects used to propagate satellite
        self.timeDeltas = TimeDelta(np.arange(0, 
                                              t2propagate.to(u.s).value,
                                              tStep.to(u.s).value) * u.s)

        #Times in UTC (default)
        self.times = satellite.epoch + self.timeDeltas

    def run_sim(self, method="J2"):
        """
        Run simulator

        Parameters
        ----------
        method: str ("J2")
            J2 to propagate using J2 perturbations
        """

        #If not maneuver schedule
        if (self.maneuverSchedule is None): 
            if method == "J2":
                def f(t0, state, k): #Define J2 perturbation
                    du_kep = func_twobody(t0, state, k)
                    ax, ay, az = J2_perturbation(
                        t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
                    )
                    du_ad = np.array([0, 0, 0, ax, ay, az])

                    return du_kep + du_ad
                coords = propagate(
                    self.initSat,
                    self.timeDeltas,
                    method=cowell,
                    f=f,
                    )
            else:
                coords = propagate(
                    self.initSat,
                    timeDeltas,
                    )

        satECI = GCRS(coords.x, coords.y, coords.z, representation_type="cartesian", obstime = self.times)
        satECISky = SkyCoord(satECI)
        satECEF = satECISky.transform_to(ITRS)
        ## Turn coordinates into an EarthLocation object
        satEL = EarthLocation.from_geocentric(satECEF.x, satECEF.y, satECEF.z)
        ## Convert to LLA
        lla_sat = satEL.to_geodetic() #to LLA
        self.rvECI = coords #ECI coordinates
        self.LLA = lla_sat #Lat long alt of satellite
        self.rvECEF = satECEF #ECEF coordinates


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
        assert isinstance(deltaV[0], astropy.units.quantity.Quantity), ('0 index deltaV value' 
                                         ' must be astropy.units.quantity.Quantity')
        assert isinstance(deltaV[1], astropy.units.quantity.Quantity), ('1 index deltaV value' 
                                         ' must be astropy.units.quantity.Quantity')
        assert isinstance(deltaV[2], astropy.units.quantity.Quantity), ('2 index deltaV value' 
                                         ' must be astropy.units.quantity.Quantity')
        self.time = time
        self.deltaV = deltaV
        self.satID = satID
        self.note = note

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
