import os
import numpy as np
from poliastro import constants
from poliastro.earth import Orbit
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody.propagation import propagate, func_twobody
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation
from poliastro.util import norm
from poliastro.frames.equatorial import GCRS
from poliastro.czml.extract_czml import CZMLExtractor
import matplotlib.pyplot as plt
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.plotting import OrbitPlotter3D, OrbitPlotter2D
import seaborn as sns
import astropy.units as u
import astropy
from astropy import time
from astropy.coordinates import EarthLocation
import OpticalLinkBudget.OLBtools as olb
import utils as utils

## Constellation Class
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
	def from_walker(cls, i, t, p, f, alt, epoch = False):
		"""
		Generate a walker constellation
		Outputs a set of satellite orbits 

		Inputs:
		i (rad)                  : inclination
		t (int)                  : total number of satellites
		p (int)                  : total number of planes
		f (int between 0 and p-1): determines relative spacing between satellites in adjacent planes
		alt (km)                 : altitude of orbit
		epoch (astropy time)     : epoch to initiate satellites. Default of False defines satellites at J2000

		Outputs:
		sats (list)              : List of satellite objects (SatClasses). Satellites organized by plane
		constClass (object)      : Constellation class from satClasses
		"""

		#Check for astropy classes
		if not isinstance(i, astropy.units.quantity.Quantity):
			i = i * u.rad
		if not isinstance(alt, astropy.units.quantity.Quantity):
			alt = alt * u.km
		#Check f is bettween 0 and p-1
		assert f >= 0 and f <= p-1, "f must be between 0 and p-1"

		s = t/p #number of satellites per plane
		pu = 360 * u.deg / t #Pattern Unit to define other variables

		interPlaneSpacing = pu * p
		nodeSpacing = pu * s
		phaseDiff = pu * f
		

		allPlanes = []
		planeIDCounter = 0
		satIDCounter = 0
		for plane in range(0,p): #Loop through each plane
			planeSats = []
			raan = plane * nodeSpacing
			for sat in range(0,int(s)): #Loop through each satellite in a plane
				omega0 = plane * phaseDiff
				omega = omega0 + sat * interPlaneSpacing
				if epoch:
					orbLoop = Satellite.circular(Earth, alt = alt,
						 inc = i, raan = raan, arglat = omega, epoch = epoch)
				else:
					orbLoop = Satellite.circular(Earth, alt = alt,
						 inc = i, raan = raan, arglat = omega)
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
		
		
	def plan_reconfigure(self, GroundLoc, GroundStation, tInit, days, figName=None, selectionMethod=1, selectionParams=None, plot = False, savePlot = False):
		"""
		Plane reconfiguration of the constellation to visit and image a GroundLoc and downlink to a GroundStation

		Inputs
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

		Outputs
		suggestedMissions: Suggested mission
		"""
		if selectionMethod==1:
			imageWeights = selectionParams[0]
			ISLWeights = selectionParams[1]
			DLWeights = selectionParams[2]
			missionWeights = selectionParams[3]
			nSatImage = selectionParams[4]
			nSatISL = selectionParams[5]
			nSatDL = selectionParams[6]
			nSatMission = selectionParams[7]

			##Maybe get rid of ghostSatsInit and ghostSatsPass as well
			imageManeuvers, paretoSats, ghostSatsInit, ghostSatsPass, rmc = self.get_pass_maneuver(GroundLoc, 
																										tInit,
																										days, 
																										task = 'Image',
																										plot = plot, 
																										savePlot = True, 
																										figName = f'figures/{figName}ImagePass.png')
			
			imageManeuvers_flat = utils.flatten(imageManeuvers)

			[obj.get_weighted_dist_from_utopia(imageWeights[0],imageWeights[1]) for obj in imageManeuvers_flat] #Gets utopia distance
			imageManeuvers_flat.sort(key = lambda x: x.utopDistWeighted)
			selectWeightedManeuversImage = imageManeuvers_flat[0:nSatImage]

			if plot:
				tMaxPlot = (days + 1) * u.day
				xlimits = [-5, tMaxPlot.to(u.hr).value]
				ylimits = [-5, 500]
				plt.figure(constrained_layout=True)
				plt.xlabel('Time (hrs)', fontsize = 14)
				plt.ylabel('Delta V (m/s)', fontsize = 14)
				plt.title('Potential Maneuver Options\n \'o\' for ascending, \'x\' for descending passes', fontsize = 16)
				lgdCheck = []
				for man in imageManeuvers_flat:
				#     clrIdx = int(man.planeID)
					# passType = man.note
					# if passType =='a':
					#   style = 'ko'
					# elif passType == 'd':
					#   style = 'kx'
					plt.plot(man.time2Pass.to(u.hr), man.deltaV, 'ko')#,
						# label = label)
				selectTimesW = [p.time2Pass.to(u.hr).value for p in selectWeightedManeuversImage]
				selectDelVW = [p.deltaV.value for p in selectWeightedManeuversImage]
				# plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
				# plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
				plt.plot(selectTimesW, selectDelVW, 'b^', label = 'Selected Sats Weighted', markersize = 10)

				plt.xlim(xlimits)
				plt.ylim(ylimits)
				plt.plot(0,0,'g*',label='Utopia Point', markersize = 20)
				plt.legend()
				if savePlot:
					from datetime import datetime
					now = datetime.now()
					timestamp = now.isoformat()
					fname = "figures/pool/" + str(timestamp) +"imagingMans"+".png"
					plt.savefig(fname,facecolor="w", dpi=300)


			## Create new constellation objects
			origSats = self.get_sats() #Original satellites
			numOriginalPlanes = len(self.planes)

			selectSats = [s.mySat for s in selectWeightedManeuversImage]

			## Create constellation of potential relay satellites
			potentialRelaySats = [sat for sat in origSats if sat not in selectSats]
			relayPlanes = []
			for planeIdx in range(0,numOriginalPlanes):
				sats = [s for s in potentialRelaySats if s.planeID == planeIdx]
				plane2append = Plane.from_list(sats)
				relayPlanes.append(plane2append)
			potentialRelaySatsConstellation = Constellation.from_list(relayPlanes)

			## Create constellation of ghost satellites that will make the image pass
			ghostImagingSats = [s.mySatFin for s in selectWeightedManeuversImage]
			desiredPassSatsPlane = []
			for planeIdx in range(0,numOriginalPlanes):
				sats = [s for s in ghostImagingSats if s.planeID == planeIdx]
				plane2append = Plane.from_list(sats)
				desiredPassSatsPlane.append(plane2append)
			ghostImagingSatsConstellation = Constellation.from_list(desiredPassSatsPlane)

			## Might not need a lot of these outputs. TO DO eliminate needless outputs
			maneuverObjs, transfersISL, timeOfISL, paretoSatsISL, missionOptions = potentialRelaySatsConstellation.get_ISL_maneuver(ghostImagingSatsConstellation, 
																															perturbation = 'none', 
																															plotFlag=plot)

			## Weighted Choice
			[obj.get_weighted_dist_from_utopia(ISLWeights[0], ISLWeights[1]) for obj in transfersISL] #Gets utopia distance
			transfersISL.sort(key = lambda x: x.utopDistWeighted)
			selectWeightedManeuversISL = transfersISL[0:nSatISL]

			if plot:
				tMaxPlot = (days + 1) * u.day
				xlimits = [-5, tMaxPlot.to(u.hr).value]
				plt.figure(constrained_layout=True)
				plt.xlabel('Time (hrs)', fontsize = 14)
				plt.ylabel('Delta V (m/s)', fontsize = 14)
				plt.title('Potential ISL Options\n \'o\' for ascending, \'x\' for descending passes', fontsize = 16)
				lgdCheck = []
				for man in transfersISL:
				#     clrIdx = int(man.planeID)
					# passType = man.note
					# if passType =='a':
					#   style = 'ko'
					# elif passType == 'd':
					#   style = 'kx'
					plt.plot(man.time2Pass.to(u.hr), man.deltaV, 'ko')#,
						# label = label)
				selectTimesW = [p.time2Pass.to(u.hr).value for p in selectWeightedManeuversISL]
				selectDelVW = [p.deltaV.value for p in selectWeightedManeuversISL]
				# plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
				# plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
				plt.plot(selectTimesW, selectDelVW, 'b^', label = 'Selected Sats Weighted', markersize = 10)

				plt.xlim(xlimits)
				plt.ylim(ylimits)
				plt.plot(0,0,'g*',label='Utopia Point', markersize = 20)
				plt.legend()

				if savePlot:
					fname = "figures/pool/" + str(timestamp) + "ISLMans" + ".png"
					plt.savefig(fname,facecolor="w", dpi=300)


			#### Might need to uncomment this
			# for s in selectWeightedManeuversISL:
			#     s.mySat.task = 'ISL'

			for s in selectWeightedManeuversISL:
				s.mySatFin.add_previousSatInteraction(s.target) #Adds target satellite(ground image) to previous interactions with the downlink satellite

			## Create constellation of ghost ISL satellites
			ghostISLSatsWeighted = [s.mySatFin for s in selectWeightedManeuversISL] #Final orbital config
			selectPlanes_f = []
			for planeIdx in range(0,numOriginalPlanes):
				sats_f = [s for s in ghostISLSatsWeighted if s.planeID == planeIdx]
				plane2append_f = Plane.from_list(sats_f, planeID = planeIdx)
				selectPlanes_f.append(plane2append_f)
			selectSatsISL_f = Constellation.from_list(selectPlanes_f) #Satellite orbits after ISL maneuver

			##Find downlink capable satellites
			downlinkManeuvers, pSats_GP, ghostSatsInit_GP, ghostSatsPass_GP, runningMissionCost = selectSatsISL_f.get_pass_maneuver(GroundStation,
																																tInit, 
																																days, 
																																useSatEpoch=True,
																																task = 'Downlink',
																																plot = True, 
																																savePlot = plot,
																																figName = f'figures/{figName}groundPassPlanner.png')

			downlinkManeuvers_flat = utils.flatten(downlinkManeuvers)

			[obj.get_weighted_dist_from_utopia(DLWeights[0],DLWeights[1]) for obj in downlinkManeuvers_flat] #Gets utopia distance
			downlinkManeuvers_flat.sort(key = lambda x: x.utopDistWeighted)
			selectWeightedDownlinkManeuvers = downlinkManeuvers_flat[0:nSatMission]
			
			if plot:
				tMaxPlot = (days + 1) * u.day
				xlimits = [-5, tMaxPlot.to(u.hr).value]
				plt.figure(constrained_layout=True)
				plt.xlabel('Time (hrs)', fontsize = 14)
				plt.ylabel('Delta V (m/s)', fontsize = 14)
				plt.title('Potential Downlink Options\n \'o\' for ascending, \'x\' for descending passes', fontsize = 16)
				lgdCheck = []
				for man in downlinkManeuvers_flat:
				#     clrIdx = int(man.planeID)
					# passType = man.note
					# if passType =='a':
					#   style = 'ko'
					# elif passType == 'd':
					#   style = 'kx'
					plt.plot(man.time2Pass.to(u.hr), man.deltaV, 'ko')#,
						# label = label)
				selectTimesW = [p.time2Pass.to(u.hr).value for p in selectWeightedDownlinkManeuvers]
				selectDelVW = [p.deltaV.value for p in selectWeightedDownlinkManeuvers]
				# plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
				# plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
				plt.plot(selectTimesW, selectDelVW, 'b^', label = 'Selected Sats Weighted', markersize = 10)

				plt.xlim(xlimits)
				plt.ylim(ylimits)
				plt.plot(0,0,'g*',label='Utopia Point', markersize = 20)
				plt.legend()
				if savePlot:
					fname = "figures/pool/" + str(timestamp) + "DLMans" +".png"
					plt.savefig(fname,facecolor="w", dpi=300)

			##Find best overall mission
			missionCosts = utils.flatten(runningMissionCost)
			[obj.get_weighted_dist_from_utopia(missionWeights[0], missionWeights[1]) for obj in missionCosts] #Gets utopia distance
			missionCosts.sort(key = lambda x: x.utopDistWeighted)
			selectMissionOptions = missionCosts[0:nSatMission]

			if plot:
				tMaxPlot = (days + 1) * u.day
				xlimits = [-5, tMaxPlot.to(u.hr).value]
				plt.figure(constrained_layout=True)
				plt.xlabel('Time (hrs)', fontsize = 14)
				plt.ylabel('Delta V (m/s)', fontsize = 14)
				plt.title('Potential Mission Options\n \'o\' for ascending, \'x\' for descending passes', fontsize = 16)
				lgdCheck = []
				for man in missionCosts:
				#     clrIdx = int(man.planeID)
					# passType = man.note
					# if passType =='a':
					#   style = 'ko'
					# elif passType == 'd':
					#   style = 'kx'
					plt.plot(man.dlTime.to(u.hr), man.totalCost, 'ko')#,
						# label = label)
				selectTimesW = [p.dlTime.to(u.hr).value for p in selectMissionOptions]
				selectDelVW = [p.totalCost.value for p in selectMissionOptions]
				# plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')
				# plt.plot(selectTimes, selectDelV, 'go', label = 'Selected Sats', markersize = 10)
				plt.plot(selectTimesW, selectDelVW, 'b^', label = 'Selected Sats Weighted', markersize = 10)

				plt.xlim(xlimits)
				plt.ylim(ylimits)
				plt.plot(0,0,'g*',label='Utopia Point', markersize = 20)
				plt.legend()
				if savePlot:
					fname = "figures/pool/" + str(timestamp) + "missionMans" + ".png"
					plt.savefig(fname,facecolor="w", dpi=300)

			return selectMissionOptions



	def get_pass_maneuver(self, GroundLoc, tInit, days, useSatEpoch = False, task = None, plot = False, savePlot = False, figName = None):
		"""
		Gets set a maneuvers that will pass a specific ground location on
		select days

		Inputs:
		GroundLoc (Obj): Ground location object to be passed over
		tInit (astropy time object): Time to initialize planner
		days (int): Amount of days ahead to for the scheduler to plan for
		useSatEpoch (Bool): If true, uses individual satellite current epoch as tInit. Useful for downklink portion of planner
		task (string): The assigned task for the desired satellite. Options: 'Image', 'ISL', 'Downlink'
		plot (Bool): Plots maneuvers if True
		savePlot(Bool): Saves plot if True
		figName (str): file name for figure


		Outputs:
		maneuverPoolCut (list of Maneuver objects): Physically viable maneuvers (selected because some maneuvers would cause satellites to crash into Earth)
		maneuverPoolAll (list of Maneuver objects): All potential maneuvers
		paretoSats (list of pareto front satellites): Satellite objects on the pareto front
		ghostSatsInitAll (array of Satellite objects): Orbit of ghost satellite if it were propagated back to initial time
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
			if not plane: #Check if plane is empty
				continue
			maneuverPlaneCut, maneuverPlaneAll, ghostSatsPass, ghostSatsInit, runningMissionCost = plane.get_pass_details(GroundLoc, tInit, days, useSatEpoch, task)
			maneuverPoolAll.append(maneuverPlaneAll)
			maneuverPoolCut.append(maneuverPlaneCut)
			ghostSatsPassAll.append(ghostSatsPass)
			ghostSatsInitAll.append(ghostSatsInit)
			missionCosts.append(runningMissionCost)

		semiFlatCut = [item for sublist in maneuverPoolCut for item in sublist] #Flatten list
		flatCut = [item for sublist in semiFlatCut for item in sublist]

		paretoSats = utils.find_non_dominated_time_deltaV(flatCut)
		ax = None
		if plot:

			colors = sns.color_palette('tab10', len(self.planes))
			sns.set_style('whitegrid')
			style = ['o', 'x']
			# fig = plt.figure(constrained_layout=True)
			fig, ax = plt.subplots(constrained_layout=True)
			plt.xlabel('Time (hrs)', fontsize = 14)
			plt.ylabel('Delta V (m/s)', fontsize = 14)
			plt.title('Potential Maneuver Options\n \'o\' for ascending, \'x\' for descending passes', fontsize = 16)

			lgdCheck = []
			for man in flatCut:

				clrIdx = int(man.planeID)
				passType = man.note
				if passType =='a':
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
				plt.plot(man.time2Pass.to(u.hr), man.deltaV, style, color = colors[clrIdx],
						label = label)
			paretoTimes = [p.time2Pass.to(u.hr).value for p in paretoSats]
			paretoDelV = [p.deltaV.value for p in paretoSats]
			plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')

			plt.plot(0,0,'g*',label='Utopia Point', markersize = 20)
			plt.legend(title = 'Plane Number')
			# plt.grid()
			if savePlot:
				plt.savefig(figName, facecolor="w", dpi=300, bbox_inches='tight')
			# plt.tight_layout()
			plt.show()

		return maneuverPoolCut, paretoSats, ghostSatsInitAll, ghostSatsPassAll, missionCosts
	
	def get_ISL_maneuver(self, Constellation, perturbation='J2', plotFlag=False, savePlot = False, figName = None):
		"""
		Calculate potential ISL maneuvers between all satellites in current constellation with
		all satellites of another satellite
		
		Inputs
		self (Constellation object): Constellation making the maneuvers
		Constellation (Constellation object): Constellation to intercept
		perturbation (string): 'J2' for J2 perturbation, else 'none'
		plotFlag (boolean): plot if True
		savePlot(Bool): Saves plot if True
		figName (str): file name for figure

		Output
		maneuverObjs [list] : list of maneuver objects to create ISL opportunity
		deltaVs [list] : list of total deltaVs
		timeOfISL [list] : list of times that ISL opportunity will happen
		paretoSats [list] : list of satellites on the pareto front
		"""
		if savePlot:
			assert figName, "Need to name your figure using figName input"

		selfSats = self.get_sats()
		targetSats = Constellation.get_sats()
		tInit = selfSats[0].epoch #Get time at beginnnig of simulation

		maneuverObjs = []
		transfers = []
		timeOfISL = []
		islOrbs = []
		missionOptions = []
		for satInt in selfSats:
			for satTar in targetSats:
				output = satInt.schedule_coplanar_intercept(satTar, perturbation=perturbation)
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
			plt.xlabel('Time (hrs)', fontsize = 14)
			plt.ylabel('Delta V (m/s)', fontsize = 14)
			plt.title('Potential Maneuver Options\n for ISL data transfer', fontsize = 16)

			lgdCheck = []
			paretoTimes = []
			for man in transfers:
				clrIdx = int(man.planeID)
				
				if clrIdx in lgdCheck:
					label = None
				else:
					label = f'{clrIdx}'
					lgdCheck.append(clrIdx)
				timeAtPass = man.time - tInit #man.time gets time at ISL, tInit is the current time
				# import ipdb; ipdb.set_trace()
				plt.plot(timeAtPass.to(u.hr), man.deltaV, 'o',color = colors[clrIdx],
						label = label)
			paretoTimes = [(p.time-tInit).to(u.hr).value for p in paretoSats]
			paretoDelV = [p.deltaV.value for p in paretoSats]
			plt.plot(paretoTimes, paretoDelV, 'r-', label = 'Pareto Front')

			plt.plot(0,0,'g*',label='Utopia Point', markersize = 20)
			plt.legend(title = 'Plane Number')
			# plt.grid()
			if savePlot:
				plt.savefig(figName, facecolor="w", dpi=300, bbox_inches='tight')
			plt.show()

		return maneuverObjs, transfers, timeOfISL, paretoSats, missionOptions

	def calc_access(self, GroundLoc):
		pass
	
	def propagate(self, time):
		"""
		Propagates satellites in the constellation to a certain time.

		Inputs
		time(astropy time object): Time to propagate to
		"""
		planes2const = []
		for plane in self.planes:
			if not plane: #Continue if empty
				continue

			planeSats = []
			for satIdx, sat in enumerate(plane.sats):
				satProp = sat.propagate(time)
				planeSats.append(satProp)
				# plane.sats[satIdx] = satProp
			plane2append = Plane.from_list(planeSats)
			planes2const.append(plane2append)
		# import ipdb; ipdb.set_trace()
		# breakpoint()
		# planes2const.append(plane2append)
		return Constellation.from_list(planes2const)

	def get_rv_from_propagate(self, timeDeltas, method="J2"):
		"""
		Propagates satellites and returns position (R) and Velocity (V) values
		at the specific timeDeltas input. Defaults to propagation using J2 perturbation

		Inputs
		timeDeltas (astropy TimeDelta object): Time intervals to get position/velocity data
		"""
		planes2const = []

		for plane in self.planes:
			if not plane: #Continue if empty
				constrained_layout

			planeSats = []
			for satIdx, sat in enumerate(plane.sats):
				if method == "J2":
					def f(t0, state, k): #Define J2 perturbation
						du_kep = func_twobody(t0, state, k)
						ax, ay, az = J2_perturbation(
							t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
						)
						du_ad = np.array([0, 0, 0, ax, ay, az])

						return du_kep + du_ad
					coords = propagate(
						sat,
						timeDeltas,
						method=cowell,
						f=f,
						)
				else:
					coords = propagate(
						sat,
						timeDeltas,
						)

				sat.rvCoords = coords
				planeSats.append(sat)
			planes2append = Plane.from_list(planeSats)
			planes2const.append(planes2append)
			# breakpoint()
		return Constellation.from_list(planes2const)


	def get_relative_velocity_analysis(self):
		"""
		Gets relative velocities between satellites in the constellation

		Needs to run get_rv_from_propagate first to get position/velocity values first

		Outputs
		outputData (dict) : First layer key are the satellites being compared i.e. '4-10'
							means that satellite 4 is compared to satellite 10. Secoond layer
							key are the specific data types described below:

							LOS (Bool): Describes if there is a line of sight between the satellites
							pDiff : Relative position (xyz)
							pDiffNorm : magnitude of relative positions
							pDiffDot : dot product of subsequent relative position entries (helps determine if there is a 180 direct crossinig)
							flag180 : Flag to determine if there was a 180 degree 'direct crossing'
							velDiffNorm : relative velocities
							slewRate : slew rates required to hold pointing between satellites
							dopplerShift : Effective doppler shifts due to relative velocities
		"""

		#Check if first satellite has rvCoords Attribute
		# assert self.planes[0].sats[0].rvCoords, "Run self.get_rv_from_propagate first to get rv values"
		if not hasattr(self.planes[0].sats[0], 'rvCoords'):
			print("Run self.get_rv_from_propagate first to get rv values")
			return
		c = 3e8 * u.m / u.s

		sats = self.get_sats()

		outputData = {}
		for satRef in sats:
			for sat in sats:

				if satRef.satID == sat.satID:
					continue

				# dictEntry = {}

				#Reference orbit RV values
				satRef_r = satRef.rvCoords.without_differentials()
				satRef_v = satRef.rvCoords.differentials                

				#Comparison orbit RV values
				sat_r = sat.rvCoords.without_differentials()
				sat_v = sat.rvCoords.differentials

				#Determine LOS availability (Vallado pg 306 5.3)
				adotb = sat_r.dot(satRef_r)
				aNorm = sat_r.norm()
				bNorm = satRef_r.norm()
				theta = np.arccos(adotb/(aNorm * bNorm))
				
				theta1 = np.arccos(constants.R_earth / aNorm)
				theta2 = np.arccos(constants.R_earth / bNorm)
				
				LOSidx = (theta1 + theta2) > theta




				
				#Relative positions
				pDiff = satRef_r - sat_r
				pDiffNorm  = pDiff.norm()
				
				pDiffDot = pDiff[:-1].dot(pDiff[1:])
				if min(pDiffDot) < 0: ## Checks for 180 deg crossinig
					flag180 = 1
				else:
					flag180 = 0

				velDiff = satRef_v["s"] - sat_v["s"]
				velDiffNorm = velDiff.norm()

				#Slew Equations
				## Do it using the slew equation (From Trevor Dahl report)
				rCV = pDiff.cross(velDiff)
				slewRateOrb = rCV / pDiffNorm**2
				slewRateOrbNorm = slewRateOrb.norm()


				#Doppler shift
				pDiffU = pDiff/pDiffNorm #unit vector direction of relative position
				rdDot = satRef_v["s"].to_cartesian() #Get velocity of destination satellite (Reference orbit)
				numTerm = rdDot.dot(pDiffU)
				rsDot = sat_v["s"].to_cartesian()
				denTerm = rsDot.dot(pDiffU)
				num = c - numTerm
				den = c - denTerm
				fd_fs = num/den

				dictEntry = {
							'LOS':LOSidx,
							'pDiff':pDiff,
							'pDiffNorm':pDiffNorm,
							'pDiffDot': pDiffDot,
							'flag180': flag180,
							'relVel': velDiffNorm,
							'slewRate': slewRateOrbNorm,
							'dopplerShift':fd_fs,
				}

				dictKey = str(satRef.satID) + '-' + str(sat.satID)
				outputData[dictKey] = dictEntry
		return outputData

	
	def get_sats(self):
		"""
		Gets a list of all the satetllite objects in the constellation
		"""
		satList = []
		planes = self.planes
		for plane in planes: #Continue if empty
			if not plane:
				continue
			for sat in plane.sats:
				satList.append(sat)
		return satList
	
	def plot(self):
		"""
		Plot constellation using Poliastro interface
		"""
		## Doesn't seem to work in jupyter notebooks. Must return plot object and show in jupyter notebook
		sats = self.get_sats()
		op = OrbitPlotter3D()
		for sat in sats:
			op.plot(sat, label=f"ID {sat.satID:.0f}")
		# op.show()
		return op ##

	def plotSats(self, satIDs):
		"""
		Plot individual satellites using Poliastro interface

		Inputs
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

	def generate_czml_file(self, fname, prop_duration, sample_points):
		"""
		Generates CZML file for the constellation for plotting

		Inputs
		fname (string): File name (including path to file) for saved czml file. Currently plots in a directory czmlFiles
		"""
		seedSat = self.get_sats()[0]
		start_epoch = seedSat.epoch #iss.epoch

		end_epoch = start_epoch + prop_duration

		earth_uv = "https://earthobservatory.nasa.gov/ContentFeature/BlueMarble/Images/land_shallow_topo_2048.jpg"
		extractor = CZMLExtractor(start_epoch, end_epoch, sample_points,
								  attractor=Earth, pr_map=earth_uv)
		for plane in self.planes: #Loop through each plane
			for sat in plane.sats: #Loop through each satellite in a plane
				extractor.add_orbit(sat, groundtrack_show=False,
							groundtrack_trail_time=0, path_show=True)
				
		testL = [str(x) for x in extractor.packets]
		toPrint = ','.join(testL)
		toPrint = '[' + toPrint + ']'
		cwd = os.getcwd()

		czmlDir = os.path.join(cwd, "czmlFiles")

		## Check if directory is available
		czmlDirExist = os.path.isdir(czmlDir)
		os.makedirs(path, exist_ok=True) 

		fileDir = os.path.join(cwd, czmlDir, fname + ".czml")
		f = open(fileDir, "w")
		f.write(toPrint)
		f.close()

# ## Constellation data class that holds data from analyzing a constellation
# class ConstellationData(Constellation):
#   """
#   This class holds the analysis conducted on a constellation class
#   """
#   def __init__(self):
#       super.__init__()

## Plane class
class Plane():
	"""
	Defines an orbital plane object that holds a set of satellite objects
	"""
	def __init__(self, sats, a = None, e = None, i = None, planeID = None, passDetails = None):
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
	
	@classmethod
	def from_list(cls, satList, planeID = None):
		"""
		Create plane from a list of satellite objects
		"""
		if not satList: #Empty list
			plane = []
			print("Empty list entered")
		else:
			for idx, sat in enumerate(satList):
				if idx == 0:
					plane = cls(sat, planeID = planeID)
				else:
					plane.add_sat(sat)
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
		
	def get_pass_details(self, GroundLoc, tInit, days, useSatEpoch, task):
		"""
		Gets set a maneuvers that will pass a specific ground location on
		select days

		Inputs:
		GroundLoc (Obj): Ground location object to be passed over
		tInit (astropy time object): Time to initialize planner
		days (int): Amount of days ahead to for the scheduler to plan for
		useSatEpoch (Bool): If true, uses individual satellite current epoch as tInit. Useful for downklink portion of planner
		task (string): The assigned task for the desired satellite. Options: 'Image', 'ISL', 'Downlink'

		Outputs:
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
		
## Satellite class 
class Satellite(Orbit):
	"""
	Defines the satellite class
	"""
	def __init__(self, state, epoch, satID = None, dataMem = None, schedule = None, 
				 commsPayload = None, previousTransfer = None, planeID = None,
				 previousSatInteractions = None, note = None, task = None):
		super().__init__(state, epoch) #Inherit class for Poliastro orbits (allows for creation of orbits)
		self.satID = satID
		self.planeID = planeID
		self.note = note        
		self.task = task
		self.alt = self.a - constants.R_earth

		if dataMem is None: #Set up ability to hold data
			self.dataMem = []
		else:
			self.dataMem = dataMem
			
		if schedule is None: #Set up ability to hold a schedule
			self.schedule = []
		else:
			self.schedule = schedule

		if commsPayload is None: #Set up ability to hold an optical payload
			self.commsPayload = []
		else:
			self.commsPayload = commsPayload
	
		if previousTransfer is None: #Set up ability to hold potential maneuver schedule
			self.previousTransfer = []
		else:
			self.previousTransfer = previousTransfer

		if previousSatInteractions is None: #Set up ability to hold history of previous interactions with other objects
			self.previousSatInteractions = []
		else:
			self.previousSatInteractions = previousSatInteractions

			
	def calc_link_budget_tx(self, rx_object, l_atm = 0, l_pointing = 0, l_pol = 0, txID = 0, rxID = 0, drReq = None, verbose = False):
		"""
		Calculate link budget with self as transmitter
		
		Inputs
		rx_object: Satellite or GroundStation Object
		l_atm (dB): Atmospheric loss
		l_pointing (dB): pointing loss
		l_pol (dB): Polarization loss
		txID (int): Choose tx communications payload. Default to 0 since we will most likely just have 1 payload
		rxID (int): Choose rx communications payload. Default to 0 since we will most likely just have 1 payload
		drReq (bytes): Required data rate in bytes. Default to None
		verbose (Bool): If True, print components of link budget
	
		Outputs (dictionary)
		P_rx (dBW): Received power
		ConN0 (dBHz): Signal to noise ratio
		EbN0 (dB): Energy per bit. Calculated if a required data rate is given "drReq"

		"""
		assert hasattr(self, 'commsPayload'), "Need to add rx_object to Transmitting Satellite"
		assert hasattr(rx_object, 'commsPayload'), "Need to add rx_object to rx_object Satellite"

		#Get tx payload
		txPL = self.commsPayload[txID]
		EIRP_dB = txPL.get_EIRP()
		wl = txPL.wavelength #wavelength

		#Get rx payload
		rxPL = rx_object.commsPayload[rxID]
		G_rx = rxPL.g_rx

		posVecTx = self.r #position of tx satellite (ECI)

		if rx_object.__class__.__name__ == 'Satellite':
			posVecRx = rx_object.r #Position of rx satellite (ECI)
		elif rx_object.__class__.__name__ == 'GroundStation':
			posVecRx = rx_object.propagate_to_ECI(self.epoch).data.without_differentials().xyz
		else:
			print("rx_object Class not recognized")

		posVecDiff = astropy.coordinates.CartesianRepresentation(posVecTx - posVecRx)
		dist = posVecDiff.norm().to(u.m)

		GonT_dB = rxPL.get_GonT()
		L_fs = (4 * np.pi * dist / wl) ** 2 #Free space path loss
		L_fs_dB = 10 * np.log10(L_fs.value)
		if L_fs_dB == float("inf") or L_fs_dB == float("-inf"):
			L_fs_dB = 0

		L_other_dB = l_pointing + l_pol

		L_u = L_fs_dB + l_atm


		P_rx = EIRP_dB - L_fs_dB + G_rx - L_other_dB

		k_dB = 228.6 #Boltzmann constant in dB

		if verbose:
			print("L_fs:    ", L_fs_dB)
			print("EIRP:    ", EIRP_dB)
			print("L_u:     ", L_u)
			print("L_other: ", L_other_dB)
			print("Gont:    ", GonT_dB)
			print("K:       ", k_dB)

		#Signal to noise ratio
		ConN0 = EIRP_dB - L_u - L_other_dB + GonT_dB + k_dB

		output = {
					"p_rx" : P_rx,
					"ConN0": ConN0,
		}

		if drReq: #Calculate EbNo if data rate required is input
			dr_dB = 10 * np.log10(drReq) #data rate in decibel
			EbN0 = ConN0 - dr_dB
			output["EbN0"] = EbN0

		return output
	
	def schedule_intercept(sat):
		pass
	
	def add_data(self, data): #Add data object to Satellite
		self.dataMem.append(data)
	
	def remove_data(self, data): #Remove data object from Satellite (i.e. when downlinked)
		if data in self.dataMem:
			self.dataMem.remove(data)
			
	def add_schedule(self, sch_commands): #Adds a schedule to the satellite
		self.schedule.append(sch_commands)
	
	def add_previousTransfer(self, prevSch):
		self.previousTransfer.append(prevSch)

	def add_previousSatInteraction(self, object):
		if not self.previousSatInteractions or not isinstance(object, list): #If empty
			self.previousSatInteractions.append(object)
		else:
			self.previousSatInteractions.extend(object)

	def add_comms_payload(self, commsPL):
		self.commsPayload.append(commsPL)
		
	def reset_payload(self):
		self.commsPayload = []
		
	# def set_task(self, task):
	#   self.task = task

	def calc_access(self, GroundLoc, timeSpan):
		pass
	
	def get_desired_pass_orbs(self, GroundLoc, tInitSim, days, useSatEpoch, task = None,
								 refVernalEquinox = astropy.time.Time("2021-03-20T0:00:00", format = 'isot', scale = 'utc')):
		"""
		Given an orbit and ground site, gets the pass time and true anomaly 
		required to complete a pass. This is a rough estimate of the time and good
		for quick but not perfectly accurate scheduling. 
		Loosely based on Legge's thesis section 3.1.2

		Inputs:
		GroundLoc (Obj): GroundLoc class. This is the ground location that you want to get a pass from
		tInit (astropy time object): Time to initialize planner
		days (list of astropy.time obj): Amount of days ahead to for the scheduler to plan for
		useSatEpoch (Bool): Use satellite epoch for start of planner (useful for subsequent stages of planner)
		task (string): The assigned task for the desired satellite. Options: 'Image', 'ISL', 'Downlink'
		refVernalEquinox (astropy time object): Date of vernal equinox. Default is for 2021

		Outputs:
		desiredGhostOrbits_atPass (array of Satellite objects): Orbit of satellite at ground pass (potential position aka ghost position)
		desiredGhostOrbits_tInit (array of Satellite objects): Orbit of ghost satellite if it were propagated back to initial time
		maneuverObjects (array of Manuever objects): Maneuver object class that holds time and cost of maneuvers
		"""

		## TODO: Doesn't yet account for RAAN drift due to J2 perturbation
		if useSatEpoch:
			tInit = self.epoch #Sets tInit to current satellite epoch
			satInit = self
		else:
			tInit = tInitSim #Sets tInit to the starting point of simiulation
			satInit = self.propagate(tInitSim)
		
		tInitMJDRaw = tInit.mjd
		tInitMJD = int(tInitMJDRaw)

		dayArray = np.arange(0, days + 1)
		days2InvestigateMJD = list(tInitMJD + dayArray) #Which days to plan over
		days = [time.Time(dayMJD, format='mjd', scale='utc')
							for dayMJD in days2InvestigateMJD]
		
		## Extract relevant orbit and ground station parameters
		i = satInit.inc
		lon = GroundLoc.lon
		lat = GroundLoc.lat
		raan = satInit.raan
		delLam = np.arcsin(np.tan(lat) / np.tan(i)) #Longitudinal offset
		theta_GMST_a = raan + delLam - lon #ascending sidereal angle of pass
		theta_GMST_d = raan - delLam - lon - np.pi * u.rad #deescending sidereal angle of pass


		delDDates = [day - refVernalEquinox for day in days] #Gets difference in time from vernal equinox
		delDDateDecimalYrList = [delDates.to_value('year') for delDates in delDDates] #Gets decimal year value of date difference
		delDDateDecimalYr = np.array(delDDateDecimalYrList)
	
		#Get solar time values for ascending and descending pass
		theta_GMT_a_raw = theta_GMST_a - 2*np.pi * delDDateDecimalYr * u.rad + np.pi * u.rad
		theta_GMT_d_raw = theta_GMST_d - 2*np.pi * delDDateDecimalYr * u.rad + np.pi * u.rad

		theta_GMT_a = np.mod(theta_GMT_a_raw, 360 * u.deg)
		theta_GMT_d = np.mod(theta_GMT_d_raw, 360 * u.deg)

		angleToHrs_a = astropy.coordinates.Angle(theta_GMT_a).hour
		angleToHrs_d = astropy.coordinates.Angle(theta_GMT_d).hour
		
		tPass_a = []
		tPass_d = []
		for d_idx, day in enumerate(days):
			timePass_a = day + angleToHrs_a[d_idx] * u.hr
			timePass_d = day + angleToHrs_d[d_idx] * u.hr
			if timePass_a > self.epoch: #Make sure time is in the future
				tPass_a.append(timePass_a)
			if timePass_d > self.epoch:
				tPass_d.append(timePass_d)
		# tPass_a = [day + angleToHrs_a[idx] * u.hr for idx, day in enumerate(days)]
		# tPass_d = [day + angleToHrs_d[idx] * u.hr for idx, day in enumerate(days)]
		timesRaw = [tPass_a, tPass_d]

		note_a = f"Potential ascending pass times for Satellite: {self.satID}"
		note_d = f"Potential descending pass times for Satellite: {self.satID}"
		scheduleItms_a = [PointingSchedule(tPass) for tPass in tPass_a]
		for schItm in scheduleItms_a:
			schItm.note = note_a
			schItm.passType = 'a'
		
		scheduleItms_d = [PointingSchedule(tPass) for tPass in tPass_d]
		for schItm in scheduleItms_d:
			schItm.note = note_d
			schItm.passType = 'd'
		
		scheduleItms = scheduleItms_a + scheduleItms_d
		
		desiredGhostOrbits_atPass = []
		desiredGhostOrbits_tInit = []
		
		maneuverObjectsAll = []
		maneuverObjectsCut = []
		runningMissionCosts = []
		## TODO: This will have to be redone once we get J2 perturbations in as RAAN will drift. We'll need multiple raans.
		for idx, sch in enumerate(scheduleItms):
			raans, anoms = self.desired_raan_from_pass_time(sch.time, GroundLoc) ##Only need one time to find anomaly since all passes should be the same geometrically
			if sch.passType == 'a': #Ascending argument of latitude
				omega = anoms[0]
				raan = raans[0]
			elif sch.passType == 'd': #Descending argument of latitude
				omega = anoms[1]
				raan = raans[1]
			
			#Get desired satellite that will make pass of ground location
			ghostSatFuture = Satellite.circular(Earth, alt = satInit.alt,
				 inc = satInit.inc, raan = raan, arglat = omega, epoch = sch.time)
			ghostSatFuture.satID = self.satID #tag with satID
			ghostSatFuture.note = sch.passType #Tag with ascending or descending
			ghostSatInit = ghostSatFuture.propagate(tInit) ## TO DO propagate with j2 perturbation
			ghostSatInit.satID = self.satID #tag with satID
			
			##Tag satellite with maneuver number
			ghostSatFuture.manID = idx
			ghostSatInit.manID = idx

			## Tag ghost satellites with sat and plane IDs
			ghostSatFuture.satID = self.satID
			ghostSatFuture.satID = self.planeID

			ghostSatInit.satID = self.satID
			ghostSatInit.satID = self.planeID

			desiredGhostOrbits_atPass.append(ghostSatFuture)
			desiredGhostOrbits_tInit.append(ghostSatInit)
			## Get rendezvous time and number of orbits to phase
			time2pass = ghostSatFuture.epoch - tInitSim

			rvdOrbsRaw = time2pass.to(u.s) / satInit.period
			rvdOrbsInt = rvdOrbsRaw.astype(int)
			
			## Get relative anomaly for rendezvous
			nuGhost = ghostSatInit.nu
			nuInit = satInit.nu
			phaseAng = nuInit - nuGhost
		
			## TO DO get deltaV direction
			tPhase, delVTot, delV1, delV2, a_phase, passFlag = self._coplanar_phase(ghostSatFuture.a, 
									phaseAng,
									rvdOrbsInt,
									rvdOrbsInt)
			


			## Create maneuver object
			manObj1 = ManeuverObject(tInit, delV1, time2pass, self.satID,
								   GroundLoc, note=sch.passType, mySat = self)
			manObj2 = ManeuverObject(time2pass, delV2, time2pass, self.satID,
								   GroundLoc, note=sch.passType, mySat = self)

			transferObj = Transfer(time2pass, delVTot, time2pass, self.satID,
								   GroundLoc, note=sch.passType, mySat = self)
			transferObj.task = task
			transferObj.maneuvers = [manObj1, manObj2]
			## Check if satellite interacted with any satellites before
			runningMissionCost = []
			if self.previousSatInteractions is not None:
				for sat in self.previousSatInteractions:
					previousMans = sat.previousTransfer

					## Add mission cost from previous satellite maneuvers
					runningMissionCost.extend(previousMans)

			## Add maneuverObject to ghostSatFuture
			if self.previousTransfer:
				#Add existing schedule to ghost satellite
				ghostSatFuture.add_previousTransfer(self.previousTransfer) 
				runningMissionCost.extend(self.previousTransfer)

			ghostSatFuture.task = task #Attach the task to the satellite
			ghostSatFuture.add_previousTransfer(transferObj)
			


			## Add ghostSatFuture to transferObj
			transferObj.mySatFin = ghostSatFuture

			## Tag transfer object with maneuver number (to match ghost orbits later)
			transferObj.manID = idx

			## Tag transfer object with planeID and satID
			transferObj.satID = self.satID
			# manObj.planeID = self.planeID

			maneuverObjectsAll.append(transferObj)
			if passFlag:
				maneuverObjectsCut.append(transferObj)

			## Add current satellitte cost to other external mission costs
			runningMissionCost.append(transferObj)

			missionOption = MissionOption(runningMissionCost, task)
			missionOption.get_mission_specs()
			runningMissionCosts.append(missionOption)

		return desiredGhostOrbits_atPass, desiredGhostOrbits_tInit, maneuverObjectsAll, maneuverObjectsCut, runningMissionCosts
		
	def desired_raan_from_pass_time(self, tPass, GroundLoc):
		"""        Gets the desired orbit specifications from a desired pass time and groundstation
		Based on equations in section 3.1.2 in Legge's thesis (2014)

		Inputs
		tPass (astropy time object): Desired time of pass. Local UTC time preferred
		GroundLoc (Obj): GroundLoc class. This is the ground location that you want to get a pass from

		Outputs
		raans [List]: 2 element list where 1st element corresponding to RAAN in the ascending case
						and the 2nd element correspond to RAAN in the descending case
		Anoms [List]: 2 element list where the elements corresponds to true Anomalies (circular orbit)
						of the ascending case and descending case respectively"""
	
		 ## Check if astropy class. Make astropy class if not
		if not isinstance(GroundLoc.lat, astropy.units.quantity.Quantity):
			GroundLoc.lat = GroundLoc.lat * u.deg
		if not isinstance(GroundLoc.lon, astropy.units.quantity.Quantity):
			GroundLoc.lon = GroundLoc.lon * u.deg
#         if not isinstance(i, astropy.units.quantity.Quantity):
#             i = i * u.rad
		tPass.location = GroundLoc.loc #Make sure location is tied to time object
		theta_GMST = tPass.sidereal_time('mean', 'greenwich') #Greenwich mean sidereal time
		
		i = self.inc
		dLam = np.arcsin(np.tan(GroundLoc.lat.to(u.rad)) / np.tan(i.to(u.rad)))

		#From Legge eqn 3.10 pg.69
		raan_ascending = theta_GMST - dLam + np.deg2rad(GroundLoc.lon)
		raan_descending = theta_GMST + dLam + np.deg2rad(GroundLoc.lon) + np.pi * u.rad


		#Get mean anomaly (assuming circular earth): https://en.wikipedia.org/wiki/Great-circle_distance
		n1_ascending = np.array([np.cos(raan_ascending),np.sin(raan_ascending),0]) #RAAN for ascending case in ECI norm
		n1_descending = np.array([np.cos(raan_descending),np.sin(raan_descending),0]) #RAAN for descending case in ECI norm

		n2Raw = GroundLoc.loc.get_gcrs(tPass).data.without_differentials() / GroundLoc.loc.get_gcrs(tPass).data.norm() #norm of ground station vector in ECI
		n2 = n2Raw.xyz

		n1_ascendingXn2 = np.cross(n1_ascending, n2) #Cross product
		n1_descendingXn2 = np.cross(n1_descending, n2)

		n1_ascendingDn2 = np.dot(n1_ascending, n2) #Dot product
		n1_descendingDn2 = np.dot(n1_descending, n2)

		n1a_X_n2_norm = np.linalg.norm(n1_ascendingXn2)
		n1d_X_n2_norm = np.linalg.norm(n1_descendingXn2)
		if GroundLoc.lat > 0 * u.deg and GroundLoc.lat < 90 * u.deg: #Northern Hemisphere case
			ca_a = np.arctan2(n1a_X_n2_norm, n1_ascendingDn2) #Central angle ascending
			ca_d = np.arctan2(n1d_X_n2_norm, n1_descendingDn2) #Central angle descending
		elif GroundLoc.lat < 0 * u.deg and GroundLoc.lat > -90 * u.deg: #Southern Hemisphere case
			ca_a = 2 * np.pi * u.rad - np.arctan2(n1a_X_n2_norm, n1_ascendingDn2) 
			ca_d = 2 * np.pi * u.rad - np.arctan2(n1d_X_n2_norm, n1_descendingDn2)
		elif GroundLoc.lat == 0 * u.deg: #Equatorial case
			ca_a = 0 * u.rad
			ca_d = np.pi * u.rad
		elif GroundLoc.lat == 90 * u.deg or GroundLoc.lat == -90 * u.deg: #polar cases
			ca_a = np.pi * u.rad
			ca_d = 3 * np.pi / 2 * u.rad
		else:
			print("non valid latitude")
			
		raans = [raan_ascending, raan_descending]
		Anoms = [ca_a, ca_d]

		return raans, Anoms
	
	@staticmethod
	def _coplanar_phase(a_tgt, theta, k_tgt, k_int, mu = constants.GM_earth):
		"""
		Get time to phase, deltaV and semi-major axis of phasing orbit in a coplanar phasing maneuver (same altitude)
		From Vallado section 6.6.1 Circular Coplanar Phasing. Algorithm 44

		Inputs:
		a_tgt (m) : semi-major axis of target orbit
		theta (rad): phase angle measured from target to the interceptor. Positive in direction of target motion.
		k_tgt : integer that corresponds to number of target satellite revolutions before rendezvous
		k_int : integer that corresponds to number of interceptor satellite revolutions before rendezvous
		mu (m^3 / s^2) : Gravitational constant of central body. Default is Earth

		Outputs:
		t_phase (s) : time to complete phasing maneuver
		deltaV (m/s) : Delta V required to complete maneuver
		delV1 (m/s) : Delta V of first burn (positive in direction of orbital velocity)
		delV2 (m/s) : Delta V of second burn which recircularies (positive in direction of orbital velocity)
		a_phase (m) : Semi-major axis of phasing orbit
		passFlag (bool) : True if orbit is valid i.e. rp > rPlanet so satellite won't crash into Earth
		"""
		if not isinstance(a_tgt, u.quantity.Quantity):
			a_tgt = a_tgt * u.m
		if not isinstance(theta, u.quantity.Quantity):
			theta = theta * u.rad

		w_tgt = np.sqrt(mu/a_tgt**3)
		t_phase = (2 * np.pi * k_tgt + theta.to(u.rad).value) / w_tgt
		a_phase = (mu * (t_phase / (2 * np.pi * k_int))**2)**(1/3)

		rp = 2 * a_phase - a_tgt #radius of perigee
		passFlag = rp > constants.R_earth #Check which orbits are non-feasible due to Earth's radius
		deltaV = 2 * np.abs(np.sqrt(2*mu / a_tgt - mu / a_phase) - np.sqrt(mu / a_tgt)) #For both burns. One to go into ellipse, one to recircularize
		
		#Get individual burn values
		delV1Raw = deltaV / 2

		#If theta > 0, interceptor in front of target, first burn is to increase semi-major axis to slow down (positive direction for burn in direction of velocity)
		thetaMask = theta >= 0 
		thetaMaskPosNeg = thetaMask * 2 - 1 #Create mask to determine first burn is in direction of velocity if theta > 0, negative if not

		delV1 = delV1Raw * thetaMaskPosNeg #Apply mask to delV1Raw
		delV2 = -delV1 # Second burn is negative of first burn

		return t_phase.to(u.s), deltaV.to(u.m/u.s), delV1.to(u.m/u.s), delV2.to(u.m/u.s), a_phase.to(u.km), passFlag

	def get_nu_intercept(self, satellite):
		"""
		Given two orbits, determine true anomaly (argument of latitude)
		of each circular orbit for the position of closest approach of the orbits
		themselves (not the individual satellites). This will help
		determine which point in the orbit is desired for rendezvous for ISL.

		Input
		self (Satellite object): Self, which is one of the satellites in question
		satellite (Satellite object): The target satellite of the reendezvous

		Output
		nus: Set of 2 true anomalies for each respective orbit for optimal ISL distance closing

		"""

		## TO DO: Build out the timed perturbation variation

		L1 = np.cross(self.r, self.v)  # Angular momentum of first orbit
		L2 = np.cross(satellite.r, satellite.v)
		L1hat = L1/norm(L1)
		L2hat = L2/norm(L2)
		intersectLine = np.cross(L1hat, L2hat) #Line that passes through both intersection points

		# Check for co-planar orbits
		Lcheck = np.isclose(L1hat, L2hat)  # Check if angular momentum is the same
		if np.all(Lcheck):  # Coplanar orbits
			print("Orbits are co-planar")
			nus1 = None
			nus2 = None
			return nus1, nus2
	
		# Check to see if intersect line vector points to first close pass after RAAN
		if intersectLine[-1] > 0:  # Intersection line points to first crossing after RAAN
			pass
		# Flip intersection lilne to point to first crossing after RAAN
		elif intersectLine[-1] < 0:
			intersectLine = -intersectLine
		# Get Raan direction GCRF
		khat = np.array([0, 0, 1])  # Z direction
		raanVec1 = np.cross(khat, L1hat)
		raanVec2 = np.cross(khat, L2hat)
		# Angle formula: cos(theta) = (a dot b) / (norm(a) * norm(b))
		# Gets angle from True anomaly at RAAN (0 deg) to intersect point
		ab1 = np.dot(raanVec1, intersectLine) / \
			(norm(raanVec1) * norm(intersectLine))
		ab2 = np.dot(raanVec2, intersectLine) / \
			(norm(raanVec2) * norm(intersectLine))
		nu1 = np.arccos(ab1)
		nu2 = np.arccos(ab2)

		# Check for equatorial orbits
		L1Eq = np.isclose(self.inc, 0 * u.deg)
		L2Eq = np.isclose(satellite.inc, 0 * u.deg)
		if L1Eq:
			nu1 = satellite.raan
		elif L2Eq:
			nu2 = self.raan
		else:
			pass
		# return true anomaly of second crossing as well
		nus1raw = [nu1, nu1 + np.pi * u.rad]
		nus2raw = [nu2, nu2 + np.pi * u.rad]

		# Make sure all values under 360 deg
		nus1 = [np.mod(nu, 360*u.deg) for nu in nus1raw]
		nus2 = [np.mod(nu, 360*u.deg) for nu in nus2raw]

		return nus1, nus2

	def schedule_coplanar_intercept(self, targetSat, task='ISL', perturbation='J2', debug=False):
		"""
		Schedule coplanar intercept (2 satellites are in the same orbital plane)
		
		Inputs
		targetSat (Satellite object): Target satellite
		task (String): task to be completed, default is ISL
		perturbation options (string): 'none', 'J2'
		Only no perturbation is currently implemented

		Outputs
		Dictionary with the following
		maneuverObjects [ManeuverObjects]: Maneuver objects for both phasing burns
		deltaVTot : total deltaV of phasing maneuver
		islTime : time of ISL
		islOrb (Satellite object): Orbit after ISL maneuvers 
		debug (optional if debug=True): list of satellites to plot to ensure ISL
		posDiff (optional if debug=True): difference in position between each of the satellites in debug and the initial satellite at time of maneuver
		"""
		if perturbation != 'J2':
			nusInit, nusPass = self.get_nu_intercept(targetSat) #Get anomalies for intercept

			## Get mean anomaly of targetSat 
			nuAtPass = targetSat.nu
			## Check if any of the angles are less than the satellite's anomaly
			lessMask = [nu < nuAtPass for nu in nusPass]
			## Add 2 pi to all angles that are less than targetSat anomaly
			nusIntersect = [nu + 2*np.pi * u.rad if mask else nu for nu, mask in zip(nusPass,lessMask)]

			## Find the next mean anomaly (aka first ISL opportunity after image acquisition)
			nuDiffs = [nu - nuAtPass for nu in nusIntersect]

			# Find minimum difference (First ISL opportunity after ground pass)
			minNu = min(nuDiffs)
			minIdx = nuDiffs.index(minNu)

			targetNu = nusPass[minIdx]

			targetSatAtISL = targetSat.propagate_to_anomaly(targetNu)
			tAtISL = targetSatAtISL.epoch
			tNow = self.epoch

			##Propagate init sat to rendezvous anomaly
			satInitAtAnom = self.propagate_to_anomaly(nusInit[minIdx])
			tInitAtAnom = satInitAtAnom.epoch

			tMan = tAtISL - tInitAtAnom #Time of maneuver

			## Number of orbits for maneuver
			rvdOrbsRaw = tMan.to(u.s) / self.period #Peg orbital maneuvers to relay satellite initial orbit period
			rvdOrbsInt = rvdOrbsRaw.astype(int)

			## Get phase for maneuver
			targetSatAtManInit = targetSat.propagate(satInitAtAnom.epoch) #Match epochs so we get a difference in anomalies
			targetNuInitSat = nusInit[minIdx] #Get mean anomaly required for initial satellite to reach rvd point

			delPhase = targetNu - targetSatAtManInit.nu

			tPhase, delVTot, delV1, delV2, a_phase, passFlag = self._coplanar_phase(self.a, 
						delPhase,
						rvdOrbsInt,
						rvdOrbsInt)

			v_dir1 = satInitAtAnom.v/np.linalg.norm(satInitAtAnom.v)
			delVVec1 = v_dir1 * delV1
			deltaV_man1 = Maneuver.impulse(delVVec1)

			orb1_phase_i = satInitAtAnom.apply_maneuver(deltaV_man1) #Applied first burn
			orb1_phase_f = orb1_phase_i.propagate(tPhase) #Propagate through phasing orbit

			v_dir2 = orb1_phase_f.v/np.linalg.norm(orb1_phase_f.v) #Get direction of velocity at end of phasing
			delVVec2 = v_dir2 * delV2
			deltaV_man2 = Maneuver.impulse(v_dir2 * delV2) #Delta V  of recircularizing orbit
			orb1_rvd = orb1_phase_f.apply_maneuver(deltaV_man2) #Orbit at rendezvous

			## Changed deltaV_man1 to delV1 here. Removed directional information
			manObj1 = ManeuverObject(tInitAtAnom, delV1, tMan, self.satID, targetSat, planeID = self.planeID)
			manObj2 = ManeuverObject(tAtISL, delV2, tMan, self.satID, targetSat, planeID = self.planeID)

			#Create a Transfer object that isn't used for scheduling, but will be used for the planner
			transferObj = Transfer(tAtISL, delVTot, tMan, self.satID, targetSat, planeID = self.planeID)
			transferObj.task = task
			## Reapply attributes that were lost during poliastro propagation
			orb1_rvd.satID = self.satID
			orb1_rvd.planeID = self.planeID
			orb1_rvd.task = task

			orb1_phase_i.satID = self.satID
			orb1_phase_i.planeID = self.planeID
			orb1_phase_i.task = task

			orb1_phase_f.satID = self.satID
			orb1_phase_f.planeID = self.planeID
			orb1_phase_f.task = task

			satInitAtAnom.satID = self.satID
			satInitAtAnom.planeID = self.planeID
			satInitAtAnom.task = task

			##Attribute beginning and ending orbits to maneuvers
			transferObj.mySat = self
			transferObj.planeID = self.planeID
			transferObj.satID = self.satID
			# totManObj.mySatFin = orb1_rvd

			manObj1.mySat = satInitAtAnom
			manObj1.mySatFin = orb1_phase_i

			manObj2.mySat = orb1_phase_f
			manObj2.mySatFin = orb1_rvd


			manList = [manObj1, manObj2]

			## Add required schedule to get to orb1_rvd
			# orb1_rvd.add_previousSchedule(manObj1)
			# orb1_rvd.add_previousSchedule(manObj2)
			transferObj.mySatFin = orb1_rvd
			transferObj.maneuvers = manList #Add the components to teh transffer
			orb1_rvd.add_previousTransfer(transferObj)


			if debug:
				satsList = [satInitAtAnom, orb1_phase_i, orb1_phase_f, orb1_rvd, targetSatAtISL]
				posDiffs = [satInitAtAnom.r - sat.r for sat in satsList]
				output = {'maneuverObjects': manList,
						  'transfer': transferObj,
						  'islTime': tAtISL,
						  'debug': satsList,
						  'posDiff': posDiffs}  
			else:
				output = {'maneuverObjects': manList,
						  'transfer': transferObj,
						  'islTime': tAtISL}

			return output


	def get_pass_from_plane(self, Plane):
		"""
		Get pass details from plane (Also will output mean anomaly and time of pass)
		"""
		pass

	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

## Ground location class
class GroundLoc():
	def __init__(self, lon, lat, h, groundID = None, name = None):
		"""
		Define a ground location. Requires astopy.coordinates.EarthLocation
		lon (deg): longitude
		lat (deg): latitude 
		h (m): height above ellipsoid
		loc (astropy object) : location in getdetic coordinates

		name: is the name of a place (i.e. Boston)
		
		**Methods**
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
		
		def propagate_to_ECI(self, obstime): #Gets GCRS coordinates (ECI)
			return self.loc.get_gcrs(obstime) 
		def get_ECEF(self):
			return self.loc.get_itrs()

## Ground station class
class GroundStation(GroundLoc):
	def __init__(self, lon, lat, h, data, commsPayload = None, groundID = None, name = None):
		GroundLoc.__init__(self, lon, lat, h, groundID, name)
		self.data = data
		self.commsPayload = commsPayload

## Schedule Item class
class ScheduleItem():
	def __init__(self, time, note = None):
		self.time = time
		self.note = note

## Pointing schedule class
class PointingSchedule(ScheduleItem):
	def __init__(self, time, passType = None,direction = None, action = None):
		ScheduleItem.__init__(self, time)
		self.direction = direction
		self.action = action
		self.passType = passType

## ManeuverSchedule Class
class ManeuverSchedule(ScheduleItem):
	def __init__(self, time, deltaV):
		ScheduleItem.__init__(self, time)
		self.deltaV = deltaV

## ManeuverObject Class
class ManeuverObject(ManeuverSchedule):
	"""
	Not to be confused with the poliastro class 'Maneuver'
	"""
	def __init__(self, time, deltaV, time2Pass, satID, target, note=None, 
				 planeID = None, mySat=None, mySatFin=None):
		"""
		time: time of maneuver
		deltaV: cost of manuever (fuel)
		time2Pass: time 
		satID: satellite ID of satellite to make maneuver (TO DO, remove and just use mySat attribute)
		target: target of manuever (ground location or satellite for ISL)
		note: Any special notes to include
		planeID: plane ID of satellite to make maneuver (TO DO, remove and just use mySat attribute)
		mySat: points to the satellite object making the maneuver
		mySatFin: Final orbit after maneuver
		"""
		ManeuverSchedule.__init__(self, time, deltaV)
		self.time2Pass = time2Pass
		self.satID = satID
		self.target = target
		self.note = note
		self.planeID = planeID
		self.mySat = mySat
		self.mySatFin = mySatFin

	def get_dist_from_utopia(self):
		squared = self.time2Pass.to(u.hr).value**2 + self.deltaV.to(u.m/u.s).value**2
		dist = np.sqrt(squared)
		self.utopDist = dist

	def get_weighted_dist_from_utopia(self, w_delv, w_t):
		"""
		Get weighted distance from utopia point 

		Inputs
		w_delv: delV weight
		w_t: time weight
		"""
		squared = self.time2Pass.to(u.hr).value**2 * w_t + self.deltaV.to(u.m/u.s).value**2 * w_delv
		dist = np.sqrt(squared)
		self.utopDistWeighted = dist

class Transfer(ManeuverObject):
	def __init__(self, time, deltaV, time2Pass, satID, target, note=None, 
				 planeID = None, mySat=None, mySatFin=None, maneuvers=None,
				 task=None):
		ManeuverObject.__init__(self, time, deltaV, time2Pass, satID, target, 
			note=note, planeID = planeID, mySat=mySat, mySatFin=mySatFin)
		self.maneuvers = maneuvers
		self.task = task


## Data class
class Data():
	def __init__(self, tStamp, dSize):
		self.tStamp = tStamp
		self.dSize = dSize

## CommsPayload class
class CommsPayload():
	"""
	Inputs
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
	def __init__(self, freq = None, wavelength = None, p_tx = None, 
				 g_tx = None, sysLoss_tx = None, l_tx = None, 
				 pointingErr = None, g_rx = None, t_sys = None,
				 sysLoss_rx = None,  beamWidth = None, aperture = None,
				 l_line = None):
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

		Inputs
		T_ant (K) : Temperature of antenna
		T_feeder (K) : Temperature of feeder
		T_receiver (K) : Temperature of receiver
		L_feeder (dB) : Feeder loss
		"""
		feederTerm = 10**(L_feeder/10) #Convert from dB
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

class MissionOption():
	"""
	Holder for mission options
	"""                       
	def __init__(self, listOfTransfers, maneuverGoal):
		self.listOfTransfers = listOfTransfers     
		self.maneuverGoal = maneuverGoal

	def get_mission_specs(self):
		self.maneuverCosts = [m.deltaV for m in self.listOfTransfers]
		self.totalCost = sum(self.maneuverCosts)

		#Find downlink satellite
		dlSat = [s for s in self.listOfTransfers if s.mySatFin.task == 'Downlink']
		if dlSat:
			self.dlTime = dlSat[0].time2Pass
		else:
			self.dlTime = 'Not scheduled yet'

	def get_weighted_dist_from_utopia(self, w_delv, w_t):
		"""
		Get weighted distance from utopia point 

		Inputs
		w_delv: delV weight
		w_t: time weight
		"""
		squared = self.dlTime.to(u.hr).value**2 * w_t + self.totalCost.to(u.m/u.s).value**2 * w_delv
		dist = np.sqrt(squared)
		self.utopDistWeighted = dist