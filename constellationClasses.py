import numpy as np
from poliastro import constants
from poliastro.earth import Orbit
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody.propagation import propagate
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation
from poliastro.util import norm
from poliastro.frames.equatorial import GCRS
import matplotlib.pyplot as plt
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.plotting import OrbitPlotter3D, OrbitPlotter2D
import astropy.units as u
import astropy
from astropy import time
from astropy.coordinates import EarthLocation
import OpticalLinkBudget.OLBtools as olb
import utils as ut

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
		
		
	def reconfigure(self, GroundLoc, GroundStation):
		pass
	
	def get_pass_maneuver(self, GroundLoc, tInit, days, plot = False, savePlot = False, figName = None):
		"""
		Gets set a maneuvers that will pass a specific ground location on
		select days

		Inputs:
		GroundLoc (Obj): Ground location object to be passed over
		tInit (astropy time object): Time to initialize planner
		days (int): Amount of days ahead to for the scheduler to plan for
		plot (Bool): Plots maneuvers if True
		savePlot(Bool): Saves plot if True
		figName (str): file name for figure

		Outputs:
		maneuverPoolCut (list of Maneuver objects): Physically viable maneuvers (selected because some maneuvers would cause satellites to crash into Earth)
		maneuverPoolAll (list of Maneuver objects): All potential maneuvers
		paretoSats (list of pareto front satellites): Satellite objects on the pareto front
		"""
		if savePlot:
			assert figName, "Need to name your figure using figName input"

		maneuverPoolAll = []
		maneuverPoolCut = []
		ghostSatsPassAll = []
		ghostSatsInitAll = []
		for plane in self.planes:
			maneuverPlaneCut, maneuverPlaneAll, ghostSatsPass, ghostSatsInit = plane.get_pass_details(GroundLoc, tInit, days)
			maneuverPoolAll.append(maneuverPlaneAll)
			maneuverPoolCut.append(maneuverPlaneCut)
			ghostSatsPassAll.append(ghostSatsPass)
			ghostSatsInitAll.append(ghostSatsInit)

		semiFlatCut = [item for sublist in maneuverPoolCut for item in sublist] #Flatten list
		flatCut = [item for sublist in semiFlatCut for item in sublist]

		paretoSats = ut.find_non_dominated_time_deltaV(flatCut)

		if plot:
			from matplotlib import cm


			c = cm.get_cmap('tab10', len(self.planes)) # color map for plots

			colors = c.colors

			style = ['o', 'x']
			plt.figure()
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
			plt.grid()
			if savePlot:
				plt.savefig(figName, facecolor="w", dpi=300, bbox_inches='tight')
			plt.show()

		return maneuverPoolCut, maneuverPoolAll, paretoSats, ghostSatsInitAll, ghostSatsPassAll
	
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
		deltaVs = []
		timeOfISL = []
		islOrbs = []
		for satInt in selfSats:
			for satTar in targetSats:
				output = satInt.schedule_coplanar_intercept(satTar, perturbation)
				maneuverObjs.append(output['maneuverObjects'])
				deltaVs.append(output['delVTot'])
				timeOfISL.append(output['islTime'])

		paretoSats = ut.find_non_dominated_time_deltaV(deltaVs)

		if plotFlag:
			from matplotlib import cm


			c = cm.get_cmap('tab10', len(self.planes)) # color map for plots

			colors = c.colors

			# style = ['o', 'x']
			plt.figure()
			plt.xlabel('Time (hrs)', fontsize = 14)
			plt.ylabel('Delta V (m/s)', fontsize = 14)
			plt.title('Potential Maneuver Options\n for ISL data transfer', fontsize = 16)

			lgdCheck = []
			paretoTimes = []
			for man in deltaVs:
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
			plt.grid()
			if savePlot:
				plt.savefig(figName, facecolor="w", dpi=300, bbox_inches='tight')
			plt.show()

		return maneuverObjs, deltaVs, timeOfISL, paretoSats

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
		planes2const.append(plane2append)
		return Constellation.from_list(planes2const)
	
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
		## Doesn't seem to work in jupyter notebooks
		sats = self.get_sats()
		op = OrbitPlotter3D()
		for sat in sats:
			op.plot(sat)
		op.show()

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
	def from_list(cls, satList):
		"""
		Create plane from a list of satellite objects
		"""
		if not satList: #Empty list
			plane = []
			print("Empty list entered")
		else:
			for idx, sat in enumerate(satList):
				if idx == 0:
					plane = cls(sat)
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
		
	def get_pass_details(self, GroundLoc, tInit, days):
		"""
		Gets set a maneuvers that will pass a specific ground location on
		select days

		Inputs:
		GroundLoc (Obj): Ground location object to be passed over
		tInit (astropy time object): Time to initialize planner
		days (int): Amount of days ahead to for the scheduler to plan for

		Outputs:
		maneuverListAll (list of Maneuver objects): All potential maneuvers
		maneuverListCut (list of Maneuver objects): Physically viable maneuvers (selected because some maneuvers would cause satellites to crash into Earth)
		"""
		maneuverListAll = []
		maneuverListCut = []
		ghostSatsPassAll = []
		ghostSatsInitAll = []
		for sat in self.sats:
			ghostSatsPass, ghostSatsInit, potentialManeuversAll, potentialManeuversCut = sat.get_desired_pass_orbs(GroundLoc, tInit, days)
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
		return maneuverListCut, maneuverListAll, ghostSatsPassAll, ghostSatsInitAll

## Satellite class 
class Satellite(Orbit):
	"""
	Defines the satellite class
	"""
	def __init__(self, state, epoch, satID = None, dataMem = None, schedule = None, 
				 commsPayload = None, potentialManSchedule = None, planeID = None,
				 note = None):
		super().__init__(state, epoch) #Inherit class for Poliastro orbits (allows for creation of orbits)
		self.satID = satID
		self.planeID = planeID
		self.note = note        
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
	
		if potentialManSchedule is None: #Set up ability to hold potential maneuver schedule
			self.potentialManSchedule = []
		else:
			self.potentialManSchedule = potentialManSchedule
			
	def calc_link_budget_tx(self, rx_comms_payload):
		pass
	
	def schedule_intercept(sat):
		pass
	
	def add_data(self, data): #Add data object to Satellite
		self.dataMem.append(data)
	
	def remove_data(self, data): #Remove data object from Satellite (i.e. when downlinked)
		if data in self.dataMem:
			self.dataMem.remove(data)
			
	def add_schedule(self, sch_commands): #Adds a schedule to the satellite
		self.schedule.append(sch_commands)
	
	def add_comms_payload(self, commsPL):
		self.commsPayload.append(commsPL)
		
	def reset_payload(self):
		self.commsPayload = []
		
	def calc_access(self, GroundLoc, timeSpan):
		pass
	
	def get_desired_pass_orbs(self, GroundLoc, tInit, days,
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
		refVernalEquinox (astropy time object): Date of vernal equinox. Default is for 2021

		Outputs:
		desiredGhostOrbits_atPass (array of Satellite objects): Orbit of satellite at ground pass (potential position aka ghost position)
		desiredGhostOrbits_tInit (array of Satellite objects): Orbit of ghost satellite if it were propagated back to initial time
		maneuverObjects (array of Manuever objects): Maneuver object class that holds time and cost of maneuvers
		"""

		## TODO: Doesn't yet account for RAAN drift due to J2 perturbation
		
		satInit = self.propagate(tInit)
		
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
		
		tPass_a = [day + angleToHrs_a[idx] * u.hr for idx, day in enumerate(days)]
		tPass_d = [day + angleToHrs_d[idx] * u.hr for idx, day in enumerate(days)]
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

			desiredGhostOrbits_atPass.append(ghostSatFuture)
			desiredGhostOrbits_tInit.append(ghostSatInit)
			
			## Get rendezvous time and number of orbits to phase
			time2pass = ghostSatFuture.epoch - tInit
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
			manObj = ManeuverObject(tInit, delVTot, time2pass, self.satID,
								   ghostSatFuture, note=sch.passType)

			## Tag manuever object with maneuver number (to match ghost orbits later)
			manObj.manID = idx

			maneuverObjectsAll.append(manObj)
			if passFlag:
				maneuverObjectsCut.append(manObj)
			

			
			
		return desiredGhostOrbits_atPass, desiredGhostOrbits_tInit, maneuverObjectsAll, maneuverObjectsCut
		
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

	def schedule_coplanar_intercept(self, targetSat, perturbation='J2', debug=False):
		"""
		Schedule coplanar intercept (2 satellites are in the same orbital plane)
		
		Inputs
		targetSat (Satellite object): Target satellite
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

			manObj1 = ManeuverObject(tInitAtAnom, deltaV_man1, tMan, self.satID, targetSat, planeID = self.planeID)
			manObj2 = ManeuverObject(tAtISL, deltaV_man2, tMan, self.satID, targetSat, planeID = self.planeID)

			#Create a maneuver object that isn't used for scheduling, but will be used for the planner
			totManObj = ManeuverObject(tAtISL, delVTot, tMan, self.satID, targetSat, planeID = self.planeID)
			totManObj.mySat = self

			## Reapply attributes that were lost during poliastro propagation
			orb1_rvd.satID = self.satID
			orb1_rvd.planeID = self.planeID

			totManObj.mySatFin = orb1_rvd

			manList = [manObj1, manObj2]

			if debug:
				satsList = [satInitAtAnom, orb1_phase_i, orb1_phase_f, orb1_rvd, targetSatAtISL]
				posDiffs = [satInitAtAnom.r - sat.r for sat in satsList]
				output = {'maneuverObjects': manList,
				          'delVTot': totManObj,
				          'islTime': tAtISL,
				          'debug': satsList,
				          'posDiff': posDiffs}	
			else:
				output = {'maneuverObjects': manList,
				          'delVTot': totManObj,
				          'islTime': tAtISL}

			return output


	def get_pass_from_plane(self, Plane):
		"""
		Get pass details from plane (Also will output mean anomaly and time of pass)
		"""
		pass

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
				 pointingErr = None, g_rx = None, tSys = None,
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
		self.tSys = tSys
		self.sysLoss_rx = sysLoss_rx
		self.beamWidth = beamWidth
		self.aperture = aperture
		self.l_line = l_line
		
	
	def get_EIRP(self):
		# TRY WITH ASTROPY UNITS
		P_tx = self.P_tx
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
		self.T_sys = Tsys
		
	def get_GonT(self):
		# TRY WITH ASTROPY UNITS
		T_dB = 10 * np.log10(self.T_sys)
		GonT_dB = self.G_r - T_dB - self.L_line
		return GonT_dB                                