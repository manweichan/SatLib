import numpy as np

def exhaustVel_fromISP(isp, g0 = 9.80665):
	"""
	Calculate exhaust velocity from Isp
	Inputs (isp, g0)
	isp : specific impulse in seconds
	g0 : gravitational constant, default is 9.80665 m/s^2

	Outputs exhaust velocity in m/s
	"""
	v = g0 * isp
	return v

def ISP_fromExhaustVel(ve, g0 = 9.80665):
	"""
	Calculate exhaust Isp from exhaust velocity
	Inputs (ve, g0)
	ve : exhaust velocity in m/s
	g0 : gravitational constant, default is 9.80665 m/s^2

	Outputs exhaust velocity in m/s
	"""
	isp = ve / g0
	return isp

def delV_rocketEqn(isp, m0, mf, g0 = 9.80665):
	"""
	Calculate delta V using the rocket eqn
	Inputs (isp, m0, mf, g0)
	isp : specific impulse in seconds
	m0  : initial mass (wet mass)
	mf  : final mass (dry mass)
	g0 : gravitational constant, default is 9.80665 m/s^2
	"""
	delV = isp * g0 * np.log(m0/mf)
	return delV

def m0_rocketEqn(mf, delV, isp, g0 = 9.80665):
	"""
	Calculate initial mass using the rocket eqn
	Inputs (isp, delV, mf, g0)
	isp  : specific impulse in seconds
	delV : delta V of maneuver (m/s)
	mf   : final mass (dry mass)
	g0   : gravitational constant, default is 9.80665 m/s^2

	Output
	m0   : initial mass (kg)
	"""
	m0 = mf * np.exp(delV/(isp * g0))
	return m0

def mf_rocketEqn(m0, delV, isp, g0 = 9.80665):
	"""
	Calculate final mass using the rocket eqn
	Inputs (isp, delV, m0, g0)
	isp  : specific impulse in seconds
	delV : delta V of maneuver (m/s)
	m0   : initial mass (wet mass)
	g0   : gravitational constant, default is 9.80665 m/s^2

	Output
	mf   : final mass (kg)
	"""
	mf = m0 * np.exp(-delV/(isp * g0))
	return mf