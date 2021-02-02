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
import OpticalLinkBudget.OLBtools as olb

class Satellite(Orbit):
    def __init__(self, state, epoch, dataMem = None, schedule = None, txOpticalPayload = None, rxOpticalPayload = None,
    txRfPayload = None, rxRfPayload = None):
        super().__init__(state, epoch) #Inherit class for Poliastro orbits (allows for creation of orbits)
        
        if dataMem is None: #Set up ability to hold data
            self.dataMem = []
        else:
            self.dataMem = dataMem
            
        if schedule is None: #Set up ability to hold a schedule
            self.schedule = []
        else:
            self.schedule = schedule

        if txOpticalPayload is None: #Set up ability to hold an optical payload
            self.txOpticalPayload = []
        else:
            self.txOpticalPayload = txOpticalPayload

        if rxOpticalPayload is None: #Set up ability to hold an optical payload
            self.rxOpticalPayload = []
        else:
            self.rxOpticalPayload = rxOpticalPayload

        if txRfPayload is None: #Set up ability to hold an optical payload
            self.txRfPayload = []
        else:
            self.txRfPayload = txRfPayload

        if rxRfPayload is None: #Set up ability to hold an optical payload
            self.rxRfPayload = []
        else:
            self.rxRfPayload = rxRfPayload
    
    # # @classmethod
    # def propagate(self, value, method=farnocchia, rtol=1e-10, **kwargs):
    #     # print("godo")
    #     super().propagate(self, value, method=farnocchia, rtol=1e-10, **kwargs)

    def add_data(self, data): #Add data object to Satellite
        self.dataMem.append(data)
    
    def remove_data(self, data): #Remove data object from Satellite (i.e. when downlinked)
        if data in self.dataMem:
            self.dataMem.remove(data)
            
    def add_schedule(self, sch_commands): #Adds a schedule to the satellite
        self.schedule.append(sch_commands)

    def add_txOpticalPayload(self, txOpticalPL): #Add a tx optical payload
        self.txOpticalPayload.append(txOpticalPL)

    def add_rxOpticalPayload(self, rxOpticalPL): #Add an rx optical payload
        self.rxOpticalPayload.append(rxOpticalPL)

    def add_txRfPayload(self, txRfPL): #Add a tx optical payload
        self.txRfPayload.append(txRfPL)

    def add_rxRfPayload(self, rxRfPL): #Add an rx optical payload
        self.rxRfPayload.append(rxRfPL)

    def resetPayloads(self):
        self.rxOpticalPayload = []
        self.txOpticalPayload = []
        self.rxRFPayload = []
        self.txRFPayload = []

    def calc_rfLinkBudgetAsTx(self, RxObject, L_atm = 0, L_pointing = 0, L_pol = 0, txID = 0, rxID = 0, verbose = False): 
        """
        Calculate link budget with self as transmitter
        
        Inputs
        RxObject: Satellite or GroundStation Object
        L_atm (dB): Atmospheric loss
        L_pointint (dB): pointing loss
        L_pol (dB): Polarization loss
        txID: Which Tx payload to use. Default to 0 since we will most likely just have 1 payload
        rxID: Which Rx payload to use. Default to 0 since we will most likely just have 1 payload
        verbose (Bool): If True, print components of link budget

        Outputs
        P_rx (dBW): Received power
        ConN0 (dBHz): Signal to noise ratio

        """
        # try: 
        assert hasattr(self, 'txRfPayload'), "Need to add txRFPayload to Transmitting Satellite"
        assert hasattr(RxObject, 'rxRfPayload'), "Need to add rxRFPayload to RxObject Satellite"

        #Get tx payload
        txPL = self.txRfPayload[txID]
        EIRP_dB = txPL.get_EIRP()
        wl = txPL.wavelength #wavelength

        #Get rx payload
        rxPL = RxObject.rxRfPayload[rxID]
        G_rx = rxPL.G_r

        posVecTx = self.r #position of tx satellite (ECI)

        if RxObject.__class__.__name__ == 'Satellite':
            posVecRx = RxObject.r #Position of rx satellite (ECI)
        elif RxObject.__class__.__name__ == 'GroundStation':
            posVecRx = RxObject.propagate_to_ECI(self.epoch).data.without_differentials().xyz
        else:
            print("RxObject Class not recognized")


        posVecDiff = astropy.coordinates.CartesianRepresentation(posVecTx - posVecRx)
        dist = posVecDiff.norm().to(u.m)

        GonT_dB = rxPL.get_GonT()
        L_fs = (4 * np.pi * dist / wl) ** 2 #Free space path loss
        L_fs_dB = 10 * np.log10(L_fs.value)
        if L_fs_dB == float("inf") or L_fs_dB == float("-inf"):
            L_fs_dB = 0

        L_other_dB = L_pointing + L_pol

        L_u = L_fs_dB + L_atm


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



        return P_rx, ConN0
        # except AttributeError:
        #     print("Need to add txRFPayload to Transmitting Satellite")

    def calc_optLinkBudgetAsTx(self, RxObject, txID = 0, rxID = 0, verbose = False):
        """
        Calculate optical link budget with self as transmitter

        Inputs
        RxObject: Satellite or GroundStation Object
        txID: Which Tx payload to use. Default to 0 since we will most likely just have 1 payload
        rxID: Which Rx payload to use. Default to 0 since we will most likely just have 1 payload
        """

        txPL = self.txOpticalPayload[txID]

        #Get TxPayload parameters
        P_tx = txPL.P_tx
        beam_width = txPL.beam_width
        lambda_gl = txPL.wavelength
        pointing_error = txPL.pointingErr
        tx_system_loss = txPL.sysLoss

        #Get rx payload
        rxPL = RxObject.rxOpticalPayload[rxID]
        rx_system_loss = rxPL.sysLoss
        aperture = rxPL.aperture

        #Get position vectors
        posVecTx = self.r #position of tx satellite (ECI)

        if RxObject.__class__.__name__ == 'Satellite':
            posVecRx = RxObject.r #Position of rx satellite (ECI)
        elif RxObject.__class__.__name__ == 'GroundStation':
            posVecRx = RxObject.propagate_to_ECI(self.epoch).data.without_differentials().xyz
        else:
            print("RxObject Class not recognized")

        posVecDiff = astropy.coordinates.CartesianRepresentation(posVecTx - posVecRx)
        dist = posVecDiff.norm().to(u.m)

        #Position error at receiver
        r = np.tan(txPL.pointingErr)*dist

        # beam waist
        W_0 = olb.fwhm_to_radius(beam_width,lambda_gl)

        # Angular wave number, = 2*pi/lambda
        k = olb.angular_wave_number(lambda_gl)

        range_loss = olb.path_loss_gaussian(W_0, lambda_gl, dist.value, aperture, pointing_error)

        all_losses = range_loss - tx_system_loss - rx_system_loss

        P_tx_dB = 10 * np.log10(P_tx)

        P_rx_dB = P_tx_dB + all_losses

        P_rx = P_tx*10**(all_losses/10)

        if verbose:
            print("P_tx: ", P_tx)
            print("Distance: ", dist)
            print("Position error at receiver",  r)
            print("Beam waist: ",W_0)
            print("Wave number: ", k)
            print("range loss: ",range_loss)
            print("All losses: ", all_losses)
            print("P_rx_dB: ",P_rx_dB)

        return P_rx_dB, P_rx


class TxOpticalPayload(): #Defines tx optical payload
    def __init__(self, P_tx, wavelength, beam_width, pointingErr = 0, sysLoss = 0): # Add tx optical payload
        """
        Defines an optical payload tx
        P_tx (W): Transmit power
        wavelength (m): wavelength in meters
        beam_width (rad): FWMH, Beam width
        pointingErr (rad): pointing error
        sysLoss (dB): System loss
        """
        self.P_tx = P_tx
        self.wavelength = wavelength
        self.beam_width = beam_width
        self.pointingErr = pointingErr
        self.sysLoss = sysLoss

class RxOpticalPayload(): #Defines tx optical payload
    def __init__(self, aperture, sysLoss = 0): # Add tx optical payload
        """
        Defines an optical payload rx
        aperture (m): aperture diameter
        sysLoss (dB): System loss
        """
        self.aperture = aperture
        self.sysLoss = sysLoss

class TxRfPayload(): #Define EIRP of the txRFPayload
    def __init__(self, P_tx, G_tx, wavelength, L_tx = 0):
        """
        Defines an RF tx payload
        P_tx (W): Transmit power
        G_tx (dB): Gain of the transmitter
        wavelength (m): wavelength in meters
        L_tx (dB): System loss like line loss
        """
        self.P_tx = P_tx
        self.G_tx = G_tx
        self.wavelength = wavelength
        self.L_tx = L_tx

    def get_EIRP(self):
        P_tx = self.P_tx
        P_tx_dB = 10 * np.log10(P_tx)

        L_tx_dB = self.L_tx
        G_tx_dB = self.G_tx

        EIRP_dB = P_tx_dB - L_tx_dB + G_tx_dB
        return EIRP_dB

class RxRfPayload(): #Define RF receiver
    def __init__(self, G_r, T_sys, L_line = 0):
        """
        Defines an RF rx payload
        G_r (dB): Gain of the receiver
        T_sys (K): Temperature of the system
        L_line (dB): Line loss at receiver
        """
        self.G_r = G_r
        self.T_sys = T_sys
        self.L_line = L_line

    def set_sysTemp(self, T_ant, T_feeder, T_receiver, L_feeder):
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
        # GonT = self.G_r / self.T_sys
        # GonT_dB = 10 * np.log10(GonT)
        T_dB = 10 * np.log10(self.T_sys)
        # print(T_dB)
        GonT_dB = self.G_r - T_dB - self.L_line
        return GonT_dB


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
        
class XLinksSchedule():
    def __init__(self, time, direction):
        """
        Defnes crosslinks for schedules
        time: time of xLink ####TODO#### Figure out if we want to do this by epoch
        direction (ECI Frame): pointing direction of xLink
        """
        self.time = time
        self.direction = direction
        
class DLinksSchedule():
    def __init__(self, time, direction):
        """
        Defnes downlinks for schedules
        time: time of xLink ####TODO#### Figure out if we want to do this by epoch
        direction (ECI Frame): pointing direction of xLink
        """
        self.time = time
        self.direction = direction
        
class ImageSchedule():
    def __init__(self, time, direction):
        """
        Defnes image taking schedule
        time: time of xLink ####TODO#### Figure out if we want to do this by epoch
        direction (ECI Frame): pointing direction of xLink
        """
        self.time = time
        self.direction = direction
        
class GroundStation():
    def __init__(self, lon, lat, h, data = None, txOpticalPayload = None, rxOpticalPayload = None,
    txRfPayload = None, rxRfPayload = None):
        """
        Define a ground station. Requires astopy.coordinates.EarthLocation
        lat (deg): latitude 
        lon (deg): longitude
        h (m): height above ellipsoid
        loc (astropy object) : location in getdetic coordinates
        data (data obj): Data transferred to ground station

        Methods
        propagate_to_ECI(self, obstime)
        get_ECEF(self)
        add_txOptical(self, txOpticalPL)
        add_rxOptical(self, rxOpticalPL)
        add_txRf(self, txRfPL): #Add a tx optical payload
        add_rxRf(self, rxRfPL): #Add an rx optical payload
        """
#         super().__init__()
        self.lon = lon
        self.lat = lat
        self.h = h
        self.loc = EarthLocation.from_geodetic(lon, lat, height=h, ellipsoid='WGS84')
        
        if data is None: #keep track of downloaded data
            self.data = []
        else:
            self.data = data

        if txOpticalPayload is None: #Set up ability to hold an optical payload
            self.txOpticalPayload = []
        else:
            self.txOpticalPayload = txOpticalPayload

        if rxOpticalPayload is None: #Set up ability to hold an optical payload
            self.rxOpticalPayload = []
        else:
            self.rxOpticalPayload = rxOpticalPayload

        if txRfPayload is None: #Set up ability to hold an optical payload
            self.txRfPayload = []
        else:
            self.txRfPayload = txRfPayload

        if rxRfPayload is None: #Set up ability to hold an optical payload
            self.rxRfPayload = []
        else:
            self.rxRfPayload = rxRfPayload
    def propagate_to_ECI(self, obstime): #Gets GCRS coordinates (ECI)
        return self.loc.get_gcrs(obstime) 
    
    def get_ECEF(self):
        return self.loc.get_itrs()

    def add_txOptical(self, txOpticalPL): #Add a tx optical payload
        self.txOpticalPayload.append(txOpticalPL)

    def add_rxOptical(self, rxOpticalPL): #Add an rx optical payload
        self.rxOpticalPayload.append(rxOpticalPL)

    def add_txRf(self, txRfPL): #Add a tx optical payload
        self.txRfPayload.append(txRfPL)

    def add_rxRf(self, rxRfPL): #Add an rx optical payload
        self.rxRfPayload.append(rxRfPL)
    
    #Note ITRF is body fixed (ECEF)
        