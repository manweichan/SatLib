import math
import numpy as np
import matplotlib.pyplot as plt

def calcMinMaxParams(alt=None, disThres=None, numSats= None):
    """
    Given two of the three paramters, the third will be calculated. If an error message is outputed instead, it could be that the calculation lead to a negative number or the communication link is only feasible if it intersets the earth.
    
    Paramters:
    ----------
    alt (int): altitude of satellites
    dist_thres (int): distance threshold for ISL
    numSats (int): number of satellites
    """
    if (numSats==None and disThres==None) or (numSats==None and alt==None) or (alt==None and disThres==None) or (alt==None and numSats==None and disThres==None):
        return 'ERROR!!! Please input two of three available parameters! Or the function will not run!'
    if numSats==None:
        r = 6371 + alt
        if (disThres/(2*r)) > 1:
            return 'ERROR!!! Please increase the minimum altitude or decrease the maximum distance threshold in order to avoid mathematical errors.'
        cent_ang = 2*(math.asin(disThres/(2*r)))
        numSats = (2*math.pi)/cent_ang
        numSats = math.ceil(numSats)
        a = math.sqrt((r**2)-((disThres/2)**2))
        if a < 6371 or numSats <= 0:
            return('ERROR!!! This set of parameters will not allow for continuous communication between satellites. Please do the following to fix the problem: (1) Increase altitude and/or (2) Decrease the distance threshold.')
        else:
            return f'Minimum number of satellites required: {numSats}'
    elif disThres==None:
        r = 6371 + alt
        cent_ang = (2*math.pi)/numSats
        disThres = math.ceil(math.sin(cent_ang/2)*2*r)
        a = math.sqrt((r**2)-((disThres/2)**2))
        if a < 6371 or disThres <= 0:
            return('ERROR!!! This set of parameters will not allow for continuous communication between satellites. Please do the following to fix the problem: (1) Increase altitude and (2) Increase the number of satellites.')
        else:
            return f'Minimum distance threshold required: {disThres}'
    elif alt==None:
        cent_ang = (2*math.pi)/numSats
        r = disThres/(2*math.sin(cent_ang/2))
        alt = math.ceil(r - 6371)
        a = math.sqrt((r**2)-((disThres/2)**2))
        if a < 6371 or alt <= 0:
            return('ERROR!!! This set of parameters will not allow for continuous communication between satellites. Please do the following to fix the problem: (1) Decrease distance threshold and/or (2) Increase the number of satellites.')
        else:
            return f'Maximum altitude allowed: {alt}'

def graphNumSatsHelperFunc(alt, disThres):
    """
    Calculates the minimum number of satellites needed given the altitude and threshold. Helper function for graphNumSats.

    Paramters:
    ----------
    alt (int): altitude of satellites
    disThres (int): distance threshold for ISL
    """
    r = 6371 + alt
    cent_ang = 2*(math.asin(disThres/(2*r)))
    numSats = math.ceil((2*math.pi)/cent_ang)
    a = math.sqrt((r**2)-((disThres/2)**2))
    return numSats, a

def graphNumSats(alt, disThres, varyAlt=False, varyDisThres=False, yLim=None, xLim=None):
    """
    Graphs the number of satellites as y with either the altitude or the distance threshold as x. The remaining parameter will be varied discretly. If you want the x value to be distance treshold and vary the altitude discretly, set varyAlt to True. If you want the x value to be altitude and vary the distance threshold discretly, set varyDisThres to True. Only set either varyAlt or varyDisThres to True. Do not leave both False or set both to True.

    Paramters:
    ----------
    alt (array): altitude of satellites
    disThres (array): distance threshold for ISL
    varyAlt (boolean): whether to vary Alt
    varyDisThres (boolean): whether to vary DisThres
    yLim (list): list of y limits
    xLim (list): list of x limits
    """
    plt.figure()
    if (varyAlt==True and varyDisThres==True) or (varyAlt==False and varyDisThres==False):
        return 'ERROR!!! Please select a parameter to vary by setting either varyAlt or varyDisThres to True.'
    maxDisThres = np.max(disThres)
    minAlt =  np.min(alt)
    minR = minAlt + 6371
    if (maxDisThres/(2*minR)) > 1:
        return 'ERROR!!! Please increase the minimum altitude or decrease the maximum distance threshold in order to avoid mathematical errors.'
    f =  np.vectorize(graphNumSatsHelperFunc)
    if varyAlt:
        for i in alt:
            numSats, a = f(i, disThres)
            #print(f'a for {i}km: {a}')
            for j in range(len(a)):
                if a[j] < 6371:
                    r = 6371 + i
                    disThres[j] = 2*math.sqrt((r**2)-(6371**2))
                    cent_ang = 2*(math.asin(disThres[j]/(2*r)))
                    numSats[j] = math.ceil((2*math.pi)/cent_ang)
            plt.scatter(disThres, numSats, label='alt={}km'.format(i))
            #print(f'a for {i}km: {a}')
        plt.xlabel('Distance Threshold')
        plt.ylabel('Number of Satellites Required')
        plt.grid()
        plt.legend()
        if yLim is not None:
            plt.ylim(yLim)
        if xLim is not None:
            plt.xlim(xLim)
        plt.show()
    else:
        for i in disThres:
            numSats, a = f(alt, i)
            #print(f'a for {i}km: {a}')
            for j in range(len(a)):
                if a[j] < 6371:
                    r = math.sqrt(((i/2)**2)+(6371**2))
                    alt[j] = r - 6371
                    cent_ang = 2*(math.asin(i/(2*r)))
                    numSats[j] = math.ceil((2*math.pi)/cent_ang)
            plt.scatter(alt, numSats, label='disThres={}km'.format(i))
            #print(f'a for {i}km: {a}')
        plt.xlabel('Altitude')
        plt.ylabel('Number of Satellites Required')
        plt.grid()
        if yLim is not None:
            plt.ylim(yLim)
        if xLim is not None:
            plt.xlim(xLim)
        plt.legend()
        plt.show()