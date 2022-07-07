import math

def calcMinMaxParams(alt=None, dist_thres=None, numSats= None):
    """
    Paramters:
    ----------
    alt (int): altitude of satellites
    dist_thres (int): distance threshold for ISL
    numSats (int): number of satellites
    """
    if (numSats==None and dist_thres==None) or (numSats==None and alt==None) or (alt==None and dist_thres==None) or (alt==None and numSats==None and dist_thres==None):
        return 'Please input two of three available parameters!'
    elif numSats==None:
        C = 2*math.pi*(6371+alt)
        sats = C/dist_thres
        sats = math.ceil(sats)
        if sats < 0:
            return 'These parameters do not produce viable results.'
        else:
            return 'Minimum number of satellites required: {}'.format(sats) 
    elif dist_thres==None:
        C = 2*math.pi*(6371+alt)
        dist = C/numSats 
        dist = math.ceil(dist)
        if dist < 0:
            return 'These parameters do not produce viable results.'
        else:
            return 'Minimum distance_threshold required: {}'.format(dist)
    elif alt==None:
        alt = ((numSats*dist_thres)/(2*math.pi)) - 6371
        alt = math.ceil(alt)
        if alt < 0:
            return 'These parameters do not produce viable results.'
        else:
            return 'Maximum altitude allowed: {}'.format(alt)