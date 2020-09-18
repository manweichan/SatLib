import numpy as np

def ECI2LVLH_mat(r,v):
    """
    Get rotation matrix to convert position/velocity in ECI frame to 
    Local Vertical Local Horizontal (LVLH) frame
    ** No translation **
    Coordinates in LVLH
    X: Complete coordinates as defined by Y and Z (velocity direction)
    Y: Orbit normal
    Z: Radial direction away from Earth
    """
    zHat = r/np.linalg.norm(r)
    yHat = np.cross(r,v)/np.linalg.norm(np.cross(r,v))
    xHat = np.cross(yHat,zHat)
    ECI2LVLH = np.stack((xHat, yHat, zHat)) #Stack in rows
    return ECI2LVLH