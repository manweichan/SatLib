import satbox as sb
import utils as utils
import astropy.units as u



##################################################################################
############################ MISCELLANEOUS FUNCTIONS #############################
##################################################################################
def check_if_all_none(array_of_elem):
    """ Check if all elements in list are None """
    result = True
    for index in array_of_elem:
        for elem in index:
            if elem is not None:
                return False
    return result





##################################################################################
############# FUNCTIONS FOR FINDING ISL USING DISTANCE_THRESHOLD ISL #############
##################################################################################
def get_relative_position_data_ISL(constellation,distance_threshold_isl):
    """
    Returns relative position data (array?)

    Parameters
    ----------
    constellation (obj): the constellation object
    distance_threshold_isl (int) : threshold (kilometers)
    """

    #Get relative position data between satellites in the constellation
    relative_position_data = constellation.get_relative_velocity_analysis()
    #Get feasibility of intersatellite links/getting the true/false array for each satellite-to-satellite pair
    distanceThreshold = distance_threshold_isl * u.km
    utils.get_isl_feasibility(relative_position_data,
                        distanceConstraint=distanceThreshold)
    return relative_position_data



def find_feasible_links_ISL(num_sats, relative_position_data):
    """
        Returns a list of strings with the pairs of satellites that can communicate during the set time interval
        
        Parameters
        ----------
        num_sats (int) : the number of satellites in the constellation
        relative_position_data (array?) : the array of relative_position_data
            
    """
    #Takes the number of satellites and gives a list of every possible pair of satellites
    L=[]
    for i in range(num_sats):
        L.append(str(i))

    L1=[]
    for i in range(len(L)):
        j=i+1
        while j != len(L):
            L1.append(L[i]+'-'+L[j])
            j+=1
    
    
    L2 = L1.copy()

    #Eliminating the pairs from the list that are never able to communicate
    #iterate through the pairs satellites
    for i in L2: 
       
        #True false mask of ISL opportunities
        mask = relative_position_data['satData'][i]['islFeasible']
       
        #If this pair has no True time intervals, eliminate it from the list
        if any(mask) == False:
            L1.remove(i)

    return L1



def get_availability_ISL(satellites,relative_position_data):
    
    
    """
        Ruturns a list of availiabilities for the list of pairs of satellites
        
        Parameters
        ----------
        satellites (list of strings) : the pairs of satellites get availabilty for
        relative_position_data (array?) : the array of relative_position_data
            
    """
    #True false mask of ISL opportunities
    #Gets satellite data between sateelites 
    #Iterates through the pairs of satellites
    L_avail=[]
    for i in satellites:
        mask = relative_position_data['satData'][i]['islFeasible'] 
        timeArray = relative_position_data['satData'][i]['times']

        #Apply true false mask to time array
        intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
        #If there are no True intervals, we let user know
        if any(mask)==False:
            return 'No Intervals Found: Intersatellite Comunication Link is not Feasible'
        else:
            L=[]
            for idx, interval in enumerate(intervalsISL):
                start_time = interval[0]
                stop_time = interval[1]
    
                #Format time to isot format
                L.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')

        L_avail.append(L)
    
    return L_avail



def get_polyline_ISL(satellites, relative_position_data):
        
    
    """
        Returns of list of the polyline intervals for the list of pairs of satellites.
        
        Parameters
        ----------
        satellites (list of strings): the pairs of satellites to find polyline intervals for
        relative_position_data (array?) : the array of relative_position_data
    """

    #True false mask of ISL opportunities
    #Gets satellite data between sateelites
    #Iterate throught the pairs of satellites 
    L_poly=[]
    for i in satellites:
        mask = relative_position_data['satData'][i]['islFeasible'] 
        timeArray = relative_position_data['satData'][i]['times']

        #If there are no True intervals, we let user know
        if any(mask)==False:
            return 'Intersatellite Comunication Link is not Feasible'
    
        #Get true intervals
        true_intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
        L_true=[]
        for idx, interval in enumerate(true_intervalsISL):
            start_time = interval[0]
            stop_time = interval[1]
    
            L_true.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')
        
        #Get false intervals
        false_intervalsISL = utils.get_false_intervals(mask, timeArray)
        L_false=[]
        #If there are no False Intervals
        if check_if_all_none(false_intervalsISL):
            pass
        else:
            for idx, interval in enumerate(false_intervalsISL):
                start_time = interval[0]
                stop_time = interval[1]
    
                L_false.append(start_time.isot + 'Z/' + stop_time.isot + 'Z')
    
        #Creating dictionary
        L_final=[]
        if mask[0]==True:
            for i in range(len(L_true)):
                try:
                    D1={}
                    D1['interval']=L_true[i]
                    D1['boolean']='true'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_false[i]
                    D2['boolean']='false'
                    L_final.append(D2)
                except:
                    pass
        if mask[0]==False:
            for i in range(len(L_false)):
                try:
                    D1={}
                    D1['interval']=L_false[i]
                    D1['boolean']='false'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_true[i]
                    D2['boolean']='true'
                    L_final.append(D2)
                except:
                    pass
        L_poly.append(L_final)

    return L_poly





##################################################################################
####### FUNCTIONS FOR FINDING GS TO SAT LINKS USING DISTANCE_THRESHOLD ISL #######
##################################################################################
def get_relative_position_data_GS(constell, GS_pos, elevation_angle_threshold_gs):
    """
        Returns a dictionary with all the necessary information to visualize links
        
        Parameters
        ----------
        constellation (obj) : the constellation object we are looking at 
        GS_pos (list of list) : the list of GS long and lat coordinates
        elevation_angle_threshold_gs (int) : the ange threshold
            
    """
# Visualizing GS to sat links using elevation_angle_threshold_gs
    L = []
    for gs in GS_pos:
        gs[0] = gs[0] *u.deg
        gs[1] = gs[1] *u.deg
        h = 0*u.km
        L.append(sb.GroundLoc(gs[0],gs[1],h))
    n = 0
    for i in L:
        i.groundID = n
        n += 1
# GS can only see satellite when it's above 20 deg in elevation
    constraint_type = 'elevation'
    constraint_angle = elevation_angle_threshold_gs * u.deg
#Create an access object
    accessObject = sb.DataAccessConstellation(constell, L)
    accessObject.calc_access(constraint_type, constraint_angle)

    dict = {}
    for i in accessObject.allAccessData:
        sat_ID = str(i.satID)
        GS_ID = str(i.groundLocID)
        mask = i.accessMask
        dict[sat_ID + '-GS' + GS_ID] = mask
    dict['times'] = constell.constellation.planes[0].sats[0].rvECEF.obstime.isot  

    return dict



def find_feasible_links_GS(relative_position_data):
    
    """
        Returns a list of strings with the pairs of satellites that can communicate during the set time interval
        
        Parameters
        ----------
        relative_position_data (dict) : the dict of relative_position_data
            
    """

    #All possible pairs of GS and Sats
    L = []
    for i in relative_position_data.keys():
        L.append(i)
    L.remove('times')
    #Eliminating the pairs from the list that are never able to communicate
    #iterate through the pairs satellites
    for i in L: 
        #True false mask of ISL opportunities
        mask = relative_position_data[i]
        #If this pair has no True time intervals, eliminate it from the list
        if any(mask) == False:
            L.remove(i)

    return L



def get_availability_GS(objects, relative_position_data):
    
    """
        Returns a list of strings with the time intervals for communication avaialability
        
        Parameters
        ----------
        objects (list) : list of object pairs
        relative_position_data (array?) :the array of relative_position_data
            
    """

    #True false mask of ISL opportunities
    #Gets satellite time interval data between objects
    #Iterates through the pairs of objects
    L_avail=[]
    for i in objects:
        mask = relative_position_data[i] 
        timeArray = relative_position_data['times']

    #Apply true false mask to time array
        intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
    #If there are no True intervals, let user know
        if any(mask)==False:
            return 'No Intervals Found: Intersatellite Comunication Link is not Feasible'
    #Reformatting the data into a format czml can use
        else:
            L=[]
            for idx, interval in enumerate(intervalsISL):
                start_time = interval[0]
                stop_time = interval[1]
    
        
                L.append(start_time + 'Z/' + stop_time + 'Z')

        L_avail.append(L)
    
    return L_avail



def get_polyline_GS(objects, relative_position_data):
    
    """
        Returns of list of the polyline intervals for the list of pairs of objects.
        
        Parameters
        ----------
        objects (list) : list of object pairs
        relative_position_data (array?) : the array of relative_position_data
    """

    #True false mask of ISL opportunities
    #Gets position data
    #Iterate throught the pairs of satellites 
    L_poly=[]
    for i in objects:
        mask = relative_position_data[i] 
        timeArray = relative_position_data['times']

         #If there are no True intervals, we let user know
        if any(mask)==False:
            return 'Intersatellite Comunication Link is not Feasible'

        #Get true intervals
        true_intervalsISL = utils.get_start_stop_intervals(mask, timeArray)
        L_true=[]
        for idx, interval in enumerate(true_intervalsISL):
            start_time = interval[0]
            stop_time = interval[1]
    
            L_true.append(start_time + 'Z/' + stop_time + 'Z')
        
        #Get false intervals
        false_intervalsISL = utils.get_false_intervals(mask, timeArray)
        L_false=[]
        #If there are no False Intervals
        if check_if_all_none(false_intervalsISL):
            pass
        else:
            for idx, interval in enumerate(false_intervalsISL):
                start_time = interval[0]
                stop_time = interval[1]
    
                L_false.append(start_time + 'Z/' + stop_time + 'Z')
    
        #Creating dictionary
        L_final=[]
        if mask[0]==True:
            for i in range(len(L_true)):
                try:
                    D1={}
                    D1['interval']=L_true[i]
                    D1['boolean']='true'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_false[i]
                    D2['boolean']='false'
                    L_final.append(D2)
                except:
                    pass
        if mask[0]==False:
            for i in range(len(L_false)):
                try:
                    D1={}
                    D1['interval']=L_false[i]
                    D1['boolean']='false'
                    L_final.append(D1)
                    D2={}
                    D2['interval']=L_true[i]
                    D2['boolean']='true'
                    L_final.append(D2)
                except:
                    pass
        L_poly.append(L_final)

    return L_poly




