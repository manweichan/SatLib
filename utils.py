import operator
import copy

def find_non_dominated_time_deltaV(flatArray):
    keyfun = operator.attrgetter('time2Pass.value')
    flatSort = copy.deepcopy(flatArray)
    flatSort.sort(key=keyfun) # Sort list by key value (usually time2Pass.value)
    next_idx = 0
    paretoList = copy.deepcopy(flatSort) #List of efficient points that will continually be cut
    paretoSats = [] #List of satellites on pareto front
    while next_idx < len(paretoList) and next_idx < 100:
        best = paretoList[next_idx] #Best satellite, lowest value by key
        paretoList = [f for f in paretoList if f.deltaV < best.deltaV] #Cut out all longer times that take more deltaV
        paretoSats.append(best) #Add best satellite to list
    return paretoSats
    