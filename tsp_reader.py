import re

def opt_reader(opt_name):
    opt = open(opt_name, 'r')
    #index 2 because spaces between ":"
    NAME = opt.readline().strip().split()[2] # NAME
    COMMENT = opt.readline().strip().split()[2]

    TYPE = opt.readline().strip().split()[2]

    DIMENSION = int(opt.readline().strip().split()[2]) # DIMENSION    
    opt.readline()
    path = []
    for i in range(DIMENSION):
        path.append(int(opt.readline().strip())-1)
    opt.close
    return path


def tsp_reader(tspName):
    tsp = open(tspName, 'r')
# Read instance header
    #if I ever want to make an better parser
    NAME = tsp.readline().strip().split()[1] # NAME
    TYPE = tsp.readline().strip().split()[1] # TYPE
    COMMENT = tsp.readline().strip().split()[1] # COMMENT
    DIMENSION = int(tsp.readline().strip().split()[1]) # DIMENSION
    EDGE_WEIGHT_TYPE = tsp.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
    EDGE_WEIGHT_FORMAT = tsp.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
    DISPLAY_DATA_TYPE = tsp.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
    EDGE_WEIGHT_SECTION = tsp.readline()
    a = []
    line = tsp.readline()
    for i in range(DIMENSION):
        a.append([x for x in re.split(r'\s+',line.strip())])
        line = tsp.readline()
    tsp.close()
    return a

#tsp_reader("bays29.tsp")
