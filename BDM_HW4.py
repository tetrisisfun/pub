from pyspark import SparkContext
import sys

def createIndex(shapefile):
    '''
    This function takes in a shapefile path, and return:
    (1) index: an R-Tree based on the geometry data in the file
    (2) zones: the original data of the shapefile
    
    Note that the ID used in the R-tree 'index' is the same as
    the order of the object in zones.
    '''
    import rtree
    import fiona.crs
    import geopandas as gpd
    
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)


def findZone(p, index, zones):
    '''
    findZone returned the ID of the shape (stored in 'zones' with
    'index') that contains the given point 'p'. If there's no match,
    None will be returned.
    '''
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None

def processTrips(pid, records):
    '''
    Our aggregation function that iterates through records in each
    partition, checking whether we could find a zone that contain
    the pickup location.
    '''
    import csv
    import pyproj
    import shapely.geometry as geom  
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index, zones = createIndex('neighborhoods.geojson')    
    
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    counts = {}
    
    for row in reader:
        p = geom.Point(proj(float(row[9]), float(row[10]))) 
        # Look up a matching zone, and update the count accordly if such a match is found
        zone = findZone(p, index, zones) 
        if zone:
            counts[zone] = counts.get(zone, 0) + 1
    return counts.items()
            
if __name__=='__main__':
   
    sc = SparkContext()
    input_file = sys.argv[1]
    yellow_taxi = sc.textFile(input_file)

    counts = yellow_taxi.mapPartitionsWithIndex(processTrips) \
        .reduceByKey(lambda x,y: x+y) \
        .collect()

    countsPerNeighborhood = map(lambda x: (zones['neighborhood'][x[0]], zones['borough'][x[0]],x[1]), counts)

    rddattempt = sc.parallelize(countsPerNeighborhood)

    rddattempt_1 = rddattempt.map(lambda x:((x[1]),(x[2],x[0])))\
            .groupByKey()\
            .map(lambda x:((x[0]), sorted(x[1],reverse=True)))\
            .sortByKey()\
            .map(lambda x:((x[0]), (x[1][0:3])))\
            .map(lambda x:((x[0] + "," + x[1][0][1] + "," + str(x[1][0][0]) + "," + x[1][1][1] + "," + str(x[1][1][0]) + "," + x[1][2][1] + "," + str(x[1][2][0]))))
    
    rddattempt_1.write.csv(output_file)