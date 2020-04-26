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

    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return zones.neighborhood[idx]
    return None

def findBorough(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return zones.borough[idx]
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
        try:

            orig = geom.Point(proj(float(row[5]), float(row[6])))
            dest = geom.Point(proj(float(row[9]), float(row[10])))
        
        
            # Look up a matching zone, and update the count accordingly if such a match is found
            zone = findZone(dest, index, zones)
            borough = findBorough(orig, index, zones)
        
            if zone and borough:
                counts[borough, zone] = counts.get((borough, zone), 0) + 1

        except (ValueError, IndexError):
            pass
    
    return counts.items()
            
if __name__=='__main__':
   
    sc = SparkContext()
    input_file = sys.argv[1]
    yellow_taxi = sc.textFile(input_file)

    counts = yellow_taxi.mapPartitionsWithIndex(processTrips).reduceByKey(lambda x,y: x+y).collect()

    rddattempt = sc.parallelize(counts)

    rddattempt_1 = rddattempt.map(lambda x: (x[0][0],((x[0][1], x[1]))))\
        .sortBy(lambda x: x[1][1], ascending=False)\
        .groupByKey()\
        .mapValues(list)\
        .reduceByKey(lambda x,y: x+y)\
        .sortByKey()\
        .map(lambda x: (x[0], x[1][0:3]))\
        .map(lambda x:((x[0] + "," + x[1][0][0] + "," + str(x[1][0][1]) + "," + x[1][1][0] + "," + str(x[1][1][1]) + "," + x[1][2][0] + "," + str(x[1][2][1]))))

    print(rddattempt.take(5))
    rddattempt_1.saveAsTextFile(sys.argv[2])