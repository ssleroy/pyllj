import os
import numpy as np

#  Physical parameters. 

Rearth = 6378.135e3             # Equatorial radius of Earth [m]
Rideal = 8.31446261815324       # Ideal gas constant [J/mole/K]
gravity = 9.80665               # WMO standard gravitational acceleration [J/kg/m]
muvap = 18.015e-3               # Mean moledular mass of water vapor [kg/mole]
mudry = 28.965e-3               # Mean molecular mass of dry air [kg/mole]

aws_region = "us-east-1"
bucket = "aer-sleroy-llj"

default_dataroot = os.getenv( "DATAROOT" )
if default_dataroot is None: 
    default_dataroot = "/fg/Data"

#  Zenodo version. 

# zenodoversion = "19740758"      # Version 2
zenodoversion = "20142370"      # Version 3

#  Define regions. 

regions = [ 
        { 'name': "north-america", 'longituderange': np.array( [ -135.0, -60.0 ] ), 'latituderange': np.array( [ 15.0, 55.0 ] ) }, 
        { 'name': "great-plains", 'longituderange': np.array( [ -100.0, -92.0 ] ), 'latituderange': np.array( [ 30.0, 37.0 ] ) } ]
#       { 'name': "great-plains", 'longituderange': np.array( [ -102.0, -95.0 ] ), 'latituderange': np.array( [ 30.0, 37.0 ] ) } ]

#  Boundaries used for evaluating cross-boundary column water fluxes. 

boundaries = {
        'great-plains': {
                'lons': [ -97.69, -97.46, -97.17, -96.28, -93.39, -90.24, -89.23, -87.18 ],
                'lats': [ 23.29, 26.82, 27.87, 28.56, 29.67, 29.19, 30.25, 30.30 ] }
            }

