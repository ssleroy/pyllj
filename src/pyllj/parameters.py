import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
from cartopy.feature import STATES, OCEAN
import warnings

warnings.filterwarnings('ignore')

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

#  Pyplot settings. 

axeslinewidth = 0.5 
plt.rcParams.update( {
  'font.family': "Times New Roman", 
  'font.size': 8,
  'font.weight': "normal",
  'text.usetex': True,
  'xtick.major.width': axeslinewidth,
  'xtick.minor.width': axeslinewidth, 
  'ytick.major.width': axeslinewidth, 
  'ytick.minor.width': axeslinewidth, 
  'axes.linewidth': axeslinewidth } ) 

#  Zenodo version. 

# zenodoversion = "19740758"      # Version 2
zenodoversion = "20142370"      # Version 3

#  Define regions. 

regions = [ 
        # { 'name': "north-america", 'longituderange': np.array( [ -135.0, -60.0 ] ), 'latituderange': np.array( [ 15.0, 55.0 ] ) }, 
        { 'name': "sgp", 'longituderange': np.array( [ -97.480 ] ), 'latituderange': np.array( [ 36.620 ] ) }, 
        { 'name': "great-plains", 'longituderange': np.array( [ -102.0, -92.0 ] ), 'latituderange': np.array( [ 30.0, 47.0 ] ) }, 
        { 'name': "southern-plains", 'longituderange': np.array( [ -100.5, -94.5 ] ), 'latituderange': np.array( [ 34.6, 38.6 ] ) } ]

#  Boundaries used for evaluating cross-boundary column water fluxes. 

boundaries = [
        { 'name': "gulf-coast", 
            'lons': [ -97.69, -97.46, -97.17, -96.28, -93.39, -90.24, -89.23, -87.18 ],
            'lats': [ 23.29, 26.82, 27.87, 28.56, 29.67, 29.19, 30.25, 30.30 ] }
        ]
boundarynames = sorted( [ b['name'] for b in boundaries ] )


def plot_regions( extent=[-130,-60,20,55], projection=ccrs.PlateCarree(), legend=False, 
                  outputfile="regions.pdf" ): 
    """Plot the boundaries and regions as defined in pyllj.parameters on a 
    map. 

    Arguments (all optional)
    ========================
    extent          A four-element list defining the longitude and latitude 
                    extent of the figure's map: 
                    [ lon_min, lon_max, lat_min, lat_max ]

    projection      The cartopy projection to be used for the map

    legend          Set to true to generate a plot legend

    outputfile      The name of the printable outpuut file
    """

    fontsize = None
    fig = plt.figure( figsize=(3,2) )

    ax = fig.add_axes( [0.01,0.01,0.98,0.98], projection=projection )
    ax.set_extent( extent, ccrs.PlateCarree() )
    ax.add_feature( OCEAN, facecolor='paleturquoise', alpha=0.4 )
    ax.add_feature( STATES, lw=0.1 )
    ax.coastlines( lw=0.5 )

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                      x_inline=False, y_inline=False, linewidth=0.33, 
                      color='k', alpha=0.5 )
    gl.right_labels = gl.top_labels = False
    gl.ylocator = ticker.FixedLocator( np.arange(20,60,10) )
    gl.xlocator = ticker.FixedLocator( np.arange(-120,-59,15) )
    if fontsize is not None: 
        gl.xlabel_style = {'size': fontsize }
        gl.ylabel_style = {'size': fontsize }

    #  Color scheme. 

    cmap = plt.get_cmap( "brg" )

    #  Count regions and boundaries. 

    nregions = len( regions )
    nboundaries = len( boundaries )
    ncurves = nregions + nboundaries

    for icurve in range(ncurves): 
        color = cmap( (icurve+0.5) / ncurves )

        if icurve < nregions: 

            #  Regions. 

            iregion = icurve
            region = regions[iregion]
            lonrange, latrange = region['longituderange'], region['latituderange']
            if lonrange.size == 2: 
                lons = np.array( [ lonrange[0], lonrange[1], lonrange[1], lonrange[0], lonrange[0] ] )
                lats = np.array( [ latrange[0], latrange[0], latrange[1], latrange[1], latrange[0] ] )
                ax.plot( lons, lats, color=color, lw=1.5, label=region['name'] )
            elif lonrange.size == 1: 
                ax.scatter( lonrange, latrange, color=color, s=1.5, label=region['name'] )

        else: 

            #  Boundaries. 

            iboundary = icurve - nregions
            boundary = boundaries[iboundary]
            lons = boundary['lons']
            lats = boundary['lats']
            ax.plot( lons, lats, color=color, lw=1.5, label=boundary['name'] )

    #  Legend. 

    if legend: 
        ax.legend()

    #  Save to output. 

    print( f'Saving to {outputfile}' )
    fig.savefig( outputfile )

    return


