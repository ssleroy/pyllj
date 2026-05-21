import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from pyllj.parameters import regions, boundaries

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


def plot_regions( legend=False, outputfile="regions.pdf" ): 

    fontsize = None
    fig = plt.figure( figsize=(3,2) )

    ax = fig.add_axes( [0.01,0.01,0.98,0.98], projection=ccrs.PlateCarree() )
    ax.set_extent( [-130,-60,20,55], ccrs.PlateCarree() )
    ax.add_feature( cfeature.OCEAN, facecolor='paleturquoise', alpha=0.4 )
    ax.add_feature( cfeature.STATES, lw=0.1 )
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

    cmap = plt.get_cmap( "jet" )

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
            lons = np.array( [ lonrange[0], lonrange[1], lonrange[1], lonrange[0], lonrange[0] ] )
            lats = np.array( [ latrange[0], latrange[0], latrange[1], latrange[1], latrange[0] ] )
            ax.plot( lons, lats, color=color, lw=1.5, label=region['name'] )

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

