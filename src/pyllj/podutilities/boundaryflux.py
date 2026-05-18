# #  Column Water Vapor Flux Across Gulf Coast
# 
# Analyze the column water vapor flux across the Gulf Coast as computed from 
# NARR, MERRA2, and ERA5. This will include annual cycle, summertime diurnal cycle, 
# and inter-annual variability. 

import os
import re
import argparse 
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt 
from .libpod import get_metricpath


#  Plotting defaults. 

axeslinewidth = 0.5 
plt.rcParams.update( {
  'font.family': "Times New Roman", 
  'font.size': 9,  
  'font.weight': "normal", 
  'text.usetex': True, 
  'xtick.major.width': axeslinewidth, 
  'xtick.minor.width': axeslinewidth, 
  'ytick.major.width': axeslinewidth, 
  'ytick.minor.width': axeslinewidth, 
  'axes.linewidth': axeslinewidth } ) 


def create_figure( reanalyses:{tuple,list}=["narr","merra2","era5"], layer:str="column", 
                  modelroot:{str,None}=None, modellabel:{str,None}=None, 
                  outputfile="boundaryflux.pdf" ): 

    data = []

    for reanalysis in reanalyses: 

        path = get_metricpath( "boundaryflux", reanalysis )
        print( f'Reading {path}' )

        d = Dataset( path, 'r' )

        groupnames = list( d.groups.keys() )
        if layer not in groupnames: 
            print( f'Layer "{layer}" not available; available layers are ' + ", ".join( [ f'"{l}"' for l in groupnames ] ) )
            return

        v = d.variables['time']
        units = v.getncattr( "units" )
        m = re.search( r'^(\w+) since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$', units )
        time_unit = m.group(1)
        epoch = datetime.strptime( m.group(2), '%Y-%m-%d %H:%M:%S' )
        times = [ epoch + timedelta( **{ time_unit: int(x) } ) for x in v[:] ]
        watervaporflux = d.groups[layer].variables['watervaporflux'][:]
        boundary = { 'lons': d.variables['longitude'][:], 'lats': d.variables['latitude'][:] }
        d.close()

        rec = { 'label': reanalysis.upper(), 'times': times, 'watervaporflux': watervaporflux } 
        data.append( rec )

    if modelroot is not None: 

        path = os.path.join( modelroot, "boundaryflux", "boundaryflux.great-plains.nc" )
        if not os.path.exists( path ): 
            print( f'Path "{path}" does not exist. Exiting.' )
            return None

        print( f'Reading {path}' )
    
        d = Dataset( path, 'r' )
        v = d.variables['time']
        units = v.getncattr( "units" )
        m = re.search( r'^(\w+) since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$', units )
        time_unit = m.group(1)
        epoch = datetime.strptime( m.group(2), '%Y-%m-%d %H:%M:%S' )
        times = [ epoch + timedelta( **{ time_unit: int(x) } ) for x in v[:] ]

        if layer in d.groups.keys(): 
            g = d.groups[layer]
        else: 
            layers = sorted( list( d.groups.keys() ) )
            print( f'Layer "{layer}" not available in {path}. ' + \
                    'Available layers are ' + ', '.join( [ f'"{lay}"' for lay in layers ] ) )
            return

        watervaporflux = g.variables['watervaporflux'][:]
        boundary = { 'lons': d.variables['longitude'][:], 'lats': d.variables['latitude'][:] }
        d.close()

        if modellabel is None: 
            label = os.path.split( modelroot )[-1]
        else: 
            label = modellabel

        rec = { 'label': label, 'times': times, 'watervaporflux': watervaporflux } 
        data.append( rec )

    #  Generate figure. 

    fig = plt.figure( figsize=(6,2.5) )

    units = 3600 / 1.0e12
    print( 'Units are Gtons per hour' )
    cmap = plt.get_cmap( "gist_ncar" )
    nrecs = len( data )
    colors = [ cmap( (i+0.5)/nrecs ) for i in range(nrecs) ]

    #  Seasonal cycle. 

    ax = fig.add_axes( [0.10,0.16,0.42,0.70] )

    xlim = ( 0, 12 )
    ylim = ( -1, 5 )
    yticks = np.arange( 0, 4.1, 2 )

    ax.set_xlim( *xlim )
    ax.set_xticks( np.arange(0.5,12,1) )
    ax.set_xticklabels( "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split() )
    ax.tick_params(axis='x', labelrotation=-60)

    ax.set_ylim( *ylim )
    ax.set_yticks( yticks )
    ax.yaxis.set_minor_locator( MultipleLocator(0.5) )
    # ax.set_ylabel( 'Column water vapor flux [Gtons/hr]' )

    x = ( xlim[0] + 0.02 * 0.5 / ax.get_position().width * (xlim[1]-xlim[0]) ) 
    y = ( ylim[0] + 0.92 * (ylim[1]-ylim[0]) ) 
    ax.text( x, y, '(a)' )

    for ytick in yticks: 
        ax.plot( xlim, [ytick,ytick], ls="--", lw=0.5, color="#808080" )
    
    for irec, rec in enumerate( data ): 
        vals = rec['watervaporflux']
        times = rec['times']
        dx = ( irec - len(data)*0.5 + 0.5 ) * 0.01 * ( xlim[1] - xlim[0] ) 
        x, y, ysdev = [], [], []
        for month in range(1,13): 
            ii = np.array( [ i for i, t in enumerate( times ) if t.month == month ] )
            x.append( month - 0.5 )
            yt = vals[ii] * units
            y.append( yt.mean() )
            ysdev.append( np.sqrt( ( ( yt - yt.mean() )**2 ).mean() ) )

        ax.errorbar( np.array(x)+dx, y, ysdev, lw=1.5, elinewidth=0.8, capsize=3.0, \
                color=colors[irec], label=rec['label'] )

    #  Mean summer diurnal cycle. 

    ax = fig.add_axes( [0.56,0.16,0.42,0.70] )

    xlim = [ -0.5, 21.5 ]

    ax.set_xlim( *xlim )
    ax.set_xticks( np.arange(0,24,3) )
    ax.set_xticklabels( [ f'{hr:02d}' for hr in range(0,24,3) ] )
    ax.set_xlabel( "UTC" )

    ax.set_ylim( *ylim )
    ax.set_yticks( yticks )
    ax.set_yticklabels( [] )
    ax.yaxis.set_minor_locator( MultipleLocator(0.5) )

    x = ( xlim[0] + 0.02 * 0.5 / ax.get_position().width * (xlim[1]-xlim[0]) ) 
    y = ( ylim[0] + 0.92 * (ylim[1]-ylim[0]) ) 
    ax.text( x, y, '(b)' )

    for ytick in yticks: 
        ax.plot( xlim, [ytick,ytick], ls="--", lw=0.5, color="#808080" )

    for irec, rec in enumerate( data ): 
        vals = rec['watervaporflux']
        times = rec['times']
        dx = ( irec - len(data)*0.5 + 0.5 ) * 0.01 * ( xlim[1] - xlim[0] ) 
        x, y, ysdev = [], [], []
        for hour in range(0,24,3): 
            ii = np.array( [ i for i, t in enumerate( times ) if t.hour==hour and t.month in [6,7,8] ] )
            x.append( hour )
            yt = vals[ii] * units
            y.append( yt.mean() )
            ysdev.append( np.sqrt( ( ( yt - yt.mean() )**2 ).mean() ) )
        ax.errorbar( np.array(x)+dx, y, ysdev, lw=1.5, elinewidth=0.8, capsize=3.0, \
                color=colors[irec], label=rec['label'] )

    ax.legend( loc="upper right", fontsize=8 )

    fig.supylabel( "Water vapor flux [ $\mathrm{Gtons\ hr}^{-1}$ ]", 
                  x=0.05, y=0.52, va="center", ha="right" )

    print( f'Saving to {outputfile}' )
    fig.savefig( outputfile )

    return


def main(): 

    parser = argparse.ArgumentParser( prog="boundaryflux", 
                 description="Compose a figure diagnosting the climatology of the flux " + \
                 "of water vapor in the PBL crossing a specific boundary" )

    parser.add_argument( "modelroot", type=str, 
            help="The root directory of the model output" )

    default_layer = "column"
    parser.add_argument( "--layer", dest="layer", default="column", 
            help=f'The name of the layer to diagnose; the default is "{default_layer}".' )

    parser.add_argument( "--label", dest="label", default="", 
            help="The nominal label of the model run" )

    default_outputfile = "boundaryflux.pdf"
    parser.add_argument( "--output", "-o", dest="outputfile", default=default_outputfile,
            help=f'The output figure file; the default is "{default_outputfile}".' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true", 
            help="Run in the Python line debugger PDB." )

    args = parser.parse_args()


    if args.pdb: 
        import pdb
        pdb.set_trace()

    if args.label == "": 
        label = os.path.split( args.modelroot )[-1]
    else: 
        label = str( args.label )

    #  Generate figure. 

    create_figure( layer=args.layer, modelroot=args.modelroot, modellabel=label, 
            outputfile=args.outputfile )

    return 


if __name__ == "__main__": 
    main()
    pass
