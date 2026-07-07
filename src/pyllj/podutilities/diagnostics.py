import os
import re
import argparse 
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, STATES, OCEAN
from pyllj.libutils import RetClass
from .libpod import get_metricpath
from .pyukmo import UKMOcolorMaps


#  Pyplot defaults. 

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


def plot_summer_llj_diagnostics( reanalyses:list=["narr","merra2","era5"], 
                                 modelroot:{str,None}=None, 
                                 modellabel:{str,None}=None, 
                                 outputfile:str="diagnostics.pdf" ): 
    """Generate a figure showing the diagnostics of the LLJ for a set of 
    reanalyses. Each row of the plot corresponds one reanalyses, and the 
    three columns correspond to (1) maps of the daily probability of LLJ 
    occurrence, (2) mean height above the surface of the LLJ core, and (3) 
    mean wind speed of the LLJ core. Output plot is outputfile."""

    ret = RetClass()

    #  Define colormaps. 

    ukm = UKMOcolorMaps()

    #  Configure dimensions. 

    nreanalyses = len( reanalyses )

    lowermargin = 0.6  # inches
    xsize = 6.5
    if modelroot is None: 
        nrows = nreanalyses
    else: 
        nrows = nreanalyses + 1
    ysize = 1.5 * nrows + lowermargin

    fig = plt.figure( figsize=(xsize,ysize) )
    proj = ccrs.PlateCarree

    #  Select summer months. 

    imonths = np.arange( 5, 8, dtype='i' )      #  Summer

    #  Loop over reanalyses. 

    for irow in range( nrows ): 

        iy = nrows - irow - 1
        if irow < nreanalyses: 
            ireanalysis = irow
            reanalysis = reanalyses[ireanalysis]
        else: 
            reanalysis = None

        #  Open and read analysisfiles. 

        if reanalysis is None: 
            path = os.path.join( modelroot, "diagnostics", "diagnostics.nc" )
        else: 
            path = get_metricpath( "diagnostics", reanalysis )

        if not os.path.exists( path ): 
            print( f'File {path} is not available. Aborting.' )
            exit()

        print( f'Reading {path}' )
        d = Dataset( path, 'r' )
        lons = d.variables['longitude'][:]
        lats = d.variables['latitude'][:]

        d_eventProbability = d.variables['eventProbability'][imonths,:,:]
        d_dailyeventProbability = d.variables['dailyeventProbability'][imonths,:,:]
        d_meanheight = d.variables['meanheight'][imonths,:,:]
        d_meanwind = d.variables['meanwind'][imonths,:,:]

        nevents = d_eventProbability.data.sum( axis=0 )
        eventProbability = nevents / imonths.size
        dailyeventProbability = d_dailyeventProbability.data.sum( axis=0 ) / imonths.size
        meanheight = ( d_meanheight * d_eventProbability.data ).sum( axis=0 ) / nevents
        meanwind = ( d_meanwind * d_eventProbability.data ).sum( axis=0 ) / nevents

        #  Mask. 

        mask = ( eventProbability < 0.05 )
        meanheight = np.ma.masked_where( np.logical_or( mask, np.ma.getmask( meanheight ) ), meanheight )
        meanwind = np.ma.masked_where( np.logical_or( mask, np.ma.getmask( meanwind ) ), meanwind )

        #  Define diagnostic plots and their parameters. 

        plotmeta = [ 
            { 
              # 'field': dailyeventProbability * 100, 
              # 'label': r'Probability of Daily Occurrence [\%]', 
              # 'levels': np.arange(0,100.1,5), 
              # 'ticks': np.arange(0,100.1,20), 
              # 'extend': "neither", 
              'field': eventProbability * 100, 
              'label': r'Probability of Occurrence [\%]', 
              'levels': np.arange(0,30.1,1), 
              'ticks': np.arange(0,30.1,10), 
              'colorscale': 11, 
              'extend': "max", 
              'tophalf': True 
            }, 
            { 'field': meanheight, 
              'label': "Height of LLJ [m]", 
              'levels': np.arange(300,800.1,25), 
              'ticks': np.arange(400,800.1,200), 
              'colorscale': 25, 
              'extend': "max", 
              'tophalf': False 
            }, 
            { 'field': meanwind, 
              'label': "Speed of LLJ [m/s]", 
              'levels': np.arange(12,20.1,0.5), 
              'ticks': np.arange(12,20.1,2), 
              'colorscale': 0, 
              'extend': "max", 
              'tophalf': False 
            }, 
        ]

        nmeta = len( plotmeta )

        #  Plots for current reanalysis. 

        if reanalysis is not None: 
            print( f'Generating {reanalysis} plots' )
            title = '({:}) {:}'.format( chr(ord("a")+irow), reanalysis.upper() )
        else: 
            print( f'Generating model plots ({modelroot})' )
            if modellabel is None: 
                title = '({:}) {:}'.format( chr(ord("a")+irow), os.path.split( modelroot )[-1] )
            else: 
                title = '({:}) {:}'.format( chr(ord("a")+irow), modellabel )

        #  Loop over diagnostic. 

        for ix, meta in enumerate(plotmeta): 

            #  Position and limits of contour plot. 

            pos = np.array( [ 0.04, 0.02, 0.92, 0.80 ] )
            pos = ( pos + np.array( [ ix, iy, 0, 0 ] ) ) / np.array( [ nmeta, nrows, nmeta, nrows ] )
            pos = pos * np.array( [ 1.0, 1-lowermargin/ysize, 1.0, 1-lowermargin/ysize ] ) \
                    + np.array( [ 0.0, lowermargin/ysize, 0.0, 0.0 ] )

            ax = fig.add_axes( pos, projection=proj() )
            ax.set_xlim( -130, -60 )
            ax.set_ylim( 20, 50 )
            ax.set_aspect('auto')

            if ix == 0: 
                ax.text( -130, 51, title, horizontalalignment="left", verticalalignment="bottom", clip_on=False )

            #  Map features. 

            ax.coastlines( resolution="50m", linewidth=0.1, color="black" )
            ax.add_feature( BORDERS, linewidth=0.1 )
            ax.add_feature( STATES, linewidth=0.1 )

            #  Contour plot. 

            cmap = ukm.get_cmap( meta['colorscale'] )
            colors = []
            if meta['tophalf']: 
                for ilevel, level in enumerate(meta['levels']): 
                    colors.append( cmap( 0.5 + 0.5 * ( level - meta['levels'].min() ) / ( meta['levels'].max() - meta['levels'].min() ) ) )
            else: 
                for ilevel, level in enumerate(meta['levels']): 
                    colors.append( cmap( ( level - meta['levels'].min() ) / ( meta['levels'].max() - meta['levels'].min() ) ) )

            cax = ax.contourf( lons, lats, meta['field'], levels=meta['levels'], colors=colors, extend=meta['extend'], transform=proj() )
            ax.contour( lons, lats, meta['field'], levels=meta['ticks'], linewidths=0.6, colors="k", transform=proj() )

            #  Colorbar. 

            if iy == 0: 

                pos = np.array( [ 0.04, 0.70, 0.92, 0.12 ] )
                pos = ( pos + np.array( [ ix, 0, 0, 0 ] ) ) / np.array( [ nmeta, 1.0, nmeta, 1.0 ] )
                pos = pos * np.array( [ 1.0, lowermargin/ysize, 1.0, lowermargin/ysize ] ) 

                cbar = fig.add_axes( pos )
                fig.colorbar( cax, cbar, orientation="horizontal", ticks=meta['ticks'], label=meta['label'] )

    #  Done with plot. Write to output. 

    print( f'Creating {outputfile}' )
    fig.savefig( outputfile )

    ret.update( success=True )
    return ret


def main(): 

    parser = argparse.ArgumentParser( prog="diagnostics", 
                 description="Compose a figure of LLJ diagnostics" )

    parser.add_argument( "modelroot", type=str, 
            help="The root directory of the model output" )

    parser.add_argument( "--label", dest="label", default="", 
            help="The nominal label of the model run" )

    default_output = "diagnostics.pdf"
    parser.add_argument( "--output", "-o", dest="outputfile", default=default_output, 
            help=f'The output figure file. The default is "{default_output}".' )

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

    plot_summer_llj_diagnostics( modelroot=args.modelroot, modellabel=label, 
            outputfile=args.outputfile )
    return 


if __name__ == "__main__": 
    main()
    pass
