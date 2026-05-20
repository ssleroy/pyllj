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
from pyllj.parameters import regions 
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
                                 region:str=None, 
                                 outputfile:str="diagnostics.pdf" ): 
    """Generate a figure showing the diagnostics of the LLJ for a set of 
    reanalyses. Each row of the plot corresponds one reanalyses, and the 
    three columns correspond to (1) maps of the daily probability of LLJ 
    occurrence, (2) mean height above the surface of the LLJ core, and (3) 
    mean wind speed of the LLJ core. Output plot is outputfile."""

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
              'field': eventProbability * 100, 
              'label': r'Probability of Occurrence [\%]', 
              'levels': np.arange(0,100.1,5), 
              'ticks': np.arange(0,100.1,20), 
              'colorscale': 11, 
              'extend': "neither", 
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

            if ix == 0 and region is not None: 
                rs = [ r for r in regions if r['name'] == region ]
                if len( rs ) == 1: 
                    r = rs[0]
                    rlons = [ r['longituderange'][0], r['longituderange'][1], r['longituderange'][1], r['longituderange'][0], r['longituderange'][0] ]
                    rlats = [ r['latituderange'][0], r['latituderange'][0], r['latituderange'][1], r['latituderange'][1], r['latituderange'][0] ]
                    ax.plot( rlons, rlats, lw=1.5, color="#0000C0" )
                else: 
                    print( f'Region "{region}" is not available in pyllj.parameters.regions' )

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

    return


def plot_wind_height_distributions( reanalyses:list=["narr","merra2","era5"], 
                                 outputfile:str="wind_height_distributions.png" ): 
    """Probability density functions of LLJ wind speed-jet height joint distribution 
    for the reanalyses. Output written to outputfile."""

    #  Bounding box for region of interest. 

    lonbounds = [ 251, 272.5 ]
    latbounds = [ 24, 49 ]

    print( 'Setting region mask to lonbounds = [ {:}, {:} ] and latbounds = [ {:}, {:} ]'.format( *lonbounds, *latbounds ) )

    #  Configure dimensions. 

    nreanalyses = len( reanalyses )

    xmargin = 0.45    # inches
    ymargin = 0.8   # inches
    xsize = xmargin + 2.0 * nreanalyses
    ysize = ymargin + 2.2 * 1
    
    fig = plt.figure( figsize=(xsize,ysize) )

    histmeta = { 'range': ( (12,24), (0,1000) ), 'bins': (24,100), 'density': False }

    levels = np.arange( 0.00, 0.6001, 0.01 )
    clevels = np.arange( 0.00, 0.6001, 0.20 )

    #  Loop over reanalyses. 

    for ireanalysis, reanalysis in enumerate( reanalyses ): 

        #  Open and read analysisfiles. 

        local_paths = [ os.path.join( DATAROOT, reanalysis.upper(), "diagnostics", 
                f'{reanalysis}_diagnostics.{year:04d}{month:02d}.nc' ) \
                    for year in range(2000,2025) for month in range(6,9) ]

        hist2d = np.zeros( histmeta['bins'], dtype=np.float32 )

        for ifile, local_path in enumerate( local_paths ): 
            print( f'Reading {local_path}' )
            a = Dataset( local_path, 'r' )

            if ifile == 0: 

                if "lons" in a.variables.keys(): 
                    lons = a.variables['lons'][:]
                    lats = a.variables['lats'][:]
                elif "longitude" in a.variables.keys(): 
                    lons = a.variables['longitude'][:]
                    lats = a.variables['latitude'][:]

                if len( lons.shape ) == 1: 
                    glons, glats = np.meshgrid( lons, lats )
                elif len( lons.shape ) == 2: 
                    glons, glats = lons, lats

                #  Justify longitude range. 

                glons[ glons < 0.0 ] += 360

                #  Mask to the Great Plains. 

                gp = np.logical_and( \
                        np.logical_and( glons > lonbounds[0], glons < lonbounds[1] ), \
                        np.logical_and( glats > latbounds[0], glats < latbounds[1] ) )

            ndays, nhours, ny, nx = a.variables['wind'].shape
            for iday in range( ndays ): 
                for ihour in range( nhours ): 
                    wind = a.variables['wind'][iday,ihour,:,:].squeeze()
                    height = a.variables['height'][iday,ihour,:,:].squeeze()
                    wind = wind[gp].flatten().compressed()
                    height = height[gp].flatten().compressed()
                    h, xedges, yedges = np.histogram2d( wind, height, **histmeta )
                    hist2d += h.astype( hist2d.dtype )

            a.close()

        print( f'Computing joint distribution' )
        hist2d /= hist2d.sum() * ( xedges[1] - xedges[0] ) * ( yedges[1] - yedges[0] ) / 1000
        # hist2d *= 100.0

        #  Plot histogram. 

        pos = np.array( [ 0.02 + ireanalysis, 0.12, 0.92, 0.76 ] ) 
        pos = pos / np.array( [ nreanalyses, 1, nreanalyses, 1 ] )
        pos = pos * np.array( [ 1-xmargin/xsize, 1-ymargin/ysize, 1-xmargin/xsize, 1-ymargin/ysize ] ) \
                + np.array( [ xmargin/xsize, ymargin/ysize, 0, 0 ] )
        ax = fig.add_axes( pos )

        ax.set_xlim( 12, 24 )
        ax.set_xticks( np.arange( 12, 25, 4 ) )
        ax.xaxis.set_minor_locator( MultipleLocator( 1 ) )
        ax.set_xlabel( 'Jet speed [m s$^{-1}$]' )

        ax.set_ylim( 0, 1 )
        ax.set_yticks( np.arange( 0, 1.1, 0.5 ) )
        ax.yaxis.set_minor_locator( MultipleLocator( 0.1 ) )
        if ireanalysis == 0: 
            ax.set_ylabel( 'Jet height [km]' )
        else: 
            ax.set_yticklabels( [] )

        ax.text( 12, 1.010, '({:}) {:}'.format( chr(ord('a')+ireanalysis), reanalysis.upper() ), 
                verticalalignment="bottom", clip_on=False )

        xmid = 0.5 * ( xedges[:-1] + xedges[1:] )
        ymid = 0.5 * ( yedges[:-1] + yedges[1:] ) / 1000
        hist2d = hist2d.T

        cax = ax.contourf( xmid, ymid, hist2d, levels=levels, cmap='OrRd', extend='max' )
        ax.contour( xmid, ymid, hist2d, levels=clevels, linewidths=0.2, colors="#808080" )

    #  Colorbar. 

    pos = [ 0.02, 0.53, 0.96, 0.12 ]
    pos = pos * np.array( [ 1-xmargin/xsize, ymargin/ysize, 1-xmargin/xsize, ymargin/ysize ] ) \
                + np.array( [ xmargin/xsize, 0, 0, 0 ] )

    cbar = fig.add_axes( pos )
    fig.colorbar( cax, cbar, orientation="horizontal", ticks=clevels, label='Probability [ ( km m/s )$^{-1}$ ]' )

    #  Done. 

    print( f'Saving to {outputfile}' )
    fig.savefig( outputfile )

    return


def main(): 

    parser = argparse.ArgumentParser( prog="diagnostics", 
                 description="Compose a figure of LLJ diagnostics" )

    parser.add_argument( "modelroot", type=str, 
            help="The root directory of the model output" )

    parser.add_argument( "--label", dest="label", default="", 
            help="The nominal label of the model run" )

    valid_regions = [ r['name'] for r in regions ]
    parser.add_argument( "--region", "-r", dest="region", default="", 
            help="The name of the region to outline in the first plot of the last row.  " + \
                    "Valid regions are " + ", ".join( [ f'"{s}"' for s in valid_regions ] ) + "." )

    parser.add_argument( "--output", "-o", dest="outputfile", default="diagnostics.pdf", 
            help="The output figure file." )

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
