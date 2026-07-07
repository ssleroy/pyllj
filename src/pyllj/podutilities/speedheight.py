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
from pyllj.parameters import regions, default_dataroot
from pyllj.libutils import RetClass 
from .libpod import get_metricpath
from .pyukmo import UKMOcolorMaps


#  Pyplot defaults. 

axeslinewidth = 0.5
plt.rcParams.update( {
    'font.family': "DejaVu Serif",
    'mathtext.fontset': "dejavuserif"
    'font.size': 9, 
    'font.weight': "normal", 
    'text.usetex': True, 
    'xtick.major.width': axeslinewidth, 
    'xtick.minor.width': axeslinewidth, 
    'ytick.major.width': axeslinewidth, 
    'ytick.minor.width': axeslinewidth, 
    'axes.linewidth': axeslinewidth } )


def plot_speed_height_distributions( region:str, 
            reanalyses:list=["narr","merra2","era5"], 
            modelroot:str=None, modellabel:str=None, 
            outputfile:str="speedheight.pdf" ): 
    """Probability density functions of LLJ wind speed-jet height joint distribution 
    for reanalyses and model output. Output written to outputfile."""

    ret = RetClass()

    #  Determine region and longitude and latitude bounds. 

    rs = [ r for r in regions if r['name']==region ]
    if len( rs ) != 1: 
        ret.update( success=False, messages="InvalidArgument", comments=f'Region "{region}" is unrecognized.' )
        return ret

    lonbounds = rs[0]['longituderange'] * 1.0
    latbounds = rs[0]['latituderange'] * 1.0

    #  Justify longitude range. 

    lonbounds[ lonbounds < 0.0 ] += 360

    #  Configure dimensions. 

    if modelroot is None: 
        nplots = len( reanalyses )
    else: 
        nplots = len( reanalyses ) + 1

    xmargin = 0.45    # inches
    ymargin = 0.8   # inches
    xsize = xmargin + 2.0 * nplots 
    ysize = ymargin + 2.2 * 1

    fig = plt.figure( figsize=(xsize,ysize) )

    histmeta = { 'range': ( (12,24), (0,1000) ), 'bins': (24,100), 'density': False }

    levels = np.arange( 0.00, 1.0001, 0.02 )
    clevels = np.arange( 0.00, 1.0001, 0.20 )

    #  Loop over reanalyses (and model). 

    for iplot in range( nplots ): 

        if iplot < len( reanalyses ): 
            plotlabel = reanalyses[iplot].upper()
            p = os.path.join( default_dataroot, plotlabel, "diagnostics" )
            local_paths = sorted( [ os.path.join(p,f) for f in os.listdir(p) \
                    if re.search( r'diagnostics.\d{6}.nc$', f ) ] )

        else: 
            if modellabel is None: 
                plotlabel = os.path.split( modelroot )[-1]
            else: 
                plotlabel = modellabel
            p = os.path.join( modelroot, "diagnostics" )
            local_paths = sorted( [ os.path.join(p,f) for f in os.listdir(p) \
                    if re.search( r'diagnostics.\d{6}.nc$', f ) ] )

        #  Open and read analysisfiles. 

        hist2d = np.zeros( histmeta['bins'], dtype=np.float32 )
        first = True

        for ifile, local_path in enumerate( local_paths ): 

            #  Only summer. 

            m = re.search( r'diagnostics.(\d{4})(\d{2}).nc$', local_path )
            if int( m.group(2) ) not in [ 6, 7, 8 ]: 
                continue

            print( f'Reading {local_path}' )
            a = Dataset( local_path, 'r' )

            if first: 

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

                #  Mask to the desired region. 

                if lonbounds[1] > lonbounds[0]: 
                    gp = np.logical_and( \
                            np.logical_and( glons > lonbounds[0], glons < lonbounds[1] ), \
                            np.logical_and( glats > latbounds[0], glats < latbounds[1] ) )
                else: 
                    gp = np.logical_and( \
                            np.logical_or( glons > lonbounds[0], glons < lonbounds[1] ), \
                            np.logical_and( glats > latbounds[0], glats < latbounds[1] ) )

                first = False 

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

        #  Plot histogram. 

        pos = np.array( [ 0.02 + iplot, 0.12, 0.92, 0.76 ] ) 
        pos = pos / np.array( [ nplots, 1, nplots, 1 ] )
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
        if iplot == 0: 
            ax.set_ylabel( 'Jet height [km]' )
        else: 
            ax.set_yticklabels( [] )

        ax.text( 12, 1.010, '({:}) {:}'.format( chr(ord('a')+iplot), plotlabel ), 
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
    fig.colorbar( cax, cbar, orientation="horizontal", ticks=clevels, label='Probability [(km m/s)$^{-1}$]' )

    #  Done. 

    print( f'Saving to {outputfile}' )
    fig.savefig( outputfile )

    return


def main(): 

    parser = argparse.ArgumentParser( prog="speedheight", 
                 description="Compose a figure of density distribution of LLJ wind speed and height" )

    parser.add_argument( "modelroot", type=str, 
            help="The root directory of the model output" )

    valid_regions = [ r['name'] for r in regions ]
    parser.add_argument( "region", type=str,  
            help="The name of the region over which to count LLJs, their speed and core height.  " + \
                    "Valid regions are " + ", ".join( [ f'"{s}"' for s in valid_regions ] ) + "." )

    parser.add_argument( "--label", dest="label", default="", 
            help="The nominal label of the model run" )

    default_output = "speedheight.pdf"
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

    plot_speed_height_distributions( args.region, 
            modelroot=args.modelroot, modellabel=label, 
            outputfile=args.outputfile )

    return 


if __name__ == "__main__": 
    main()
    pass
