import os 
from datetime import datetime, timedelta 
import boto3
from pprint import pprint
import warnings
from netCDF4 import Dataset 
import numpy as np 
from ..libutils import LambertConformalProjection
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, STATES, OCEAN
from pyukmo import UKMOcolorMaps 

warnings.filterwarnings('ignore')

#  Plotting defaults. 

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

#  Other settings. 

monthstrings = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()

DATAROOT = os.getenv( "DATAROOT" )
print( f'DATAROOT = {DATAROOT}' )

tmpdir = "tmp"
os.makedirs( tmpdir, exist_ok=True )


def plot_narr_llj_diagnostics( analysisfiles:{str,list}, pdffile:str, title=None ): 
    """Generate a four-plot figure on the diagnostics of the LLJ over the 
    U.S. Great Plains. The analysisfiles defines the path(s) to the output of 
    pyllj.narr.compute_diagnostics and may be in an s3 bucket. 
    The output is written to a PDF file (pdffile).

    Arguments
    =========
    analysisfiles       A string or list of strings defining the paths to 
                        analysis files output by pyllj.compute_diagnostics; 
                        they exist in S3 buckets, in which case the string(s) 
                        should be prefixed by "s3://".

    pdffile             A string defining the path to the output PDF file on 
                        the local file system. 

    title               An optional title to put at the top of the figure.
    """

    #  Get S3 object as needed. 

    if isinstance(analysisfiles,list): 
        infiles = analysisfiles
    elif isinstance(analysisfiles,str): 
        infiles = [ analysisfiles ]
    else: 
        print( 'The input analysisfiles must be an instance of str or list.' )
        return


    #  Check to see if all or none of the input files are in an S3 bucket.

    stest = np.array( [ infile[:5]=="s3://" for infile in infiles ], dtype=np.int8 )
    if int( stest.sum() ) not in [ 0, len(infiles) ]: 
        print( 'Either all or none of the input analysisfiles should be in an S3 bucket.' )
        return
    s3 = ( stest.sum() > 0 )

    #  S3 prep...

    if s3: 

        local_paths = []
        bucket_name = infiles[0][5:].split( "/" )[0]

        #  Check to see that all infiles are in the same S3 bucket. 

        for infile in infiles[1:]: 
            ss1 = infile[5:].split( "/" )
            if ss1[0] != bucket_name: 
                print( f'S3 bucket name mismatch: {ss[0]}' )
                return

        #  Access to S3 bucket. 

        session = boto3.Session( region_name="us-east-1" )
        s3 = session.resource( "s3" ).Bucket( bucket_name )

        #  Download S3 objects. 

        for infile in infiles: 

            ss = infile[5:].split( "/" )
            bucket_path = "/".join( ss[1:] )
            objs = [ obj for obj in s3.objects.all().filter( Prefix=bucket_path ) ]

            if len(objs) == 0: 
                print( f'No S3 object {infile} was found.' )
                return
            elif len(objs) > 1: 
                print( f'More than one object {infile} was found.' )
                return
            else: 
                obj = objs[0]

            local_path = os.path.join( tmpdir, ss[-1] )
            if not os.path.exists( local_path ): 
                print( f'Downloading {local_path}' )
                s3.download_file( obj.key, local_path )
            local_paths.append( local_path )

    else: 

        local_paths = infiles

    #  Open and read analysisfiles. 

    wind, height = [], []

    for local_path in local_paths: 
        print( f'Reading {local_path}' )
        a = Dataset( local_path, 'r' )
        lons = a.variables['lons'][:]
        lats = a.variables['lats'][:]
        wind.append( a.variables['wind'][:] )
        height.append( a.variables['height'][:] )
        a.close()

    wind = np.ma.concatenate( wind )
    height = np.ma.concatenate( height )

    ndays, nhours, ny, nx = wind.shape

    #  Count events; compute mean heights and winds. 

    print( f'Computing diagnostics' )

    events = np.logical_not( wind.mask )
    nevents = events.sum(axis=1).sum(axis=0)
    ndailyevents = events.any(axis=1).sum(axis=0)

    eventProbability = nevents / ( ndays * nhours )
    dailyeventProbability = ndailyevents / ndays
    meanheight = height.reshape( (ndays*nhours,ny,nx) ).mean(axis=0) 
    meanwind = wind.reshape( (ndays*nhours,ny,nx) ).mean(axis=0) 

    #  Mask. 

    mask = ( nevents == 0 )
    imask = ( eventProbability < 0.05 )
    eventProbability = np.ma.masked_where( mask, eventProbability )
    dailyeventProbability = np.ma.masked_where( mask, dailyeventProbability )
    meanheight = np.ma.masked_where( imask, meanheight )
    meanwind = np.ma.masked_where( imask, meanwind )

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

    proj = ccrs.PlateCarree

    #  Set up figure. 

    print( f'Generating figure' )

    cm = 2.54
    # fig = plt.figure( figsize=(16/cm,14/cm) )
    fig = plt.figure( figsize=(6.5,2.0) )
    # nxplots, nyplots = 2, 2
    nxplots, nyplots = 3, 1
    nplots = nxplots * nyplots

    subpos = np.array( [ 0.02, 0.30, 0.98, 0.98 ] )
    wpos = np.array( [ 0.10, 0.22, 0.90, 0.25 ] )

    if title is None: 
        plotwindow = { 'pos': np.array( [ 0.0, 0.0, 1.0, 1.0 ] ) }
    else: 
        plotwindow = { 'pos': [ 0.0, 0.0, 1.0, 0.95 ] }
        ax = fig.add_axes( [ 0.0, 0.0, 1.0, 0.95 ] )
        ax.set_axis_off()
        ax.set_title( title )

    p = plotwindow['pos']
    plotwindow.update( { 'offset': p[1] + (p[0]-p[1]) * np.array([1,0,1,0]) } )
    plotwindow.update( { 'scale': ( p[2] - p[0] ) * np.array([1,0,1,0]) + ( p[3] - p[1] ) * np.array([0,1,0,1]) } )

    #  Define colormaps. 

    ukm = UKMOcolorMaps()

    #  Define plots. 

    plotmeta = [ 
            { 'field': eventProbability * 100, 
              'label': r'Probability of Occurrence [\%]', 
              'levels': np.arange(0,50.1,2), 
              'ticks': np.arange(0,50.1,10), 
              'colorscale': 9, 
              'extend': "max", 
              'tophalf': True
            }, 
            { 'field': dailyeventProbability * 100, 
              'label': r'Probability of Daily Occurrence [\%]', 
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

    plotmeta = plotmeta[1:]
    
    #  Loop over axes. 

    for iax, meta in enumerate(plotmeta): 

        ix = iax % nxplots 
        iy = nyplots - 1 - int(iax/nxplots)

        #  Position and limits of contour plot. 

        pos = ( subpos + np.array( [ix,iy,ix,iy] ) ) / np.array([nxplots,nyplots,nxplots,nyplots])
        pos = plotwindow['offset'] + plotwindow['scale'] * pos
        pos = np.array( [ pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1] ] )

        ax = fig.add_axes( pos, projection=proj() )
        ax.set_xlim( -130, -60 )
        ax.set_ylim( 20, 50 )
        ax.set_aspect('auto')

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

        pos = ( wpos + np.array( [ix,iy,ix,iy] ) ) / np.array([nxplots,nyplots,nxplots,nyplots])
        pos = plotwindow['offset'] + plotwindow['scale'] * pos
        pos = np.array( [ pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1] ] )

        cbar = fig.add_axes( pos )
        fig.colorbar( cax, cbar, orientation="horizontal", ticks=meta['ticks'], label=meta['label'] )

    #  Done with plot. Write to output. 

    print( f'Creating {pdffile}' )
    fmt = pdffile.split(".")[-1]
    fig.savefig( pdffile, format=fmt )

    #  Remove local paths if they were downloaded from an S3 bucket. 

    if s3: 
        for local_path in local_paths: 
            os.unlink( local_path )
            pass 

    return


def plot_llj_diagnostics( analysisfiles:{str,list}, pdffile:str, title=None ): 
    """Generate a four-plot figure on the diagnostics of the LLJ over the 
    U.S. Great Plains. The analysisfiles defines the path(s) to the output of 
    ERA5 or MERRA-2 compute_diagnostics. The output is written to a PDF file 
    (pdffile).

    Arguments
    =========
    analysisfiles       A string or list of strings defining the paths to 
                        analysis files output by pyllj.{model}.compute_diagnostics. 

    pdffile             A string defining the path to the output PDF file on 
                        the local file system. 

    title               An optional title to put at the top of the figure.
    """

    if isinstance(analysisfiles,list): 
        infiles = analysisfiles
    elif isinstance(analysisfiles,str): 
        infiles = [ analysisfiles ]
    else: 
        print( 'The input analysisfiles must be an instance of str or list.' )
        return

    #  Open and read analysisfiles. 

    wind, height = [], []

    for analysisfile in analysisfiles: 
        print( f'Reading {analysisfile}' )
        a = Dataset( analysisfile, 'r' )
        lons = a.variables['lons'][:]
        lats = a.variables['lats'][:]
        wind.append( a.variables['wind'][:] )
        height.append( a.variables['height'][:] )
        a.close()

    wind = np.ma.concatenate( wind )
    height = np.ma.concatenate( height )

    ndays, nhours, nlats, nlons = wind.shape

    #  Count events; compute mean heights and winds. 

    print( f'Computing diagnostics' )

    events = np.logical_not( wind.mask )
    nevents = events.sum(axis=1).sum(axis=0)
    ndailyevents = events.any(axis=1).sum(axis=0)

    eventProbability = nevents / ( ndays * nhours )
    dailyeventProbability = ndailyevents / ndays
    meanheight = height.reshape( (ndays*nhours,nlats,nlons) ).mean(axis=0) 
    meanwind = wind.reshape( (ndays*nhours,nlats,nlons) ).mean(axis=0) 

    #  Mask. 

    mask = ( nevents == 0 )
    imask = ( eventProbability < 0.05 )
    eventProbability = np.ma.masked_where( mask, eventProbability )
    dailyeventProbability = np.ma.masked_where( mask, dailyeventProbability )
    meanheight = np.ma.masked_where( imask, meanheight )
    meanwind = np.ma.masked_where( imask, meanwind )

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

    proj = ccrs.PlateCarree

    #  Set up figure. 

    print( f'Generating figure' )

    cm = 2.54
    # fig = plt.figure( figsize=(16/cm,14/cm) )
    fig = plt.figure( figsize=(6.5,2.0) )
    
    # nxplots, nyplots = 2, 2
    nxplots, nyplots = 3, 1
    nplots = nxplots * nyplots

    subpos = np.array( [ 0.02, 0.30, 0.98, 0.98 ] )
    wpos = np.array( [ 0.10, 0.22, 0.90, 0.25 ] )

    if title is None: 
        plotwindow = { 'pos': np.array( [ 0.0, 0.0, 1.0, 1.0 ] ) }
    else: 
        plotwindow = { 'pos': [ 0.0, 0.0, 1.0, 0.95 ] }
        ax = fig.add_axes( [ 0.0, 0.0, 1.0, 0.95 ] )
        ax.set_axis_off()
        ax.set_title( title )

    p = plotwindow['pos']
    plotwindow.update( { 'offset': p[1] + (p[0]-p[1]) * np.array([1,0,1,0]) } )
    plotwindow.update( { 'scale': ( p[2] - p[0] ) * np.array([1,0,1,0]) + ( p[3] - p[1] ) * np.array([0,1,0,1]) } )

    #  Define colormaps. 

    ukm = UKMOcolorMaps()

    #  Define plots. 

    plotmeta = [ 
            { 'field': eventProbability * 100, 
              'label': r'Probability of Occurrence [%]', 
              'levels': np.arange(0,50.1,2), 
              'ticks': np.arange(0,50.1,10), 
              'colorscale': 9, 
              'extend': "max", 
              'tophalf': True
            }, 
            { 'field': dailyeventProbability * 100, 
              'label': r'Probability of Daily Occurrence [%]', 
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

    plotmeta = plotmeta[1:]
    
    #  Loop over axes. 

    for iax, meta in enumerate(plotmeta): 

        ix = iax % nxplots 
        iy = nyplots - 1 - int(iax/nxplots)

        #  Position and limits of contour plot. 

        pos = ( subpos + np.array( [ix,iy,ix,iy] ) ) / np.array([nxplots,nyplots,nxplots,nyplots])
        pos = plotwindow['offset'] + plotwindow['scale'] * pos
        pos = np.array( [ pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1] ] )

        ax = fig.add_axes( pos, projection=proj() )
        ax.set_xlim( -130, -60 )
        ax.set_ylim( 20, 50 )
        ax.set_aspect('auto')

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

        pos = ( wpos + np.array( [ix,iy,ix,iy] ) ) / np.array([nxplots,nyplots,nxplots,nyplots])
        pos = plotwindow['offset'] + plotwindow['scale'] * pos
        pos = np.array( [ pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1] ] )

        cbar = fig.add_axes( pos )
        fig.colorbar( cax, cbar, orientation="horizontal", ticks=meta['ticks'], label=meta['label'] )

    #  Done with plot. Write to output. 

    print( f'Creating {pdffile}' )
    fmt = pdffile.split(".")[-1]
    fig.savefig( pdffile, format=fmt )

    #  Remove local paths if they were downloaded from an S3 bucket. 

    if s3: 
        for local_path in local_paths: 
            os.unlink( local_path )
            pass 

    return


# Define a class for vector field plotting. 

class WindField(): 

    def __init__( self, analysisfile, dx=6, dy=6, scale=300.0 ): 

        self.dx = dx
        self.dy = dy
        self.scale = scale

        print( f'Reading coordinate metadata from {analysisfile}' )
        d = Dataset( analysisfile, 'r' )

        #  Get coordinates. 

        self.months = d.variables['month'][:]
        self.levels = d.variables['level'][:]
        self.hours = d.variables['hour'][:]

        if "lon" in d.variables.keys(): 
            self.lons = d.variables['lon'][:]
        elif "longitude" in d.variables.keys(): 
            self.lons = d.variables['longitude'][:]

        if "lat" in d.variables.keys(): 
            self.lats = d.variables['lat'][:]
        elif "latitude" in d.variables.keys(): 
            self.lats = d.variables['latitude'][:]

        self.lons[ self.lons < 0 ] += 360
        self.lons[ self.lons >= 360.0 ] -= 360

        if len( self.lons.shape ) == 2: 
            self.lambert = True
            self.nx = d.dimensions['x'].size
            self.ny = d.dimensions['y'].size
            self.nz = self.levels.size
            self.mlons = self.lons
            self.mlats = self.lats
        else: 
            self.lambert = False
            self.nx = self.lons.size
            self.ny = self.lats.size
            self.nz = self.levels.size
            mlons, mlats = np.meshgrid( self.lons, self.lats )
            self.mlons = mlons
            self.mlats = mlats

        print( f'Lambert = {self.lambert}, nx = {self.nx}, ny={self.ny}, nmonths={self.months.size}, ' + \
                    f'nlevels={self.levels.size}, nhours={self.hours.size}' )

        #  Get projection. 

        if self.lambert: 
            v = d.variables['Lambert_Conformal']
            atts = { attname: v.getncattr(attname) for attname in v.ncattrs() }
            print( 'Projection:' )
            print( '\n'.join( [ f'  {key}: {value}' for key, value in atts.items() ] ) )

            #  Map using Lambert Conformal. 

            self.projection = ccrs.LambertConformal( 
                    central_longitude=atts['longitude_of_central_meridian'], 
                    central_latitude=atts['latitude_of_projection_origin'], 
                    false_easting=atts['false_easting'], 
                    false_northing=atts['false_northing'], 
                    standard_parallels=atts['standard_parallel'] 
                    ) 

        d.close()

        #  Define Great Plains LLJ mask. 

        self.bbox = { 'lonrange': [ 360-102, 360-95 ], 'latrange': [ 30, 37 ] }

        dlons0 = self.mlons - self.bbox['lonrange'][0]
        dlons1 = self.mlons - self.bbox['lonrange'][1]

        dlats0 = self.mlats - self.bbox['latrange'][0]
        dlats1 = self.mlats - self.bbox['latrange'][1]

        if self.bbox['lonrange'][1] > self.bbox['lonrange'][0]: 
            self.mask = np.logical_and( np.logical_and( dlons0 >= 0.0, dlons1 <= 0.0 ), \
                    np.logical_and( dlats0 >= 0.0, dlats1 <= 0.0 ) ).astype( np.int8 )
        else: 
            self.mask = np.logical_and( np.logical_or( dlons0 >= 0.0, dlons1 <= 0.0 ), \
                    np.logical_and( dlats0 >= 0.0, dlats1 <= 0.0 ) ).astype( np.int8 )

        print( 'LLJ bounding box:' )
        print( "  lonrange = " + ", ".join( [ f'{float(lon):.1f}' for lon in self.bbox['lonrange'] ] ) )
        print( "  latrange = " + ", ".join( [ f'{float(lat):.1f}' for lat in self.bbox['latrange'] ] ) )

        return

    def __call__( self, axes_limits, uwnd, vwnd, labels=True, bbox=True ): 

        ax = fig.add_axes( axes_limits, projection=self.projection )
        ax.set_aspect('auto')

        #  Map properties. 

        ax.set_extent([-130,-70,23,50], ccrs.PlateCarree())
        ax.add_feature(OCEAN.with_scale('10m'),lw=0.5,facecolor='lightblue',alpha=0.4)
        ax.add_feature(STATES.with_scale('10m'),lw=0.5,facecolor='#FFCC00',alpha=0.4)
        ax.add_feature(STATES.with_scale('10m'),lw=0.5,edgecolor='black')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=labels, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
        if labels: 
            gl.right_labels = gl.top_labels = False
            gl.ylocator = ticker.FixedLocator( np.arange( 20, 51, 5 ) )
            gl.xlocator = ticker.FixedLocator( np.arange( -130, -70+1, 10 ) )

        #  Draw vector field. 

        ax.quiver( self.mlons[::self.dy,::self.dx], self.mlats[::self.dy,::self.dx], 
               uwnd[::self.dy,::self.dx], vwnd[::self.dy,::self.dx], 
               scale=self.scale, transform=ccrs.PlateCarree(), color="#DD0000" )

        #  Reference wind speed barb. 

        refspeed = 10.0
        print( f'Reference wind barb = {refspeed:4.1f} m/s' )

        refspeed = 10.0
        rlons = np.zeros( (1,1), np.float32 ) - 122.0
        rlats = np.zeros( (1,1), np.float32 ) + 27.0
        ru = np.zeros( (1,1), np.float32 )
        rv = np.zeros( (1,1), np.float32 ) + refspeed

        ax.quiver( rlons, rlats, ru, rv, scale=self.scale, transform=ccrs.PlateCarree(), color="#000000", lw=2.0 )

        #  Draw region bounding box. 

        if bbox: 
            xb, yb = self.bbox['lonrange'], self.bbox['latrange']
            x = [ xb[0], xb[1], xb[1], xb[0], xb[0] ]
            y = [ yb[0], yb[0], yb[1], yb[1], yb[0] ]
            ax.plot( x, y, transform=ccrs.PlateCarree(), lw=1.2, color='#0000FF' )

        return ax


def compute_wind_barbs( model:{"narr","merra2","era5"} ): 
    """Compute the wind barbs for one of the models. A dictionary is returned containing 
    the related WindField instance for the model and the lengths of the u and v wind 
    components decomposed by month, hour, and height level above the surface."""

    #  Create a WindField instance. 

    analysisfile = os.path.join( DATAROOT, model.upper(), "isohypses", f"{model}_isohypses.2000-2024.nc" )
    WF = WindField( analysisfile, scale=150 )

    #  Wind barbs for region, annual cycle, diurnal cycle

    d = Dataset( analysisfile, 'r' )
    uwnd = np.ma.zeros( ( 12, 8, WF.nz ), np.float32 )
    vwnd = np.ma.zeros( ( 12, 8, WF.nz ), np.float32 )

    for imonth in range(12): 
        for ihour in range(8): 
            uwnd[imonth,ihour,:] = ( d.variables['uwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()
            vwnd[imonth,ihour,:] = ( d.variables['vwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()

    d.close()

    ret = { 'model': model, 'WF': WF, 'uwnd': uwnd, 'vwnd': vwnd }
    return ret


def plot_wind_barbs( wind_barbs ): 
    """Generate a wind barb plot based on an isohypsic climatology analysis. 
    The input argument is a dictionar output by compute_wind_barbs."""

    model = wind_barbs['model']
    uwnd = wind_barbs['uwnd']
    vwnd = wind_barbs['vwnd']
    WF = wind_barbs['WF']

    outputfile = f"{model}_wind_profiles.great-plains.pdf"
    nx, ny = 4, 3

    xticks = np.arange( 0, 24, 3, dtype=np.int32 )
    yticks = np.arange( 0, 3.01, 1, dtype=np.int32 )
    cmap1 = plt.get_cmap( 'gist_ncar' )
    colors = [ cmap1( (i+0.5)/8 ) for i in range(8) ]
    meta = { 'scale': 25, 'scale_units': "inches" }
    scale_length = 5.0

    print( f'Scale vector is {scale_length:5.2f} m/s' )

    fig = plt.figure( figsize=(9,6.5) )

    xmin, xmax = 1.0, 0.0
    ymin, ymax = 1.0, 0.0

    for imonth in range(12): 

        ix = imonth % nx
        iy = ny - int(imonth/nx) - 1
        pos = ( np.array( [0.05,0.02,0.90,0.83] ) + np.array( [ix,iy,0,0] ) ) / np.array( [nx,ny,nx,ny] )
        pos = np.array( [0.96,0.94,0.96,0.94] ) * pos + np.array( [0.04,0.06,0,0] )
        ax = fig.add_axes( pos )

        xmin = min( [ xmin, pos[0] ] )
        xmax = max( [ xmax, pos[0] + pos[2] ] )
        ymin = min( [ ymin, pos[1] ] )
        ymax = max( [ ymax, pos[1] + pos[3] ] )

        ax.set_xlim( -3, 24 )
        ax.set_xticks( xticks )
        if iy == 0: 
            ax.set_xticklabels( [ f'{int(x):02d}' for x in xticks ] ) 
        else: 
            ax.set_xticklabels( [] )

        ax.set_ylim( 0, 3 )
        ax.set_yticks( yticks )
        ax.yaxis.set_minor_locator( ticker.MultipleLocator(0.2) )
        if ix == 0: 
            ax.set_yticklabels( yticks )
        else: 
            ax.set_yticklabels( [] )

        ax.set_title( monthstrings[imonth] )

        for ihour in range(8): 
            x = np.repeat( WF.hours[ihour], WF.levels.size )
            y = WF.levels/1000
            ax.quiver( x, y, uwnd[imonth,ihour,:], vwnd[imonth,ihour,:], color=colors[ihour], **meta )

        ax.quiver( 23, 0.2, 0.0, scale_length, **meta, color="#000000" )

    fig.supxlabel( 'UTC hour', x=(xmin+xmax)/2, y=0.01, ha='center', va='bottom' )
    fig.supylabel( 'Height above surface [km]', x=0.01, y=(ymin+ymax)/2, ha='left', va='center' )

    print( f'Generating {outputfile}' )
    plt.savefig( outputfile )

    return outputfile

