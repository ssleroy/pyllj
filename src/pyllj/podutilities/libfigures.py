import os 
from datetime import datetime, timedelta 
from netCDF4 import Dataset 
import numpy as np 
from pyllj.parameters import regions, boundaries
from pyllj.podutilities.libpod import get_metricpath
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
from cartopy.feature import STATES, OCEAN
import warnings

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

warnings.filterwarnings('ignore')


# Define a class for vector field plotting. 

class WindField(): 

    def __init__( self, analysisfile, dx=6, dy=6, region:str="great-plains", scale=300.0 ): 

        self.dx = dx
        self.dy = dy
        self.scale = scale
        self.region = region

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

        #  Define the mask. 

        rs = [ r for r in regions if r['name']==region ]
        if len( rs ) == 1: 
            r = rs[0]
        else: 
            print( f'Region "{region}" is unavailable' )
            return None

        lonrange, latrange = r['longituderange'] * 1, r['latituderange'] * 1
        lonrange[ lonrange<0 ] += 360
        self.bbox = { 'lonrange': lonrange, 'latrange': latrange }

        if lonrange.size == 1 and latrange.size == 1: 

            #  Select nearest gridpoint. 

            mlons = np.deg2rad( self.mlons )
            mlats = np.deg2rad( self.mlats )
            lon = np.deg2rad( lonrange[0] )
            lat = np.deg2rad( latrange[0] )

            mp = np.array( [ np.cos(mlons) * np.cos(mlats), np.sin(mlons) * np.cos(mlats), np.sin(mlats) ] )
            p = np.array( [ np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat) ] )
            pmp = np.matmul( mp.T, p ).T
            ii = np.argmax( pmp ).squeeze()
            ilat, ilon = int( ii / mlons.shape[1] ), ( ii % mlons.shape[1] )
            self.mask = np.zeros( mlons.shape, np.int8 )
            self.mask[ilat,ilon] = 1

        else: 

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


def compute_wind_barbs( model, modellabel:str=None, region:str="southern-plains" ): 
    """Compute the wind barbs for one of the models. A dictionary is returned containing 
    the related WindField instance for the model and the lengths of the u and v wind 
    components decomposed by month, hour, and height level above the surface."""

    #  Create a WindField instance. 

    analysisfile = get_metricpath( "isohypses", model )
    if analysisfile is None: 
        analysisfile = os.path.join( model, "isohypses", "isohypses.nc" )
        if modellabel is None: 
            ss = model.split( "/" )
            if ss[-1] != "": 
                modelname = ss[-1]
            else: 
                modelname = ss[-2]
        else: 
            modelname = modellabel

    WF = WindField( analysisfile, region=region, scale=150 )

    #  Wind barbs for region, annual cycle, diurnal cycle

    d = Dataset( analysisfile, 'r' )
    hours = d.variables['hour'][:]
    ncp = hours.size
    uwnd = np.ma.zeros( ( 12, ncp, WF.nz ), np.float32 )
    vwnd = np.ma.zeros( ( 12, ncp, WF.nz ), np.float32 )

    for imonth in range(12): 
        for ihour in range(ncp): 
            uwnd[imonth,ihour,:] = ( d.variables['uwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()
            vwnd[imonth,ihour,:] = ( d.variables['vwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()

    d.close()

    ret = { 'model': model, 'modelname': modelname, 'WF': WF, 'uwnd': uwnd, 'vwnd': vwnd, 'hours': hours }
    return ret


def plot_wind_barbs( wind_barbs ): 
    """Generate a wind barb plot based on an isohypsic climatology analysis. 
    The input argument is a dictionar output by compute_wind_barbs."""

    model = wind_barbs['model']
    uwnd = wind_barbs['uwnd']
    vwnd = wind_barbs['vwnd']
    WF = wind_barbs['WF']

    outputfile = f"{model}_wind_profiles.{WF.region}.pdf"
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


