import os 
from datetime import datetime, timedelta 
from netCDF4 import Dataset 
import numpy as np 
from pyllj.parameters import regions
from pyllj.podutilities.libpod import get_metricpath
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
from cartopy.feature import STATES, OCEAN
import warnings

warnings.filterwarnings('ignore')


# Define a class for vector field plotting. 

class WindField(): 

    def __init__( self, analysisfile, dx=6, dy=6, region:str="great-plains", scale=300.0 ): 

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

        rs = [ r for r in regions if r['name']==region ]
        if len( rs ) == 1: 
            r = rs[0]
        else: 
            print( f'Region "region" is unavailable' )
            return None

        self.region = region
        lonrange, latrange = r['longituderange'] * 1, r['latituderange'] * 1
        lonrange[ lonrange<0 ] += 360
        self.bbox = { 'lonrange': lonrange, 'latrange': latrange }

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


def compute_wind_barbs( model:str, region:str="southern-plains" ): 
    """Compute the wind barbs for one of the models. A dictionary is returned containing 
    the related WindField instance for the model and the lengths of the u and v wind 
    components decomposed by month, hour, and height level above the surface."""

    #  Create a WindField instance. 

    analysisfile = get_metricpath( "isohypses", model )
    if analysisfile is None: 
        analysisfile = os.path.join( DATAROOT, model, "isohypses", "isohypses.nc" )
        modelname = os.path.split( model )[-1]
    else: 
        modelname = model

    WF = WindField( analysisfile, region=region, scale=150 )

    #  Wind barbs for region, annual cycle, diurnal cycle

    d = Dataset( analysisfile, 'r' )
    uwnd = np.ma.zeros( ( 12, 8, WF.nz ), np.float32 )
    vwnd = np.ma.zeros( ( 12, 8, WF.nz ), np.float32 )

    for imonth in range(12): 
        for ihour in range(8): 
            uwnd[imonth,ihour,:] = ( d.variables['uwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()
            vwnd[imonth,ihour,:] = ( d.variables['vwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()

    d.close()

    ret = { 'model': modelname, 'WF': WF, 'uwnd': uwnd, 'vwnd': vwnd }
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

