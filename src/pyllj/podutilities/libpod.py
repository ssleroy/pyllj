"""Library of tools useful in POD analysis and plotting. 

Authors: Stephen Leroy (stephen.leroy@janusresearch.us), 
        Sara Vannah (sara.vannah@janusresearch.us)
Date: May 22, 2026

Contents
========
class ModelOutput               Creates a portal to the atmospheric model output
class WindField                 Defines a wind field and provides a method to plot it
function get_metricpath         Retrieves and references an LLJ metric
"""

import os
import re
import json
import netrc
import requests 
from datetime import datetime, timedelta, timezone
from netCDF4 import Dataset
import numpy as np
from tqdm import tqdm
from time import time
from scipy.interpolate import interp1d, CubicHermiteSpline
from ..libutils import RetClass 
from ..parameters import Rearth, Rideal, gravity, muvap, mudry, default_dataroot, \
        aws_region, bucket, zenodoversion, regions, boundaries
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
from cartopy.feature import STATES, OCEAN
import warnings

warnings.filterwarnings('ignore')

#  Pyplot settings. 

axeslinewidth = 0.5 
plt.rcParams.update( {
  'font.family': "DejaVu Serif",
  'mathtext.fontset': "dejavuserif",
  'font.size': 8,
  'font.weight': "normal",
  'text.usetex': True,
  'xtick.major.width': axeslinewidth,
  'xtick.minor.width': axeslinewidth, 
  'ytick.major.width': axeslinewidth, 
  'ytick.minor.width': axeslinewidth, 
  'axes.linewidth': axeslinewidth } ) 

#  Constants and defaults. 

fill_float = -1.0e20
epoch = datetime( year=1980, month=1, day=1 )
output_time_units = "hours"

#  Caching, download. 

rcfile = os.path.expanduser( "~/.pylljrc" )

#  Subdirectory of post-processed model output. 

modeloutput_subdir = "work/pp"

#  Exception handling. 

class Error( Exception ): 
    pass

class libpodError( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment 


#  Configure cache. 

def configure_cache( cachedir:str ): 
    """This function should be used just once by any individual user. It establishes 
    the directory where cached data should reside. Presently, it should reside on the 
    local file system with ~4 GB storage available. The cache directory name is 
    written in ~/.pylljrc, which contains JSON-formatted data."""

    ret = RetClass()
    acachedir = os.path.abspath( os.path.expanduser( cachedir ) ) 

    try: 
        os.makedirs( acachedir, exist_ok=True )
    except: 
        ret.update( success=False, messages="InvalidPath", comments=f'Unable to make directory {acachedir}.' )
        return ret

    rc = {}
    if os.path.exists( rcfile ): 
        with open( rcfile, 'r' ) as f: 
            rc = json.load( f )

    rc.update( { 'cachedir': acachedir } )

    with open( rcfile, 'w' ) as f: 
        json.dump( rc, f, indent=2 )

    ret.update( success=True )
    return ret


#  Get paths to metric data files. 

def get_metricpath( metric:str, reanalysis:str ): 
    """Get the absolute path to a data file to be used as the reference for 
    a POD metric. 

    Arguments
    =========
    metric      A string that selects the metric: "diagnostics", "watervaporflux", 
                "boundaryflux", or "isohypses". 

    reanalysis  A string naming the reanalysis to be used as reference: "narr", 
                "merra2", "era5". """

    valid_metrics = [ "diagnostics", "watervaporflux", "boundaryflux", "isohypses" ]
    if metric not in valid_metrics: 
        print( 'Metric must be one of ' + ', '.join( 
                [ f'"{vmetric}"' for vmetric in valid_metrics ] ) )
        return None

    valid_reanalyses = [ "narr", "merra2", "era5" ]
    if reanalysis not in valid_reanalyses: 
        print( 'Reanalysis must be one of ' + ', '.join( 
                [ f'"{vreanalysis}"' for vreanalysis in valid_reanalyses ] ) )
        return None

    rc = {}
    if os.path.exists( rcfile ): 
        with open( rcfile, 'r' ) as f: 
            rc = json.load( f )

    if "cachedir" in rc.keys(): 
        cachedir = rc['cachedir']
    else: 
        print( 'cachedir not in resource file. Use configure_cache to configure the cache directory.' )
        return None

    if metric == "boundaryflux": 
        m = "boundaryflux.great-plains"
    else:
        m = metric

    file = f'{reanalysis}_{m}.nc' 
    local_path = os.path.join( cachedir, file )

    #  Check if data file already resides on local file system. 

    if not os.path.exists( local_path ): 

        #  Download data file. 

        print( f'Downloading {file}' )

        remote_url = f'https://zenodo.org/records/{zenodoversion}/files/{file}?download=1'
        with requests.get( remote_url, stream=True ) as r: 
            r.raise_for_status()
            with open( local_path, 'wb' ) as f: 
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    return local_path


class ModelOutput(): 
    """A class to interact with output files of the GFDL AM4 or the NCAR CAM7
    models."""

    def __init__( self, dataroot:str ): 
        """Instantiate the interface to the model run output."""

        if not os.path.isdir( dataroot ): 
            raise libpodError( message="InvalidPath", comment=f'"{dataroot}" is not a valid directory' )

        self.dataroot = dataroot

        #  Get a comprehensive listing of output files. 


        # TODO: don't require dataroot, make model identification indenepdent of dir structure. 
        if "AM4" in rootpath:
            rootpath = os.path.join( self.dataroot, "work", "pp", "atmos_8xdaily_inst" )
        elif "CAM7" in rootpath: 
            rootpath = os.path.join( self.dataroot, "llj.01", "downloads", "gridded" )
        ncfiles = []

        for root, subdirs, files in os.walk( rootpath ): 
            subdirs.sort()
            files.sort()
            ncfiles += [ os.path.join( root, f ) for f in files if re.search( r'\.nc$', f ) ]

        #  Scan output files. 

        self.filemeta = []
        self.toffset = None
        self.tdelta = None
        self.lons = None
        self.lats = None
        self.hyai = None
        self.hybi = None
        self.z3 = None 
        self.zg = None
        self.zsurg = None

        for ncfile in ncfiles: 
            d = Dataset( ncfile, 'r' )

            #  Get surface orography. 

            if "zsurf" in d.variables.keys() and "zg" in d.variables.keys()  and self.z3 is None: 
                # AM4 holds zg and zsurf as separate variables. 
                self.zg = d.variables["zg"]
                self.zsurf = d.variables["zsurf"]

            elif"Z3" in d.variables.keys() and self.z3 is None:
                # CAM7 height variable. 
                self.z3 = d.variables["Z3"]

            #  Process times. 

            if 'time' not in d.variables.keys(): 
                d.close()
                continue

            v = d.variables['time']
            attr = v.getncattr( "units" )
            m = re.search( r'^(\w+) since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', attr )
            time_units = timedelta( **{ m.group(1): 1.0 } )
            epoch = datetime.strptime( m.group(2), "%Y-%m-%d %H:%M:%S" )
            times = [ epoch + float(dt) * time_units for dt in v[:] ]

            #  Compute the time offset from the beginning of the month, midnight. 

            if self.toffset is None: 
                self.toffset = times[0] - datetime( year=times[0].year, month=times[0].month, day=1 )

            if self.tdelta is None: 
                self.tdelta = times[1] - times[0] 

            #  Get latitudes and longitudes. Justify longitudes. 

            if self.lons is None: 
                self.lons = d.variables['lon'][:]
                self.lons[ self.lons < 0 ] += 360

            if self.lats is None: 
                self.lats = d.variables['lat'][:]

            #  Find the atmospheric variable. 

            coords = { "lon", "lat", "time" }
            vn = None

            for vname, v in d.variables.items(): 
                if coords.issubset( set( v.dimensions ) ): 
                    vn = vname
                    break 

            #  Find vertical ordering. 

            if "levhalf" in d.variables.keys(): 
                z = d.variables['levhalf'][:]
                ascending = ( z[1] < z[0] )
            elif "pfull" in d.variables.keys(): 
                z = d.variables['pfull'][:]
                ascending = ( z[1] < z[0] )
            else: 
                ascending = None

            #  Record metadata. 

            if vn is not None: 
                self.filemeta.append( { 
                        'path': ncfile, 
                        'times': np.array( times ), 
                        'time_range': ( times[0], times[-1] ), 
                        'variable': vn, 
                        'ascending': ascending, 
                        'open': None } )

            #  Done with file. 

            d.close()

        #  Catalog of variable names. 

        self.variable_names = sorted( list( { fm['variable'] for fm in self.filemeta } ) )

        #  Get hybrid vertical coefficients. 

        rootpath = os.path.join( self.dataroot, "work", "history" )
        self.bk = None
        self.pk = None

        for root, subdirs, files in os.walk( rootpath ): 
            files.sort()
            subdirs.sort()

            ncfiles = [ os.path.join( root, f ) for f in files if re.search( r'atmos_8xdaily_inst\.tile.*\.nc\.\d{4}$', f ) ]
            for ncfile in ncfiles: 
                with Dataset( ncfile, 'r' ) as d: 
                    self.bk = d.variables['bk'][:]
                    self.pk = d.variables['pk'][:]
                if self.bk is not None and self.pk is not None: 
                    break

            if self.bk is not None and self.pk is not None: 
                break

        return

    def getvar( self, variable, datetime ): 
        """This method returns a pointer to netCDF4 variable and the time index 
        of the corresponding datatime in that variable. 

        The specific return is a 3-tuple, ( variable_pointer, itime, ascending )

        variable_pointer            A pointer to a netCDF4.Variable object

        itime                       The time index in the variable corresponding 
                                    to datetime

        ascending                   A boolean or None indicating if the vertical 
                                    coordinate of the atmospheric variable is 
                                    ascending (True) or not (False)
        """

        eps = timedelta( seconds=10 )
        vpointer, itime, ascending = None, None, None

        for fm in self.filemeta: 
            if fm['variable'] == variable: 
                if datetime >= fm['time_range'][0]-eps and datetime <= fm['time_range'][1]+eps: 
                    if fm['open'] is None: 
                        fm['open'] = Dataset( fm['path'], 'r' )
                    vpointer = fm['open'].variables[variable]
                    ascending = fm['ascending']

                    itimes = np.argwhere( np.abs( fm['times'] - datetime ) < eps ).squeeze()
                    if itimes.size == 1: 
                        itime = itimes

        return vpointer, itime, ascending

    
    def close( self ): 
        """Close all open files."""

        for fm in self.filemeta: 
            if fm['open'] is not None: 
                fm['open'].close()
                fm['open'] = None


################################################################################
#  Define a class for vector field plotting. 
################################################################################

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

