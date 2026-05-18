import os
import stat
import re
import json
from datetime import datetime, timezone
import importlib
from tqdm import tqdm
from time import time
from datetime import datetime, timedelta, timezone
import earthaccess
import numpy as np
from scipy.interpolate import interp1d, CubicHermiteSpline
from netCDF4 import Dataset
from ..libutils import RetClass
from ..parameters import Rearth, Rideal, gravity, muvap, mudry, default_dataroot, aws_region, bucket

#  Suppress warnings. 

import warnings
warnings.filterwarnings( "ignore" )

#  AWS and storage path settings. 

cycle_time = 3      #  hours
fill_float = -1.0e20
downloads_subdir = "MERRA2/downloads" 
epoch = datetime( year=1980, month=1, day=1 )
output_time_units = "hours"
ncformat = "NETCDF4_CLASSIC"

#  Restrict to North America. 

region = { 
        'regionname': "NorthAmerica", 
        'longituderange': np.array( [ -135.0, -60.0 ] ), 
        'latituderange': np.array( [ 15.0, 55.0 ] )
        }

#  Authenticate earthaccess

auth = earthaccess.login( persist=True )

#  Exception handling. 

class Error( Exception ): 
    pass

class libmerra2Error( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment 


def truncate( inputfile, outputfile, longituderange, latituderange ): 
    """Truncate the variables and area of an MERRA2 file to longituderange, 
    latituderange, each being two-element tuples or lists defining the lower 
    and upper extents of the longitude and latitude ranges of the area. The 
    inputfile is the ERA5 download with full global grid; outputfile is the 
    output with area truncation."""

    print( f'Creating {outputfile}' )

    os.makedirs( os.path.dirname( outputfile ), exist_ok=True )
    d = Dataset( inputfile, 'r' )
    e = Dataset( outputfile, 'w', format=ncformat )

    #  Get longitude, latitude grid. 

    lons = d.variables['lon'][:].squeeze()
    lats = d.variables['lat'][:].squeeze()

    alongituderange = np.array( longituderange ).squeeze()
    alatituderange = np.array( latituderange ).squeeze()

    #  Justify longituderange. 

    alongituderange[ alongituderange >= 180.0 ] -= 360.0
    alongituderange[ alongituderange < -180.0 ] += 360.0

    #  Indices for subsetting in longitude and latitude. 

    if alongituderange[0] < alongituderange[1]: 
        ilons = np.argwhere( np.logical_and( lons >= alongituderange[0], lons <= alongituderange[1] ) ).squeeze()
    else: 
        ilons = np.argwhere( np.logical_or( lons >= alongituderange[0], lons <= alongituderange[1] ) ).squeeze()

    ilats = np.argwhere( np.logical_and( lats >= latituderange.min(), lats <= latituderange.max() ) ).squeeze()

    #  Dimensions. 

    for dimname in [ 'lon', 'lat', 'lev', 'time' ]: 
        if dimname == "lon": 
            e.createDimension( dimname, ilons.size )
        elif dimname == "lat": 
            e.createDimension( dimname, ilats.size )
        elif dimname == "time": 
            e.createDimension( dimname )
        else: 
            e.createDimension( dimname, d.dimensions[dimname].size )

    #  Global attributes. 

    e.setncatts( { att: d.getncattr(att) for att in d.ncattrs() } )

    #  Coordinate variables. 

    ss = "%Y-%m-%d %H:%M:%S"

    for varname in [ 'lon', 'lat', 'lev', 'time' ]: 
        vin = d.variables[varname]
        vout = e.createVariable( varname, vin.dtype, dimensions=vin.dimensions )
        vout.setncatts( { att: vin.getncattr(att) for att in vin.ncattrs() if att not in [ "valid_range" ] } )

        #  Coordinate variables. 

        if varname == "lon": 
            vout[:] = vin[ilons]
        elif varname == "lat": 
            vout[:] = vin[ilats]
        elif varname == "time": 
            m = re.search( r'^([a-z]+) since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$', vin.units )
            units = m.group(1)
            vout.setncattr( "units", f"minutes since " + epoch.strftime(ss) )
            depoch = int( ( datetime.strptime( m.group(2), ss ) - epoch ) / timedelta( minutes=1 ) )
            vals = vin[:] * ( timedelta( **{units:1} ) / timedelta( minutes=1 ) ) + depoch
            # vout.setncattr( "valid_range", [ vals.min(), vals.max() ] )
            vout[:] = vals
        else: 
            vout[:] = vin[:]

    #  Data field variables. 

    for varname in [ 'PHIS', 'PL', 'PS', 'H', 'QV', 'T', 'U', 'V' ]: 
        vin = d.variables[varname]

        vout = e.createVariable( varname, vin.dtype, dimensions=vin.dimensions )
        attdict = { att: vin.getncattr(att) for att in vin.ncattrs() if att not in [ "valid_range" ] \
                if att not in [ '_FillValue', 'missing_value', 'fmissing_value', 'scale_factor', 'add_offset' ] } 
        vout.setncatts( attdict )

        if vin.ndim == 2: 
            vout[:,:] = vin[ilats,ilons]
        elif vin.ndim == 3: 
            ni = d.dimensions[ vin.dimensions[0] ].size
            for i in range(ni): 
                vout[i,:,:] = vin[i,ilats,ilons]
        elif vin.ndim == 4: 
            ni = d.dimensions[ vin.dimensions[0] ].size
            for i in range(ni): 
                vout[i,:,:,:] = vin[i,:,ilats,ilons]

    #  Done with truncation. 

    d.close()
    e.close()

    return outputfile 


################################################################################
#  Define a class useful for dealing with ERA5 data files. 
################################################################################

class MERRA2file(): 
    """A class to download, remove, and provide a local path to a North American 
    Regional Reanalysis (ERA5) data file."""

    def __init__( self, dtime:datetime, dataroot:str=default_dataroot ): 
        """Access a MERRA2 data file for a particular time. If the file is not currently 
        available on the file system, then download it through NASA Earthdata. 

        Arguments
        =========
        dtime           An instance of datetime.datetime for the reanalysis fields
        dataroot        The root directory for all data used in LLJ analysis."""

        self.time = dtime
        self.dataroot = dataroot
        self.dataset = None
        self.request = {}
        self.success = True
        self.message = None
        self.comment = None

        #  Justify longituderange. 

        ii = ( region['longituderange'] >= 180.0 )
        region['longituderange'][ii] -= 360.0
        ii = ( region['longituderange'] < -180.0 )
        region['longituderange'][ii] += 360.0

        #  Final path. 

        downloadsroot = os.path.join( dataroot, downloads_subdir )
        basename = f'merra2.M2I3NVASM.{regionname}.{dtime.year:4d}{dtime.month:02d}{dtime.day:02d}.nc'
        self.localpath = os.path.join( downloadsroot, f'{dtime.year:4d}', f'{dtime.month:02d}', basename )

        if os.path.exists( self.localpath ): 
            try: 
                d = Dataset( self.localpath, 'r' )
                d.close()
                download = False
                print( f'{self.localpath} exists and is readable NetCDF' )
            except: 
                print( f'{self.localpath} exists but is unreadable NetCDF' )
                download = True
        else: 
            download = True

        if download: 

            t0 = time()

            temporal = ( dtime.strftime( "%Y-%m-%d" ), dtime.strftime( "%Y-%m-%d" ) )
            try:
                recs = earthaccess.search_data( short_name="M2I3NVASM", temporal=temporal )
            except:
                recs = []

            if len( recs ) == 0: 
                raise libmerra2Error( "DataUnavailable", 'MERRA2 data for {:} is unavailable'.format( dtime.strftime( "%Y-%m-%d %H:%M:%S" ) ) )

            #  Download. 

            tmpdir = os.path.join( downloadsroot, "tmp" )
            os.makedirs( tmpdir, exist_ok=True )
            inputfile = os.path.join( tmpdir, recs[0]['umm']['DataGranule']['Identifiers'][0]['Identifier'] )

            if not os.path.exists( inputfile ): 
                earthaccess.download( recs, local_path=tmpdir )

            #  Subset downloaded file. 

            ret = truncate( inputfile, self.localpath, region['longituderange'], region['latituderange'] )

            #  Remove temporary file. 

            os.unlink( inputfile )

            t1 = time()
            dt = int( t1 - t0 )
            hrs, mins, secs = int(dt/3600), int(dt/60) % 60, dt % 60
            print( f'  elapsed time = {hrs:d} hrs, {mins:2d} mins, {secs:2d} secs\n' )

        return

    def open( self ): 
        """Open the NetCDF file and return a netCDF4.Dataset object."""

        ret = Dataset( self.localpath, 'r' )
        return ret

    def remove( self ): 
        os.unlink( self.localpath )


def netrcauth( authstr:str ): 
    """Create a .netrc file for authentication of NASA Earthdata. The input string should 
    contain the username and then the password, separated by a single space."""

    username, password = authstr.split(" ")

    rootdir = os.path.expanduser("~")
    path = os.path.join( rootdir, ".netrc" )
    with open( path, 'w' ) as f: 
        f.write( f'machine urs.earthdata.nasa.gov login {username} password {password}\n' )

    os.chmod( path, stat.S_IRUSR )

    return

