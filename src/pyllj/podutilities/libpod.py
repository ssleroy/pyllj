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
from ..parameters import Rearth, Rideal, gravity, muvap, mudry, default_dataroot, aws_region, bucket, zenodoversion

cycle_time = 3      #  hours
fill_float = -1.0e20
epoch = datetime( year=1980, month=1, day=1 )
output_time_units = "hours"


#  Caching, download. 

rcfile = os.path.expanduser( "~/.pylljrc" )

#  Subdirectory of post-processed model output. 

modeloutput_subdir = "work/pp"

#  Define North America region. 

region = { 'name': "North America", 'longitude_range': [230,300], 'latitude_range': [20,50] }


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

        rootpath = os.path.join( self.dataroot, "work", "pp", "atmos_8xdaily_inst" )
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
        self.zsurf = None

        for ncfile in ncfiles: 
            d = Dataset( ncfile, 'r' )

            #  Get surface orography. 

            if "zsurf" in d.variables.keys() and self.zsurf is None: 
                self.zsurf = d.variables['zsurf'][:]

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

