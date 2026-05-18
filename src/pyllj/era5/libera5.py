import os
import json
from datetime import datetime, timezone
import importlib
from tqdm import tqdm
from time import time
from datetime import datetime, timedelta, timezone
import cdsapi
import numpy as np
from scipy.interpolate import interp1d, CubicHermiteSpline
from netCDF4 import Dataset
from ..parameters import Rearth, Rideal, gravity, muvap, mudry, default_dataroot, aws_region, bucket
from ..libutils import RetClass

#  AWS and storage path settings. 

cycle_time = 3      #  hours
fill_float = -1.0e20
downloads_subdir = os.path.join( "ERA5", "downloads" )

#  Epoch for output files. 

epoch = datetime( year=1980, month=1, day=1 )
output_time_units = "hours"

#  Get hybrid levels. 

with importlib.resources.open_text( "pyllj.era5", "hybrid.json" ) as d: 
    hybrid = json.load( d )
    for key in hybrid.keys(): 
        hybrid[key] = np.array( hybrid[key] )

#  Exception handling. 

class Error( Exception ): 
    pass

class libera5Error( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment 


#  CDS API client

def initCDS( key ): 
    if key != "": 
        root = os.path.expanduser( "~" )
        path = os.path.join( root, ".cdsapirc" )
        if os.path.exists( path ):  
            print( f'Error: {path} already exists; exiting' )
            exit()

        with open( path, 'w' ) as f:  
            print( f'Creating {path} with given key' )
            f.write( 'url: https://cds.climate.copernicus.eu/api\n' )
            f.write( f'key: {key}\n' )

    client = cdsapi.Client()
    return client


def area_truncate( inputfile, outputfile, longituderange, latituderange ): 
    """Truncate the area of an ERA5 file to longituderange, latituderange, each being 
    two-element tuples or lists defining the lower and upper extents of the 
    longitude and latitude ranges of the area. The inputfile is the ERA5 download with 
    full global grid; outputfile is the output with area truncation."""
    #  THIS FUNCTION IS BUGGY, CURRENTLY UNUSED - ssl, 23 Oct 2025

    print( f'Creating {outputfile}' )

    d = Dataset( inputfile, 'r' )
    e = Dataset( outputfile, 'w', format="NETCDF4" )

    #  Get longitude, latitude grid. 

    lons = d.variables['longitude'][:].squeeze()
    lats = d.variables['latitude'][:].squeeze()

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

    for dimname, dim in d.dimensions.items(): 
        if dimname == "longitude": 
            e.createDimension( dimname, ilons.size )
        elif dimname == "latitude": 
            e.createDimension( dimname, ilats.size )
        else: 
            e.createDimension( dimname, dim.size )

    #  Global attributes. 

    e.setncatts( { att: d.getncattr(att) for att in d.ncattrs() } )

    #  Coordinate variables. 

    for varname, var in d.variables.items(): 

        if var.ndim == 1: 

            vout = e.createVariable( varname, var.dtype, dimensions=var.dimensions )
            vout.setncatts( { att: var.getncattr(att) for att in var.ncattrs() } )

            #  Coordinate variables. 

            if varname == "longitude": 
                vout[:] = var[ilons]
            elif varname == "latitude": 
                vout[:] = var[ilats]
            else: 
                vout[:] = var[:]

    #  Data field variables. 

    for varname, var in d.variables.items(): 

        if var.ndim > 1: 

            vout = e.createVariable( varname, var.dtype, dimensions=var.dimensions )
            vout.setncatts( { att: var.getncattr(att) for att in var.ncattrs() } )

            if var.ndim == 2: 
                vout[:,:] = var[ilats,ilons]
            elif var.ndim == 3: 
                ni = d.dimensions[ var.dimensions[0] ].size
                for i in range(ni): 
                    vout[i,:,:] = var[i,ilats,ilons]
            elif var.ndim == 4: 
                ni = d.dimensions[ var.dimensions[0] ].size
                for i in range(ni): 
                    vout[i,:,:,:] = var[i,:,ilats,ilons]

    #  Done with truncation. 

    d.close()
    e.close()

    return outputfile 


################################################################################
#  Define a class useful for dealing with ERA5 data files. 
################################################################################

class ERA5file(): 
    """A class to download, remove, and provide a local path to a North American 
    Regional Reanalysis (ERA5) data file."""

    def __init__( self, var:str, client:cdsapi.Client, dtime:datetime=None, dataroot:str=default_dataroot ): 
        """Instantiate a portal to an ERA5file used for LLJ analysis. 

        Arguments
        =========
        var             A string defining a variable, drawn from upperair_vars | surface_vars
        client          An instance of cdsapi.Client, giving access to the Copernicus Climate Data 
                        Store to enable downloads of ECMWF ERA5 data
        dtime           An instance of datetime.datetime indicating the time interval of interest in 
                        ERA5 analyses
        dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the 
                        FileGateway Data directory
        """

        self.var = var
        self.client = client
        self.time = dtime
        self.dataroot = dataroot
        self.localpath = None
        self.dataset = None
        self.request = {}
        self.success = True
        self.message = None
        self.comment = None
        self.downloadsroot = os.path.join( dataroot, downloads_subdir )

        upperair_vars = [ "temp", "geop", "etadot", "shum", "uwnd", "vwnd" ]
        surface_vars = [ "pres.sfc", "geop.sfc", "temp.2m", "dewpt.2m", "uwnd.10m", "vwnd.10m" ]

        #  Restrict to North America. 

        longituderange = np.array( [ -135.0, -60.0 ] )
        latituderange = np.array( [ 15.0, 55.0 ] )

        #  Justify longituderange. 

        ii = ( longituderange >= 180.0 )
        longituderange[ii] -= 360.0
        ii = ( longituderange < -180.0 )
        longituderange[ii] += 360.0

        #  Define area. 

        area = "/".join( [ f'{x:.2f}'.strip() for x in [ latituderange.max(), longituderange[0], latituderange.min(), longituderange[1] ] ] )

        #  date-string definition. 

        ff = "%Y-%m-%d"

        if var in upperair_vars: 

            #  One month of data. 

            d0 = datetime( year=dtime.year, month=dtime.month, day=1 )
            d1 = d0 + timedelta(days=31)
            d1 = datetime( year=d1.year, month=d1.month, day=1 ) - timedelta(days=1)

            basename = f'{var}.{dtime.year:4d}{dtime.month:02d}.nc'
            self.localpath = os.path.join( self.downloadsroot, "modellevels", basename )

            self.dataset = "reanalysis-era5-complete"
            self.request = { 
                            'class': "ea", 
                            'date': "{:}/to/{:}".format( d0.strftime(ff), d1.strftime(ff) ), 
                            'levtype': "ml", 
                            'levelist': "1/to/137", 
                            'param': None, 
                            'stream': "oper", 
                            'expver': "1", 
                            'time': "00/to/23/by/3", 
                            'type': 'an', 
                            'grid': "0.25/0.25", 
                            'format': "netcdf" 
                            }

            #  Restrict to North America. 

            self.request.update( { 'area': area } ) 

            #  Variable identifiers. 

            if var == "temp": 
                self.request['param'] = "130"

            elif var == "shum": 
                self.request['param'] = "133"

            elif var == "uwnd": 
                self.request['param'] = "131"

            elif var == "vwnd": 
                self.request['param'] = "132"

            elif var == "geop": 
                self.request['param'] = "129"

            elif var == "etadot": 
                self.request['param'] = "77"

        elif var in surface_vars: 

            #  One year of data. 

            d0 = datetime( year=dtime.year, month=1, day=1 )
            d1 = datetime( year=dtime.year, month=12, day=31 )

            self.dataset = "reanalysis-era5-complete"
            self.request = { 
                            'class': "ea", 
                            'date': "{:}/to/{:}".format( d0.strftime(ff), d1.strftime(ff) ), 
                            'levtype': "sfc", 
                            'param': None, 
                            'stream': "oper", 
                            'time': "00/to/23/by/3", 
                            'type': 'an', 
                            'grid': "0.25/0.25", 
                            'format': "netcdf" 
                            }

            if var == "geop.sfc": 
                self.request = { 
                            'date': "2000-01-01", 
                            'levtype': "sfc", 
                            'param': "129.128", 
                            'stream': "oper", 
                            'time': "00", 
                            'type': 'an', 
                            'grid': "0.25/0.25", 
                            'format': "netcdf" 
                            }

            elif var == "pres.sfc": 
                self.request['param'] = "134.128"

            elif var == "temp.2m": 
                self.request['param'] = "167.128"

            elif var == "dewpt.2m": 
                self.request['param'] = "168.128"

            elif var == "uwnd.10m": 
                self.request['param'] = "165.128"

            elif var == "vwnd.10m": 
                self.request['param'] = "166.128"

            #  Restrict to North America. 

            self.request.update( { 'area': area } ) 

            #  Define subdirectory. 

            if var in [ "geop.sfc" ]: 
                basename = f'{var}.nc'
                self.localpath = os.path.join( self.downloadsroot, "time_invariant", basename )

            else: 
                basename = f'{var}.{dtime.year:4d}.nc'
                self.localpath = os.path.join( self.downloadsroot, "monolevel", basename )

        else: 

            self.success = False
            self.message = "UnrecognizedVariable"
            self.comment = f'Variable {var} is not recognized; recognized variables are ' + \
                    ', '.join( upperair_vars + surface_vars ) + '.' 

        #  Check if the file already exists, and, if it exists, if it is a valid 
        #  NetCDF file. If the file doesn't exist or is not a NetCDF file, then 
        #  self.success = False and a download is attempted. 

        if os.path.isfile( self.localpath ): 
            try: 
                d = Dataset( self.localpath, 'r' )
                d.close()
                self.success = True
            except: 
                self.success = False
        else: 
            self.success = False 

        #  Attempt download. 

        if not self.success : 

            if var == "geop": 

                #  Things are different for upper air geopotential, which ERA5 does not serve up. Instead, 
                #  it requires that it be computed from upper air temperature and humidity data and the 
                #  surface geopotential. 

                ret = self.compute_geop()

            else: 

                self.success, ntries = False, 0

                #  Attempt download 10 times. 

                t0 = time()

                print( f'  Downloading {self.localpath}' )

                while not self.success and ntries < 10: 
                    ntries += 1

                    #  Test for validity of download. 

                    try: 
                        os.makedirs( os.path.dirname( self.localpath ), exist_ok=True )
                        client.retrieve( self.dataset, self.request, self.localpath ) 

                        d = Dataset( self.localpath, 'r' )
                        d.close()
                        self.success = True

                    except: 
                        self.success = False
                        continue

                if self.success: 
                    self.comment = f"Downloaded {self.localpath} after {ntries} tries"
                else: 
                    self.message = "RemoteFileUnavailable"
                    self.comment = f"Unable to download remote file {self.localpath}."
                    raise libera5Error( self.message, self.comment )

                t1 = time()
                dt = int( t1 - t0 )
                hrs, mins, secs = int(dt/3600), int(dt/60) % 60, dt % 60
                print( f'  elapsed time = {hrs:d} hrs, {mins:2d} mins, {secs:2d} secs\n' )

    def open( self ): 
        """Open the NetCDF file and return a netCDF4.Dataset object."""

        ret = Dataset( self.localpath, 'r' )
        return ret

    def remove( self ): 
        os.unlink( self.localpath )

    def compute_geop( self ): 
        """Compute upper air geopotential based on temperature, specific humidity, and 
        surface pressure and geopotential fields."""

        print( "  Computing upper air geopotential for {:}".format( self.time.strftime( "%Y-%m" ) ) )

        temp = ERA5file( "temp", self.client, self.time, dataroot=self.dataroot )
        shum = ERA5file( "shum", self.client, self.time, dataroot=self.dataroot )
        ps = ERA5file( "pres.sfc", self.client, self.time, dataroot=self.dataroot )
        hs = ERA5file( "geop.sfc", self.client, self.time, dataroot=self.dataroot )

        #  Open dependent files. 

        t0 = time()

        d = { 
             'temp': temp.open(), 
             'shum': shum.open(), 
             'ps': ps.open(), 
             'hs': hs.open()
             }

        #  Get dimensions and coordinates. 

        dd = d['temp']
        nlons = dd.dimensions['longitude'].size 
        nlats = dd.dimensions['latitude'].size
        nlevels = dd.dimensions['model_level'].size
        ntimes = dd.dimensions['valid_time'].size
        nlevels1 = nlevels + 1

        #  Coordinates. 

        coords = {}
        for var in [ 'longitude', 'latitude', 'model_level', 'valid_time' ]: 
            v = dd.variables[var]
            coords[var] = { 
                    'dtype': v.dtype, 
                    'dims': v.dimensions, 
                    'atts': { att: v.getncattr(att) for att in v.ncattrs() }, 
                    'vals': v[:]
                }

        coords['model_level1'] = { 
                    'dtype': np.int32, 
                    'dims': ( "model_level1", ), 
                    'atts': { 
                             'long_name': "hybrid interface level", 
                             'units': "1", 
                             'positive': coords['model_level']['atts']['positive'], 
                             'standard_name': "atmosphere_hybrid_sigma_interface_pressure_coordinate" }, 
                    'vals': np.arange( 1, nlevels1+1, dtype=np.int32 )
                }

        #  Get dimensions of 4D field. 

        dims = d['temp'].variables['t'].dimensions
        dims4d = ( dims[0], 'model_level1', dims[2], dims[3] )

        #  Create output file. 

        d['h'] = Dataset( self.localpath, 'w', "NETCDF4" )
        dd = d['h']

        #  Create dimensions and coordinates. 

        dd.createDimension( "longitude", nlons )
        dd.createDimension( "latitude", nlats )
        dd.createDimension( "model_level1", nlevels1 )
        dd.createDimension( "valid_time" )

        for key in dims4d: 
            val = coords[key]
            v = dd.createVariable( key, val['dtype'], dimensions=val['dims'] )
            v.setncatts( val['atts'] )
            v[:] = val['vals']

        #  Define output variable. 

        v = dd.createVariable( "z", np.float32, dimensions=dims4d )
        v.setncatts( { 
                'description': "Upper air geopotential", 
                'units': "J/kg", 
                '_FillValue': np.float32(fill_float) } )

        #  Global attributes. 

        dd.setncatts( { 
            'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ), 
            'file_type': "ERA5 upper air geopotential at layer interfaces" } )

        #  Loop over time. 

        hs = d['hs'].variables['z'][:].squeeze()

        for itime in tqdm( range(ntimes), desc="Time" ): 

            v_temp = d['temp'].variables['t'][itime,:,:,:].squeeze()
            v_shum = d['shum'].variables['q'][itime,:,:,:].squeeze()
            v_ps = d['ps'].variables['sp'][itime,:,:].squeeze()

            #  Compose pressure profiles at layer interfaces. 

            v_p = np.outer( hybrid['hybi'], v_ps.flatten() ).reshape( nlevels+1, nlats, nlons )
            v_p = ( v_p.transpose( (1,2,0) ) + hybrid['hyai'] ).transpose( (2,0,1) )

            #  Compose delta log-pressure for each layer. And scaleheight. 

            ii = ( v_p[:] <= 0.0 )
            pmin = v_p[ np.logical_not(ii) ].min()
            v_p[ii] = pmin / np.exp(1.0)
            dlnp = np.abs( np.log( v_p[1:,:,:] / v_p[:-1,:,:] ) )
            H = Rideal * v_temp * ( (v_shum)/muvap + (1-v_shum)/mudry )

            #  Integrate the hypsometric equation. 

            Dh = np.zeros( (nlevels+1,nlats,nlons), np.float32 )
            Dh[1:,:,:] = -( H * dlnp ).cumsum( axis=0 )

            #  Turn upside down if necessary; add to surface geopotential. 

            if dlnp[3,0,0] > 0.0: 
                h = Dh - Dh[-1,:,:] + hs
            else: 
                h = Dh + hs

            #  Write to output. 

            dd.variables['z'][itime,:,:,:] = h

        #  Done. 

        for key, val in d.items(): 
            val.close()

        t1 = time()
        dt = int( t1 - t0 )
        hrs, mins, secs = int(dt/3600), int(dt/60) % 60, dt % 60
        print( f'  elapsed time = {hrs:d} hrs, {mins:2d} mins, {secs:2d} secs\n' )

        return

