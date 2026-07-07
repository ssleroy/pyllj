import os
import re
import sys
import shutil
import argparse
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
import boto3
from time import time
from .libera5 import RetClass, cycle_time, ERA5file, downloads_subdir, initCDS
from ..parameters import default_dataroot, bucket, aws_region, gravity 


output_subdir = "ERA5/isohypses"

#  AWS communication. Check if available. 

try: 
    session = boto3.Session( region_name=aws_region )
    s3 = session.client( "s3" )
    ret = s3.list_objects_v2( Bucket=bucket )

except: 
    s3 = None

#  The fixed output heights above the surface [in meters]. 

olevels = np.arange( 100.0, 3001.0, 100.0 )


def compute_isohypses( yearmonth:str, client, dataroot:str=default_dataroot, clobber:bool=False ): 
    """Generate the monthly average diurnal cycle of horizontal winds as generated 
    by the ERA5 reanalysis on the ERA5 Lambert Conformal grid and on fixed 
    height-above-the surface levels. The output is written to the S3 bucket 'bucket'.

    Arguments
    =========

    yearmonth       A string of the form "YYYY-MM" indicating the year and the 
                    month for which to compute the monthly average diurnal cycle.

    client          An instance of cdsapi.Client used to retrieve data from
                    the Copernicus Climate Data Store.

    dataroot        If AWS S3 is not available, this is the root of all LLJ research
                    data.

    clobber         Set to true to clobber previously existing output files. 
    """

    t0 = time()
    ret = RetClass()

    #  Processing in cloud?

    if s3:
        ret.update( comments='Processing in AWS S3.' )
    else:
        ret.update( comments='Processing in local file system.' )

    #  Set paths, etc.

    year, month = int(yearmonth[0:4]), int(yearmonth[5:7])
    outputfile = f'isohypses.{year:04d}{month:02d}.nc'

    if s3: 
        outputpath = os.path.join( "Data", output_subdir, outputfile )
        r = s3.list_objects_v2( Bucket=bucket, Prefix=outputpath )
        if r['KeyCount'] == 1: 
            if clobber: 
                ret.update( comments=f's3://{bucket}/{outputpath} already exists. Clobbering.' )
            else: 
                ret.update( comments=f's3://{bucket}/{outputpath} already exists. Exiting.' )
                return ret
    else: 
        outputpath = os.path.join( dataroot, output_subdir, outputfile )
        if os.path.exists( outputpath ): 
            if clobber: 
                print( f'{outputpath} already exists. Clobbering.' )
            else: 
                print( f'{outputpath} already exists. Exiting.' )
                return

    #  Retrieve surface geopotential file.

    lpath = f'geop.sfc.nc'

    if s3: 
        rpath = f'Data/{downloads_subdir}/time_invariant/{lpath}'
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
            if r['KeyCount'] == 1:
                print( f'Downloading s3://{bucket}/{rpath}' )
                s3.download_file( bucket, rpath, lpath )
        phis_path = lpath
    else: 
        phis_path = os.path.join( dataroot, downloads_subdir, "time_invariant", lpath )

    #  Retrieve geopotential file.

    lpath = f'geop.{year:04d}{month:02d}.nc'

    if s3: 
        rpath = f'Data/{downloads_subdir}/modellevels/{lpath}'
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
            if r['KeyCount'] == 0:
                resp = ERA5file( "geop", client, datetime(year,month,1) )
            print( f'Downloading s3://{bucket}/{rpath}' )
            s3.download_file( bucket, rpath, lpath )
        geop_path = lpath
    else: 
        geop_path = os.path.join( dataroot, downloads_subdir, "modellevels", lpath )
        
    #  Retrieve u file.

    lpath = f'uwnd.{year:04d}{month:02d}.nc'

    if s3: 
        rpath = f'Data/{downloads_subdir}/modellevels/{lpath}'
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
            if r['KeyCount'] == 0:
                resp = ERA5file( "uwnd", client, datetime(year,month,1) )
            print( f'Downloading s3://{bucket}/{rpath}' )
            s3.download_file( bucket, rpath, lpath )
        uwnd_path = lpath
    else: 
        uwnd_path = f'{dataroot}/ERA5/downloads/modellevels/{lpath}'

    #  Retrieve v file.

    lpath = f'vwnd.{year:04d}{month:02d}.nc'

    if s3: 
        rpath = f'Data/ERA5/downloads/modellevels/{lpath}'
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
            if r['KeyCount'] == 0:
                resp = ERA5file( "vwnd", client, datetime(year,month,1) )
            print( f'Downloading s3://{bucket}/{rpath}' )
            s3.download_file( bucket, rpath, lpath )
        vwnd_path = lpath
    else: 
        vwnd_path = f'{dataroot}/ERA5/downloads/modellevels/{lpath}'

    #  Read in surface geopotential. Collect data on grid.

    d = Dataset( phis_path, 'r' )
    phis = d.variables['z'][:].squeeze()
    d.close()

    #  Open height and wind files.

    d_geop = Dataset( geop_path, 'r' )
    geop = d_geop.variables['z']
    d_uwnd = Dataset( uwnd_path, 'r' )
    uwnd = d_uwnd.variables['u']
    d_vwnd = Dataset( vwnd_path, 'r' )
    vwnd = d_vwnd.variables['v']

    #  Get dimensions, times, coordinate data and metadata.

    v = d_geop.variables['valid_time']
    m = re.search( r'^([a-z]+) since (\d{4}-\d{2}-\d{2})$', v.getncattr("units") )
    time_units, epoch = m.group(1), datetime.fromisoformat( m.group(2) )

    ntimes = d_geop.dimensions['valid_time'].size
    nhours = int( 24.0 / cycle_time )
    # nlevels = d_geop.dimensions['model_level'].size
    nlons = d_geop.dimensions['longitude'].size
    nlats = d_geop.dimensions['latitude'].size

    grid = {}
    for var in [ 'longitude', 'latitude' ]: 
        v = d_geop.variables[var]
        grid[var] = {
            'dims': v.dimensions,
            'dtype': v.dtype,
            'atts': { att: v.getncattr(att) for att in v.ncattrs() },
            'vals': v[:]
        }

    #  Create output file.

    print( f'Creating {outputfile}' )
    sys.stdout.flush()

    d_out = Dataset( outputfile, 'w', format="NETCDF4" )

    d_out.createDimension( 'longitude', nlons )
    d_out.createDimension( 'latitude', nlats )
    d_out.createDimension( 'level', olevels.size )
    d_out.createDimension( 'hour', nhours )

    #  Create new variables.

    for var, v in grid.items():
        vout = d_out.createVariable( var, v['dtype'], dimensions=v['dims'] )
        vout.setncatts( v['atts'] )
        if var not in [ 'time', 'Lambert_Conformal' ]:
            vout[:] = v['vals'][:]

    vout = d_out.createVariable( 'level', np.float32, ('level',) )
    vout.setncatts( { 'long_name': "Height above the surface",
                    'units': "m" } )

    vout = d_out.createVariable( 'hour', np.float32, ('hour',) )
    vout.setncatts( { 'long_name': "Zulu hour of the day", 
                    'units': "hours" } )

    vout = d_out.createVariable( 'uwnd', np.float32, ('hour','level','latitude','longitude') )
    vout.setncatts( { 'long_name': "Zonal component of wind",
                    'units': "m/s" } )

    vout = d_out.createVariable( 'vwnd', np.float32, ('hour','level','latitude','longitude') )
    vout.setncatts( { 'long_name': "Meridional component of wind",
                    'units': "m/s" } )

    vout = d_out.createVariable( 'year', np.int32 )
    vout.setncatts( { 'long_name': "Year of the monthly average of the diurnal cycle in horizontal winds", 
                    'units': "none" } )

    vout = d_out.createVariable( 'month', np.int32 )
    vout.setncatts( { 'long_name': "Month of the monthly average of the diurnal cycle in horizontal winds", 
                    'units': "none", 
                    'valid_range': np.array( [1,12], dtype=np.int32 ) } )

    #  Global attributes. 

    d_out.setncatts( { 
                    'file_type': "era5_isohypsic_diurnal_wind_climatology", 
                    'description': "The diurnal cycle of horizontal winds from ERA5 " + \
                            "at discrete heights above the surface is averaged over one month", 
                    'author': "Stephen Leroy (stephen.leroy@janusresearch.us)" 
                    } )

    #  Write levels to output.

    d_out.variables['level'][:] = olevels
    d_out.variables['hour'][:] = np.arange( 0, 24, 3 )
    d_out.variables['year'][:] = year
    d_out.variables['month'][:] = month

    #  Loop over time.

    u = np.ma.zeros( (nhours,olevels.size,nlats,nlons), np.float32 ) 
    v = np.ma.zeros( (nhours,olevels.size,nlats,nlons), np.float32 ) 

    print( 'Computing isohypsic analysis' )

    for itime in range(ntimes): 
        ihour = ( itime % nhours )

        dt = epoch + timedelta( **{ time_units: int(d_geop.variables['valid_time'][itime]) } )
        print( '  Time ' + dt.strftime( "%Y-%m-%d %H:%M" ) )
        sys.stdout.flush()

        #  Get height profiles, subtract surface.

        print( '    Reading input fields' )
        sys.stdout.flush()

        h1 = ( geop[itime,:,:,:] - phis ) / gravity 
        uwnd1 = uwnd[itime,:,:,:]
        vwnd1 = vwnd[itime,:,:,:]

        #  Interpolate onto new levels.

        for iolevel in tqdm( range(olevels.size), desc="    Computing iolevel" ): 
            olevel = olevels[iolevel]
            ii = np.argmin( ( h1[1:,:,:] - olevel ) * ( h1[:-1,:,:] - olevel ), axis=0 )
            for ilat in range(nlats): 
                for ilon in range(nlons): 
                    i = ii[ilat,ilon]
                    t = ( olevel - h1[i,ilat,ilon] ) / ( h1[i+1,ilat,ilon] - h1[i,ilat,ilon] )
                    u[ihour,iolevel,ilat,ilon] += uwnd1[i,ilat,ilon] * (1-t) + uwnd1[i+1,ilat,ilon] * t
                    v[ihour,iolevel,ilat,ilon] += vwnd1[i,ilat,ilon] * (1-t) + vwnd1[i+1,ilat,ilon] * t

    #  Average over month. 

    ndays = int( ntimes / nhours )
    u /= ndays
    v /= ndays

    #  Write to output.

    d_out.variables['uwnd'][:,:,:,:] = u
    d_out.variables['vwnd'][:,:,:,:] = v

    #  Done with computations.

    d_out.setncatts( { 'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ) } )
    d_out.close()

    if s3: 
        print( f'Uploading to s3://{bucket}/{outputpath}' )
        sys.stdout.flush()
        s3.upload_file( outputfile, bucket, outputpath )
    else: 
        print( f'Moving to {outputpath}' )
        sys.stdout.flush()
        os.makedirs( os.path.dirname( outputpath ), exist_ok=True )
        shutil.copy( outputfile, outputpath )
        os.unlink( outputfile )

    #  Timing. 

    t1 = time()
    dt = t1 - t0
    minutes = int( dt / 60 )
    seconds = int( dt - minutes*60 )

    print( f'Elapsed time = {minutes:d} mins, {seconds:02d} secs' )
    sys.stdout.flush()

    return


def main(): 

    parser = argparse.ArgumentParser( prog="compute_era5_isohypses", 
            description="Generate a monthly average diurnal cycle of ERA5 horizontal " + \
                    "winds on fixed heights above the surface" )

    parser.add_argument( "yearmonth", type=str, help='Year-month of ERA5 output to process, format "YYYY-MM".' )

    parser.add_argument( "-k", "--key", dest="key", default="", 
            help="""The key for the Copernicus Data Store account; can be found in ~/.cdsapirc file; 
            the default is to simply use the .cdsapirc file""" )

    if not s3: 
        parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
                default=default_dataroot,
                help="""The root directory where the LLJ ERA5 files are stored and where
                    results will be written. """ + f'The default is {default_dataroot}, and ' + \
                    f'output will be written to the subdirectory {output_subdir}.' )

    parser.add_argument( "--clobber", "-c", default=False, action="store_true", 
            help='Clobber a pre-existing output file; the default is not to clobber' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Instantiate access to Copernicus Climate Data Store. 

    client = initCDS( args.key )

    #  Run computations. 

    if s3: 
        kwargs = { 'clobber': args.clobber }
    else: 
        kwargs = { 'dataroot': args.dataroot, 'clobber': args.clobber }

    ret = compute_isohypses( args.yearmonth, client, **kwargs )
    print( ret )

    return


if __name__ == "__main__": 
    main()

