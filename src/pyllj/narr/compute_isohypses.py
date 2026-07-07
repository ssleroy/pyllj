import os
import sys
import shutil 
import argparse
from datetime import datetime, timezone
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
import boto3
from time import time
from .libnarr import cycle_time, RetClass, NARRfile, downloads_subdir
from ..parameters import bucket, aws_region, default_dataroot


output_subdir = "NARR/isohypses"

#  AWS communication. Check if available.

try:
    session = boto3.Session( region_name=aws_region )
    s3 = session.client( "s3" )
    ret = s3.list_objects_v2( Bucket=bucket )

except:
    s3 = None

#  The fixed output heights above the surface [in meters]. 

olevels = np.arange( 100.0, 3001.0, 100.0 )


def compute_isohypses( yearmonth:str, dataroot:str=default_dataroot, clobber:bool=False ): 
    """Generate the monthly average diurnal cycle of horizontal winds as generated 
    by the North American Regional Reanalysis (NARR) on the NARR Lambert Conformal 
    grid and on fixed height-above-the surface levels. The output is written to the 
    S3 bucket 'bucket'.

    Arguments
    =========

    yearmonth       A string of the form "YYYY-MM" indicating the year and the
                    month for which to compute the monthly average diurnal cycle.

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory

    clobber         Set to true to clobber previously existing output files."""

    t0 = time()
    ret = RetClass()

    #  Processing in cloud? 

    if s3: 
        ret.update( comments='Processing in AWS S3.' )
    else: 
        ret.update( comments='Processing in local file system.' )

    #  Various parameters and settings. 

    dt = datetime.fromisoformat( yearmonth + "-01" )
    outputfile = f'isohypses.{dt.year:04d}{dt.month:02d}.nc'

    #  Check for pre-existing output.

    if s3: 
        outputpath = os.path.join( "Data", output_subdir, outputfile )
        r = s3.list_objects_v2( Bucket=bucket, Prefix=outputpath )
        if r['KeyCount'] == 1: 
            if clobber: 
                print( f's3://{bucket}/{outputpath} already exists. Clobbering.' )
            else: 
                print( f's3://{bucket}/{outputpath} already exists. Exiting.' )
                return

    else: 
        outputpath = os.path.join( dataroot, output_subdir, outputfile )
        if os.path.exists( outputpath ): 
            if clobber: 
                ret.update( comments=f'{outputpath} already exists. Clobbering.' )
            else: 
                ret.update( comments=f'{outputpath} already exists. Exiting.' )
                return ret

    #  Retrieve surface geopotential file.

    lpath = f'hgt.sfc.nc'

    if s3: 
        rpath = f'Data/{downloads_subdir}/time_invariant/lpath' 
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
        if r['KeyCount'] == 1:
            print( f'Downloading s3://{bucket}/{rpath}' )
            s3.download_file( bucket, rpath, lpath )
        phis_path = lpath

    else: 
        phis_path = os.path.join( dataroot, downloads_subdir, "time_invariant", lpath )

    #  Retrieve geopotential file.

    lpath = f'hgt.{year:04d}{month:02d}.nc'

    if s3: 
        rpath = f'Data/{downloads_subdir}/pressure/{lpath}' 
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
            if r['KeyCount'] == 0:
                resp = NARRfile( "hgt", datetime(year,month,1) )
            print( f'Downloading s3://{bucket}/{rpath}' )
            s3.download_file( bucket, rpath, lpath )
        hgt_path = lpath
    else: 
        hgt_path = os.path.join( dataroot, downloads_subdir, "pressure", lpath )

    #  Retrieve u file.

    lpath = f'uwnd.{year:04d}{month:02d}.nc'

    if s3: 
        rpath = f'Data/{downloads_subdir}/pressure/{lpath}'
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
            if r['KeyCount'] == 0:
                resp = NARRfile( "uwnd", datetime(year,month,1) )
            print( f'Downloading s3://{bucket}/{rpath}' )
            s3.download_file( bucket, rpath, lpath )
        uwnd_path = lpath
    else: 
        uwnd_path = os.path.join( dataroot, downloads_subdir, "pressure", lpath )

    #  Retrieve v file.

    lpath = f'vwnd.{year:04d}{month:02d}.nc'

    if s3: 
        rpath = f'Data/{downloads_subdir}/pressure/{lpath}'
        if not os.path.exists( lpath ):
            r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
            if r['KeyCount'] == 0:
                resp = NARRfile( "vwnd", datetime(year,month,1) )
            print( f'Downloading s3://{bucket}/{rpath}' )
            s3.download_file( bucket, rpath, lpath )
        vwnd_path = lpath
    else: 
        vwnd_path = os.path.join( dataroot, downloads_subdir, "pressure", lpath )

    #  Read in surface geopotential. Collect data on grid.

    d = Dataset( phis_path, 'r' )
    phis = d.variables['hgt'][:].squeeze()
    d.close()

    #  Open height and wind files.

    d_hgt = Dataset( hgt_path, 'r' )
    hgt = d_hgt.variables['hgt']
    d_uwnd = Dataset( uwnd_path, 'r' )
    uwnd = d_uwnd.variables['uwnd']
    d_vwnd = Dataset( vwnd_path, 'r' )
    vwnd = d_vwnd.variables['vwnd']

    #  Get dimensions, times, coordinate data and metadata.

    ntimes = d_hgt.dimensions['time'].size
    nhours = int( 24.0 / cycle_time )
    nlevels = d_hgt.dimensions['level'].size
    nx = d_hgt.dimensions['x'].size
    ny = d_hgt.dimensions['y'].size

    grid = {}
    for var in [ 'x', 'y', 'lon', 'lat', 'Lambert_Conformal' ]:
        v = d_hgt.variables[var]
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

    d_out.createDimension( 'x', nx )
    d_out.createDimension( 'y', ny )
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

    vout = d_out.createVariable( 'uwnd', np.float32, ('hour','level','y','x') )
    vout.setncatts( { 'long_name': "Zonal component of wind",
                    'units': "m/s" } )

    vout = d_out.createVariable( 'vwnd', np.float32, ('hour','level','y','x') )
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
                    'file_type': "narr_isohypsic_diurnal_wind_climatology", 
                    'description': "The diurnal cycle of horizontal winds from NARR " + \
                            "at discrete heights above the surface is averaged over one month", 
                    'author': "Stephen Leroy (stephen.leroy@janusresearch.us)" 
                    } )

    #  Write levels to output.

    d_out.variables['level'][:] = olevels
    d_out.variables['hour'][:] = np.arange( 0, 24, 3 )
    d_out.variables['year'][:] = year
    d_out.variables['month'][:] = month

    #  Loop over time.

    u = np.ma.zeros( (nhours,olevels.size,ny,nx), np.float32 ) 
    v = np.ma.zeros( (nhours,olevels.size,ny,nx), np.float32 ) 

    for itime in range(ntimes): 
        ihour = ( itime % nhours )

        print( f'itime = {itime}/{ntimes}' )
        sys.stdout.flush()

        #  Get height profiles, subtract surface.

        h1 = hgt[itime,:,:,:] - phis  
        uwnd1 = uwnd[itime,:,:,:]
        vwnd1 = vwnd[itime,:,:,:]

        #  Interpolate onto new levels.

        for iolevel in tqdm( range(olevels.size), desc="iolevel" ): 
            olevel = olevels[iolevel]
            ii = np.argmin( ( h1[1:,:,:] - olevel ) * ( h1[:-1,:,:] - olevel ), axis=0 )
            for iy in range(ny): 
                for ix in range(nx): 
                    i = ii[iy,ix]
                    t = ( olevel - h1[i,iy,ix] ) / ( h1[i+1,iy,ix] - h1[i,iy,ix] )
                    u[ihour,iolevel,iy,ix] += uwnd1[i,iy,ix] * (1-t) + uwnd1[i+1,iy,ix] * t
                    v[ihour,iolevel,iy,ix] += vwnd1[i,iy,ix] * (1-t) + vwnd1[i+1,iy,ix] * t

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

    ret.update( success=True )

    return ret 


def main(): 

    parser = argparse.ArgumentParser( prog="compute_narr_isohypses", 
            description="Generate a monthly average diurnal cycle of NARR horizontal " + \
                    "winds on fixed heights above the surface" )

    parser.add_argument( "yearmonth", type=str, help='Year-month of NARR output to process, format "YYYY-MM".' )

    if not s3: 
        parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
                default=default_dataroot,
                help="""The root directory where the LLJ NARR files are stored and where
                    results will be written. """ + f'The default is {default_dataroot}, and ' + \
                    f'output will be written to the subdirectory {output_subdir}.' )

    parser.add_argument( "--clobber", "-c", default=False, action="store_true", 
            help='Clobber a pre-existing output file; the default is not to clobber' )

    args = parser.parse_args()

    if s3:
        kwargs = { 'clobber': args.clobber }
    else:
        kwargs = { 'dataroot': args.dataroot, 'clobber': args.clobber }

    ret = compute_isohypses( args.yearmonth, client, **kwargs )
    print( ret )

    return


if __name__ == "__main__": 
    main()

