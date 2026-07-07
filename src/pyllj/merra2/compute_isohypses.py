import os
import re
import sys
import shutil 
import argparse
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from netCDF4 import Dataset, MFDataset
import numpy as np
import boto3
from time import time
from .libmerra2 import RetClass, cycle_time, downloads_subdir, region, gravity, netrcauth
from ..parameters import bucket, aws_region, default_dataroot, gravity 


output_subdir = "MERRA2/isohypses"

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
    by the MERRA2 reanalysis grid and on fixed height-above-the surface levels. The 
    output is written to the S3 bucket 'bucket'.

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

    output_time_units = "hours"
    nhours = int( 24.0 / cycle_time + 0.001 )

    #  Define outputfile, outputpath, etc. 

    dt = datetime.fromisoformat( yearmonth + "-01" )
    outputfile = f'isohypses.{dt.year:04d}{dt.month:02d}.nc'

    #  Check for pre-existing output. 

    if s3: 
        outputpath = f'Data/{output_subdir}/outputfile'
        resp = s3.list_objects_v2( Bucket=bucket, Prefix=outputpath )
        if resp['KeyCount'] == 1: 
            if clobber: 
                ret.update( comments=f's3://{bucket}/{outputpath} already exists. Clobbering.' )
            else: 
                ret.update( success=False, messages="NoClobber", comments=f's3://{bucket}/{outputpath} already exists. Exiting.' )
                return ret
    else: 
        outputpath = os.path.join( dataroot, output_subdir, outputfile )
        if os.path.exists( outputpath ): 
            if clobber: 
                ret.update( comments=f'{outputpath} already exists. Clobbering.' )
            else: 
                ret.update( success=False, messages="NoClobber", comments=f'{outputpath} already exists. Exiting.' )
                return ret

    #  Collect input files for the month. 

    if s3: 
        prefix = f'Data/{downloads_subdir}/{year:4d}/{month:02d}' 
        resp = s3.list_objects_v2( Bucket=bucket, Prefix=prefix )

        if resp['KeyCount'] < 21: 
            ret.update( success=False, messages="InsufficientData", \
                    comments=f"Insufficient number of input files in s3://{bucket}/{prefix}, only {resp['KeyCount']} found" )
            return ret

        inputfiles = sorted( [ Item['Key'] for Item in resp['Contents'] \
                if re.search( r'merra2\..*\.nc$', Item['Key'] ) and re.search( region['regionname'], Item['Key'] ) ] )
        ret.update( success=True, comments=f's3://{bucket}/{prefix} contains {len(inputfiles)} files' )

    else: 
        prefix = os.path.join( dataroot, downloads_subdir, f'{year:4d}', f'{month:02d}' )
        r = [ f for f in os.listdir( prefix ) if re.search( r'\w+', f ) ]
        if len( r ) < 21: 
            ret.update( success=False, messages="InsufficientData", \
                    comments=f"Insufficient number of input files in {prefix}, only {len(r)} found" )
            return ret

        inputfiles = sorted( [ os.path.join( prefix, f ) for f in r \
                if re.search( r'merra2\..*\.nc$', f ) and re.search( region['regionname'], f ) ] )
        ret.update( success=True, comments=f'{prefix} contains {len(inputfiles)} files' )

    for iinputfile, inputfile in enumerate(inputfiles): 

        #  Download and open inputfile. 

        if s3: 
            local_inputfile = os.path.basename( inputfile )
            print( f'Downloading {inputfile}' )
            sys.stdout.flush()
            s3.download_file( bucket, inputfile, local_inputfile )
        else: 
            local_inputfile = inputfile

        ds = Dataset( local_inputfile, 'r' )

        #  Get time epoch and units.

        ts = ds.variables['time'][:]
        m = re.search( r'^([a-z]+) since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$', ds.variables['time'].units )
        input_time_units = m.group(1)
        input_epoch = datetime.strptime( m.group(2), "%Y-%m-%d %H:%M:%S" )

        #  Get dimensions, times, coordinate data and metadata.

        if iinputfile == 0: 
            lons = ds.variables['lon'][:]
            lats = ds.variables['lat'][:]
            levs = ds.variables['lev'][:]
            nlons, nlats, nlevs = lons.size, lats.size, levs.size
            ntimes = ts.size * nhours

            #  Define grid. 

            grid = {}
            for var in [ 'lon', 'lat' ]: 
                v = ds.variables[var]
                grid[var] = {
                    'dims': v.dimensions,
                    'dtype': v.dtype,
                    'atts': { att: v.getncattr(att) for att in v.ncattrs() },
                    'vals': v[:]
                }

            #  Create output file.

            print( f'Creating {outputfile}' )
            sys.stdout.flush()

            d_out = Dataset( outputfile, 'w', format="NETCDF3_CLASSIC" )

            d_out.createDimension( 'longitude', nlons )
            d_out.createDimension( 'latitude', nlats )
            d_out.createDimension( 'level', olevels.size )
            d_out.createDimension( 'hour', nhours )

            #  Create new variables.

            v = grid['lon']
            vout = d_out.createVariable( "longitude", v['dtype'], dimensions=("longitude",) )
            vout.setncatts( v['atts'] )
            vout[:] = v['vals'][:]

            v = grid['lat']
            vout = d_out.createVariable( "latitude", v['dtype'], dimensions=("latitude",) )
            vout.setncatts( v['atts'] )
            vout[:] = v['vals'][:]

            vout = d_out.createVariable( 'level', np.float32, ('level',) )
            vout.setncatts( { 'long_name': "Height above the surface", 'units': "m" } )

            vout = d_out.createVariable( 'hour', np.float32, ('hour',) )
            vout.setncatts( { 'long_name': "Zulu hour of the day", 'units': "hours" } )

            vout = d_out.createVariable( 'uwnd', np.float32, ('hour','level','latitude','longitude') )
            vout.setncatts( { 'long_name': "Zonal component of wind", 'units': "m/s" } )

            vout = d_out.createVariable( 'vwnd', np.float32, ('hour','level','latitude','longitude') )
            vout.setncatts( { 'long_name': "Meridional component of wind", 'units': "m/s" } )

            vout = d_out.createVariable( 'year', np.int32 )
            vout.setncatts( { 'long_name': "Year of the monthly average of the diurnal cycle in horizontal winds", 
                    'units': "none" } )

            vout = d_out.createVariable( 'month', np.int32 )
            vout.setncatts( { 'long_name': "Month of the monthly average of the diurnal cycle in horizontal winds", 
                    'units': "none", 
                    'valid_range': np.array( [1,12], dtype=np.int32 ) } )

            #  Global attributes. 

            d_out.setncatts( { 
                    'file_type': "merra2_isohypsic_diurnal_wind_climatology", 
                    'description': "The diurnal cycle of horizontal winds from MERRA2 " + \
                            "at discrete heights above the surface is averaged over one month", 
                    'author': "Stephen Leroy (stephen.leroy@janusresearch.us)" 
                    } )

            #  Write levels to output.

            d_out.variables['level'][:] = olevels
            d_out.variables['hour'][:] = np.arange( 0, 24, cycle_time )
            d_out.variables['year'][:] = year
            d_out.variables['month'][:] = month

            #  Initialize u, v profiles. 

            u = np.ma.zeros( (nhours,olevels.size,nlats,nlons), np.float32 ) 
            v = np.ma.zeros( (nhours,olevels.size,nlats,nlons), np.float32 ) 

        #  Loop over time.

        hour_iterator = range(nhours)
        # hour_iterator = tqdm( range(nhours), desc="ihour" )

        for ihour in hour_iterator: 

            t = input_epoch + timedelta( **{input_time_units: int(ts[ihour])} )

            #  Get height profiles, subtract surface.

            h1 = ds.variables['H'][ihour,:,:,:] - ds.variables['PHIS'][ihour,:,:] / gravity
            uwnd1 = ds.variables['U'][ihour,:,:,:]
            vwnd1 = ds.variables['V'][ihour,:,:,:]

            #  Interpolate onto new levels.

            level_iterator = tqdm( range(olevels.size), desc='  Processing ' + t.strftime( "%Y-%m-%d %H:%M" ) )
            # level_iterator = range(olevels.size)

            for iolevel in level_iterator: 
                olevel = olevels[iolevel]
                ii = np.argmin( ( h1[1:,:,:] - olevel ) * ( h1[:-1,:,:] - olevel ), axis=0 )
                for ilat in range(nlats): 
                    for ilon in range(nlons): 
                        i = ii[ilat,ilon]
                        t = ( olevel - h1[i,ilat,ilon] ) / ( h1[i+1,ilat,ilon] - h1[i,ilat,ilon] )
                        u[ihour,iolevel,ilat,ilon] += uwnd1[i,ilat,ilon] * (1-t) + uwnd1[i+1,ilat,ilon] * t
                        v[ihour,iolevel,ilat,ilon] += vwnd1[i,ilat,ilon] * (1-t) + vwnd1[i+1,ilat,ilon] * t

        #  Done with input file. 

        ds.close()
        if s3: 
            os.unlink( local_inputfile )

    #  Average over month. 

    ndays = len( inputfiles )
    u /= ndays
    v /= ndays

    #  Write to output.

    d_out.variables['uwnd'][:,:,:,:] = u
    d_out.variables['vwnd'][:,:,:,:] = v

    #  Done with computations.

    d_out.setncatts( { 'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ) } )
    d_out.close()

    #  Upload to S3. 

    if s3: 
        print( f'Uploading to s3://{bucket}/{outputpath}' )
        sys.stdout.flush()
        s3.upload_file( outputfile, bucket, outputpath )
        os.unlink( outputfile )

    else: 
        print( f'Moving to {outputpath}' )
        sys.stdout.flush()
        os.makedirs( os.path.dirname( outputpath ), exist_ok=True )
        shutil.copy( outputfile, outputpath )
        os.unlink( outputfile )

    #  Timing. 

    t1 = time()
    dt = t1 - t0
    hours, minutes, seconds = int( dt / 3600 ), int( dt / 60 ) % 60, int( dt ) % 60

    print( f'Elapsed time = {hours:d} hrs, {minutes:2d} mins, {seconds:2d} secs' )
    sys.stdout.flush()

    return ret


def main(): 

    parser = argparse.ArgumentParser( prog="compute_merra2_isohypses", 
            description="Generate a monthly average diurnal cycle of MERRA2 horizontal " + \
                    "winds on fixed heights above the surface" )

    parser.add_argument( "yearmonth", type=str, help='Year-month of MERRA2 output to process, format "YYYY-MM".' )

    if not s3: 
        parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
                default=default_dataroot,
                help="""The root directory where the LLJ MERRA2 files are stored and where
                    results will be written. """ + f'The default is {default_dataroot}, and ' + \
                    f'output will be written to the subdirectory {output_subdir}.' )

    parser.add_argument( "--auth", "-a", dest="auth", default="",
            help='String containing the username and password for authentication to NASA Earthdata; the
            username and password should be separated by a single space' )

    parser.add_argument( "--clobber", "-c", default=False, action="store_true", 
            help='Clobber a pre-existing output file; the default is not to clobber' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    #  Debug? 

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Authentication.

    if args.auth != "":
        netrcauth( args.auth )

    if s3: 
        kwargs = { 'clobber': args.clobber }
    else: 
        kwargs = { 'dataroot': args.dataroot, 'clobber': args.clobber }

    ret = compute_isohypses( args.yearmonth, **kwargs )
    print( ret )

    pass


if __name__ == "__main__": 
    main()

