import os
import re
import sys
import shutil
from datetime import datetime, timezone
from time import time
import argparse
from netCDF4 import Dataset
import numpy as np
import boto3
from .libera5 import RetClass 
from ..parameters import aws_region, bucket, default_dataroot


#  Establish root of data. If S3 is authenticated, use it. If not use the 
#  argument dataroot. 

try: 
    session = boto3.Session( region_name=aws_region )
    s3 = session.client( "s3" )
    ret = s3.list_objects_v2( Bucket=bucket, Prefix="Data" )

except: 
    s3 = None

output_subdir = "ERA5/isohypses"


def compute_isohypses_climatology( yearrange:str, dataroot:str=default_dataroot ): 
    """Compute a climatology of isohypsic analysis over range of years previously 
    generated compute_era5_isohypses. 

    yearrange       A 2-tuple or 2-element list of integers prescribing the range 
                    of years over which to compute the climatology

    dataroot        If AWS S3 is not available, this is the root of all LLJ research 
                    data. 

    The output is written into the S3 bucket with a filename defined in part by 
    the range of years."""

    time0 = time()
    ret = RetClass()

    #  Processing in cloud?

    if s3:
        ret.update( comments='Processing in AWS S3.' )
    else:
        ret.update( comments='Processing in local file system.' )

    #  Define the output path. 

    outputfile = 'isohypses.nc' 
    if s3: 
        outputpath = f'Data/{output_subdir}/{outputfile}'
    else: 
        outputpath = f'{dataroot}/{output_subdir}/{outputfile}'

    #  Create output file. 

    print( f'Creating {outputfile}' )
    sys.stdout.flush()

    o = Dataset( outputfile, 'w', format="NETCDF4" )

    #  Scan over years. 

    for year in range( yearrange[0], yearrange[1]+1 ): 
        for imonth in range(12): 
            month = imonth + 1

            #  Remote path for input file. 

            if s3: 
                found, comments = True, []
                for base in [ "isohypses", "era5_isohypses" ]: 
                    rpath = f'Data/{output_subdir}/{base}.{year:4d}{month:02d}.nc'
                    lpath = os.path.basename( rpath )
                    r = s3.list_objects_v2( Bucket=bucket, Prefix=rpath )
                    if r['KeyCount'] == 1: 
                        found = True
                        break 
                    else: 
                        comments.append( f'File not found: s3://{bucket}/{rpath}' )
                ret.update( success=False, messages='UnavailableFile', comments=comments )
                return ret

            else: 
                found, comments = True, []
                for base in [ "isohypses", "era5_isohypses" ]: 
                    rpath = f'{dataroot}/{output_subdir}/{base}.{year:4d}{month:02d}.nc'
                    lpath = os.path.basename( rpath )
                    if os.path.exists( rpath ): 
                        found = True
                        break
                    else: 
                        comments.append( f'File not found: {rpath}' )
                if not found: 
                    ret.update( success=False, messages='UnavailableFile', comments=comments )
                    return ret

            if s3: 
                print( f'Downloading s3://{bucket}/{rpath}' )
                sys.stdout.flush()
                s3.download_file( bucket, rpath, lpath )
            else: 
                lpath = rpath

            #  Open input data file. 

            print( f'Processing {lpath}' )
            d = Dataset( lpath, 'r' )

            #  Initialize. 

            if year == yearrange[0] and imonth == 0: 

                #  Dimensions. 

                dimensions = [ 'longitude', 'latitude', 'level', 'hour' ]
                for dim in dimensions: 
                    o.createDimension( dim, d.dimensions[dim].size )
                o.createDimension( 'month', 12 )

                #  Metadata variables. 

                metadatavars = [ 'longitude', 'latitude', 'level', 'month', 'hour' ]
                for mvar in metadatavars: 
                    dv = d.variables[mvar]
                    if mvar == "month": 
                        ov = o.createVariable( mvar, dv.dtype, dimensions=("month",) )
                    else: 
                        ov = o.createVariable( mvar, dv.dtype, dimensions=dv.dimensions )
                    atts = { att: dv.getncattr( att ) for att in dv.ncattrs() }
                    ov.setncatts( atts )

                #  Metadata values. 

                for mvar in metadatavars: 
                    if mvar == "month": 
                        o.variables[mvar][:] = np.arange(1,13,dtype='i')
                    else: 
                        o.variables[mvar][:] = d.variables[mvar][:]

                #  Climatological variables. 

                nlons = d.dimensions['longitude'].size
                nlats = d.dimensions['latitude'].size
                nlevels = d.dimensions['level'].size
                nhours = d.dimensions['hour'].size
                nmonths = 12

                for var in [ 'uwnd', 'vwnd' ]: 
                    dv = d.variables[var]
                    ov = o.createVariable( var, dv.dtype, dimensions=('month','hour','level','latitude','longitude') )
                    atts = { att: dv.getncattr( att ) for att in dv.ncattrs() }
                    ov.setncatts( atts )

                #  Local variables. 

                uwnd = np.ma.zeros( (nmonths,nhours,nlevels,nlats,nlons), dtype=np.float32 )
                vwnd = np.ma.zeros( (nmonths,nhours,nlevels,nlats,nlons), dtype=np.float32 )
                count = np.zeros( 12, dtype=np.int32 )

                #  Done with initialization. 

            uwnd[imonth,:,:,:] += d.variables['uwnd'][:]
            vwnd[imonth,:,:,:] += d.variables['vwnd'][:]
            count[imonth] += 1

            d.close()

            if s3: 
                print( f'Removing {lpath}' )
                sys.stdout.flush()
                os.unlink( lpath )

    #  Normalize. 

    for imonth in range(12): 
        uwnd[imonth,:,:,:] /= count[imonth]
        vwnd[imonth,:,:,:] /= count[imonth]

    #  Write to output. 

    o.variables['uwnd'][:] = uwnd
    o.variables['vwnd'][:] = vwnd

    #  Global attributes. 

    o.setncatts( { 
        'file_type': "era5_isohypsic_analysis_climatology", 
        'year_range': [ np.int32(yearrange[0]), np.int32(yearrange[1]) ], 
        'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ), 
        'author': "Stephen Leroy (stephen.leroy@janusresearch.us" } )

    #  Done. Close and upload. 

    o.close()

    if s3: 
        print( f'Uploading to s3://{bucket}/{outputpath}' )
        sys.stdout.flush()
        s3.upload_file( outputfile, bucket, outputpath )
    else: 
        print( f'Moving to {outputpath}' )
        sys.stdout.flush()
        shutil.copy( outputfile, outputpath )
        os.unlink( outputfile )

    time1 = time()

    #  Elapsed time. 

    dt = int( time1 - time0 )
    hours = int( dt / 3600 )
    minutes = int( dt / 60 ) % 60
    seconds = dt % 60

    print( f'Elapsed time = {hours} hrs, {minutes:02d} mins, {seconds:02d} secs' )

    return ret 


def main(): 

    parser = argparse.ArgumentParser( description="Generate an annual cycle, diurnal " + \
            "cycle climatology of horizontal winds at fixed heights above the surface " + \
            "based on the output of compute_era5_isohypses" )

    parser.add_argument( 'yearrange', type=str, help='The range of years over which to ' + \
            'compute the climatology, format "YYYY:YYYY"' )

    if not s3: 
        parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
                default=default_dataroot,
                help="""The root directory where the LLJ ERA5 files are stored and where
                    results will be written. """ + f'The default is {default_dataroot}, and ' + \
                    f'output will be written to the subdirectory {output_subdir}.' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
    if not m: 
        print( 'Be sure that yearrange has format "YYYY:YYYY".' )
        sys.stdout.flush()
        return 

    yearrange = [ int( m.group(1) ), int( m.group(2) ) ]
    if s3: 
        kwargs = {}
    else: 
        kwargs = { 'dataroot': args.dataroot }

    ret = compute_isohypses_climatology( yearrange, **kwargs )   
    print( ret )

    return 


if __name__ == "__main__": 
    main()
    pass

