import os
import re
import sys
from datetime import datetime, timezone
from time import time
import argparse
from netCDF4 import Dataset
import numpy as np
from .libmerra2 import RetClass, default_dataroot


output_subdir = "MERRA2/isohypses"


def compute_isohypses_climatology( yearrange, dataroot:str=default_dataroot ): 
    """Compute a climatology of isohypsic analysis over range of years previously 
    generated compute_merra2_isohypses. 

    Arguments
    =========

    yearrange       A 2-tuple or 2-element list of integers prescribing the range 
                    of years over which to compute the climatology

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory

    The output is written into the S3 bucket with a filename defined in part by 
    the range of years."""

    time0 = time()

    ret = RetClass()

    #  Define the output path. 

    outputpath = os.path.join( dataroot, output_subdir, 'isohypses.nc' )

    #  Create output file. 

    print( f'Creating {outputpath}' )
    sys.stdout.flush()
    os.makedirs( os.path.dirname( outputpath ), exist_ok=True )

    o = Dataset( outputpath, 'w', format="NETCDF4" )

    #  Scan over years. 

    for year in range( yearrange[0], yearrange[1]+1 ): 
        for imonth in range(12): 
            month = imonth + 1

            #  Remote path for input file. 

            found, comments = False, []

            for base in [ "isohypses", "merra2_isohypses" ]
                inputpath = os.path.join( dataroot, output_subdir, f'isohypses.{year:4d}{month:02d}.nc' )
                if  os.path.exists( inputpath ): 
                    found = True
                    break 
                else: 
                    comments.append( f'{inputpath} does not exist' )

            if not found: 
                ret.update( comments=comments )
                continue

            #  Open input data file. 

            try: 
                d = Dataset( inputpath, 'r' )
            except: 
                ret.update( comments=f'{inputpath} is unreadable' )
                continue

            print( f'Processing {inputpath}' )
            sys.stdout.flush()

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

    #  Normalize. 

    for imonth in range(12): 
        uwnd[imonth,:,:,:] /= count[imonth]
        vwnd[imonth,:,:,:] /= count[imonth]

    print( f'Normalizing' )
    sys.stdout.flush()

    #  Write to output. 

    o.variables['uwnd'][:] = uwnd
    o.variables['vwnd'][:] = vwnd

    #  Global attributes. 

    o.setncatts( { 
        'file_type': "merra2_isohypsic_analysis_climatology", 
        'yearrange': [ np.int32(yearrange[0]), np.int32(yearrange[1]) ], 
        'author': "Stephen Leroy (stephen.leroy@janusresearch.us)", 
        'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ) } )

    #  Done. 

    o.close()

    print( f'Created {outputpath}' )

    #  Elapsed time. 

    time1 = time()

    dt = int( time1 - time0 )
    hours = int( dt / 3600 )
    minutes = int( dt / 60 ) % 60
    seconds = dt % 60

    ret.update( success=True, comments=f'Elapsed time = {hours} hrs, {minutes:2d} mins, {seconds:2d} secs' )
    return ret 


def main(): 

    default_diagnostics_dir = os.path.join( default_dataroot, output_subdir )

    parser = argparse.ArgumentParser( description="Generate an annual cycle, diurnal " + \
            "cycle climatology of horizontal winds at fixed heights above the surface " + \
            "based on the output of compute_merra2_isohypses" )

    parser.add_argument( 'yearrange', type=str, help='The range of years over which to ' + \
            'compute the climatology, format "YYYY:YYYY"' )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
            default=default_dataroot,
            help="""The root directory where the LLJ files are stored and where
                results will be written. """ + f'The default is {default_dataroot}, and ' + \
                f'output will be written to the subdirectory {default_diagnostics_dir}.' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    #  Debug? 

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Parse year range. 

    m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
    if not m: 
        print( 'Be sure that yearrange has format "YYYY:YYYY".' )
        sys.stdout.flush()
        return 

    yearrange = [ int( m.group(1) ), int( m.group(2) ) ]
    ret = compute_isohypses_climatology( yearrange, dataroot=args.dataroot ) 

    print( ret )
    return 


if __name__ == "__main__": 
    main()
    pass

