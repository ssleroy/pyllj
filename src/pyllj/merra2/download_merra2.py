import os
import re
import argparse
from datetime import datetime, timedelta
from time import time
import earthaccess
from netCDF4 import Dataset
from .libmerra2 import RetClass, region, default_dataroot, truncate, netrcauth, downloads_subdir

#  Authenticate earthaccess. 

auth = earthaccess.login( persist=True )

#  Justify region longituderange.

ii = ( region['longituderange'] >= 180.0 )
region['longituderange'][ii] -= 360.0
ii = ( region['longituderange'] < -180.0 )
region['longituderange'][ii] += 360.0

#  Maximum number of files to download and process simulataneously. 

nmaxrecs = 8


def download( datetimerange:{tuple,list}, dataroot:str=default_dataroot, clobber:bool=False ): 
    """Download MERRA2 3-hourly reanalysis of temperature, humidity, winds, surface pressure.

    Arguments
    =========

    datetimerange   A 2-element tuple or list of instances of datetime.datetime that designate 
                    the day range over which to download data. 

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory
    """

    t0 = time()

    ret = RetClass()

    #  Check input. 

    if not isinstance( datetimerange, tuple ) and not isinstance( datetimerange, list ): 
        ret.update( success=False, messages="InvalidArgument", comments="libmerra2.download: datetimerange must be a tuple or list" )
        return ret

    if len( datetimerange ) != 2: 
        ret.update( success=False, messages="InvalidArgument", comments="libmerra2.download: datetimerange must have length 2" )
        return ret

    if not isinstance( datetimerange[0], datetime ) or not isinstance( datetimerange[1], datetime ): 
        ret.update( success=False, messages="InvalidArgument", comments="libmerra2.download: datetimerange must be composed of datetime.datetime instances" )
        return ret

    if datetimerange[1] < datetimerange[0]: 
        ret.update( success=False, messages="InvalidArgument", comments="libmerra2.download: datetimerange[1] must be greater than datetimerange[0]" )
        return ret

    #  Formulate temporal argument.  Search Earthdata CMR. 

    temporal = tuple( [ dt.strftime( "%Y-%m-%d" ) for dt in datetimerange ] )
    try:
        recs = earthaccess.search_data( short_name="M2I3NVASM", temporal=temporal )
    except:
        recs = []

    #  Loop over returned records. 

    downloadsroot = os.path.join( dataroot, downloads_subdir )
    tmpdir = os.path.join( downloadsroot, "tmp" )
    os.makedirs( tmpdir, exist_ok=True )

    #  Eliminate recs that have already been downloaded. precs is the subset of recs found by earthaccess
    #  that have yet to be processed because the truncated version of those files do not yet exist in S3. 

    precs = []
    for rec in recs: 
        dtime = datetime.fromisoformat( rec['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'][:-1] )
        basename = f'merra2.M2I3NVASM.{region["regionname"]}.{dtime.year:4d}{dtime.month:02d}{dtime.day:02d}.nc'
        localpath = os.path.join( downloadsroot, f'{dtime.year:4d}', f'{dtime.month:02d}', basename )
        if clobber: 
            precs.append( rec )
        elif not os.path.exists( localpath ): 
            precs.append( rec )
        else: 
            try: 
                d = Dataset( localpath, 'r' )
                d.close()
            except: 
                precs.append( rec )

    #  Segment the precs that have yet to be processed. srecs are a subset of precs, with nmaxrecs that 
    #  maximum length of precs. This is done for the sake of efficiency in calling earthaccess.download. 

    while len( precs ) > 0: 

        if len( precs ) > nmaxrecs: 
            srecs = precs[:nmaxrecs]
            precs = precs[nmaxrecs:]
        else: 
            srecs = precs[:]
            precs = []

        #  Download only those files that do not exist in the temporary storage directory. 

        drecs = []
        for rec in srecs: 
            f = os.path.join( tmpdir, rec['umm']['DataGranule']['Identifiers'][0]['Identifier'] )
            if not os.path.exists( f ): 
                drecs.append( rec )

        #  Do the download. 

        if len( drecs ) > 0: 
            files = [ rec['umm']['DataGranule']['Identifiers'][0]['Identifier'] for rec in drecs ]
            print( 'Downloading ' + ", ".join( files ) )
            earthaccess.download( drecs, local_path=tmpdir )

        #  Truncate. 

        for rec in srecs: 

            #  MERRA2 date. 

            dtime = datetime.fromisoformat( rec['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'][:-1] )
            basename = f'merra2.M2I3NVASM.{region["regionname"]}.{dtime.year:4d}{dtime.month:02d}{dtime.day:02d}.nc'
            localpath = os.path.join( downloadsroot, f'{dtime.year:4d}', f'{dtime.month:02d}', basename )
            tmpfile = os.path.join( tmpdir, rec['umm']['DataGranule']['Identifiers'][0]['Identifier'] )

            #  Check for successful download. 

            ntries = 1
            success_message = "Successful"

            while True: 

                status = success_message
                if os.path.exists( tmpfile ): 
                    try: 
                        d = Dataset( tmpfile, 'r' )
                        d.close()
                    except: 
                        status = f'Unable to read {basename}'
                else: 
                    status = f'Unable to download {basename}'

                #  Break out of indefinite loop if the download was successful (and readable). 
                #  Break out of loop also if too many (10) failures to download. 

                if status == success_message or ntries >= 10: break 

                #  Another attempt at downloading. 

                earthaccess.download( [ rec ], local_path=tmpdir )
                ntries += 1

            #  If the download was not successful, figure out why, log it, and continue loop 
            #  over records. 

            if status != success_message: 
                ret.update( comments=status )
                continue

            #  Truncate.

            outputfile = truncate( tmpfile, localpath, region['longituderange'], region['latituderange'] )
            os.unlink( tmpfile )

    t1 = time()
    dt = int( t1 - t0 )
    days, hrs, mins, secs = int(dt/86400), int(dt/3600) % 24, int(dt/60) % 60, dt % 60
    print( f'Total elapsed time = {days:d} days, {hrs:d} hrs, {mins:2d} mins, {secs:2d} secs' )

    return ret


def main(): 
    
    parser = argparse.ArgumentParser( prog="download_merra2", description="Download all meteorological fields relevant to Great Plains LLJ research. " + \
            "The MERRA2 data files will be truncated to the area of the North America." )

    parser.add_argument( "monthrange", type=str, help='The range of months over which to download MERRA2 data, format ' + \
            '"YYYY-MM:YYYY-MM" for the first and last month to download' )

    parser.add_argument( "-d", "--dataroot", dest="dataroot", default=default_dataroot, 
            help=f'The root for the project; the default is "{default_dataroot}"' )

    parser.add_argument( "--auth", "-a", dest="auth", default="",
            help='String containing the username and password for authentication to NASA Earthdata; the ' + \
            'username and password should be separated by a single space' )

    parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true", 
            help='Clobber pre-existing output files; no clobbering by default' )

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

    #  Check month range. 

    m = re.search( r'^(\d{4}-\d{2}):(\d{4}-\d{2})$', args.monthrange )

    if not m: 
        print( 'Be sure that the monthrange has format "YYYY-MM:YYYY-MM".' )
        exit()

    dt0 = datetime.fromisoformat( m.group(1)+"-01" )
    dt1 = datetime.fromisoformat( m.group(2)+"-01" ) + timedelta( days=31 )
    dt1 = datetime( year=dt1.year, month=dt1.month, day=1 ) - timedelta( days=1 )
    ret = download( (dt0,dt1), dataroot=args.dataroot, clobber=args.clobber )

    if ret.success: 
        print( 'Normal successful completion' )
        print( 'Messages = ' + ', '.join( ret.messages ) )
        print( 'Comments = \n  ' + '\n  '.join( ret.comments ) )
    
    else: 
        print( 'Unsuccessful completion' )
        print( 'Messages = ' + ', '.join( ret.messages ) )
        print( 'Comments = \n  ' + '\n  '.join( ret.comments ) )

    pass


if __name__ == "__main__": 
    main()
    pass

