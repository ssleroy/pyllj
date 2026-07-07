import os
import sys
import argparse
import re
from datetime import datetime, timedelta
from time import time
from .libera5 import RetClass, default_dataroot, ERA5file, initCDS
from threading import Thread


#  Capture returned objects. This code is downloaded from 
#  https://medium.com/@birenmer/threading-the-needle-returning-values-from-python-threads-with-ease-ace21193c148


class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        # Initializing the Thread class
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    # Overriding the Thread.run function
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return



def download_month( monthstr:str, client, dataroot:str=default_dataroot ): 
    """Download one month of ERA5 3-hourly reanalysis of temperature, humidity, winds, 
    surface pressure. Also compute geopotential from these data. Return an instance 
    of RetClass. 

    Arguments
    =========

    monthstr        The month to download, should have format "YYYY-MM". 

    client          An instance of cdsapi.Client used to retrieve data from the Copernicus 
                    Climate Data Store.

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory

    """

    time0 = time()
    ret = RetClass()
    ret.update( comments=f'Running download_month for "{monthstr}"' )

    dmonth = datetime.fromisoformat( monthstr+"-01" )
    for var in [ 'uwnd.10m', 'vwnd.10m', 'temp.2m', 'dewpt.2m', 'uwnd', 'vwnd', 'temp', 'shum', 'geop' ]: 
        try: 
            d = ERA5file( var, client, dmonth, dataroot=dataroot )
            ret.update( comments=f'Successfully downloaded/computed {var}' )
        except Exception as e: 
            ret.update( comments=f'Unable to download/compute {var} for {monthstr}: {e}' )

    time1 = time()
    ret.update( comments=f'Total time elpased = {time1-time0} secs' )

    return ret



def download( datetimerange:tuple, client, dataroot:str=default_dataroot ): 
    """Download ERA5 3-hourly reanalysis of temperature, humidity, winds, surface pressure. 
    Compute geopotential from these data. Return an instance of RetClass.

    Arguments
    =========

    datetimerange   A 2-tuple of datetime.datetime instance defining the range 
                    of times of ERA5 files to download. 

    client          An instance of cdsapi.Client used to retrieve data from
                    the Copernicus Climate Data Store.

    """

    t0 = time()

    ret = RetClass()

    #  Check input. 

    if not isinstance( datetimerange, tuple ): 
        ret.update( success=False, messages="InvalidArgument", comments="libera5.download: datetimerange must be a tuple" )
        return ret

    if len( datetimerange ) != 2: 
        ret.update( success=False, messages="InvalidArgument", comments="libera5.download: datetimerange must have length 2" )
        return ret

    if not isinstance( datetimerange[0], datetime ) or not isinstance( datetimerange[1], datetime ): 
        ret.update( success=False, messages="InvalidArgument", comments="libera5.download: datetimerange must be composed of datetime.datetime instances" )
        return ret

    if datetimerange[1] < datetimerange[0]: 
        ret.update( success=False, messages="InvalidArgument", comments="libera5.download: datetimerange[1] must be greater than datetimerange[0]" )
        return ret


    dmonth = datetime( year=datetimerange[0].year, month=datetimerange[0].month, day=1 )
    while dmonth <= datetimerange[1]: 
        for var in [ 'uwnd.10m', 'vwnd.10m', 'temp.2m', 'dewpt.2m', 'uwnd', 'vwnd', 'temp', 'shum', 'geop' ]: 
            d = ERA5file( var, client, dmonth, dataroot=dataroot )
        dt = dmonth + timedelta(days=31)
        dmonth = datetime( year=dt.year, month=dt.month, day=1 )

    t1 = time()
    dt = int( t1 - t0 )
    days, hrs, mins, secs = int(dt/86400), int(dt/3600) % 24, int(dt/60) % 60, dt % 60
    print( f'Total elapsed time = {days:d} days, {hrs:d} hrs, {mins:2d} mins, {secs:2d} secs' )

    ret.update( success=True )

    return ret


def main(): 
    
    parser = argparse.ArgumentParser( prog="download_era5", description="Download all meteorological fields relevant to Great Plains LLJ research. " + \
            "The ERA5 data files will be truncated to the area of the North America." )

    parser.add_argument( "monthrange", type=str, help='The range of months over which to download ERA5 data, format ' + \
            '"YYYY-MM:YYYY-MM" for the first and last month to download' )

    parser.add_argument( "-d", "--dataroot", dest="dataroot", default=default_dataroot, 
            help=f'The root for the project; the default is "{default_dataroot}"' )

    parser.add_argument( "-k", "--key", dest="key", default="", 
            help='The key for the Copernicus Data Store account; can be found in ~/.cdsapirc file; ' + \
                    'the default is to simply use the .cdsapirc file' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    m = re.search( r'^(\d{4}-\d{2}):(\d{4}-\d{2})$', args.monthrange )

    if not m: 
        print( 'Be sure that the monthrange has format "YYYY-MM:YYYY-MM".' )
        return

    #  Initialize access to Copernicus Climate Data Store. 

    client = initCDS( args.key  )

    #  Execute. 

    dt0 = datetime.fromisoformat( m.group(1)+"-15" )
    dt1 = datetime.fromisoformat( m.group(2)+"-15" )
    ret = download( (dt0,dt1), client, dataroot=args.dataroot )

    if ret.success: 
        print( 'Normal successful completion' )
    
    else: 
        print( 'Unsuccessful' )
        print( 'Messages = ' + ', '.join( ret.messages ) )
        print( 'Comments = \n  ' + '\n  '.join( ret.comments ) )

    return




if __name__ == "__main__": 
    main()
    pass

