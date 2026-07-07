import re
import os
from datetime import datetime, timedelta, timezone
import argparse
from .libera5 import cycle_time, default_dataroot, downloads_subdir, ERA5file, hybrid, initCDS
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from tqdm import tqdm
from time import time
from .libera5 import RetClass
from ..parameters import gravity


#  Subdirectory for diagnostics output. 

output_subdir = "ERA5/diagnostics"

#  Physical constants. 

nzulu = int( 24.0 / cycle_time + 0.001 )    # Number of model cycles per day


################################################################################
#  Define a function that can be used to produce a cubic Hermite spline
#  instance, which computes derivatives as input to 
#  scipy.interpolate.CubicHermiteSpline. 
################################################################################

def CHS( x, y ): 
    """Create an instance of a CubicHermiteSpline. It estimates the derivative of 
    y in x as an intermediate step."""

    deriv = ( y[1:] - y[:-1] ) / ( x[1:] - x[:-1] )
    midx = 0.5 * ( x[1:] + x[:-1] )
    midy = 0.5 * ( y[1:] + y[:-1] ) 

    f = CubicHermiteSpline( midx, midy, deriv )
    return f


################################################################################
#  Compute Great Plains LLJ diagnostics. 
################################################################################

def compute_diagnostics( month:str, client, dataroot:str=default_dataroot, clobber:bool=False ): 
    """This function computes diagnostics of the Great Plains LLJ in a time 
    interval defined by datetimerange as realized in the ERA5 reanalysis. The 
    diagnostics are written into the NetCDF file diagnosticsfile.

    Arguments
    =========

    month           A string of format "YYYY-MM" defining the month 
                    over which to compute diagnostics. 

    client          An instance of cdsapi.Client used to retrieve data from 
                    the Copernicus Climate Data Store. 

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory
    """

    time1 = time()
    ret = RetClass()

    dt = datetime.fromisoformat( month + "-01" )
    outputpath = os.path.join( dataroot, output_subdir, 'diagnostics.{:}.nc'.format( dt.strftime("%Y%m") ) )

    if os.path.exists( outputpath ) and not clobber:
        print( f'{outputpath} alread exists. Exiting.' )
        return

    #  Create datetimerange, a list of instances of datetime.datetime 
    #  corresponding to the daterange. 

    dt1 = datetime.fromisoformat( month+"-01" )
    dt2 = dt1 + timedelta(days=31)
    dt2 = datetime( year=dt2.year, month=dt2.month, day=1 ) - timedelta(days=1)
    datetimerange = [ dt1, dt2 ]

    #  Get surface height (fixed) file. Get longitudes and latitudes as well. 

    geopsfcfile = ERA5file( "geop.sfc", client, dt1, dataroot=dataroot  )
    d = geopsfcfile.open()
    hgtsfc = d.variables['z'][:].squeeze() / gravity
    lons = d.variables['longitude'][:].squeeze()
    lats = d.variables['latitude'][:].squeeze()
    d.close()

    #  Dimensions. 

    nlats, nlons = lats.size, lons.size
    ndays = ( datetimerange[1] - datetimerange[0] ).days + 1

    #  Dimension arrays. 

    datestrings = []
    llj_exists = np.zeros( (ndays,nzulu,nlats,nlons), np.int8 )
    llj_height = np.zeros( (ndays,nzulu,nlats,nlons), np.float32 )
    llj_wnd = np.zeros( (ndays,nzulu,nlats,nlons), np.float32 )

    #  Initialize loop over time. 

    d = { 'dt': datetime(1800,1,1) }
    dt = datetimerange[0] + timedelta(days=0)
    iday = 0

    #  Loop over time. 

    while dt <= datetimerange[1]: 

        datestrings.append( dt.strftime( "%Y-%m-%d" ) )

        if dt.year != d['dt'].year or dt.month != d['dt'].month: 

            #  Get paths to wind and height files. 

            for key, val in d.items(): 
                if key in [ 'uwnd', 'vwnd', 'geop' ]: 
                    val.close()

            uwndfile = ERA5file( "uwnd", client, dt, dataroot=dataroot )
            vwndfile = ERA5file( "vwnd", client, dt, dataroot=dataroot )
            geopfile = ERA5file( "geop", client, dt, dataroot=dataroot )

            #  Open files. 

            d.update( { 
                    'uwnd': uwndfile.open(), 
                    'vwnd': vwndfile.open(), 
                    'geop': geopfile.open()
                } )

        d.update( { 'dt': dt } )

        #  Scan over time intervals in day. 

        for ihour in range(nzulu): 

            itime = ( dt.day - 1 ) * nzulu + ihour
            wnd = np.sqrt( d['uwnd'].variables['u'][itime,:,:,:]**2 + \
                    d['vwnd'].variables['v'][itime,:,:,:]**2 )
            dh = d['geop'].variables['z'][itime,:,:,:]/gravity - hgtsfc
            dh = 0.5 * ( dh[:-1] + dh[1:] )
            ascending = ( dh[1,0,0] > dh[0,0,0] )

            #  Bonner criteria. Find wind speed maxima and minima below 3 km height above the surface. 

            print( 'Processing for {:}'.format( dt.strftime( "%Y-%m-%d %H:%M" ) ) )
            latiter = tqdm( range(nlats), desc='Latitude' )

            for ilat in latiter: 
                for ilon in range(nlons): 
                    ih = np.argwhere( dh[:,ilat,ilon] >= 0.0 ).flatten()
                    if ih.size < 4: 
                        continue
                    if ascending: 
                        f = CHS( dh[ih,ilat,ilon], wnd[ih,ilat,ilon] )
                    else: 
                        f = CHS( np.flip( dh[ih,ilat,ilon] ), np.flip( wnd[ih,ilat,ilon] ) )

                    #  Find extrema below 3 km height above the surface. 

                    fd = f.derivative()
                    roots = fd.roots()
                    if len(roots) < 2: 
                        continue

                    i = np.argwhere( np.logical_and( roots > 0.0, roots < 3.0e3 ) ).flatten()
                    roots = roots[i]

                    #  Separate maxima. 

                    fdd = fd.derivative()
                    deriv2 = fdd( roots )

                    i = np.argwhere( deriv2 < 0 ).flatten()
                    if i.size > 0: 
                        roots_maxima = roots[i]
                        vals_maxima = f( roots_maxima )
                    else: 
                        roots_maxima = None
                        vals_maxima = None

                    if roots_maxima is None: 
                        continue

                    #  Find wind maximum below 3 km height above the surface. 

                    imax = np.argmax( vals_maxima )
                    root_max = roots[imax]
                    wnd_max = vals_maxima[imax]

                    #  Find wind minima above the wind maximum. 

                    i = np.argwhere( np.logical_and( roots > root_max, deriv2 > 0 ) ).flatten()
                    if i.size > 0: 
                        root_min = min( roots[i] )
                        wnd_min = f( root_min )
                    else: 
                        root_min = None
                        wnd_min = None

                    if root_min is None: 
                        continue

                    #  Apply Bonner criteria. 

                    if wnd_max > 20.0: 
                        llj = ( wnd_min < 10.0 )
                    elif wnd_max > 16.0: 
                        llj = ( wnd_min < 8.0 )
                    elif wnd_max > 12.0: 
                        llj = ( wnd_min < 6.0 )
                    else: 
                        llj = False

                    if llj: 
                        llj_exists[iday,ihour,ilat,ilon] = 1
                        llj_height[iday,ihour,ilat,ilon] = root_max
                        llj_wnd[iday,ihour,ilat,ilon] = wnd_max

            #  Next time in file. 

            dt += timedelta(hours=cycle_time)

        #  Next day. 

        iday += 1

    #  Close files. 

    for key, val in d.items(): 
        if key in [ 'uwnd', 'vwnd', 'geop' ]: 
            val.close()

    #  Turn llj_exists into a mask for the other variables. 

    llj_height = np.ma.masked_where( np.logical_not(llj_exists), llj_height )
    llj_wnd = np.ma.masked_where( np.logical_not(llj_exists), llj_wnd )

    #  Write to output. 

    print( f'Creating {outputpath}' )
    os.makedirs( os.path.dirname( outputpath ), exist_ok=True )
    d = Dataset( outputpath, 'w', format="NETCDF4" )

    #  Create dimensions. 

    d.createDimension( "lon", nlons )
    d.createDimension( "lat", nlats )
    d.createDimension( "hr", nzulu )
    d.createDimension( "day", ndays )
    d.createDimension( "daystr", 10 )

    #  Create variables. 

    outs = {}

    var = "lons"
    x = d.createVariable( var, 'f4', dimensions=("lon",) )
    x.setncatts( { 
            'description': "East longitude array of grid", 
            'units': "degrees"
        } )
    outs.update( { var: x } )

    var = "lats"
    x = d.createVariable( var, 'f4', dimensions=("lat",) )
    x.setncatts( { 
            'description': "North latitude array of grid", 
            'units': "degrees"
        } )
    outs.update( { var: x } )

    var = "date"
    x = d.createVariable( var, 'c', dimensions=("day","daystr") )
    x.setncatts( { 
            'description': "Date as YYYY-MM-DD", 
            'units': "date"
        } )
    outs.update( { var: x } )

    var = "hour"
    x = d.createVariable( var, 'i2', dimensions=("hr",) )
    x.setncatts( { 
            'description': "UTC hour of day", 
            'units': "hours"
        } )
    outs.update( { var: x } )

    var = "height"
    x = d.createVariable( var, 'f4', dimensions=("day","hr","lat","lon") )
    x.setncatts( { 
            'description': "Height of the LLJ above the surface", 
            'units': "gpm"
        } )
    outs.update( { var: x } )

    var = "wind"
    x = d.createVariable( var, 'f4', dimensions=("day","hr","lat","lon") )
    x.setncatts( { 
            'description': "Wind speed of the LLJ", 
            'units': "m s**-1"
        } )
    outs.update( { var: x } )

    #  Global attributes. 

    d.setncatts( { 
            'file_type': "ERA5-LLJ-diagnostics", 
            'month': month, 
            'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ), 
            'author': "Stephen Leroy (stephen.leroy@janusresearch.us)"
        } )

    #  Write variables to output. 

    outs['lons'][:] = lons
    outs['lats'][:] = lats
    for i in range(ndays): 
        outs['date'][i,:] = datestrings[i]
    outs['hour'][:] = np.arange( 0, 24, 3 )
    outs['height'][:] = llj_height[:]
    outs['wind'][:] = llj_wnd[:]

    d.close()

    #  Done. 

    time2 = time()
    dt = time2 - time1
    hours, minutes, seconds = int(dt/3600), int(dt/60) % 60, int(dt) % 60 
    print( f'Elapsed time = {hours} hrs, {minutes:2d} mins, {seconds:2d} secs' )

    return ret



def main(): 

    default_diagnostics_dir = os.path.join( default_dataroot, output_subdir )

    parser = argparse.ArgumentParser( prog="compute_era5_diagnostics", 
            description="""Compute diagnostics of the Great Plains low-level jet (LLJ) based 
            on the ERA5 reanalysis. The diagnostics include the frequency 
            of occurrence of the LLJ, the height of the LLJ wind maximum, and the wind speed 
            maximum.""" )

    parser.add_argument( "month", type=str, 
            help="""A string defining the month for which to compute diagnostics, format 
            "YYYY-MM".""" )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str, 
            default=default_dataroot, 
            help="""The root directory where the LLJ files are stored and where 
                results will be written. """ + f'The default is {default_dataroot}, and ' + \
                f'output will be written to the subdirectory {default_diagnostics_dir}.' )

    parser.add_argument( "-k", "--key", dest="key", default="",
            help='The key for the Copernicus Data Store account; can be found in ~/.cdsapirc file; ' + \
                    'the default is to simply use the .cdsapirc file' )

    parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Date range. 

    try: 
        dtmonth = datetime.fromisoformat( args.month+"-01" )
    except: 
        print( "Month must have format YYYY-MM" )
        return

    client = initCDS( args.key )
    ret = compute_diagnostics( args.month, client, dataroot=args.dataroot, clobber=args.clobber )

    return


if __name__ == "__main__": 
    main()
    pass


