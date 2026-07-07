import re
import os
from datetime import datetime, timedelta, timezone
import argparse
from .libnarr import cycle_time, default_dataroot, NARRfile, RetClass, gravity
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from tqdm import tqdm
from time import time


#  Subdirectory for diagnostics output. 

output_subdir = "NARR/diagnostics"

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

def compute_diagnostics( month:{str}, dataroot:str=default_dataroot, clobber:bool=False ): 
    """This function computes diagnostics of the Great Plains LLJ in a time 
    interval defined by datetimerange as realized in the North American 
    Regional Reanalysis. The diagnostics are written into the NetCDF file 
    outputpath.

    Arguments
    =========

    month           A string of format "YYYY-MM" defining the month 
                    over which to compute diagnostics. 

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory

    """

    time1 = time()
    ret = RetClass()

    dt = datetime.fromisoformat( month + "-01" )
    outputpath = os.path.join( dataroot, output_subdir, 'diagnostics.{:}.nc'.format( dt.strftime("%Y%m") ) )

    if os.path.exists( outputpath ): 
        if clobber: 
            print( f'compute_narr_diagnostics: {outputpath} already exists. Clobbering.' )
        else: 
            print( f'compute_narr_diagnostics: {outputpath} already exists. Exiting.' )
            return

    os.makedirs( os.path.dirname( outputpath ), exist_ok=True )

    #  Create datetimerange, a list of instances of datetime.datetime 
    #  corresponding to the daterange. 

    dt1 = datetime.fromisoformat( month+"-01" )
    dt2 = dt1 + timedelta(days=31)
    dt2 = datetime( year=dt2.year, month=dt2.month, day=1 ) - timedelta(days=1)
    datetimerange = [ dt1, dt2 ]

    #  Get surface height (fixed) file. Get longitudes and latitudes as well. 

    hgtsfcfile = NARRfile( "hgt.sfc", dataroot=dataroot )
    d = hgtsfcfile.open()
    hgtsfc = d.variables['hgt'][:].squeeze()
    lons = d.variables['lon'][:].squeeze()
    lats = d.variables['lat'][:].squeeze()
    d.close()

    #  Dimensions. 

    ny, nx = hgtsfc.shape
    ndays = ( datetimerange[1] - datetimerange[0] ).days + 1

    #  Dimension arrays. 

    datestrings = []
    llj_exists = np.zeros( (ndays,nzulu,ny,nx), np.int8 )
    llj_height = np.zeros( (ndays,nzulu,ny,nx), np.float32 )
    llj_wnd = np.zeros( (ndays,nzulu,ny,nx), np.float32 )

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
                if key in [ 'uwnd', 'vwnd', 'hgt' ]: 
                    val.close()

            uwndfile = NARRfile( "uwnd", dt, dataroot=dataroot )
            vwndfile = NARRfile( "vwnd", dt, dataroot=dataroot )
            hgtfile = NARRfile( "hgt", dt, dataroot=dataroot )

            #  Open files. 

            d.update( { 
                    'uwnd': uwndfile.open(), 
                    'vwnd': vwndfile.open(), 
                    'hgt': hgtfile.open()
                } )

        d.update( { 'dt': dt } )

        #  Scan over time intervals in day. 

        for ihour in range(nzulu): 

            itime = ( dt.day - 1 ) * nzulu + ihour
            wnd = np.sqrt( d['uwnd'].variables['uwnd'][itime,:,:,:]**2 + \
                    d['vwnd'].variables['vwnd'][itime,:,:,:]**2 )
            dh = d['hgt'].variables['hgt'][itime,:,:,:] - hgtsfc

            #  Bonner criteria. Find wind speed maxima and minima below 3 km height above the surface. 

            desc = 'Processing for {:}'.format( dt.strftime( "%Y-%m-%d %H:%M" ) )
            yiter = tqdm( range(ny), desc=desc )

            for iy in yiter: 
                for ix in range(nx): 
                    ih = np.argwhere( dh[:,iy,ix] >= 0.0 ).flatten()
                    if ih.size < 4: 
                        continue
                    f = CHS( dh[ih,iy,ix], wnd[ih,iy,ix] )

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
                        llj_exists[iday,ihour,iy,ix] = 1
                        llj_height[iday,ihour,iy,ix] = root_max
                        llj_wnd[iday,ihour,iy,ix] = wnd_max

            #  Next time in file. 

            dt += timedelta(hours=3)

        #  Next day. 

        iday += 1

    #  Close files. 

    for key, val in d.items(): 
        if key in [ 'uwnd', 'vwnd', 'hgt' ]: 
            val.close()

    #  Turn llj_exists into a mask for the other variables. 

    llj_height = np.ma.masked_where( np.logical_not(llj_exists), llj_height )
    llj_wnd = np.ma.masked_where( np.logical_not(llj_exists), llj_wnd )

    #  Write to output. 

    print( f'Creating {outputpath}' )
    d = Dataset( outputpath, 'w', format="NETCDF4" )

    #  Create dimensions. 

    d.createDimension( "x", nx )
    d.createDimension( "y", ny )
    d.createDimension( "hr", nzulu )
    d.createDimension( "day", ndays )
    d.createDimension( "daystr", 10 )

    #  Create variables. 

    outs = {}

    var = "lons"
    x = d.createVariable( var, 'f4', dimensions=("y","x") )
    x.setncatts( { 
            'description': "East longitude array of grid", 
            'units': "degrees"
        } )
    outs.update( { var: x } )

    var = "lats"
    x = d.createVariable( var, 'f4', dimensions=("y","x") )
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
    x = d.createVariable( var, 'f4', dimensions=("day","hr","y","x") )
    x.setncatts( { 
            'description': "Height of the LLJ above the surface", 
            'units': "gpm"
        } )
    outs.update( { var: x } )

    var = "wind"
    x = d.createVariable( var, 'f4', dimensions=("day","hr","y","x") )
    x.setncatts( { 
            'description': "Wind speed of the LLJ", 
            'units': "m s**-1"
        } )
    outs.update( { var: x } )

    #  Global attributes. 

    d.setncatts( { 
            'file_type': "NARR-LLJ-diagnostics", 
            'month': month, 
            'creation_time': datetime.now(timezone.utc).isoformat(timespec="seconds"), 
            'author': "Stephen Leroy (sleroy@aer.com)"
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

    parser = argparse.ArgumentParser( prog="compute_narr_diagnostics", 
            description="""Compute diagnostics of the Great Plains low-level jet (LLJ) based 
            on the North American Regional Reanalysis. The diagnostics include the frequency 
            of occurrence of the LLJ, the height of the LLJ wind maximum, and the wind speed 
            maximum.""" )

    parser.add_argument( "month", type=str, 
            help="""A string defining the month for which to compute diagnostics, format 
            "YYYY-MM".""" )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str, 
            default=default_dataroot, 
            help=f'The root data directory for the project. The default is {default_dataroot} and ' + \
                f'the output will be written to the subdirectory {default_diagnostics_dir}.' )

    args = parser.parse_args()

    #  Date range. 

    try: 
        dtmonth = datetime.fromisoformat( args.month+"-01" )
    except: 
        print( "Month must have format YYYY-MM" )
        return

    ret = compute_diagnostics( args.month, args.dataroot )

    return


if __name__ == "__main__": 
    main()
    pass


