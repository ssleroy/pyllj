import re
import os
from datetime import datetime, timedelta, timezone
import argparse
from .libpod import modeloutput_subdir, ModelOutput, region
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from tqdm import tqdm
from time import time
from ..parameters import gravity, RetClass


#  Subdirectory for diagnostics output. 

output_subdir = "diagnostics"


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

def compute_diagnostics( month:str, dataroot:str, clobber:bool=False ): 
    """This function computes diagnostics of the Great Plains LLJ in a month 
    YYYY-MM as produced by an atmospheric model (GFDL AM4, NCAR CAM7, etc.). 
    The diagnostics are written into the NetCDF file diagnosticsfile.

    Arguments
    =========

    month           A string of format "YYYY-MM" defining the month 
                    over which to compute diagnostics. 

    dataroot        The root of the model run. 

    """

    time1 = time()
    ret = RetClass()

    dt = datetime.fromisoformat( month + "-01" )
    outputpath = os.path.join( dataroot, output_subdir, 'diagnostics.{:}.nc'.format( dt.strftime("%Y%m") ) )

    if os.path.exists( outputpath ): 
        if clobber: 
            print( f'{outputpath} alread exists. Overwriting.' )
            ret.update( comments=f'{outputpath} already exists. Overwriting.' )
        else: 
            print( f'{outputpath} alread exists. Exiting.' )
            ret.update( success=False, comments=f'{outputpath} already exists. Exiting.' )
            return ret

    #  Instantiate access to the model output. 

    model = ModelOutput( dataroot )
    ncd = int( timedelta(hours=24)/model.tdelta + 0.001 )

    #  Create datetimerange, a list of instances of datetime.datetime 
    #  corresponding to the daterange. 

    dt1 = datetime.fromisoformat( month+"-01" )
    dt2 = dt1 + timedelta(days=31)
    dt2 = datetime( year=dt2.year, month=dt2.month, day=1 ) - timedelta(days=1)
    datetimerange = [ dt1, dt2 ]

    #  Dimensions. 

    nlats, nlons = model.lats.size, model.lons.size
    ndays = ( datetimerange[1] - datetimerange[0] ).days + 1

    #  Dimension arrays. 

    datestrings = []
    llj_exists = np.zeros( (ndays,ncd,nlats,nlons), np.int8 )
    llj_height = np.zeros( (ndays,ncd,nlats,nlons), np.float32 )
    llj_wnd = np.zeros( (ndays,ncd,nlats,nlons), np.float32 )

    #  Initialize loop over time. 

    dtday = datetimerange[0] + timedelta(days=0)
    iday = 0

    #  Loop over time. 

    while dtday <= datetimerange[1]: 

        datestrings.append( dtday.strftime( "%Y-%m-%d" ) )

        #  Scan over time intervals in day. 

        for izulu in range(ncd): 

            dt = dtday + izulu * timedelta(days=1) / ncd + model.toffset
            ihour = ( izulu + 1 ) % ncd

            #  Get wind and height profiles. 

            ucomp, i, ascending = model.getvar( "ucomp", dt )
            uwnd = ucomp[i,:,:,:]
            if not ascending: 
                uwnd = np.flip( uwnd, axis=0 )

            vcomp, i, ascending = model.getvar( "vcomp", dt )
            vwnd = vcomp[i,:,:,:]
            if not ascending: 
                vwnd = np.flip( vwnd, axis=0 )

            # TODO: need an if statement here to get Z3, too, if necessary. 
            zgcomp, i, ascending = model.getvar( "zg", dt )
            zg = zgcomp[i,:,:,:]
            if not ascending: 
                zg = np.flip( zg, axis=0 )

            wnd = np.sqrt( uwnd**2 + vwnd**2 )
            
            # TODO: ideally would not hardcode this and would instead check if 
            # a given model's height coord were on pressure levs vs half levs 
            # then, in a second step, subtract surface if necessary. 
            # However, this is *much* quicker...
            if model.Z3 is not None:
                # CAM7 
                dh = model.Z3
            elif model.zg is not None:
                # AM4
                dh = 0.5 * ( zg[:-1,:,:] + zg[1:,:,:] ) - model.zsurf
            else:
                raise NameError("Height variable not found for model.")

            #  Bonner criteria. Find wind speed maxima and minima below 3 km height above the surface. 

            print( 'Processing for {:}'.format( dt.strftime( "%Y-%m-%d %H:%M" ) ) )
            latiter = tqdm( range(nlats), desc='Latitude' )

            for ilat in latiter: 
                for ilon in range(nlons): 
                    ih = np.argwhere( dh[:,ilat,ilon] >= 0.0 ).flatten()
                    if ih.size < 4: 
                        continue
                    f = CHS( dh[ih,ilat,ilon], wnd[ih,ilat,ilon] )

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

            #  Next zulu time. 

        #  Next day. 

        dtday += timedelta( days=1 )
        iday += 1

    #  Close files. 

    model.close()

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
    d.createDimension( "hr", ncd )
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
            'file_type': "model-diagnostics", 
            'dataroot': dataroot, 
            'month': month, 
            'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ), 
            'author': "Stephen Leroy (stephen.leroy@janusresearch.us)"
        } )

    #  Write variables to output. 

    outs['lons'][:] = model.lons
    outs['lats'][:] = model.lats
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
    ret.update( success=True, comments=f'Elapsed time = {hours} hrs, {minutes:2d} mins, {seconds:2d} secs' )

    return ret



def main(): 

    parser = argparse.ArgumentParser( prog="compute_diagnostics", 
            description="""Compute diagnostics of the Great Plains low-level jet (LLJ) based 
            on atmospheric model output. The diagnostics include the frequency of occurrence 
            of the LLJ, the height of the LLJ wind maximum, and the wind speed maximum.""" )

    parser.add_argument( "month", type=str, 
            help="""A string defining the month for which to compute diagnostics, format 
            "YYYY-MM".""" )

    parser.add_argument( "dataroot", type=str, help="""The root directory of the model run.""" )

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

    ret = compute_diagnostics( args.month, args.dataroot, clobber=args.clobber )
    print( ret )

    return


if __name__ == "__main__": 
    main()
    pass


