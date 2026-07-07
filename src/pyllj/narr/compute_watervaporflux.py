import re
import os
import sys
import argparse
import numpy as np 
from netCDF4 import Dataset
from time import time
from datetime import datetime, timedelta, timezone
from .libnarr import RetClass, NARRfile, cycle_time, gravity 
from ..parameters import default_dataroot

#  Parameters. 

output_subdir = "NARR/watervaporflux"
pbl_depth = 3.0e3           # meters


################################################################################
#  Compute column water vapor flux. 
################################################################################

def compute_watervaporflux( month:str, dataroot:str=default_dataroot, clobber:bool=False ): 
    """This function computes the column-integrated water vapor flux over 
    the interval defined by datetimerange as realized in the North American 
    Regional Reanalysis. The component flux terms are written into the 
    NetCDF file watervaporfile.

    Arguments
    =========

    month           A string defining the month for which to calculated column-integrated 
                    water vapor flux (YYYY-MM)

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory

    """

    time1 = time()
    ret = RetClass()

    dt = datetime.fromisoformat( month + "-01" )
    outputpath = os.path.join( dataroot, output_subdir, 'watervaporflux.{:}.nc'.format( dt.strftime("%Y%m") ) )
    os.makedirs( os.path.dirname( outputpath ), exist_ok=True )

    #  Output file already exists? 

    if not clobber and os.path.exists( outputpath ): 
        print( f'Output file {watervaporfluxfile} already exists. Exiting.' )
        ret.update( success=True, messages="OutputFileExists", 
                   comments = [ f'Output file {watervaporfluxfile} already exists, do not clobber', 
                              'Exiting' ] )
        return ret

    #  Get surface height (fixed) file. Get longitudes and latitudes as well. Parameters 
    #  defining the Lambert conformal projection, too. 

    hgtsfcfile = NARRfile( "hgt.sfc", dataroot=dataroot )
    d = hgtsfcfile.open()
    lons = d.variables['lon'][:].squeeze()
    lats = d.variables['lat'][:].squeeze()
    v = d.variables['Lambert_Conformal']
    projection_attrs = { attr: v.getncattr(attr) for attr in v.ncattrs() } 
    v = d.variables['time']
    time_attrs = { attr: v.getncattr(attr) for attr in v.ncattrs() } 
    hgtsfc = d.variables['hgt'][:].squeeze()
    d.close()

    #  Dimensions. 

    ny, nx = lons.shape
    t1 = datetime.fromisoformat( month+"-01" )
    t2 = t1 + timedelta(days=31)
    t2 = datetime( year=t2.year, month=t2.month, day=1 ) 
    ncd = int( 24 / cycle_time + 0.001 )
    ntimes = int( ( t2 - t1 ) / timedelta(hours=cycle_time) + 0.001 )

    #  Dimension arrays. 

    times = np.ma.masked_array( np.zeros( ntimes, np.float64 ) )
    column, pbl = {}, {}

    for v in [ column, pbl ]: 
        v.update( { 
                'uf': np.ma.masked_array( np.zeros( (ntimes,ny,nx), np.float32 ) ), 
                'vf': np.ma.masked_array( np.zeros( (ntimes,ny,nx), np.float32 ) ), 
                'uwvf': np.ma.masked_array( np.zeros( (ntimes,ny,nx), np.float32 ) ), 
                'vwvf': np.ma.masked_array( np.zeros( (ntimes,ny,nx), np.float32 ) ), 
                'columnwater': np.ma.masked_array( np.zeros( (ntimes,ny,nx), np.float32 ) ) 
            } )

    #  Initialize loop over time. 

    d = { 'dt': datetime(1800,1,1) }
    pres = None

    #  Loop over time. 

    for itime in range(ntimes): 

        dt = t1 + itime * timedelta(hours=cycle_time)
        print( '  Time ' + dt.strftime( "%Y-%m-%d %H:%M" ) )
        sys.stdout.flush()

        if dt.year != d['dt'].year or dt.month != d['dt'].month : 

            #  Get paths to upper air humidity, wind and height files. 

            for key, val in d.items(): 
                if key in [ 'uwnd', 'vwnd', 'shum', 'psfc', 'qsfc', 'usfc', 'vsfc' ]: 
                    val.close()

            uwndfile = NARRfile( "uwnd", dt, dataroot=dataroot )
            vwndfile = NARRfile( "vwnd", dt, dataroot=dataroot )
            shumfile = NARRfile( "shum", dt, dataroot=dataroot )
            hgtfile  = NARRfile( "hgt", dt, dataroot=dataroot )
            psfcfile = NARRfile( "pres.sfc", dt, dataroot=dataroot )
            qsfcfile = NARRfile( "shum.2m", dt, dataroot=dataroot )
            usfcfile = NARRfile( "uwnd.10m", dt, dataroot=dataroot )
            vsfcfile = NARRfile( "vwnd.10m", dt, dataroot=dataroot )

            #  Open files. 

            d.update( { 
                    'uwnd': uwndfile.open(), 
                    'vwnd': vwndfile.open(), 
                    'shum': shumfile.open(), 
                    'hgt': hgtfile.open(), 
                    'psfc': psfcfile.open(), 
                    'qsfc': qsfcfile.open(), 
                    'usfc': usfcfile.open(), 
                    'vsfc': vsfcfile.open()
                } )

            if pres is None: 
                pres = d['shum']['level'][:] * 100
                dpres = np.abs( pres[1:] - pres[:-1] ) 

        d.update( { 'dt': dt } )

        #  Read fields.  Note that NARR variables are already masked if they're below the surface. 

        uwnd = d['uwnd'].variables['uwnd'][itime,:,:,:]
        vwnd = d['vwnd'].variables['vwnd'][itime,:,:,:]
        shum = d['shum'].variables['shum'][itime,:,:,:]
        psfc = d['psfc'].variables['pres'][itime,:,:]
        qsfc = d['qsfc'].variables['shum'][itime,:,:]
        usfc = d['usfc'].variables['uwnd'][itime,:,:]
        vsfc = d['vsfc'].variables['vwnd'][itime,:,:]

        #  Entire column: Compute mid-point fluxes. 

        midu = 0.5 * ( uwnd[1:,:,:] + uwnd[:-1,:,:] )
        midv = 0.5 * ( vwnd[1:,:,:] + vwnd[:-1,:,:] )
        midq = 0.5 * ( shum[1:,:,:] + shum[:-1,:,:] )

        column['uf'][itime,:,:] = np.ma.dot( np.transpose( midu, axes=(1,2,0) ), dpres ) / gravity
        column['vf'][itime,:,:] = np.ma.dot( np.transpose( midv, axes=(1,2,0) ), dpres ) / gravity
        column['uwvf'][itime,:,:] = np.ma.dot( np.transpose( midq*midu, axes=(1,2,0) ), dpres ) / gravity
        column['vwvf'][itime,:,:] = np.ma.dot( np.transpose( midq*midv, axes=(1,2,0) ), dpres ) / gravity
        column['columnwater'][itime,:,:] = np.ma.dot( np.transpose( midq, axes=(1,2,0) ), dpres ) / gravity

        #  PBL: Compute mid-point fluxes. 

        dhgt = d['hgt'].variables['hgt'][itime,:,:,:] - hgtsfc
        flag = ( dhgt[1:,:,:] <= pbl_depth )

        pbl['uf'][itime,:,:] = np.ma.dot( np.transpose( midu*flag, axes=(1,2,0) ), dpres ) / gravity
        pbl['vf'][itime,:,:] = np.ma.dot( np.transpose( midv*flag, axes=(1,2,0) ), dpres ) / gravity
        pbl['uwvf'][itime,:,:] = np.ma.dot( np.transpose( midq*midu*flag, axes=(1,2,0) ), dpres ) / gravity
        pbl['vwvf'][itime,:,:] = np.ma.dot( np.transpose( midq*midv*flag, axes=(1,2,0) ), dpres ) / gravity
        pbl['columnwater'][itime,:,:] = np.ma.dot( np.transpose( midq*flag, axes=(1,2,0) ), dpres ) / gravity

        #  PBL: Compute contribution of top PBL layer. 

        topflag = ( ( dhgt[1:,:,:] - pbl_depth ) * ( dhgt[:-1,:,:] - pbl_depth ) <= 0 )

        uf1 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        vf1 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        uwvf1 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        vwvf1 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        columnwater1 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )

        iz = 0
        done = np.zeros( (ny,nx), np.int8 )
        ascending = ( dhgt[1,0,0] > dhgt[0,0,0] )

        while iz < pres.size and np.any( done == 0 ): 
            mask = np.logical_and( done == 0, topflag[iz,:,:] ) 
            if ascending: 
                pb = pres[iz]
            else: 
                pb = pres[iz+1]

            if np.any( mask ): 
                t = ( pbl_depth - dhgt[iz,:,:] ) / ( dhgt[iz+1,:,:] - dhgt[iz,:,:] )
                midq = shum[iz,:,:] * (1-0.5*t) + shum[iz+1,:,:] * 0.5*t
                midu = uwnd[iz,:,:] * (1-0.5*t) + uwnd[iz+1,:,:] * 0.5*t
                midv = vwnd[iz,:,:] * (1-0.5*t) + vwnd[iz+1,:,:] * 0.5*t
                dp = pb - np.exp( np.log(pres[iz]) * (1-t) + np.log(pres[iz+1]) * t ) 
                uf1 += midu * dp / gravity * mask
                vf1 += midv * dp / gravity * mask
                uwvf1 += midq * midu * dp / gravity * mask
                vwvf1 += midq * midv * dp / gravity * mask
                columnwater1 += midq * dp / gravity * mask
                done += mask
            iz += 1

        #  Add top layer of PBL in with the rest of the column. 

        pbl['uf'][itime,:,:] += uf1
        pbl['vf'][itime,:,:] += vf1
        pbl['uwvf'][itime,:,:] += uwvf1
        pbl['vwvf'][itime,:,:] += vwvf1
        pbl['columnwater'][itime,:,:] += columnwater1

        #  Account for the surface layer. 

        uf0 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        vf0 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        uwvf0 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        vwvf0 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )
        columnwater0 = np.ma.masked_array( np.zeros( (ny,nx), np.float32 ) )

        iz = 0
        done = np.zeros( (ny,nx), np.int8 )

        while iz < pres.size and np.any( done == 0 ): 
            mask = np.logical_and( uwvf0 == 0.0, pres[iz] <= psfc[:,:] ) 
            if np.any( mask ): 
                midq = 0.5 * ( shum[iz,:,:] + qsfc[:,:] )
                midu = 0.5 * ( uwnd[iz,:,:] + usfc[:,:] )
                midv = 0.5 * ( vwnd[iz,:,:] + vsfc[:,:] )
                dps = np.abs( psfc[:,:] - pres[iz] ) 
                uf0 += midu * dps / gravity * mask
                vf0 += midv * dps / gravity * mask
                uwvf0 += midq * midu * dps / gravity * mask
                vwvf0 += midq * midv * dps / gravity * mask
                columnwater0 += midq * dps / gravity * mask
                done += mask
            iz += 1

        #  Add surface layer in with the rest of the column. 

        for v in [ column, pbl ]: 
            v['uf'][itime,:,:] += uf0
            v['vf'][itime,:,:] += vf0
            v['uwvf'][itime,:,:] += uwvf0
            v['vwvf'][itime,:,:] += vwvf0
            v['columnwater'][itime,:,:] += columnwater0

        #  Compute time. 

        times[itime] = int( ( dt - datetime(1800,1,1) ) / timedelta(hours=1) + 0.001 )

    #  Close files. 

    for key, val in d.items(): 
        if key not in [ 'dt' ]:
            val.close()

    #  Write to output. 

    print( f'Creating {outputpath}' )
    d = Dataset( outputpath, 'w' )

    #  Create dimensions. 

    d.createDimension( "x", nx )
    d.createDimension( "y", ny )
    d.createDimension( "time" )

    #  Create variables. 

    outs = {}

    var = "Lambert_Conformal"
    x = d.createVariable( var, 'i4' )
    x.setncatts( projection_attrs )
    outs.update( { var: x } )

    var = "time"
    x = d.createVariable( var, 'f8', dimensions=("time",) )
    time_attrs.update( { 'actual_range': [ times.min(), times.max() ] } )
    x.setncatts( time_attrs )
    outs.update( { var: x } )

    var = "lon"
    x = d.createVariable( var, 'f4', dimensions=("y","x") )
    x.setncatts( { 
            'description': "East longitude array of grid", 
            'units': "degrees"
        } )
    outs.update( { var: x } )

    var = "lat"
    x = d.createVariable( var, 'f4', dimensions=("y","x") )
    x.setncatts( { 
            'description': "North latitude array of grid", 
            'units': "degrees"
        } )
    outs.update( { var: x } )

    for groupname in [ "column", "pbl" ]: 

        g = d.createGroup( groupname )

        #  Group attributes. 

        if groupname == "column": 
            g.setncattr( "description", "All quantities calculated for the entire atmospheric column" )
        elif groupname == "pbl": 
            g.setncattr( "description", "All quantities calculated for the planetary boundary layer" )

        if groupname == "pbl": 
            var = "pbl_depth"
            x = g.createVariable( var, 'f4' )
            x.setncatts( { 
                    'description': "The height of the planetary boundary layer above the surface. " + \
                        "Note that the surface geopotential is defined by the NARR model.", 
                    'units': "m"
                } )

        var = "uf"
        x = g.createVariable( var, 'f4', dimensions=("time","y","x") )
        x.setncatts( { 
                'description': "Eastward (u) component of the atmospheric mass flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "vf"
        x = g.createVariable( var, 'f4', dimensions=("time","y","x") )
        x.setncatts( { 
                'description': "Northward (v) component of the atmospheric mass flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "uwvf"
        x = g.createVariable( var, 'f4', dimensions=("time","y","x") )
        x.setncatts( { 
                'description': "Eastward (u) component of the column-integrated water vapor flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "vwvf"
        x = g.createVariable( var, 'f4', dimensions=("time","y","x") )
        x.setncatts( { 
                'description': "Northward (v) component of the column-integrated water vapor flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "cwv"
        x = g.createVariable( var, 'f4', dimensions=("time","y","x") )
        x.setncatts( { 
                'description': "Column-integrated water vapor", 
                'units': "kg m**-2"
            } )

    #  Global attributes. 

    d.setncatts( { 
            'file_type': "NARR-watervaporflux", 
            'month': month, 
            'creation_time': datetime.now(timezone.utc).isoformat(timespec="seconds"), 
            'author': "Stephen Leroy (sleroy@aer.com)"
        } )

    #  Write variables to output. 

    d.variables['lon'][:] = lons
    d.variables['lat'][:] = lats
    d.variables['time'][:] = times
    d.groups['pbl']['pbl_depth'][:] = pbl_depth

    for groupname in [ "column", "pbl" ]: 

        g = d.groups[groupname]
        if groupname == "column": 
            vs = column
        elif groupname == "pbl": 
            vs = pbl

        g.variables['uf'][:] = vs['uf']
        g.variables['vf'][:] = vs['vf']
        g.variables['uwvf'][:] = vs['uwvf']
        g.variables['vwvf'][:] = vs['vwvf']
        g.variables['cwv'][:] = vs['columnwater']

    d.close()

    #  Done. 

    time2 = time()
    dt = time2 - time1
    hours, minutes, seconds = int(dt/3600), int(dt/60) % 60, int(dt) % 60 
    print( f'Elapsed time = {hours} hrs, {minutes:2d} mins, {seconds:2d} secs' )

    return ret




def main(): 

    default_watervaporflux_dir = os.path.join( default_dataroot, output_subdir )

    parser = argparse.ArgumentParser( prog="compute_narr_watervaporflux", 
            description="""Compute column-integrated water vapor and the component 
            column-integrated water flux over the entire domain of the North American 
            Regional Reanalysis. Column-integrated water vapor is also included in 
            the output.""" )

    parser.add_argument( "month", type=str, 
            help="""A string defining the month ifor which to compute column-integrated 
            water vapor flux. The string has format YYYY-MM.""" )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str, 
            default=default_dataroot, 
            help="""The root directory where the LLJ NARR files are stored and where 
                results will be written. """ + f'The default is {default_dataroot}, and ' + \
                f'output will be written to the subdirectory {default_watervaporflux_dir}.' )

    parser.add_argument( "--clobber", dest="clobber", default=False, action="store_true", 
            help="""Clobber a pre-existing output file. The default is not to clobber.""" )

    args = parser.parse_args()

    #  Date range. 

    emessage = "The month has to have format YYYY-MM"

    try: 
        month = datetime.strptime( args.month+"-01", '%Y-%m-%d' )
    except: 
        print( emessage )
        return

    ret = compute_watervaporflux( args.month, args.dataroot, clobber=args.clobber )

    return


if __name__ == "__main__": 
    main()
    pass

