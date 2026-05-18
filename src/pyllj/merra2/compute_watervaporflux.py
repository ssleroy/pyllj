import re
import os
import sys
import argparse
import numpy as np 
from netCDF4 import Dataset, MFDataset
from time import time
from datetime import datetime, timedelta, timezone
from .libmerra2 import RetClass, cycle_time, default_dataroot, downloads_subdir, \
        gravity, epoch, output_time_units, region, netrcauth

#  Parameters. 

output_subdir = "MERRA2/watervaporflux"
pbl_depth = 3.0e3           # meters
fill_int = np.int32( -999 )


################################################################################
#  Compute column water vapor flux. 
################################################################################

def compute_watervaporflux( month:str, dataroot:str=default_dataroot, clobber:bool=False ): 
    """This function computes the column-integrated water vapor flux over 
    the interval defined by datetimerange as realized in the MERRA2 reanalysis. 
    The component flux terms are written into the NetCDF file watervaporfile.

    Arguments
    =========

    month           A string defining the month for which to calculated column-integrated 
                    water vapor flux (YYYY-MM)

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory

    """

    time1 = time()
    ret = RetClass()

    t = datetime.fromisoformat( month+"-01" )
    outputpath = os.path.join( dataroot, output_subdir, 'watervaporflux.{:}.nc'.format( t.strftime("%Y%m") ) )
    downloadsroot = os.path.join( dataroot, downloads_subdir )

    #  Output file already exists? 

    if not clobber and os.path.exists( outputpath ): 
        print( f'Output file {outputpath} already exists. Exiting.' )
        ret.update( success=True, messages="OutputFileExists", 
                   comments = [ f'Output file {outputpath} already exists, do not clobber', 
                              'Exiting' ] )
        return ret

    #  Search downloads directory for all NetCDF files. 

    monthpath = os.path.join( downloadsroot, f'{t.year:4d}', f'{t.month:02d}' )
    if not os.path.isdir( monthpath ): 
        ret.update( success=False, messages="InvalidPath", comments=f'Path {monthpath} does not exist or is not a directory' )
        return ret

    inputfiles = sorted( [ os.path.join( monthpath, f ) for f in os.listdir( monthpath ) \
            if re.search( r'^merra2\..*\.nc$', f ) and re.search( region['regionname'], f ) ] ) 
    if len(inputfiles) < 20: 
        ret.update( success=False, messages="InsufficientData", comments=f'Path {monthpath} contains only {len(inputfiles)} files' )
        return ret
    else: 
        ret.update( success=True, comments=f'Path {monthpath} contains {len(inputfiles)} files' )

    #  Get surface height (fixed) file. Get longitudes and latitudes as well. Parameters 
    #  defining the Lambert conformal projection, too. 

    ds = MFDataset( inputfiles, 'r' )
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    levs = ds.variables['lev'][:]
    ts = ds.variables['time'][:]

    #  Get time epoch and units. 

    m = re.search( r'^([a-z]+) since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$', ds.variables['time'].units )
    input_time_units = m.group(1)
    input_epoch = datetime.strptime( m.group(2), "%Y-%m-%d %H:%M:%S" )

    #  Dimensions. 

    nlons, nlats, nlevs, ntimes = lons.size, lats.size, levs.size, ts.size
    ncd = int( 24.0 / cycle_time + 0.001 )

    #  Dimension arrays. 

    times = np.zeros( ntimes, np.int32 ) + fill_int
    column, pbl = {}, {}

    for v in [ column, pbl ]: 
        v.update( { 
                'uf': np.ma.masked_array( np.zeros( (ntimes,nlats,nlons), np.float32 ) ), 
                'vf': np.ma.masked_array( np.zeros( (ntimes,nlats,nlons), np.float32 ) ), 
                'uwvf': np.ma.masked_array( np.zeros( (ntimes,nlats,nlons), np.float32 ) ), 
                'vwvf': np.ma.masked_array( np.zeros( (ntimes,nlats,nlons), np.float32 ) ), 
                'columnwater': np.ma.masked_array( np.zeros( (ntimes,nlats,nlons), np.float32 ) ) 
            } )

    #  Initialize loop over time. 

    d, pres = {}, None

    #  Loop over time. 

    for itime in range(ntimes): 

        dt = input_epoch + timedelta( **{ input_time_units: int(ts[itime]) } )
        print( '  Time ' + dt.strftime( "%Y-%m-%d %H:%M" ) )
        sys.stdout.flush()

        d.update( { 'dt': dt } )

        #  Read fields.  Note that MERRA2 variables are already masked if they're below the surface. 

        uwnd = ds.variables['U'][itime,:,:,:]
        vwnd = ds.variables['V'][itime,:,:,:]
        shum = ds.variables['QV'][itime,:,:,:]
        pmid = ds.variables['PL'][itime,:,:,:]
        psfc = ds.variables['PS'][itime,:,:]

        #  Entire column: Compute layer fluxes. 

        pres = np.zeros( (nlevs+1,nlats,nlons), np.float32 )
        if pmid[1,0,0] > pmid[0,0,0]: 
            pres[0,:,:] = 0.0
            pres[1:-1,:,:] = 0.5 * ( pmid[1:,:,:] + pmid[:-1,:,:] )
            pres[-1,:,:] = psfc
        else: 
            pres[0,:,:] = psfc
            pres[1:-1,:,:] = 0.5 * ( pmid[1:,:,:] + pmid[:-1,:,:] )
            pres[-1,:,:] = 0.0

        dpres = np.abs( pres[1:,:,:] - pres[:-1,:,:] )

        column['uf'][itime,:,:] = ( uwnd * dpres ).sum(axis=0) / gravity
        column['vf'][itime,:,:] = ( vwnd * dpres ).sum(axis=0) / gravity
        column['uwvf'][itime,:,:] = ( shum * uwnd * dpres ).sum(axis=0) / gravity
        column['vwvf'][itime,:,:] = ( shum * vwnd * dpres ).sum(axis=0) / gravity
        column['columnwater'][itime,:,:] = ( shum * dpres ).sum(axis=0) / gravity

        #  PBL: Compute mid-point fluxes. 

        dmidgeop = ds.variables['H'][itime,:,:,:] * gravity - ds.variables['PHIS'][itime,:,:]
        dgeop = np.zeros( (nlevs+1,nlats,nlons), np.float32 )
        if dmidgeop[1,0,0] > dmidgeop[0,0,0]: 
            dgeop[0,:,:] = 0.0
            dgeop[1:-1,:,:] = 0.5 * ( dmidgeop[:-1,:,:] + dmidgeop[1:,:,:] )
            dgeop[-1,:,:] = 100.0e3 * gravity
        else: 
            dgeop[0,:,:] = 100.0e3 * gravity
            dgeop[1:-1,:,:] = 0.5 * ( dmidgeop[:-1,:,:] + dmidgeop[1:,:,:] )
            dgeop[-1,:,:] = 0.0

        flag = ( dgeop[1:,:,:] <= pbl_depth * gravity )

        pbl['uf'][itime,:,:] = ( uwnd * dpres * flag ).sum(axis=0) / gravity
        pbl['vf'][itime,:,:] = ( vwnd * dpres * flag ).sum(axis=0) / gravity
        pbl['uwvf'][itime,:,:] = ( shum * uwnd * dpres * flag ).sum(axis=0) / gravity
        pbl['vwvf'][itime,:,:] = ( shum * vwnd * dpres * flag ).sum(axis=0) / gravity
        pbl['columnwater'][itime,:,:] = ( shum * dpres * flag ).sum(axis=0) / gravity

        #  PBL: Compute contribution of top PBL layer. 

        topflag = ( ( dgeop[1:,:,:] - pbl_depth * gravity ) * ( dgeop[:-1,:,:] - pbl_depth * gravity ) <= 0 )

        uf1 = np.ma.masked_array( np.zeros( (nlats,nlons), np.float32 ) )
        vf1 = np.ma.masked_array( np.zeros( (nlats,nlons), np.float32 ) )
        uwvf1 = np.ma.masked_array( np.zeros( (nlats,nlons), np.float32 ) )
        vwvf1 = np.ma.masked_array( np.zeros( (nlats,nlons), np.float32 ) )
        columnwater1 = np.ma.masked_array( np.zeros( (nlats,nlons), np.float32 ) )

        done = np.zeros( (nlats,nlons), np.int8 )
        if dgeop[1,0,0] > dgeop[0,0,0]: 
            iziter = range( 0, nlevs-1, 1 )
        else: 
            iziter = range( nlevs-1, -1, -1 )

        for iz in iziter: 
            if not np.any( done == 0 ): break

            mask = np.logical_and( done == 0, topflag[iz,:,:] ) 
            if np.any( mask ): 
                t = ( pbl_depth * gravity - dgeop[iz,:,:] ) / ( dgeop[iz+1,:,:] - dgeop[iz,:,:] )
                dp = np.abs( np.exp( np.log(pres[iz,:,:] ) * (1-t) + np.log( pres[iz+1,:,:]) * t ) - pres[iz,:,:] )
                uf1 += uwnd[iz,:,:] * dp / gravity * mask
                vf1 += vwnd[iz,:,:] * dp / gravity * mask
                uwvf1 += shum[iz,:,:] * uwnd[iz,:,:] * dp / gravity * mask
                vwvf1 += shum[iz,:,:] * vwnd[iz,:,:] * dp / gravity * mask
                columnwater1 += shum[iz,:,:] * dp / gravity * mask
                done += mask

        #  Add top layer of PBL in with the rest of the column. 

        pbl['uf'][itime,:,:] += uf1
        pbl['vf'][itime,:,:] += vf1
        pbl['uwvf'][itime,:,:] += uwvf1
        pbl['vwvf'][itime,:,:] += vwvf1
        pbl['columnwater'][itime,:,:] += columnwater1

        #  Compute time. 

        times[itime] = int( ( dt - epoch ) / timedelta( **{ output_time_units: 1 } ) + 0.001 )

    #  Close files. 

    ds.close()

    #  Write to output. 

    print( f'Creating {outputpath}' )
    d = Dataset( outputpath, 'w', format="NETCDF4" )

    #  Create dimensions. 

    d.createDimension( "longitude", nlons )
    d.createDimension( "latitude", nlats )
    d.createDimension( "time" )

    #  Create variables. 

    outs = {}

    var = "time"
    x = d.createVariable( var, np.int32, dimensions=("time",) )
    x.setncattr( "units", output_time_units + " since " + epoch.strftime("%Y-%m-%d %H:%M:%S") )
    outs.update( { var: x } )

    var = "longitude"
    x = d.createVariable( var, np.float32, dimensions=("longitude",) )
    x.setncatts( { 
            'description': "East longitude array of grid", 
            'units': "degrees"
        } )
    outs.update( { var: x } )

    var = "latitude"
    x = d.createVariable( var, np.float32, dimensions=("latitude",) )
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
            x = g.createVariable( var, np.float32 )
            x.setncatts( { 
                    'description': "The height of the planetary boundary layer above the surface. " + \
                        "Note that the surface geopotential is defined by the MERRA2 model.", 
                    'units': "m"
                } )

        var = "uf"
        x = g.createVariable( var, np.float32, dimensions=("time","latitude","longitude") )
        x.setncatts( { 
                'description': "Eastward (u) component of the atmospheric mass flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "vf"
        x = g.createVariable( var, np.float32, dimensions=("time","latitude","longitude") )
        x.setncatts( { 
                'description': "Northward (v) component of the atmospheric mass flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "uwvf"
        x = g.createVariable( var, np.float32, dimensions=("time","latitude","longitude") )
        x.setncatts( { 
                'description': "Eastward (u) component of the column-integrated water vapor flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "vwvf"
        x = g.createVariable( var, np.float32, dimensions=("time","latitude","longitude") )
        x.setncatts( { 
                'description': "Northward (v) component of the column-integrated water vapor flux", 
                'units': "kg m**-1 s**-1"
            } )

        var = "cwv"
        x = g.createVariable( var, np.float32, dimensions=("time","latitude","longitude") )
        x.setncatts( { 
                'description': "Column-integrated water vapor", 
                'units': "kg m**-2"
            } )

    #  Global attributes. 

    d.setncatts( { 
            'file_type': "MERRA2-watervaporflux", 
            'month': month, 
            'creation_time': datetime.now(timezone.utc).strftime( "%d %b %Y %H:%M:%S UTC" ),
            'author': "Stephen Leroy (stephen.leroy@janusresearch.us" } )

    #  Write variables to output. 

    d.variables['longitude'][:] = lons
    d.variables['latitude'][:] = lats
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

    parser = argparse.ArgumentParser( prog="compute_era5_watervaporflux", 
            description="""Compute column-integrated water vapor and the component 
            column-integrated water flux over the entire domain of the North American 
            Regional Reanalysis. Column-integrated water vapor is also included in 
            the output.""" )

    parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
            default=default_dataroot,
            help="""The root directory where the LLJ files are stored and where
                results will be written. """ + f'The default is {default_dataroot}, and ' + \
                f'output will be written to the subdirectory {default_watervaporflux_dir}.' )

    parser.add_argument( "--auth", "-a", dest="auth", default="",
            help="""String containing the username and password for authentication to NASA Earthdata; the 
            username and password should be separated by a single space""" )

    parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true", 
            help="""Clobber a pre-existing output file. The default is not to clobber.""" )

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

    #  Month range. 

    m = re.search( r'^(\d{4}-\d{2}):(\d{4}-\d{2})$', args.monthrange )
    if m:
        monthrange = [ m.group(1), m.group(2) ]
    else:
        print( 'Be sure that the monthrange has format "YYYY-MM:YYYY-MM"' )
        return

    if monthrange[0] > monthrange[1]:
        print( 'Be sure that the monthrange has a first value that is less than or equal to the second value' )
        return

    os.makedirs( os.path.join( args.dataroot, output_subdir ), exist_ok=True )

    #  Loop over months. 

    dt = datetime.fromisoformat( monthrange[0]+"-01" )
    dt1 = datetime.fromisoformat( monthrange[1]+"-01" )

    while dt <= dt1: 

        print( "Processing " + dt.strftime("%Y-%m") )
        ret = compute_watervaporflux( dt.strftime("%Y-%m"), dataroot=args.dataroot, clobber=args.clobber )

        print( ret )

        #  Next month. 

        dt += timedelta( days=31 )
        dt = datetime( year=dt.year, month=dt.month, day=1 )

    return


if __name__ == "__main__": 
    main()
    pass

