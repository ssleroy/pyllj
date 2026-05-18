import re
import os
import sys
import argparse
import numpy as np 
from netCDF4 import Dataset
from time import time
from datetime import datetime, timedelta, timezone
from .libera5 import RetClass, ERA5file, hybrid, cycle_time, downloads_subdir, gravity, initCDS, \
        epoch, output_time_units
from ..parameters import default_dataroot

#  Parameters. 

output_subdir = "ERA5/watervaporflux"
pbl_depth = 3.0e3           # meters

#  Convert hybrid coefficients to numpy arrays. 

hyai, hybi = np.array( hybrid['hyai'] ), np.array( hybrid['hybi'] )
fill_int = np.int32( -999 )


################################################################################
#  Compute column water vapor flux. 
################################################################################

def compute_watervaporflux( month:str, client, dataroot:str=default_dataroot, clobber:bool=False ): 
    """This function computes the column-integrated water vapor flux over 
    the interval defined by datetimerange as realized in the ERA5 reanalysis. 
    The component flux terms are written into the NetCDF file watervaporfile.

    Arguments
    =========

    month           A string defining the month for which to calculated column-integrated 
                    water vapor flux (YYYY-MM)

    client          An instance of cdsapi.Client used to retrieve data from the Copernicus 
                    Climate Data Store.

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory

    clobber         Set to true to clobber previously existing output files. 
    """

    time1 = time()
    ret = RetClass()

    #  Output file already exists? 

    dt = datetime.fromisoformat( month + "-01" )
    outputpath = os.path.join( dataroot, output_subdir, 'watervaporflux.{:}.nc'.format( dt.strftime("%Y%m") ) )

    if not clobber and os.path.exists( outputpath ): 
        print( f'Output file {outputpath} already exists. Exiting.' )
        ret.update( success=True, messages="OutputFileExists", 
                   comments = [ f'Output file {outputpath} already exists, do not clobber', 
                              'Exiting' ] )
        return ret

    #  Get surface height (fixed) file. Get longitudes and latitudes as well. Parameters 
    #  defining the Lambert conformal projection, too. 

    geopsfcfile = ERA5file( "geop.sfc", client, datetime(year=2000,month=1,day=1), dataroot=dataroot )
    d = geopsfcfile.open()
    lons = d.variables['longitude'][:].squeeze()
    lats = d.variables['latitude'][:].squeeze()
    geopsfc = d.variables['z'][:].squeeze() 
    d.close()

    #  Dimensions. 

    nlons, nlats = lons.size, lats.size
    t1 = datetime.fromisoformat( month+"-01" )
    t2 = t1 + timedelta(days=31)
    t2 = datetime( year=t2.year, month=t2.month, day=1 ) 
    ncd = int( 24 / cycle_time + 0.001 )
    ntimes = int( ( t2 - t1 ) / timedelta(hours=cycle_time) + 0.001 )

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

    d = { 'dt': datetime(1800,1,1) }
    pres = None

    #  Loop over time. 

    print( 'Computing water vapor flux' )

    for itime in range(ntimes): 

        dt = t1 + itime * timedelta(hours=cycle_time)
        print( '  Time ' + dt.strftime( "%Y-%m-%d %H:%M" ) )
        sys.stdout.flush()

        if dt.year != d['dt'].year or dt.month != d['dt'].month : 

            #  Get paths to upper air humidity, wind and height files. 

            for key, val in d.items(): 
                if key in [ 'uwnd', 'vwnd', 'shum', 'psfc', 'qsfc', 'usfc', 'vsfc' ]: 
                    val.close()

            uwndfile = ERA5file( "uwnd", client, dt, dataroot=dataroot )
            vwndfile = ERA5file( "vwnd", client, dt, dataroot=dataroot )
            shumfile = ERA5file( "shum", client, dt, dataroot=dataroot )
            geopfile  = ERA5file( "geop", client, dt, dataroot=dataroot )
            psfcfile = ERA5file( "pres.sfc", client, dt, dataroot=dataroot )
            # qsfcfile = ERA5file( "dewpt.2m", client, dt, dataroot=dataroot )
            # usfcfile = ERA5file( "uwnd.10m", client, dt, dataroot=dataroot )
            # vsfcfile = ERA5file( "vwnd.10m", client, dt, dataroot=dataroot )

            #  Open files. 

            d.update( { 
                    'uwnd': uwndfile.open(), 
                    'vwnd': vwndfile.open(), 
                    'shum': shumfile.open(), 
                    'geop': geopfile.open(), 
                    'psfc': psfcfile.open() 
                } )

            #  Get input time epoch and units. 

            time_units_str = d['geop'].variables['valid_time'].getncattr( "units" )
            m = re.search( r'^([a-zA-Z]+) since (\d{4}-\d{2}-\d{2})', time_units_str )
            input_time_units, input_epoch = m.group(1), datetime.fromisoformat( m.group(2) )

        d.update( { 'dt': dt } )

        #  Record time. 

        fdt = input_epoch + timedelta( **{ input_time_units: float( d['geop'].variables['valid_time'][itime] + 1.e-5 ) } )
        times[itime] = np.int32( ( fdt - epoch ) / timedelta( **{ output_time_units: 1 } ) + 0.001 )

        #  Read fields.  Note that ERA5 variables are already masked if they're below the surface. 

        uwnd = d['uwnd'].variables['u'][itime,:,:,:]
        vwnd = d['vwnd'].variables['v'][itime,:,:,:]
        shum = d['shum'].variables['q'][itime,:,:,:]
        psfc = d['psfc'].variables['sp'][itime,:,:]

        #  Entire column: Compute layer fluxes. 

        pres = ( np.outer( hybi, psfc.flatten() ) + np.outer( hyai, np.repeat( 1.0, nlons*nlats ) ) ).reshape( hyai.size, nlats, nlons )
        dpres = np.abs( pres[1:,:,:] - pres[:-1,:,:] )

        column['uf'][itime,:,:] = ( uwnd * dpres ).sum(axis=0) / gravity
        column['vf'][itime,:,:] = ( vwnd * dpres ).sum(axis=0) / gravity
        column['uwvf'][itime,:,:] = ( shum * uwnd * dpres ).sum(axis=0) / gravity
        column['vwvf'][itime,:,:] = ( shum * vwnd * dpres ).sum(axis=0) / gravity
        column['columnwater'][itime,:,:] = ( shum * dpres ).sum(axis=0) / gravity

        #  PBL: Compute mid-point fluxes. 

        dgeop = d['geop'].variables['z'][itime,:,:,:] - geopsfc
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

        iz = 0
        ascending = ( dgeop[1,0,0] > dgeop[0,0,0] )
        done = np.zeros( (nlats,nlons), np.int8 )

        while iz < hyai.size and np.any( done == 0 ): 
            mask = np.logical_and( done == 0, topflag[iz,:,:] ) 

            if np.any( mask ): 
                if ascending: 
                    pb = pres[iz,:,:]
                else: 
                    pb = pres[iz+1,:,:]

                t = ( pbl_depth * gravity - dgeop[iz,:,:] ) / ( dgeop[iz+1,:,:] - dgeop[iz,:,:] )
                dp = pb - np.exp( np.log(pres[iz,:,:] ) * (1-t) + np.log( pres[iz+1,:,:]) * t ) 
                uf1 += uwnd[iz,:,:] * dp / gravity * mask
                vf1 += vwnd[iz,:,:] * dp / gravity * mask
                uwvf1 += shum[iz,:,:] * uwnd[iz,:,:] * dp / gravity * mask
                vwvf1 += shum[iz,:,:] * vwnd[iz,:,:] * dp / gravity * mask
                columnwater1 += shum[iz,:,:] * dp / gravity * mask
                done += mask
            iz += 1

        #  Add top layer of PBL in with the rest of the column. 

        pbl['uf'][itime,:,:] += uf1
        pbl['vf'][itime,:,:] += vf1
        pbl['uwvf'][itime,:,:] += uwvf1
        pbl['vwvf'][itime,:,:] += vwvf1
        pbl['columnwater'][itime,:,:] += columnwater1

        #  Compute time. 

        times[itime] = int( ( dt - datetime(1800,1,1) ) / timedelta(hours=1) + 0.001 )

    #  Close files. 

    for key, val in d.items(): 
        if key not in [ 'dt' ]:
            val.close()

    #  Write to output. 

    print( f'Creating {outputpath}' )
    os.makedirs( os.path.dirname( outputpath ), exist_ok=True )
    d = Dataset( outputpath, 'w', format="NETCDF4" )

    #  Create dimensions. 

    d.createDimension( "longitude", nlons )
    d.createDimension( "latitude", nlats )
    d.createDimension( "time" )

    #  Create variables. 

    outs = {}

    var = "time"
    x = d.createVariable( var, np.int32, dimensions=("time",) )
    x.setncattr( "units", output_time_units + " since " + epoch.strftime("%Y-%m-%d %H:%M:%S")  )
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
                        "Note that the surface geopotential is defined by the ERA5 model.", 
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
            'file_type': "ERA5-watervaporflux", 
            'month': month, 
            'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ), 
            'author': "Stephen Leroy (stephen.leroy@janusresearch.us)"
        } )

    #  Write variables to output. 

    d.variables['time'][:] = np.ma.masked_where( ( times == fill_int ), times )
    d.variables['longitude'][:] = lons
    d.variables['latitude'][:] = lats
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
            column-integrated water flux over the entire domain of ERA5. 
            Column-integrated water vapor is also included in the output.""" )

    parser.add_argument( "month", type=str, 
            help="""A string defining the month for which to compute column-integrated 
            water vapor flux. The string has format YYYY-MM.""" )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str, 
            default=default_dataroot, 
            help="""The root directory where the LLJ ERA5 files are stored and where 
                results will be written. """ + f'The default is {default_dataroot}, and ' + \
                f'output will be written to the subdirectory {default_watervaporflux_dir}.' )

    parser.add_argument( "-k", "--key", dest="key", default="",
            help='The key for the Copernicus Data Store account; can be found in ~/.cdsapirc file; ' + \
                    'the default is to simply use the .cdsapirc file' )

    parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true", 
            help="""Clobber a pre-existing output file. The default is not to clobber.""" )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Date range. 

    emessage = "The month has to have format YYYY-MM"

    try: 
        month = datetime.strptime( args.month+"-01", '%Y-%m-%d' )
    except: 
        print( emessage )
        return

    client = initCDS( args.key )
    ret = compute_watervaporflux( args.month, client, dataroot=args.dataroot, clobber=args.clobber )

    return


if __name__ == "__main__": 
    main()
    pass

