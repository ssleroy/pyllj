import os
import re
import sys
import shutil
import argparse
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
import boto3
from time import time
from .libpod import cycle_time, ModelOutput
from ..parameters import RetClass, regions, gravity 


output_subdir = "isohypses"

#  The fixed output heights above the surface [in meters]. 

olevels = np.arange( 100.0, 3001.0, 100.0 )


def compute_isohypses( yearmonth:str, dataroot:str, clobber:bool=False ): 
    """Generate the monthly average diurnal cycle of horizontal winds as generated 
    by the output of an atmospheric model on fixed height-above-the surface levels. 

    Arguments
    =========

    yearmonth       A string of the form "YYYY-MM" indicating the year and the 
                    month for which to compute the monthly average diurnal cycle.

    dataroot        Root path of the model run. 

    clobber         Set to true to clobber previously existing output files. 
    """

    t0 = time()
    ret = RetClass()

    #  Set paths, etc.

    year, month = int(yearmonth[0:4]), int(yearmonth[5:7])
    outputfile = f'isohypses.{year:04d}{month:02d}.nc'

    outputpath = os.path.join( dataroot, output_subdir, outputfile )
    if os.path.exists( outputpath ): 
        if clobber: 
            print( f'{outputpath} already exists. Clobbering.' )
        else: 
            print( f'{outputpath} already exists. Exiting.' )
            ret.update( success=False )
            return ret

    #  Access to model output. 

    model = ModelOutput( dataroot )
    lons = model.lons * 1.0
    lats = model.lats * 1.0

    #  Restrict to North America. 

    rs = [ region for region in regions if region['name'] == "north-america" ]
    if len( rs ) == 1: 
        region = rs[0]
    else: 
        ret.update( success=False, messages="InvalidRegion", comments='Region "north-america" is not available in regions' )
        model.close()
        return ret

    alongituderange = region['longituderange']
    alatituderange = region['latituderange']

    #  Justify longitudes. 

    alongituderange[ alongituderange >= 180 ] -= 360
    lons[lons >= 180] -= 360

    #  Subselect longitudes. 

    ilons0 = ( lons >= alongituderange[0] )
    ilons1 = ( lons <= alongituderange[1] )

    if alongituderange[1] > alongituderange[0]: 
        ilons = np.argwhere( np.logical_and( ilons0, ilons1 ) ).squeeze()
    else: 
        ilons = np.concatenate( np.argwhere( ilons1 ).squeeze(), np.argwhere( ilons0 ).squeeze() )

    #  Subselect latitudes. 

    ilats = np.argwhere( np.logical_and( lats >= alatituderange[0], lats <= alatituderange[1] ) ).squeeze()

    #  Dimensions. 

    nlons, nlats = ilons.size, ilats.size 
    nhours = int( 24 / cycle_time )
    dt0 = datetime( year=year, month=month, day=1 )
    dt1 = dt0 + timedelta(days=31)
    ndays = ( dt1 - dt0 ).days
    ntimes = nhours * ndays

    #  Create output file.

    print( f'Creating {outputfile}' )
    sys.stdout.flush()

    d_out = Dataset( outputfile, 'w', format="NETCDF4" )

    d_out.createDimension( 'longitude', nlons )
    d_out.createDimension( 'latitude', nlats )
    d_out.createDimension( 'level', olevels.size )
    d_out.createDimension( 'hour', nhours )

    #  Create new variables.

    vout = d_out.createVariable( "longitude", np.float32, dimensions=('longitude',) )
    vout.setncatts( { 'description': "East longitude", 
                    'units': "degrees" } )

    vout = d_out.createVariable( "latitude", np.float32, dimensions=('latitude',) )
    vout.setncatts( { 'description': "North latitude", 
                    'units': "degrees" } )

    vout = d_out.createVariable( 'level', np.float32, ('level',) )
    vout.setncatts( { 'long_name': "Height above the surface",
                    'units': "m" } )

    vout = d_out.createVariable( 'hour', np.float32, ('hour',) )
    vout.setncatts( { 'long_name': "Zulu hour of the day",
                    'units': "hours" } )

    vout = d_out.createVariable( 'uwnd', np.float32, ('hour','level','latitude','longitude') )
    vout.setncatts( { 'long_name': "Zonal component of wind",
                    'units': "m/s" } )

    vout = d_out.createVariable( 'vwnd', np.float32, ('hour','level','latitude','longitude') )
    vout.setncatts( { 'long_name': "Meridional component of wind",
                    'units': "m/s" } )

    vout = d_out.createVariable( 'year', np.int32 )
    vout.setncatts( { 'long_name': "Year of the monthly average of the diurnal cycle in horizontal winds",
                    'units': "none" } )

    vout = d_out.createVariable( 'month', np.int32 )
    vout.setncatts( { 'long_name': "Month of the monthly average of the diurnal cycle in horizontal winds",
                    'units': "none",
                    'valid_range': np.array( [1,12], dtype=np.int32 ) } )

    #  Global attributes.

    d_out.setncatts( {
                    'file_type': "model-isohypsic_diurnal_wind_climatology",
                    'dataroot': dataroot, 
                    'description': "The diurnal cycle of horizontal winds from a model run " + \
                            "at discrete heights above the surface is averaged over one month",
                    'author': "Stephen Leroy (stephen.leroy@janusresearch.us)"
                    } )


    #  Write levels to output.

    d_out.variables['longitude'][:] = lons[ilons]
    d_out.variables['latitude'][:] = lats[ilats]
    d_out.variables['level'][:] = olevels
    d_out.variables['hour'][:] = np.arange( 0, 24, cycle_time )
    d_out.variables['year'][:] = year
    d_out.variables['month'][:] = month

    #  Loop over time.

    u = np.ma.zeros( (nhours,olevels.size,nlats,nlons), np.float32 ) 
    v = np.ma.zeros( (nhours,olevels.size,nlats,nlons), np.float32 ) 

    print( 'Computing isohypsic analysis' )

    for itime in range(ntimes): 
        dt = datetime( year=year, month=month, day=1 ) + itime * timedelta( hours=int(cycle_time) ) + model.toffset
        ihour = int( dt.hour / cycle_time )

        print( '  Time ' + dt.strftime( "%Y-%m-%d %H:%M" ) )
        sys.stdout.flush()

        #  Get height profiles, subtract surface.

        print( '    Reading input fields' )
        sys.stdout.flush()

        ucomp, i, ascending = model.getvar( "ucomp", dt )
        uwnd1 = ucomp[i,:,:,:]
        if not ascending: 
            uwnd1 = np.flip( uwnd1, axis=0 )

        vcomp, i, ascending = model.getvar( "vcomp", dt )
        vwnd1 = vcomp[i,:,:,:]
        if not ascending: 
            vwnd1 = np.flip( vwnd1, axis=0 )

        zgcomp, i, ascending = model.getvar( "zg", dt )
        h1 = zgcomp[i,:,:,:]
        if not ascending: 
            h1 = np.flip( h1, axis=0 )
        h1 = h1 - model.zsurf

        #  Interpolate onto new levels.

        for iolevel in tqdm( range(olevels.size), desc="    Computing iolevel" ): 
            olevel = olevels[iolevel]
            ii = np.argmin( ( h1[1:,:,:] - olevel ) * ( h1[:-1,:,:] - olevel ), axis=0 )
            for iilat, ilat in enumerate( ilats ): 
                for iilon, ilon in enumerate( ilons ): 
                    i = ii[ilat,ilon]
                    t = ( olevel - h1[i,ilat,ilon] ) / ( h1[i+1,ilat,ilon] - h1[i,ilat,ilon] )
                    u[ihour,iolevel,iilat,iilon] += uwnd1[i,ilat,ilon] * (1-t) + uwnd1[i+1,ilat,ilon] * t
                    v[ihour,iolevel,iilat,iilon] += vwnd1[i,ilat,ilon] * (1-t) + vwnd1[i+1,ilat,ilon] * t

    #  Average over month. 

    dt1 = datetime( year=year, month=month, day=1 )
    dt2 = dt1 + timedelta( days=31 )
    dt2 = datetime( year=dt2.year, month=dt2.month, day=1 )
    ndays = int( ( dt2 - dt1 ).days )

    u /= ndays
    v /= ndays

    #  Write to output.

    d_out.variables['uwnd'][:,:,:,:] = u
    d_out.variables['vwnd'][:,:,:,:] = v

    #  Done with computations.

    d_out.setncatts( { 'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ) } )
    d_out.close()

    print( f'Moving to {outputpath}' )
    sys.stdout.flush()
    os.makedirs( os.path.dirname( outputpath ), exist_ok=True )
    shutil.copy( outputfile, outputpath )
    os.unlink( outputfile )

    #  Timing. 

    t1 = time()
    dt = t1 - t0
    minutes = int( dt / 60 )
    seconds = int( dt - minutes*60 )
    ret.update( success=True, comments=f'Elapsed time = {minutes:d} mins, {seconds:02d} secs' )

    return ret


def main(): 

    parser = argparse.ArgumentParser( prog="compute_isohypses", 
            description="Generate a monthly average diurnal cycle of model horizontal " + \
                    "winds on fixed heights above the surface" )

    parser.add_argument( "yearmonth", type=str, help='Year-month of model output to process, format "YYYY-MM".' )

    parser.add_argument( "dataroot", type=str,
                help="""The root directory of the model run""" )

    parser.add_argument( "--clobber", "-c", default=False, action="store_true", 
            help='Clobber a pre-existing output file; the default is not to clobber' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Run computations. 

    ret = compute_isohypses( args.yearmonth, args.dataroot, clobber=args.clobber )
    print( ret )

    return


if __name__ == "__main__": 
    main()

