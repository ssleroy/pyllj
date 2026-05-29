#  Imports. 

import os
import glob
import re
import argparse
from datetime import datetime, timedelta, timezone
import csv
from netCDF4 import Dataset
import numpy as np
from ..parameters import default_dataroot, sondes
from ..libutils import RetClass


downloads_subdir = "downloads"
output_subdir = "isohypses"

def compute_isohypses_climatology( sonde:str, dataroot:str=default_dataroot, 
                                  yearrange=[2000,2017], outputfile="isohypses.nc" ): 

    ret = RetClass()

    #  Find the relevant sonde records. 

    recs = [ rec for rec in sondes if rec['name']==sonde ]
    if len( recs ) != 1: 
        sonde_names = sorted( [ rec['name'] for rec in sondes ] )
        comment = f'Could not find sonde "{sonde}". Valid sondes are ' + \
                ", ".join( [ f'"{na}"' for na in sonde_names ] ) + '.'
        ret.update( status=False, messages="InvalidArgument", comments=comment )
        return ret

    rec = recs[0]

    inputpath = os.path.join( dataroot, '{:}_snd'.format( rec['station_id'] ), "downloads" )
    inputfiles = sorted( glob.glob( inputpath, "*.csv" ) )

    #  Loop over input files. 

    olevels = np.arange( 100.0, 3000.1, 100.0, np.float32 )

    data = []
    for inputfile in inputfiles: 
        file = os.path.basename( inputfile )
        m = re.search( r'(\d{8}_\d{4}).csv$', file )
        dtime = datetime.strptime( m.group(1), '%Y%m%d_%H%M' )
        with open( inputfile, 'r' ) as f: 
            drecs = csv.DictReader( f, delimiter="," )
            heights = np.array( [ float( drec['hght'] ) for drec in drecs ] )
            directions = np.array( [ float( drec['drct'] ) for drec in drecs ] )
            speeds = np.array( [ float( drec['sped'] ) for drec in drecs ] )
            
            #  Convert wind speed and direction to u, v, heights to delta-heights. 

            alpha = np.deg2rad( directions )
            u = speeds * np.sin( alpha )
            v = speeds * np.cos( alpha )
            dheights = heights - rec['surface_height']

            #  Interpolate onto delta-isohypsic grid. 

            mask = np.logical_or( olevels < dheights.min(), olevels > dheights.max() )
            intp = interp1d( dheights, u )
            ui = np.ma.masked_where( mask, intp( olevels ) )
            intp = interp1d( dheights, v )
            vi = np.ma.masked_where( mask, intp( olevels ) )

            #  Store in data. 

            data.append( { 'time': dtime, 'ui': ui, 'vi': vi } )

    #  All data read in. What hours are available? 

    hours = np.array( sorted( list( { d['time'].hour for d  in data } ) ), np.int32 )
    nhours = hours.size

    #  Initialize climatology. 

    uc = np.zeros( (12,nhours,olevels.size), np.float32 )
    ucn = np.zeros( (12,nhours,olevels.size), np.int32 )
    vc = np.zeros( (12,nhours,olevels.size), np.float32 )
    vcn = np.zeros( (12,nhours,olevels.size), np.int32 )

    #  Loop over month, hour. 

    for d in data: 

        if d['time'].year < yearrange[0] or d['time'].year > yearrange[1]: 
            continue

        imonth = d['time'].month - 1
        ihour = np.argwhere( hours == d['time'].hour ).squeeze()

        good = np.logical_not( np.get_mask( d['ui'] ) )
        uc[imonth,ihour,good] += d['ui'][good]
        ucn[imonth,ihour,good] += 1

        good = np.logical_not( np.get_mask( d['vi'] ) )
        vc[imonth,ihour,good] += d['vi'][good]
        vcn[imonth,ihour,good] += 1

    #  Normalize and mask. 

    uc /= ucn
    uc = np.ma.masked_where( ucn < 24, ucn )

    vc /= vcn
    vc = np.ma.masked_where( vcn < 24, vcn )

    #  Write to output file. 

    outputpath = os.path.join( dataroot, "isohypses", outputfile )
    os.mkdirs( os.path.dirname( outputpath ), exist_ok=True )
    ret.update( comments=f'Writing to {outputpath}' )

    d = Dataset( outputpath, 'w', format="NETCDF4" )

    #  Dimensions. 

    d.createDimension( "level", olevels.size )
    d.createDimension( "hour", nhours )
    d.createDimension( "month", 12 )

    #  Variables and variable attributes. 

    v = d.createVariable( "longitude", np.float32 )
    v.setncatts( { 
                  'units': "degrees_east", 
                  'standard_name': "longitude", 
                  'long_name': "longitude" } )

    v = d.createVariable( "latitude", np.float32 )
    v.setncatts( { 
                  'units': "degrees_north", 
                  'standard_name': "latitude", 
                  'long_name': "latitude" } )

    v = d.createVariable( "level", np.float32, dimensions=("level",) )
    v.setncatts( { 
                  'long_name': "Height above the surface", 
                  'units': "m" } )

    v = d.createVariable( "month", np.int32, dimensions=("month",) )
    v.setncatts( { 
                  'long_name': "Month of the monthly average of the diurnal cycle in horizontal winds", 
                  'units': "units", 
                  'valid_range': np.int32( [1,12] ) } )

    v = d.createVariable( "hour", np.int32, dimensions=("hour",) )
    v.setncatts( { 
                  'long_name': "Zulu hour of the day", 
                  'units': "hours", 
                  'valid_range': np.int32( [0,23] ) } )

    v = d.createVariable( "uwnd", np.float32, dimensions=("month","hour","level") )
    v.setncatts( { 
                  'long_name': "Zonal component of wind", 
                  'units': "m/s" } )

    v = d.createVariable( "vwnd", np.float32, dimensions=("month","hour","level") )
    v.setncatts( { 
                  'long_name': "Meridional component of wind", 
                  'units': "m/s" } )

    #  Global attributes. 

    d.setncatts( { 
                  'file_type': "sonde_isohypsic_analysis_climatology", 
                  'sonde_name': rec['name'], 
                  'station_id': rec['station_id'], 
                  'year_range': np.int32( yearrange ), 
                  'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ), 
                  'author': "Stephen Leroy (stephen.leroy@janusresearch.us)" } )

    #  Write to output. 

    d.variables['longitude'][:] = rec['longitude']
    d.variables['latitude'][:] = rec['latitude']
    d.variables['level'][:] = olevels
    d.variables['month'][:] = np.arange(12,np.int32) + 1
    d.variables['hour'][:] = hours
    d.variables['uwnd'][:] = uc
    d.variables['vwnd'][:] = vc

    ret.update( success=True )
    return ret

def main(): 

    parser = argparse.ArgumentParser( prog='compute_sonde_isohypses_climatology', 
                description='Compute a delta-isoshypsic horizontal wind climatology ' + \
                'for rawinsonde data at a single station' )

    parser.add_argument( 'station', type=str, help="Name of the rawinsonde station. " + \
            "Valid names are " + ", ".join( [ '"{:}"'.format( s['name'] ) for s in sondes ] ) + "." )

    parser.add_argument( 'yearrange', type=str, help='The range of years over which to ' + \
            'compute the climatology, format "YYYY:YYYY"' )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
            default=default_dataroot,
            help="""The root directory where LLJ project files are stored and where
                    results will be written. """ + f'The default is {default_dataroot}.' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true", 
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Process yearrange. 

    m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
    if m: 
        yearrange = ( int(m.group(1)), int(m.group(2)) )
    else: 
        print( 'Invalid yearrange, must have format "YYYY:YYYY"' )
        exit()

    #  Do computations. 

    ret = compute_isohypses_climatology( args.station, yearrange=yearrange )

    print( ret )
    pass

