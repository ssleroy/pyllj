import os 
import re
import argparse 
from netCDF4 import Dataset 
import numpy as np 
from datetime import datetime, timezone
import warnings

warnings.filterwarnings('ignore')


def compute_diagnostics_climatology( dataroot:str, clobber:bool=False ): 
    """Compute a climatology of LLJ diagnostics, the output of the compute_diagnostics 
    routines. It scans the diagnostics (existence of LLJ, core wind speed, core 
    height above surface) for every month of the run and writes output to 
    "summary_diagnostics.nc". 

    Arguments
    =========
    dataroot        Path to the root of the model run. 

    clobber         Set to true in order to clobber pre-existing output. """


    #  Establish directory path, outputfile. 

    analysispath = os.path.join( dataroot, "diagnostics" )

    if not os.path.isdir( analysispath ): 
        print( f'Analysis path {analysispath} does not exist. Aborting.' )
        exit()

    outputfile = os.path.join( analysispath, "diagnostics.nc" )
    if os.path.exists( outputfile ): 
        if clobber: 
            print( f'{outputfile} exists. Clobbering.' )
        else: 
            print( f'{outputfile} exists. Exiting.' )
            return

    #  Get a listing of diagnostics analysis files. 

    paths = sorted( [ os.path.join( analysispath, f ) for f in os.listdir( analysispath ) if re.search( r'diagnostics.*\.\d{6}\.nc$', f ) ] )

    if len( paths ) == 0: 
        print( f'No diagnostics analysis files found in {analysispath}. Aborting.' )
        exit()
    else: 
        print( f'Found {len(paths)} diagnostics files.' )

    analyses = []
    for path in paths: 
        m = re.search( r'diagnostics.*\.(\d{4})(\d{2})\.nc$', path )
        analyses.append( { 'year': int(m.group(1)), 'month': int(m.group(2)), 'path': path } )

    #  Loop over months. 

    diagnostics = []

    for month in range(1,13): 

        wind, height = [], []

        for analysis in analyses: 
            if analysis['month'] != month: continue

            print( 'Reading '  + analysis['path'] )
            a = Dataset( analysis['path'], 'r' )
            lons = a.variables['lons'][:]
            lats = a.variables['lats'][:]
            wind.append( a.variables['wind'][:] )
            height.append( a.variables['height'][:] )
            a.close()

        wind = np.ma.concatenate( wind )
        height = np.ma.concatenate( height )

        ndays, nhours, ny, nx = wind.shape

        #  Count events; compute mean heights and winds. 

        events = np.logical_not( wind.mask )
        nevents = events.sum(axis=1).sum(axis=0)
        ndailyevents = events.any(axis=1).sum(axis=0)

        eventProbability = nevents / ( ndays * nhours )
        dailyeventProbability = ndailyevents / ndays
        meanheight = height.reshape( (ndays*nhours,ny,nx) ).mean(axis=0) 
        meanwind = wind.reshape( (ndays*nhours,ny,nx) ).mean(axis=0) 

        #  Mask. 

        mask = ( nevents == 0 )

        diagnostics.append( { 
                'eventProbability': eventProbability, 
                'dailyeventProbability': dailyeventProbability, 
                'meanheight':  np.ma.masked_where( mask, meanheight ), 
                'meanwind': np.ma.masked_where( mask, meanwind ) } )

    #  Turn lons and lats into meshgrid if not already so. 

    if len( lons.shape ) == 2: 
        mlons = lons
        mlats = lats
    else: 
        mlons, mlats = np.meshgrid( lons, lats )

    ny, nx = mlons.shape

    #  Write to outputfile. 

    d = Dataset( outputfile, 'w', format="NETCDF4" )

    #  Dimensions. 

    d.createDimension( "nx", nx )
    d.createDimension( "ny", ny )
    d.createDimension( "month", 12 )

    #  Create variables. 

    var = d.createVariable( "longitude", np.float32, ("ny","nx") )
    var.setncatts( { 'description': "east longitude", 'units': "degrees" } )

    var = d.createVariable( "latitude", np.float32, ("ny","nx") )
    var.setncatts( { 'description': "north latitude", 'units': "degrees" } )

    var = d.createVariable( "month", np.int32, ("month",) )
    var.setncatts( { 'description': "Month of year: January=1, February=2, etc.", 
                    'units': "none", 'valid_range': [ np.int32(1), np.int32(12) ] } )

    var = d.createVariable( "eventProbability", np.float32, ("month","ny","nx") )
    var.setncatts( { 'description': "Probability that an LLJ occurs", 
                    'units': "none", 'valid_range': [ np.float32(0.0), np.float32(1.0) ] } )

    var = d.createVariable( "dailyeventProbability", np.float32, ("month","ny","nx") )
    var.setncatts( { 'description': "Probability that at least one LLJ occurs on a given day", 
                    'units': "none", 'valid_range': [ np.float32(0.0), np.float32(1.0) ] } )

    var = d.createVariable( "meanheight", np.float32, ("month","ny","nx") )
    var.setncatts( { 'description': "Mean height above the surface of the jet core", 
                    'units': "m", 'valid_range': [ np.float32(0.0), np.float32(3000.0) ] } )

    var = d.createVariable( "meanwind", np.float32, ("month","ny","nx") )
    var.setncatts( { 'description': "Mean wind speed of LLJ jet cores", 
                    'units': "m/s", 'valid_range': [ np.float32(0.0), np.float32(100.0) ] } )

    #  Global attributes. 

    yearrange = np.int32( [ analyses[0]['year'], analyses[-1]['year'] ] )
    d.setncatts( { 
                  'file_type': 'diagnostics_climatology', 
                  'dataroot': dataroot, 
                  'year_range': yearrange, 
                  'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" )
                } )

    #  Write data. 

    d.variables['longitude'][:] = mlons
    d.variables['latitude'][:] = mlats
    d.variables['month'][:] = np.arange( 1, 13 )

    for imonth, dm in enumerate(diagnostics): 
        d.variables['eventProbability'][imonth,:,:] = dm['eventProbability']
        d.variables['dailyeventProbability'][imonth,:,:] = dm['dailyeventProbability']
        d.variables['meanheight'][imonth,:,:] = dm['meanheight']
        d.variables['meanwind'][imonth,:,:] = dm['meanwind']

    #  Done. 

    d.close()
    print( f'Results written to {outputfile}.' )

    return


def main(): 

    parser = argparse.ArgumentParser( prog="compute_diagnostics_climatology", 
            description="Computing a climatology of LLJ diagnostics " + \
            "based on the output of compute_diagnostics (or one of its kin)." )

    parser.add_argument( "dataroot", type=str,
            help='The root data directory for the model output. The diagnostics ' + \
                    'files will be searched in <dataroot>/diagnostics.' )

    parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true", 
            help="Clobber pre-existing output file if it already exists. The default " + \
            "behavior is not to clobber/overwrite." )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true", 
            help="Run in the Python debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    compute_diagnostics_climatology( args.dataroot, clobber=args.clobber )
    pass

