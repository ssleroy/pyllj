"""libcli.py

Author: Stephen Leroy (stephen.leroy@janusresearch.us)
Date: January 26, 2026
Version: 

A module that contains instructions for constructing jobs to be submitted to 
batch, whether AWS Batch or a slurm batch queue. It also builds subparsers to 
aid in constructing those jobs.

This module is built specifically for analyzing the Great Plains low-level jet 
(LLJ) as contained in the North American Regional Reanalysis (NARR), the ECMWF 
5'th edition reanalysis (ERA5), and the Modern Era Retrospective Reanalysis 
version 2 (MERRA-2) as well as in climate models.
"""

import os
import re
import argparse
from netrc import netrc
from datetime import datetime, timedelta
from .parameters import default_dataroot


def earthdataauth(): 
    """Return the username and password for access to NASA Earthdata. The information is read from 
    the user's ~/.netrc file."""

    nc = netrc()
    host = "urs.earthdata.nasa.gov"
    resp = nc.authenticators( host )
    if resp is None: 
        print( f'You must provide information for machine {host} in your ~/.netrc file; it is ' + \
                'necessary to authenticate to NASA Earthdata for MERRA2 downloads.' )
        exit()

    uname, passwd = resp[0], resp[2]
    return uname, passwd


def compute_era5( monthrange:{list,tuple}, diagnostics=False, watervaporflux=False, clobber=False ): 
    """Generate a list of jobs to be submitted to batch for ERA5. 

    monthrange          A two-element tuple or list defining the month range over which 
                        to define jobs; each element must be a string of format "YYYY-MM", 
                        and the range will be inclusive. 

    diagnostics         Set to True if LLJ diagnostics should be computed. 

    watervaporflux      Set to True if column water vapor fluxes should be computed. 

    clobber             Set to true if previously existing output files should be clobbered. 
    """

    #  Get CDS authentication key. 

    path = os.path.join( os.path.expanduser("~"), ".cdsapirc" )
    if not os.path.exists( path ): 
        print( f'Cannot find file {path}, which is needed to authenticate the Copernicus Climate Data Store, the host of ERA5 data.' )
        return
    with open( path, 'r' ) as f:
        lines = f.readlines()
        key = lines[1].strip()[5:]

    #  Collect submittable jobs. 

    jobs = []
    month = monthrange[0]

    while month <= monthrange[1]: 

        #  Compute first and last day of the month and the argument. 

        day = datetime.strptime( month+"-01", "%Y-%m-%d" )
        yearmonth = day.strftime( "%Y%m" )

        if diagnostics: 
            job = [ "compute_era5_diagnostics", month, "-k", key ]
            if clobber: 
                job.append( "--clobber" )
            jobname = "compute_era5_diagnostics_" + month
            jobs.append( { 'command': job, 'name': jobname } )

        if watervaporflux: 
            outputfile = f'watervaporflux.{yearmonth}.nc' 
            job = [ "compute_era5_watervaporflux", month, "-k", key ]
            if clobber: 
                job.append( "--clobber" )
            jobname = "compute_era5_watervaporflux_" + month
            jobs.append( { 'command': job, 'name': jobname } )

        month = ( day + timedelta(days=31) ).strftime("%Y-%m")

    return jobs


def compute_narr( monthrange:{list,tuple}, diagnostics=False, watervaporflux=False, clobber=False ): 
    """Generate a list of jobs to be submitted to batch for NARR, the North American 
    Regional Reanalysis. 

    monthrange          A two-element tuple or list defining the month range over which 
                        to define jobs; each element must be a string of format "YYYY-MM", 
                        and the range will be inclusive. 

    diagnostics         Set to True if LLJ diagnostics should be computed. 

    watervaporflux      Set to True if column water vapor fluxes should be computed. 

    clobber             Set to true if previously existing output files should be clobbered. 
    """

    jobs = []
    month = monthrange[0]

    while month <= monthrange[1]: 

        #  Compute first and last day of the month and the argument. 

        day = datetime.strptime( month+"-01", "%Y-%m-%d" )
        yearmonth = day.strftime( "%Y%m" )

        if diagnostics: 
            job = [ "compute_narr_diagnostics", month ]
            if clobber: 
                job.append( "--clobber" )
            jobname = "compute_narr_diagnostics_" + month
            jobs.append( { 'command': job, 'name': jobname } )

        if watervaporflux: 
            outputfile = f'narr_watervaporflux.{yearmonth}.nc' 
            job = [ "compute_narr_watervaporflux", month ]
            if clobber: 
                job.append( "--clobber" )
            jobname = "compute_narr_watervaporflux_" + month
            jobs.append( { 'command': job, 'name': jobname } )

        month = ( day + timedelta(days=31) ).strftime("%Y-%m")

    return jobs


def build_subparsers( parser:argparse.ArgumentParser ): 
    """Build subparsers onto command line parser. These subparsers enable the specification 
    of many programs that can be run in batch mode, whether AWS Batch or Slurm batch queue. 
    The input parser must be an instance of argparse.ArgumentParser, and the full command 
    line parser is returned. """

    #  Account for dataroot. 

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
                default=default_dataroot,
                help='The root directory where the LLJ files are stored and where ' + \
                    f'results will be written. The default is {default_dataroot}.' )

    subparsers = parser.add_subparsers( dest="command" )

    #  Diagnostic analysis with NARR. 

    narr_diagnostics_parser = subparsers.add_parser( "narr_diagnostics", help='Compute the ' + \
            'diagnostics of the Great Plains low-level jet, including frequency of ' + \
            'occurrence, wind speed, etc., based on the NARR reanalysis.' )

    narr_diagnostics_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    #  Column mass and water vapor flux analysis with NARR. 

    narr_watervaporflux_parser = subparsers.add_parser( "narr_watervaporflux", help='Compute the ' + \
            'column water vapor, column mass flux, and column water vapor flux ' + \
            'over the region of NARR; a subgroup considers the planetary boundary ' + \
            'layer only.' )

    narr_watervaporflux_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    #  Column mass and water vapor flux analysis with NARR. 

    narr_climatology_parser = subparsers.add_parser( "narr_climatology", help='Compute the monthly ' + \
            'climatology of water vapor fluxes (and masss fluxes and column ' + \
            'water vapor) over a range of years, based on the output of ' + \
            'narr_watervaporflux.' )

    narr_climatology_parser.add_argument( "yearrange", type=str, help='The range ' + \
            'of years over which to compute, format "YYYY:YYYY", and the ' + \
            'range is inclusive' )

    #  Flux of water vapor across a boundary. 

    narr_boundaryflux_parser = subparsers.add_parser( "narr_boundaryflux", help='Compute the flux ' + \
            'of water vapor across a specified boundary for a given time range, based on the NARR reanalysis.' )

    narr_boundaryflux_parser.add_argument( "boundary", type=str, help='The name ' + \
            'of the boundary over which to compute the water vapor flux' )

    narr_boundaryflux_parser.add_argument( "dayrange", type=str, help='The range ' + \
            'of days over which to compute, format "YYYY-MM-DD:YYYY-MM-DD", and the ' + \
            'range is inclusive' )

    #  Isohypsic analysis (height above the surface) of horizontal winds. 

    narr_isohypses_parser = subparsers.add_parser( "narr_isohypses", help='Compute the diurnal ' + \
            'cycle of horizontal winds for each month at fixed height levels above the surface, ' + \
            'based on the NARR reanalysis.' )

    narr_isohypses_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    #  Climatology of isohypsic analyses (height above the surface) of horizontal winds. 

    narr_isohypses_climatology_parser = subparsers.add_parser( "narr_isohypses_climatology", 
            help='Compute the diurnal cycle of horizontal winds for each month at fixed ' + \
            'height levels above the surface, based on the NARR reanalysis.' )

    narr_isohypses_climatology_parser.add_argument( "yearrange", type=str, help='The range ' + \
            'of years over which to compute, format "YYYY:YYYY", and the ' + \
            'range is inclusive' )

    #  Download ERA5. 

    era5_download_parser = subparsers.add_parser( "era5_download", help='Download files ' + \
            'needed for LLJ diagnostics and climatology research.' )

    era5_download_parser.add_argument( "yearrange", type=str, help='The range ' + \
            'of years over which to compute, format "YYYY:YYYY", and the ' + \
            'range is inclusive' )

    #  Diagnostic analysis with ERA5. 

    era5_diagnostics_parser = subparsers.add_parser( "era5_diagnostics", help='Compute the ' + \
            'diagnostics of the Great Plains low-level jet, including frequency of ' + \
            'occurrence, wind speed, etc., based on the ERA5 reanalysis.' )

    era5_diagnostics_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    era5_diagnostics_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Column mass and water vapor flux analysis with ERA5. 

    era5_watervaporflux_parser = subparsers.add_parser( "era5_watervaporflux", help='Compute the ' + \
            'column water vapor, column mass flux, and column water vapor flux ' + \
            'over the region of ERA5; a subgroup considers the planetary boundary ' + \
            'layer only.' )

    era5_watervaporflux_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    era5_watervaporflux_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Column mass and water vapor flux analysis with ERA5. 

    era5_climatology_parser = subparsers.add_parser( "era5_climatology", help='Compute the monthly ' + \
            'climatology of water vapor fluxes (and masss fluxes and column ' + \
            'water vapor) over a range of years, based on the output of ' + \
            'era5_watervaporflux.' )

    era5_climatology_parser.add_argument( "yearrange", type=str, help='The range ' + \
            'of years over which to compute, format "YYYY:YYYY", and the ' + \
            'range is inclusive' )

    #  Flux of water vapor across a boundary. 

    era5_boundaryflux_parser = subparsers.add_parser( "era5_boundaryflux", help='Compute the flux ' + \
            'of water vapor across a specified boundary for a given time range, based on the ERA5 reanalysis.' )

    era5_boundaryflux_parser.add_argument( "boundary", type=str, help='The name ' + \
            'of the boundary over which to compute the water vapor flux' )

    era5_boundaryflux_parser.add_argument( "dayrange", type=str, help='The range ' + \
            'of days over which to compute, format "YYYY-MM-DD:YYYY-MM-DD", and the ' + \
            'range is inclusive' )

    #  Isohypsic analysis (height above the surface) of horizontal winds. 

    era5_isohypses_parser = subparsers.add_parser( "era5_isohypses", help='Compute the diurnal ' + \
            'cycle of horizontal winds for each month at fixed height levels above the surface, ' + \
            'based on the ERA5 reanalysis.' )

    era5_isohypses_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    era5_isohypses_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Climatology of isohypsic analyses (height above the surface) of horizontal winds. 

    era5_isohypses_climatology_parser = subparsers.add_parser( "era5_isohypses_climatology", 
            help='Compute the diurnal cycle of horizontal winds for each month at fixed ' + \
            'height levels above the surface, based on the ERA5 reanalysis.' )

    era5_isohypses_climatology_parser.add_argument( "yearrange", type=str, help='The range ' + \
            'of years over which to compute, format "YYYY:YYYY", and the ' + \
            'range is inclusive' )

    #  Download MERRA2. 

    merra2_download_parser = subparsers.add_parser( "merra2_download", help='Download MERRA2 ' + \
            'meteorological fields relevant to the Great Plains low-level jet.' )

    merra2_download_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    merra2_download_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Diagnostic analysis with MERRA2. 

    merra2_diagnostics_parser = subparsers.add_parser( "merra2_diagnostics", help='Compute the ' + \
            'diagnostics of the Great Plains low-level jet, including frequency of ' + \
            'occurrence, wind speed, etc., based on the MERRA2 reanalysis.' )

    merra2_diagnostics_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    merra2_diagnostics_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Column mass and water vapor flux analysis with NARR. 

    merra2_watervaporflux_parser = subparsers.add_parser( "merra2_watervaporflux", help='Compute the ' + \
            'column water vapor, column mass flux, and column water vapor flux ' + \
            'based on the MERRA2 reanalysis; a subgroup considers the planetary boundary ' + \
            'layer only.' )

    merra2_watervaporflux_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    merra2_watervaporflux_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Column mass and water vapor flux analysis with MERRA2. 

    merra2_climatology_parser = subparsers.add_parser( "merra2_climatology", help='Compute the monthly ' + \
            'climatology of water vapor fluxes (and masss fluxes and column ' + \
            'water vapor) over a range of years, based on the output of ' + \
            'merra2_watervaporflux.' )

    merra2_climatology_parser.add_argument( "yearrange", type=str, help='The range ' + \
            'of years over which to compute, format "YYYY:YYYY", and the ' + \
            'range is inclusive' )

    #  Flux of water vapor across a boundary. 

    merra2_boundaryflux_parser = subparsers.add_parser( "merra2_boundaryflux", help='Compute the flux ' + \
            'of water vapor across a specified boundary for a given time range, based on the MERRA2 reanalysis.' )

    merra2_boundaryflux_parser.add_argument( "boundary", type=str, help='The name ' + \
            'of the boundary over which to compute the water vapor flux' )

    merra2_boundaryflux_parser.add_argument( "dayrange", type=str, help='The range ' + \
            'of days over which to compute, format "YYYY-MM-DD:YYYY-MM-DD", and the ' + \
            'range is inclusive' )

    #  Isohypsic analysis (height above the surface) of horizontal winds. 

    merra2_isohypses_parser = subparsers.add_parser( "merra2_isohypses", help='Compute the diurnal ' + \
            'cycle of horizontal winds for each month at fixed height levels above the surface, ' + \
            'based on the MERRA2 reanalysis.' )

    merra2_isohypses_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    merra2_isohypses_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Climatology of isohypsic analyses (height above the surface) of horizontal winds. 

    merra2_isohypses_climatology_parser = subparsers.add_parser( "merra2_isohypses_climatology", 
            help='Compute the diurnal cycle of horizontal winds for each month at fixed ' + \
            'height levels above the surface, based on the MERRA2 reanalysis.' )

    merra2_isohypses_climatology_parser.add_argument( "yearrange", type=str, help='The range ' + \
            'of years over which to compute, format "YYYY:YYYY", and the ' + \
            'range is inclusive' )

    #  Diagnostic analysis for POD. 

    pod_diagnostics_parser = subparsers.add_parser( "pod_diagnostics", help='Compute the ' + \
            'diagnostics of the Great Plains low-level jet, including frequency of ' + \
            'occurrence, wind speed, etc., based on the climate model output. It is ' + \
            'required to specify the dataroot.' )

    pod_diagnostics_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    pod_diagnostics_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Column mass and water vapor flux analysis for POD. 

    pod_watervaporflux_parser = subparsers.add_parser( "pod_watervaporflux", help='Compute the ' + \
            'column water vapor, column mass flux, and column water vapor flux ' + \
            'from model output; a subgroup considers the planetary boundary ' + \
            'layer only.' )

    pod_watervaporflux_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    pod_watervaporflux_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Isohypsic analysis (height above the surface) of horizontal winds. 

    pod_isohypses_parser = subparsers.add_parser( "pod_isohypses", help='Compute the diurnal ' + \
            'cycle of horizontal winds for each month at fixed height levels above the surface, ' + \
            'based on model output.  It is required to specify the dataroot.' )

    pod_isohypses_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    pod_isohypses_parser.add_argument( "-c", "--clobber", dest="clobber", default=False, action="store_true",
            help='Clobber pre-existing output files; no clobbering by default' )

    #  Process TROPICS coverage. 

    tropics_coverage_parser = subparsers.add_parser( "tropics_coverage",
            help="Scan TROPICS L2B neural network retrieval files for soundings over Texas and record their geolocations" )

    tropics_coverage_parser.add_argument( "monthrange", type=str, help='The range ' + \
            'of months over which to compute, format "YYYY-MM:YYYY-MM", and the ' + \
            'range is inclusive' )

    #  Successful completion. 

    return parser


def process( args:argparse.Namespace, command ): 
    """Process the input specified by the command line interface.

    args        The object returned by argparse.ArgumentParser.parse_args. 

    command     A method that submits an individual job to batch, whether AWS Batch 
                or Slurm batch. The method takes two required arguments and one 
                optional: 

                    (a) The command as a parsed list of command line arguments to 
                    be run in batch; 

                    (b) The name of the job; 

                    (c) Optional "execute", set to False if the job should only be 
                    specified but not submitted. """ 

    #  Parse the time range. Sometimes it's a day range; sometimes a month range; 
    #  sometimes a year range. 

    if args.command in [ 'narr_boundaryflux', 'era5_boundaryflux', 'merra2_boundaryflux' ]: 

        #  Day range...

        m = re.search( r'^(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})$', args.dayrange )
        if m: 
            dayrange = [ m.group(1), m.group(2) ]
        else: 
            print( 'Be sure that the dayrange has format "YYYY-MM-DD:YYYY-MM-DD"' )
            return

        if dayrange[0] > dayrange[1]: 
            print( 'Be sure that the yearrange has a first value that is less than or equal to the second value' )
            return

    elif args.command in [ \
            'narr_diagnostics', 'narr_watervaporflux', 'narr_isohypses', \
            'era5_diagnostics', 'era5_watervaporflux', 'era5_isohypses', \
            'merra2_download', 'merra2_diagnostics', 'merra2_watervaporflux', 'merra2_isohypses', \
            'pod_diagnostics', 'pod_watervaporflux', 'pod_isohypses', \
            'tropics_coverage' ]: 

        #  Month range...

        m = re.search( r'^(\d{4}-\d{2}):(\d{4}-\d{2})$', args.monthrange )
        if m: 
            monthrange = [ m.group(1), m.group(2) ]
        else: 
            print( 'Be sure that the monthrange has format "YYYY-MM:YYYY-MM"' )
            return

        if monthrange[0] > monthrange[1]: 
            print( 'Be sure that the monthrange has a first value that is less than or equal to the second value' )
            return

    elif args.command in [ 'narr_climatology', 'narr_isohypses_climatology', \
            'era5_download', 'era5_climatology', 'era5_isohypses_climatology', \
            'merra2_climatology', 'merra2_isohypses_climatology' ]: 

        #  Year range...

        m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
        if m: 
            yearrange = [ int(m.group(1)), int(m.group(2)) ]
        else: 
            print( 'Be sure that the yearrange has format "YYYY:YYYY"' )
            return

        if yearrange[0] > yearrange[1]: 
            print( 'Be sure that the yearrange has a first value that is less than or equal to the second value' )
            return


    #  Process according to command. 

    #  NARR processing...

    if args.command == "narr_diagnostics": 
        jobs = compute_narr( monthrange, diagnostics=True, watervaporflux=False )
        for job in jobs: 
            command( job['command'], job['name'], dataroot=args.dataroot, execute=True )

    elif args.command == "narr_watervaporflux": 
        jobs = compute_narr( monthrange, diagnostics=False, watervaporflux=True )
        for job in jobs: 
            command( job['command'], job['name'], dataroot=args.dataroot, execute=True )

    elif args.command == "narr_climatology": 
        job = [ 'compute_narr_climatology', args.yearrange ]
        # if args.clobber: job.append( "--clobber" )
        jobName = "compute-narr-climatology"
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "narr_boundaryflux": 
        job = [ 'compute_narr_boundary_watervaporflux', args.boundary, args.dayrange ]
        jobName = f'narr_boundaryflux_{dayrange[0]}_{dayrange[1]}'
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "narr_isohypses": 

        dt = datetime.fromisoformat( monthrange[0]+"-01" )
        dt1 = datetime.fromisoformat( monthrange[1]+"-01" )

        while dt <= dt1: 

            ss = f'{dt.year:4d}-{dt.month:02d}'
            job = [ 'compute_narr_isohypses', ss ]
            jobName = f"compute-narr-isohypses-{ss}"
            command( job, jobName, dataroot=args.dataroot, execute=True )

            dt += timedelta( days=31 )
            dt = datetime( year=dt.year, month=dt.month, day=1 )

    elif args.command == "narr_isohypses_climatology": 
        job = [ 'compute_narr_isohypses_climatology', args.yearrange ]
        jobName = "compute-narr-isohypses-climatology" + str( args.yearrange ).replace( ":", "_" )
        command( job, jobName, dataroot=args.dataroot, execute=True )

    #  ERA5 processing...

    elif args.command == "era5_download": 
        path = os.path.join( os.path.expanduser("~"), ".cdsapirc" )
        if not os.path.exists( path ): 
            print( f'Cannot find file {path}, which is needed to authenticate the Copernicus Climate Data Store, the host of ERA5 data.' )
            exit()
        with open( path, 'r' ) as f:
            lines = f.readlines()
            key = lines[1].strip()[5:]

        job = [ 'download_era5', '{:4d}-01:{:4d}-12'.format( *yearrange ), '-k', key ]
        jobname = 'download_era5_{:}-{:}'.format( *yearrange )
        ret = command( job, jobname, dataroot=args.dataroot, execute=True )

    elif args.command == "era5_diagnostics": 
        jobs = compute_era5( monthrange, diagnostics=True, watervaporflux=False )
        for job in jobs: 
            if args.clobber: job['command'].append( "--clobber" )
            command( job['command'], job['name'], dataroot=args.dataroot, execute=True )

    elif args.command == "era5_watervaporflux": 
        jobs = compute_era5( monthrange, diagnostics=False, watervaporflux=True )
        for job in jobs: 
            if args.clobber: job['command'].append( "--clobber" )
            command( job['command'], job['name'], dataroot=args.dataroot, execute=True )

    elif args.command == "era5_climatology": 
        job = [ 'compute_era5_climatology', args.yearrange ]
        # if args.clobber: job.append( "--clobber" )
        jobName = "compute-era5-climatology"
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "era5_boundaryflux": 
        job = [ 'compute_era5_boundary_watervaporflux', args.boundary, args.dayrange ]
        jobName = f'era5_boundaryflux_{dayrange[0]}_{dayrange[1]}'
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "era5_isohypses": 

        dt = datetime.fromisoformat( monthrange[0]+"-01" )
        dt1 = datetime.fromisoformat( monthrange[1]+"-01" )
        u, p = earthdataauth()

        while dt <= dt1: 

            ss = f'{dt.year:4d}-{dt.month:02d}'
            job = [ 'compute_era5_isohypses', ss, "--auth", f'{u} {p}' ]
            if args.clobber: job.append( "--clobber" )
            jobName = f"compute-era5-isohypses-{ss}"
            command( job, jobName, dataroot=args.dataroot, execute=True )

            dt += timedelta( days=31 )
            dt = datetime( year=dt.year, month=dt.month, day=1 )

    elif args.command == "era5_isohypses_climatology": 
        job = [ 'compute_era5_isohypses_climatology', args.yearrange ]
        jobName = "compute-era5-isohypses-climatology" + str( args.yearrange ).replace( ":", "_" )
        command( job, jobName, dataroot=args.dataroot, execute=True )

    #  MERRA2 processing...

    elif args.command == "merra2_download": 
        u, p = earthdataauth()
        job = [ 'download_merra2', args.monthrange, "--auth", f'{u} {p}' ]
        if args.clobber: job.append( "--clobber" )
        jobName = "merra2_download_" + str( args.monthrange ).replace( ":", "_" )
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "merra2_diagnostics": 

        dt = datetime.fromisoformat( monthrange[0]+"-01" )
        dt1 = datetime.fromisoformat( monthrange[1]+"-01" )
        u, p = earthdataauth()

        while dt <= dt1: 
            ss = f'{dt.year:4d}-{dt.month:02d}'
            job = [ 'compute_merra2_diagnostics', ss, "--auth", f'{u} {p}' ]
            if args.clobber: job.append( "--clobber" )
            jobName = f'merra2_diagnostics_{ss}'
            command( job, jobName, dataroot=args.dataroot, execute=True )

            dt += timedelta( days=31 )
            dt = datetime( year=dt.year, month=dt.month, day=1 )

    elif args.command == "merra2_climatology": 
        job = [ 'compute_merra2_climatology', args.yearrange ]
        # if args.clobber: job.append( "--clobber" )
        jobName = "compute_merra2_climatology"
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "merra2_watervaporflux": 

        u, p = earthdataauth()
        job = [ 'compute_merra2_watervaporflux', args.monthrange, "--auth", f'{u} {p}' ]
        if args.clobber: job.append( "--clobber" )
        jobName = 'compute_merra2_watervaporflux_' + str( args.monthrange ).replace( ":", "_" )
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "merra2_boundaryflux": 
        job = [ 'compute_merra2_boundary_watervaporflux', args.boundary, args.dayrange ]
        jobName = f'merra2_boundaryflux_{dayrange[0]}_{dayrange[1]}'
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "merra2_isohypses": 

        dt = datetime.fromisoformat( monthrange[0]+"-01" )
        dt1 = datetime.fromisoformat( monthrange[1]+"-01" )
        u, p = earthdataauth()

        while dt <= dt1: 

            ss = f'{dt.year:4d}-{dt.month:02d}'
            job = [ 'compute_merra2_isohypses', ss, "--auth", f'{u} {p}' ]
            if args.clobber: job.append( "--clobber" )
            jobName = f"compute_merra2_isohypses_{ss}"
            command( job, jobName, dataroot=args.dataroot, execute=True )

            dt += timedelta( days=31 )
            dt = datetime( year=dt.year, month=dt.month, day=1 )

    elif args.command == "merra2_isohypses_climatology": 
        job = [ 'compute_merra2_isohypses_climatology', args.yearrange ]
        jobName = "compute-merra2-isohypses-climatology" + str( args.yearrange ).replace( ":", "_" )
        command( job, jobName, dataroot=args.dataroot, execute=True )

    elif args.command == "pod_diagnostics": 

        dt = datetime.fromisoformat( monthrange[0]+"-01" )
        dt1 = datetime.fromisoformat( monthrange[1]+"-01" )

        while dt <= dt1: 
            ss = f'{dt.year:4d}-{dt.month:02d}'
            job = [ 'compute_diagnostics', ss, args.dataroot ]
            if args.clobber: job.append( "--clobber" )
            jobName = f'compute_diagnostics_{ss}'
            command( job, jobName, execute=True )

            dt += timedelta( days=31 )
            dt = datetime( year=dt.year, month=dt.month, day=1 )

    elif args.command == "pod_watervaporflux": 

        dt = datetime.fromisoformat( monthrange[0]+"-01" )
        dt1 = datetime.fromisoformat( monthrange[1]+"-01" )

        while dt <= dt1: 
            ss = f'{dt.year:4d}-{dt.month:02d}'
            job = [ 'compute_watervaporflux', ss, args.dataroot ]
            if args.clobber: job.append( "--clobber" )
            jobName = f'compute_watervaporflux_{ss}'
            command( job, jobName, execute=True )

            dt += timedelta( days=31 )
            dt = datetime( year=dt.year, month=dt.month, day=1 )

    elif args.command == "pod_isohypses": 

        dt = datetime.fromisoformat( monthrange[0]+"-01" )
        dt1 = datetime.fromisoformat( monthrange[1]+"-01" )

        while dt <= dt1: 

            ss = f'{dt.year:4d}-{dt.month:02d}'
            job = [ 'compute_isohypses', ss, args.dataroot ]
            if args.clobber: job.append( "--clobber" )
            jobName = f"compute_isohypses_{ss}"
            command( job, jobName, execute=True )

            dt += timedelta( days=31 )
            dt = datetime( year=dt.year, month=dt.month, day=1 )

    elif args.command == "tropics_coverage": 

        yearmonth = monthrange[0]
        while yearmonth <= monthrange[1]: 

            dt0 = datetime.strptime( yearmonth+"-01", "%Y-%m-%d" )
            dt = dt0 + timedelta(days=31)
            dt = datetime( dt.year, dt.month, 1 ) 
            dt1 = dt - timedelta(days=1)
            datetimerange = dt0.strftime("%Y-%m-%d") + ":" + dt1.strftime("%Y-%m-%d")
            outfile = '/fg/Data/TROPICS/coverage.{:}.nc'.format( dt0.strftime("%Y-%m") )
            job = [ 'tropics_coverage', datetimerange, outfile ]
            jobName = 'tropics_coverage-' + dt0.strftime("%Y-%m")
            command( job, jobName, dataroot=args.dataroot, execute=True )
            yearmonth = dt.strftime("%Y-%m")

    return


if __name__ == "__main__": 
    main()
    pass

