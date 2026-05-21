import os
import re
import sys
import argparse
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta, timezone
from time import time

#  Number of model cycles per day. 

output_subdir = "watervaporflux"


def compute_climatology( yearrange:{tuple,list}, dataroot:str ): 
    """Compute the annual and diurnal cycles of atmosphere column mass 
    flux, column water vapor, and column water vapor flux by month-of-year 
    based on the output of compute_watervaporflux. 

    Arguments
    =========
    yearrange       A 2-element tuple/list of the begin and end year over which to compute 
                    the climatologies. 

    dataroot        The root of the model run.

    Output is written to "watervaporflux_climatology.nc" in the dataroot/output_subdir 
    directory."""

    #  Initialize. 

    time0 = time()

    #  Test arguments. 

    if len(yearrange) != 2: 
        print( "yearrange must be a two-element list of integers." )
        return

    if not isinstance(yearrange[0],int) or not isinstance(yearrange[1],int): 
        print( "yearrange must be a two-element list of integers." )
        return

    if yearrange[0] > yearrange[1]: 
        print( "The first elements of yearrange must be less than or equal to the second." )
        return

    #  Get a listing of watervaporflux files. 

    output_root = os.path.join( dataroot, output_subdir )
    files = [ os.path.join( output_root, f ) for f in os.listdir( output_root ) if re.search( r'watervaporflux\.\d{6}\.nc$', f ) ] 

    #  Check that there are 12 files for each year. 

    dfiles = {}

    for year in range(yearrange[0],yearrange[1]+1): 
        yfiles = [ f for f in files if re.search( r'watervaporflux\.' + str(year) + r'\d{2}\.nc$', f ) ]

        if len(yfiles) != 12: 
            print( f'Year {year} has only {len(yfiles)} files and should have 12 to be considered.' )
            return -10

        for yfile in yfiles: 
            m = re.search( r'\d{4}(\d{2})\.nc$', yfile ) 
            if not m: continue
            tag = f'{year:4d}-{m.group(1)}'
            dfiles[tag] = yfile

    #  Do processing. Scan over years. 

    out = None
    variables = [ "uf", "vf", "uwvf", "vwvf", "cwv" ]

    print( 'Scanning watervaporflux files' )
    sys.stdout.flush()

    for imonth in range(12): 
        for year in range(yearrange[0],yearrange[1]+1): 

            file = dfiles[f'{year:4d}-{imonth+1:02d}']
            print( "  " + file )
            sys.stdout.flush()
            d = Dataset( file, 'r' )

            #  Get projection info, grid dimensions; create out structure. 

            if out is None: 

                #  Dimensioning variables. 

                nlons = d.dimensions['longitude'].size 
                nlats = d.dimensions['latitude'].size 

                #  Get time interval tdelta and output cadence. 

                v = d.variables['time']
                attr = v.getncattr( "units" )
                m = re.search( r'^(\w+) since', attr )
                tdelta = float( v[1] - v[0] ) * timedelta( **{ m.group(1): 1 } )
                ncd = int( timedelta(hours=24) / tdelta + 0.001 )

                #  Initialize output dictionary. 

                out = { 'attributes': {}, 'select_variables': {}, 'pbl_depth': {}, 
                       'column_climatology': [], 'pbl_climatology': [] } 

                #  Get all info on select variables. 

                for variable_name in [ "longitude", "latitude" ]: 
                    var = d.variables[variable_name]
                    out['select_variables'].update( { variable_name: 
                            { 'dimensions': var.dimensions, 
                              'atts': { key: var.getncattr(key) for key in var.ncattrs() }, 
                              'dtype': var.dtype, 
                              'values': var[:] } 
                        } )

                var = d.groups['pbl']['pbl_depth']
                out['pbl_depth'].update( 
                        { 'dimensions': var.dimensions, 
                          'atts': { key: var.getncattr(key) for key in var.ncattrs() }, 
                          'dtype': var.dtype, 
                          'values': var[:] } ) 

                #  Record attributes of climatology variables. 

                for variable_name in variables: 
                    var = d.groups['column'].variables[variable_name]
                    out['attributes'][variable_name] = { key: var.getncattr(key) for key in var.ncattrs() } 

                #  Initialize climatologies. 

                for i in range(12): 
                    out['column_climatology'].append( 
                            { key: np.zeros( (ncd,nlats,nlons), dtype=np.float32 ) for key in variables } | \
                            { key+"_var": np.zeros( (ncd,nlats,nlons), dtype=np.float32 ) for key in variables } )
                    out['pbl_climatology'].append( 
                            { key: np.zeros( (ncd,nlats,nlons), dtype=np.float32 ) for key in variables } | \
                            { key+"_var": np.zeros( (ncd,nlats,nlons), dtype=np.float32 ) for key in variables } )

            #  Statistics. 

            ndays = int( d.dimensions['time'].size / ncd )

            for vert in [ 'column', 'pbl' ]: 
                clim = out[f'{vert}_climatology'][imonth]
                for key in variables:  
                    clim[key] += d.groups[vert].variables[key][:].reshape( ncd, ndays, nlats, nlons ).mean(axis=1)
                    clim[key+"_var"] += ( d.groups[vert].variables[key][:].reshape( ncd, ndays, nlats, nlons ).mean(axis=1) )**2 

            d.close()

    #  Normalize by number of years. 

    print( 'Computing statistics' )
    sys.stdout.flush()

    nyears = yearrange[1] - yearrange[0] + 1

    for vert in [ 'column', 'pbl' ]: 
        for imonth in range(12): 
            clim = out[f'{vert}_climatology'][imonth]
            for key in variables: 
                clim[key] /= nyears
                clim[key+"_var"] = ( clim[key+"_var"] - clim[key]**2 * nyears ) / ( nyears - 1 )

    #  Write to output. 

    output = os.path.join( dataroot, output_subdir, "watervaporflux.nc" )
    print( f'Writing to {output}' )
    sys.stdout.flush()

    d = Dataset( output, 'w' )

    #  Dimensions. 

    d.createDimension( "longitude", nlons )
    d.createDimension( "latitude", nlats )
    d.createDimension( "zulu", ncd )
    d.createDimension( "month", 12 )

    #  Define select variables. 

    for vname, value in out['select_variables'].items(): 
        v = d.createVariable( vname, value['dtype'], dimensions=value['dimensions'] )
        v.setncatts( value['atts'] )

    x = d.createVariable( "zulu", 'f', dimensions=("zulu",) )
    x.setncatts( { 
            'description': "Zulu/UTC of diurnal cycle", 
            'units': "hours" } )

    #  Define variables. 

    for vert in [ 'column', 'pbl' ]: 
        g = d.createGroup( vert )

        for varname in variables: 
            varname1 = varname + ""
            v = g.createVariable( varname1, 'f', dimensions=("month","zulu","latitude","longitude") )
            desc = out['attributes'][varname]['description']
            description = "Annual cycle, diurnal cycle of " + desc[0].lower() + desc[1:]
            units = out['attributes'][varname]['units']
            v.setncatts( { 'description': description, 'units': units } )

        for varname in variables: 
            varname1 = varname + "_var"
            v = g.createVariable( varname1, 'f', dimensions=("month","zulu","latitude","longitude") )
            desc = out['attributes'][varname]['description']
            description = "Variance of annual cycle, diurnal cycle of " + desc[0].lower() + desc[1:]
            units = "( " + out['attributes'][varname]['units'] + " )**2" 
            v.setncatts( { 'description': description, 'units': units } )

    d.groups['column'].setncatts( { 
            'description': "Computations done for the entire atmospheric column" } )

    d.groups['pbl'].setncatts( { 
            'description': "Computations done for the planetary boundary layer only, " + \
            "defined as the lowest layer of the atmosphere of thickness defined by " + \
            "pbl_depth" } )

    dd = out['pbl_depth']
    x = d.groups['pbl'].createVariable( "pbl_depth", dd['dtype'], dd['dimensions'] )
    x.setncatts( dd['atts'] )

    #  Global attributes. 

    d.setncatts( { 
            'file_type': "model-annual_diurnal_climatology", 
            'dataroot': dataroot, 
            'year_range': ( np.int32(yearrange[0]), np.int32(yearrange[1]) ), 
            'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ), 
            'author': "Stephen Leroy (stephen.leroy@janusresearch.us)" } )

    #  Write data values. 

    d.variables['zulu'][:] = np.arange( 0.0, 24.0, int( 24.0/ncd + 0.001 ) )

    for vname, value in out['select_variables'].items(): 
        d.variables[vname][:] = value['values']

    d.groups['pbl'].variables['pbl_depth'][:] = out['pbl_depth']['values']

    for vert in [ 'column', 'pbl' ]: 

        g = d.groups[vert]

        for varname in variables: 
            for imonth in range(12): 
                g.variables[varname][imonth,:,:,:] = out[f'{vert}_climatology'][imonth][varname][:]

        for varname in variables: 
            varname1 = varname + "_var"
            for imonth in range(12): 
                g.variables[varname1][imonth,:,:,:] = out[f'{vert}_climatology'][imonth][varname1][:]

    #  Done. 

    d.close()

    time1 = time()
    dt = time1 - time0
    hours, minutes, seconds = int(dt/3600), int(dt/60) % 60, int(dt) % 60
    print( f'Time elapsed = {hours:d} hrs, {minutes:2d} mins, {seconds:2d} secs' )
    sys.stdout.flush()

    return


def main(): 

    parser = argparse.ArgumentParser( prog="compute_climatology", description=
            """Compute annual cycle, diurnal cycle climatology of column mass flux, 
            column water vapor flux, and column water vapor content over a specified 
            range of years. The output is written to 
            <dataroot>/watervaporflux/watervaporflux_climatology.nc.""" )

    parser.add_argument( "yearrange", type=str, help="""Inclusive range of years over which 
            to compute annual cycle, diurnal cycle climatologies based on 
            watervaporflux files produced by compute_watervaporflux. The 
            argument must have format "YYYY:YYYY".""" )

    parser.add_argument( "dataroot", type=str,
            help="""The root directory of the model run.""" )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true", 
            help="Use this option to enter the Python line debugger" )

    #  Parse. 

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
    if m: 
        yearrange = ( int(m.group(1)), int(m.group(2)) )
    else: 
        print( 'Year range must have format "YYYY:YYYY".' )
        return

    ret = compute_climatology( yearrange, args.dataroot )


if __name__ == "__main__": 
    main()
    pass

