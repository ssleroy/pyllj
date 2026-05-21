import os
import re
import sys
from time import time 
from tqdm import tqdm 
from datetime import datetime, timedelta, timezone
import argparse
import numpy as np
from netCDF4 import Dataset 
from .libpod import RetClass, epoch, ModelOutput 
from ..libutils import GridInterpolator 
from ..parameters import boundaries, Rearth

watervaporflux_subdir = "watervaporflux"
output_subdir = "boundaryflux"
fill_float = -1.0e20


class Error( Exception ): 
    pass


def boundary_watervaporflux( boundary:str, daterange:{tuple,list}, dataroot:str ): 
    """Compute column water vapor flux across a boundary. 

    Arguments
    =========

    boundary        The boundary across which the computations should be performed; must be 
                    a key in the 'boundaries' dictionary. 

    daterange       A two-element tuple or list containing two instances of datetimes 
                    defining the time range over which computations should be done. 

    dataroot        The root of the model run. 
    """

    t0 = time()
    ret = RetClass()

    #  Check input. 

    if boundary not in boundaries.keys(): 
        ret.update( success=False, messages="InvalidBoundary", \
                comments=f'Boundary "{boundary}" not a valid value. Availables boundaries are ' + \
                ", ".join( sorted( list( boundaries.keys() ) ) ) + "." )
        return ret

    #  Date range. 

    d1, d2 = daterange[0], daterange[1]

    #  Set parameters, etc. 

    model = ModelOutput( dataroot )
    ncd = int( timedelta(hours=24) / model.tdelta + 0.001 )

    #  Establish lines of longitude, latitude along the boundaries. Keep track of normal 
    #  directions for each segment. 

    resolution = 1.0e3          #  Maximum segement length [meters]. 

    slons = np.deg2rad( boundaries[boundary]['lons'] )
    slats = np.deg2rad( boundaries[boundary]['lats'] )

    #  Wrap around date line if necessary. 

    rlon = slons[ int(slons.size/2) ]       #  Reference longitude. 
    dlons = slons - rlon
    slons = rlon + np.arctan2( np.sin(dlons), np.cos(dlons) )

    #  Compute interval position vectors (3D). 

    pvs = np.array( [ np.cos(slons) * np.cos(slats), np.sin(slons) * np.cos(slats), np.sin(slats) ] ).T 

    #  Define segment endpoints. 

    lons, lats, normals = [], [], []

    for interval in range( slons.size - 1 ): 
        ds = 2 * Rearth * np.arcsin( np.linalg.norm( pvs[interval+1,:] - pvs[interval,:] ) / 2 )
        ns = max( 2, int( ds/resolution ) )
        dlon = slons[interval+1] - slons[interval]
        dlat = slats[interval+1] - slats[interval]

        t = np.arange(ns) / ns
        lons += [ slons[interval] + tt*dlon for tt in t ]
        lats += [ slats[interval] + tt*dlat for tt in t ]

    lons.append( slons[-1] )
    lats.append( slats[-1] )

    lons = np.array( lons )
    lats = np.array( lats )

    #  Define mid-points (midlons, midlats) and normal vectors (dx). 

    midlons = 0.5 * ( lons[1:] + lons[:-1] )
    midlats = 0.5 * ( lats[1:] + lats[:-1] )

    dlons = lons[1:] - lons[:-1]
    dlats = lats[1:] - lats[:-1]

    dx = np.array( [ -dlats * Rearth, dlons * Rearth * np.cos(midlats) ] )
    dxa = np.sqrt( ( dx * dx ).sum(axis=0) )

    #  Convert to degrees. 

    midlons = np.rad2deg( midlons )
    midlats = np.rad2deg( midlats )
    lons = np.rad2deg( lons )
    lats = np.rad2deg( lats )

    #  Commence loop over day. 

    ntimes = int( (d2-d1+timedelta(days=1)) / model.tdelta + 0.001 )
    times = []
    yearmonth, proj = None, None
    gappy_months = set()

    for itime in tqdm( range(ntimes), desc="  Time" ): 
        dtroot = d1 + itime * model.tdelta 
        dt = dtroot + model.toffset
        times.append( dt )

        if dtroot.strftime( "%Y%m" ) != yearmonth: 

            #  Close and remove previous file. 

            if yearmonth is not None: 
                d.close()
            yearmonth = dtroot.strftime( "%Y%m" )

            #  Download (if necessary) and open file. 

            try: 
                path = os.path.join( dataroot, watervaporflux_subdir )
                file = [ os.path.join( path, f ) for f in os.listdir( path ) \
                        if re.search( r'watervaporflux.' + yearmonth + r'.nc$', f ) ][0]
                d = Dataset( file, 'r' )
                v = d.variables['time']
                m = re.search( r'^(\w+) since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$', v.getncattr( "units" ) )
                time_units, epoch = m.group(1), datetime.strptime( m.group(2), "%Y-%m-%d %H:%M:%S" )
                file_times = [ epoch + timedelta( **{ time_units: int(t) } ) for t in v[:] ]
            except: 
                ret.update( success=False, messages="InvalidFile", comments=f'File {file} does not exist or is not a NetCDF file' )
                return ret

            #  Get groups and their attributes.

            if itime == 0:
                groups = {}
                for groupname, group in d.groups.items():
                    atts = { attrname: group.getncattr( attrname ) for attrname in group.ncattrs() }
                    groups[groupname] = {
                            'atts': atts,
                            'massflux': np.zeros( ntimes, np.float32 ),
                            'watervaporflux': np.zeros( ntimes, np.float32 ),
                            'columnwatervapor': np.zeros( ntimes, np.float32 ) }

                    #  Get pbl_depth info if available. 

                    if "pbl_depth" in group.variables.keys(): 
                        v = group.variables['pbl_depth']
                        groups[groupname]['pbl_depth'] = {
                                     'atts': { key: v.getncattr(key) for key in v.ncattrs() }, 
                                     'dtype': v.dtype, 
                                     'value': v[:] }
                    else: 
                        groups[groupname]['pbl_depth'] = None

            if proj is None: 

                #  Get longitude latitude grid and define interpolator. 

                mlons = d.variables['longitude'][:]
                mlats = d.variables['latitude'][:]
                intp = GridInterpolator( mlons, mlats, midlons, midlats )
                proj = "model"

        #  Time index. 

        ii = [ i for i, file_time in enumerate(file_times) if file_time == dt ]
        if len( ii ) == 0: 
            gappy_months.add( dt.strftime( "%Y-%m" ) )
            continue
        else: 
            i = ii[0]

        #  Get eastward (u) and northward (v) components of flux for time interval. Also 
        #  get water vapor fluxes. 

        for groupname, g in d.groups.items():
            og = groups[groupname]          #  Output group

            uf = g.variables['uf'][i,:,:]
            vf = g.variables['vf'][i,:,:]
            uwvf = g.variables['uwvf'][i,:,:]
            vwvf = g.variables['vwvf'][i,:,:]
            cwv = g.variables['cwv'][i,:,:]

            #  Interpolate along boundary/coastline .

            ufi = intp( uf )
            vfi = intp( vf )
            uwvfi = intp( uwvf )
            vwvfi = intp( vwvf )
            cwvi = intp( cwv )

            #  Integrate along boundary.

            flux = np.array( [ ufi, vfi ] )
            wvflux = np.array( [ uwvfi, vwvfi ] )
            massflux = ( dx * flux ).sum()
            watervaporflux = ( dx * wvflux ).sum()
            columnwatervapor = ( dxa * cwvi ).sum() / dxa.sum()

            #  Store result.

            og['massflux'][itime] = massflux
            og['watervaporflux'][itime] = watervaporflux
            og['columnwatervapor'][itime] = columnwatervapor

    d.close()

    #  Write results to output.

    outputpath = os.path.join( dataroot, output_subdir, f"boundaryflux.{boundary}.nc" )
    print( f'Writing to {outputpath}' )
    sys.stdout.flush()

    dirname = os.path.dirname( outputpath )
    if dirname != "": 
        os.makedirs( dirname, exist_ok=True )
    d = Dataset( outputpath, 'w', format="NETCDF4" )

    #  Dimensions. 

    d.createDimension( "time" )
    d.createDimension( "vertex", slons.size )

    #  Variables. 

    var = "longitude"
    v = d.createVariable( var, "f4", ("vertex",) )
    v.setncatts( { 
            'description': "The longitudes of the vertices that define the coast-following " + \
                "line segment across which total water flux is computed", 
            'units': "degrees east"
        } )

    var = "latitude"
    v = d.createVariable( var, "f4", ("vertex",) )
    v.setncatts( { 
            'description': "The latitudes of the vertices that define the coast-following " + \
                "line segment across which total water flux is computed", 
            'units': "degrees north"
        } )

    var = "time"
    v = d.createVariable( var, np.int32, ("time",) )
    v.setncatts( { 'description': "The time corresponding to the data (UTC)", 
                  'units': "hours since " + epoch.strftime( "%Y-%m-%d %H:%M:%S" ) } )

    #  Groups. 

    for groupname, group in groups.items():

        g = d.createGroup( groupname )

        #  Group attributes. 

        g.setncatts( group['atts'] )

        var = "massflux"
        v = g.createVariable( var, "f4", ("time",) )
        v.setncatts( {
                'description': "Total atmospheric mass flux crossing the boundary " + \
                            "defined by the longitudes/latitudes vertices",
                'units': "kg/s"
            } )

        var = "watervaporflux"
        v = g.createVariable( var, "f4", ("time",) )
        v.setncatts( {
                'description': "Total mass flux of water vapor crossing the boundary " + \
                            "defined by the longitudes/latitudes vertices",
                'units': "kg/s"
            } )

        var = "columnwatervapor"
        v = g.createVariable( var, "f4", ("time",) )
        v.setncatts( {
                'description': "Column-integrated water vapor averaged along the boundary " + \
                            "defined by the longitudes/latitudes vertices",
                'units': "kg/m**2"
            } )

        if group['pbl_depth'] is not None: 
            var = "pbl_depth"
            v = g.createVariable( var, group['pbl_depth']['dtype'] )
            v.setncatts( group['pbl_depth']['atts'] )
            v[:] = group['pbl_depth']['value']

    #  Global attributes. 

    d.setncatts( {
            'file_type': "model-boundary_watervaporflux", 
            'description': "This file contains the flux of total column water vapor across a " + \
                "boundary defined by longitude and latitude vertices. The input is taken from " + \
                "a run of a climate/atmospheric model.", 
            'dataroot': dataroot, 
            'date_range': [ r.strftime("%Y-%m-%d") for r in daterange ], 
            'author': "Stephen Leroy (stephen.leroy@janusresearch.us)", 
            'creation_time': datetime.now( tz=timezone.utc ).strftime( "%d %b %Y %H:%M:%S UTC" ) } )

    #  Write data values. 

    d.variables['longitude'][:] = np.rad2deg( np.arctan2( np.sin(slons), np.cos(slons) ) )
    d.variables['latitude'][:] = np.rad2deg( slats )
    d.variables['time'][:] = [ np.int32( ( t - epoch ) / timedelta(hours=1) + 0.001 ) for t in times ]

    for groupname in groups.keys():
        g = groups[groupname]
        og = d.groups[groupname]

        og.variables['massflux'][:] = np.ma.masked_values( g['massflux'], fill_float )
        og.variables['watervaporflux'][:] = np.ma.masked_values( g['watervaporflux'], fill_float )
        og.variables['columnwatervapor'][:] = np.ma.masked_values( g['columnwatervapor'], fill_float )

    d.close()

    if len( gappy_months ) > 0: 
        ret.update( comments='Months with gaps in them are '+", ".join( sorted( list( gappy_months ) ) ) )

    t1 = time()
    dt = t1 - t0
    hours, minutes, seconds = int(dt/3600), int(dt/60) % 60, int(dt) % 60
    ret.update( success=True, comments=f'Total elapsed time = {hours:d} hrs, {minutes:2d} mins, {seconds:2d} secs' )

    return ret


def main(): 

    parser = argparse.ArgumentParser( prog="compute_boundarywatervaporflux", 
            description="""Compute column-integrated water vapor and the component 
            column-integrated water flux over the entire domain of a run of a 
            climate/atmospheric model. Column-integrated water vapor is also 
            included in the output.""" )

    parser.add_argument( "boundary", type=str, 
            help="""Specify which boundary across which to compute mass flux of water 
            vapor. Valid values are: """ + ", ".join( [ f'"{b}"' for b in boundaries.keys() ] ) + ". " + \
            "The actual path of the boundary will be written into the output file." )

    parser.add_argument( "daterange", type=str, 
            help="""A string defining the date range over which to compute column-integrated 
            water vapor flux. The string is composed of two dates of the form 
            "YYYY-MM-DD:YYYY-MM-DD", and the range is inclusive.""" )

    parser.add_argument( "modelroot", type=str,
            help="""The root directory of the model run.""" )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true",
            help="Use this option to enter the Python line debugger" )

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    #  Execute. 

    m = re.search( r'^(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})$', args.daterange )
    if m: 
        daterange = ( datetime.fromisoformat(m.group(1)), datetime.fromisoformat(m.group(2)) )
        ret = boundary_watervaporflux( args.boundary, daterange, args.modelroot )
        print( ret )
    else: 
        print( 'The daterange argument must have format "YYYY-MM-DD:YYYY-MM-DD".' )


    return

if __name__ == "__main__": 
    main()
    pass

