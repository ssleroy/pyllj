import os
import re
import sys
from tqdm import tqdm 
from datetime import datetime, timedelta, timezone
import argparse
import numpy as np
from netCDF4 import Dataset 
from .libnarr import cycle_time, epoch, RetClass 
from ..parameters import boundaries, default_dataroot, Rearth
from ..libutils import LambertConformalProjection, LambertConformalInterpolator 

watervaporflux_subdir = "NARR/watervaporflux" 
output_subdir = "NARR/boundaryflux"


def boundary_watervaporflux( boundary, daterange:{tuple,list}, dataroot=default_dataroot ): 
    """Compute column water vapor flux across a boundary. 

    Arguments
    =========

    boundary        The boundary across which the computations should be performed; must be 
                    a key in the 'boundaries' dictionary. 

    daterange       A two-element tuple or list containing two instances of datetimes 
                    defining the time range over which computations should be done. 

    dataroot        The root of all LLJ research data, by default /fg/Data, pointing to the
                    FileGateway Data directory
    """

    ret = RetClass()

    #  Check input. 

    if boundary not in boundaries.keys(): 
        print( f'Boundary "{boundary}" not a valid value. Availables boundaries are ' + \
                ", ".join( sorted( list( boundaries.keys() ) ) ) + "." )
        return None

    #  Date range. 

    d1, d2 = daterange[0], daterange[1]

    #  Set parameters, etc. 

    ncd = int( 24.0 / cycle_time + 0.001 )

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

    ntimes = int( (d2-d1+timedelta(days=1)) / timedelta(hours=cycle_time) + 0.001 )
    times = [ d1 + timedelta(hours=cycle_time) * itime for itime in range(ntimes) ]
    yearmonth, proj = None, None

    for itime in tqdm( range(ntimes), desc="  Time" ):
        dt = times[itime]

        if dt.strftime( "%Y%m" ) != yearmonth: 

            #  Close and remove previous file. 

            if yearmonth is not None: 
                d.close()
            yearmonth = dt.strftime( "%Y%m" )

            #  Download (if necessary) and open file. 

            try: 
                path = os.path.join( dataroot, watervaporflux_subdir )
                file = [ os.path.join( path, f ) for f in os.listdir( path ) \
                        if re.search( r'watervaporflux.' + yearmonth + r'.nc$', f ) ][0]
                d = Dataset( file, 'r' )
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

                #  Get Lambert conformal projection parameters and define projection. 

                v = d.variables['Lambert_Conformal']
                central_meridian = v.getncattr( "longitude_of_central_meridian" )
                standard_parallels = v.getncattr( "standard_parallel" )
                reference_latitude = v.getncattr( "latitude_of_projection_origin" )
                proj = LambertConformalProjection( central_meridian, standard_parallels, reference_latitude )

                #  Get longitude latitude grid and define interpolator. 

                narrlons = d.variables['lon'][:]
                narrlats = d.variables['lat'][:]
                intp = LambertConformalInterpolator( proj, narrlons, narrlats, midlons, midlats )

        #  Time index. 

        i = int( ( dt - datetime( dt.year, dt.month, 1 ) ) / timedelta(hours=cycle_time) + 0.001 )

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

    outputpath = os.path.join( dataroot, output_subdir, f'boundaryflux.{boundary}.nc' )
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
            'file_type': "NARR_boundary_watervaporflux", 
            'description': "This file contains the flux of total column water vapor across a " + \
                "boundary defined by longitude and latitude vertices. The input is taken from the " + \
                "North American Regional Reanalysis (NARR).", 
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

        og.variables['massflux'][:] = g['massflux']
        og.variables['watervaporflux'][:] = g['watervaporflux']
        og.variables['columnwatervapor'][:] = g['columnwatervapor']

    d.close()
    return


def main(): 

    parser = argparse.ArgumentParser( prog="compute_narr_boundary_watervaporflux", 
            description="""Compute column-integrated water vapor and the component 
            column-integrated water flux over the entire domain of the North American 
            Regional Reanalysis. Column-integrated water vapor is also included in 
            the output.""" )

    parser.add_argument( "boundary", type=str, 
            help="""Specify which boundary across which to compute mass flux of water 
            vapor. Valid values are: """ + ", ".join( [ f'"{b}"' for b in boundaries.keys() ] ) + ". " + \
            "The actual path of the boundary will be written into the output file." )

    parser.add_argument( "daterange", type=str,
            help="""A string defining the date range over which to compute column-integrated
            water vapor flux. The string is composed of two dates of the form
            "YYYY-MM-DD:YYYY-MM-DD", and the range is inclusive.""" )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
            default=default_dataroot,
            help="""The root directory where the LLJ files are stored; """ + \
                f'The default is {default_dataroot}.' )

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
        ret = boundary_watervaporflux( args.boundary, daterange, args.dataroot )
    else:
        print( 'The daterange argument must have format "YYYY-MM-DD:YYYY-MM-DD".' )

    return

if __name__ == "__main__": 
    main()
    pass

