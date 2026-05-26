#  Imports. 

import numpy as np
from netCDF4 import Dataset 

#  Exception handlers. 

class Error( Exception ): 
    pass

class GridInterpolatorError( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment

class LambertConformalProjectionError( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment

class LambertConformalInterpolatorError( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment


################################################################################
#  Define convenient RetClass class, useful for tracking status and 
#  commentary on program execution. 
################################################################################

class RetClass(): 

    def __init__(self): 
        self.success = True
        self.messages = []
        self.comments = []

    def update( self, success:bool=None, messages:{str,list}=None, comments:{str,list}=None ): 

        if success is not None: 
            self.success = success

        if isinstance(messages,str): 
            self.messages.append( messages )
        elif isinstance(messages,list): 
            self.messages += messages

        if isinstance(comments,str): 
            self.comments.append( comments )
        elif isinstance(comments,list): 
            self.comments += comments

        return self

    #  Magic methods. 

    def __repr__( self ): 
        return 'RetClass\n  success={:}\n  messages={:}\n  comments={:}'.format( 
            self.success, "; ".join(self.messages), "; ".join(self.comments) )

    def __padd__( self, x ): 
        self.success = ( self.success and x.success )
        self.messages += x.messages
        self.comments += x.comments

    def __add__( self, x, y ): 
        ret = RetClass( success=(x.success and y.success), 
                       messages=x.messages+y.messages, 
                       comments=x.comments+y.comments )
        return ret


class GridInterpolator(): 
    """This class creates a function that can be called to interpolate a 2-dimensional 
    model field with a rectangular longitude-latitude grid onto a specific sequence of 
    points with prescribed longitudes and latitudes. It is best used to evaluate two 
    dimensional longitude-latitude fields along just one coast boundary defined by the 
    sequence of boundary longitudes and latitudes. It is valid for only one model 
    longitude-latitude grid and one coastline; other instances are required for either a 
    different model grid or a different coastline."""

    pass

    def __init__( self, mlons, mlats, blons, blats ): 
        """Create an interpolator for a sequence of points or a rectangular grid 
        of longitudes and latitudes (blons, blats) of a model field on a rectangular 
        longitude-latitude grid (mlons, mlats).

        Arguments
        =========
        mlons           A one-dimensional numpy array defining the longitudes 
                        of the original rectangular model grid

        mlats           A one-dimensional numpy array defining the latitudes 
                        of the original rectangular model grid

        blons           The longitudes of either the one-dimensional curve onto 
                        which to interpolate or the longitudes of a rectangular 
                        grid onto which to interpolate; if the former, blons and 
                        blats must be one-dimensional; if the latter, blons and 
                        blats must be two-dimensional. In both cases, their 
                        shapes should be identical. numpy.meshgrid may be useful 
                        for gridded two-dimensional output. 

        blats           The latitudes of either the one-dimensional curve onto 
                        which to interpolate or the latitudes of a rectangular 
                        grid onto which to interpolate; if the former, blons and 
                        blats must be one-dimensional; if the latter, blons and 
                        blats must be two-dimensional. In both cases, their 
                        shapes should be identical. numpy.meshgrid may be useful 
                        for gridded two-dimensional output. 

                        """

        self.nmlons = mlons.size
        self.nmlats = mlats.size 
        blons_shape = blons.shape
        blats_shape = blats.shape

        if len( blons_shape ) != len( blats_shape ): 
            GridInterpolatorError( "InvalidArguments", "blons and blats must have the " + \
                            "same number of dimensions" )

        elif len( blons_shape ) not in [ 1, 2 ]: 
            GridInterpolatorError( "InvalidArguments", "blons and blats must have the " + \
                            "either one or two dimensions" )

        else: 
            for n, m in zip( blons_shape, blats_shape ): 
                if n != m: 
                    GridInterpolatorError( "InvalidArguments", "blons and blats must " + \
                            "have the same shape" )

        self.shape = blons_shape
        lons = ( blons * 1.0 ).flatten()
        lats = ( blats * 1.0 ).flatten()

        self.reflon = mlons[0]
        if self.reflon < -180.0: 
            self.reflon += 360
        elif self.reflon >= 180.0: 
            self.reflon -= 360

        model_dlons = mlons - self.reflon
        model_dlons[ model_dlons >= 360 ] -= 360.0
        model_dlons[ model_dlons < 0 ] += 360.0

        model_lats = mlats 

        #  Calculate the longitude interpolators. 

        intp_dlons = lons - self.reflon 
        intp_dlons[ intp_dlons >= 360 ] -= 360.0
        intp_dlons[ intp_dlons < 0 ] += 360.0

        self.ilons = np.zeros( lons.size, np.int32 )
        self.tlons = np.zeros( lons.size, np.float32 )

        for i in range( intp_dlons.size ): 
            ii = np.argwhere( ( intp_dlons[i] - model_dlons[:-1] ) * ( intp_dlons[i] - model_dlons[1:] ) <= 0.0 ).squeeze()
            if ii.size == 1: 
                j = ii
            elif ii.size > 1: 
                j = ii[0]
            else: 
                raise GridInterpolatorError( "InvalidBoundary", "Boundary falls outside model domain" )
            self.ilons[i] = j
            self.tlons[i] = ( intp_dlons[i] - model_dlons[j] ) / ( model_dlons[j+1] - model_dlons[j] )

        #  Calculate the latitude interpolators. 

        self.ilats = np.zeros( lats.size, np.int32 )
        self.tlats = np.zeros( lats.size, np.float32 )

        for i in range( lats.size ): 
            ii = np.argwhere( ( lats[i] - model_lats[:-1] ) * ( lats[i] - model_lats[1:] ) <= 0.0 ).squeeze()
            if ii.size == 1: 
                j = ii
            elif ii.size > 1: 
                j = ii[0]
            else: 
                raise GridInterpolatorError( "InvalidBoundary", "Boundary falls outside model domain" )
            self.ilats[i] = j
            self.tlats[i] = ( lats[i] - model_lats[j] ) / ( model_lats[j+1] - model_lats[j] )


    def __call__( self, field ): 
        """Interpolate the model field."""

        #  Check field dimensions. 

        if len( field.shape ) != 2: 
            raise GridInterpolatorError( "InvalidArgument", "The input field " + \
                    "must have two dimensions" )

        if field.shape[0] != self.nmlats or field.shape[1] != self.nmlons: 
            raise GridInterpolatorError( "InvalidArgument", "Dimensions of input field do " + \
                    "not match dimensions expected by the interpolator object" ) 

        #  Interpolation. 

        out = field[self.ilats,self.ilons] * (1-self.tlons) * ( 1-self.tlats ) + \
                field[self.ilats,self.ilons+1] * self.tlons * ( 1-self.tlats ) + \
                field[self.ilats+1,self.ilons] * (1-self.tlons) * self.tlats + \
                field[self.ilats+1,self.ilons+1] * self.tlons * self.tlats

        out = out.reshape( self.shape )

        return out


class LambertConformalInterpolator(): 
    def __init__( self, lambert_conformal_projection, mlons, mlats, blons, blats ): 
        """Create a function that can be used for interpolating fields defined on a 
        Lambert Conformal Conic map projection. The coordinates of the grid are defined 
        by the lons, lats grid.

        Arguments
        =========

        lambert_conformal_projection    A LambertConformalProjection object

        mlons                           A two-dimensional ndarray defining the grid 
                                        longitudes [degrees east]

        mlats                           A two-dimensional ndarray defining the grid 
                                        latitudes [degrees north]

        blons                           Either a one-dimensional or a two-dimensional 
                                        ndarray containing the longitudes at which to 
                                        interpolate a field. If both blons and blats 
                                        are one-dimensional, then interpolation returns 
                                        a one-dimensional array along the path traced 
                                        out by blons and blats. If both are two-dimensional, 
                                        then the interpolator returns a gridded field at 
                                        the locations of blons and blats. 

        blats                           Either a one-dimensional or a two-dimensional 
                                        ndarray containing the latitudes at which to 
                                        interpolate a field. If both blons and blats 
                                        are one-dimensional, then interpolation returns 
                                        a one-dimensional array along the path traced 
                                        out by blons and blats. If both are two-dimensional, 
                                        then the interpolator returns a gridded field at 
                                        the locations of blons and blats. 

        The input mlons and mlats must have the same shape. Likewise for blons and blats."""

        if not isinstance(lambert_conformal_projection,LambertConformalProjection): 
            raise LambertConformalInterpolatorError( "InvalidArgument", 
                    "first argument must be an instance of LambertConformalProjection" )

        if len( mlons.shape ) != 2 or len( mlats.shape ) != 2: 
            raise LambertConformalInterpolatorError( "InvalidArgument", 
                    "mlons and mlats must both be two-dimensional" )

        if mlons.shape[0] != mlats.shape[0] or mlons.shape[1] != mlats.shape[1]: 
            raise LambertConformalInterpolatorError( "InvalidArgument", 
                    "mlons and mlats must have the same shape" )

        if len( blons.shape ) != len( blats.shape ): 
            raise LambertConformalInterpolatorError( "InvalidArgument", 
                    "blons and blats must have the same number of dimensions" )

        if len( blons.shape ) not in [1,2]: 
            raise LambertConformalInterpolatorError( "InvalidArgument", 
                    "blons and blats must have either one or two dimensions" )

        for n, m in zip( blons.shape, blats.shape ): 
            if n != m: 
                raise LambertConformalInterpolatorError( "InvalidArgument", 
                        "blons and blats must have the same shape" )


        self.mlons = np.array( mlons )
        self.mlats = np.array( mlats )
        self.p = lambert_conformal_projection 
        self.shape = blons.shape

        #  Generate x, y coordinates. 

        x, y = self.p.lonlat2xy( self.mlons, self.mlats )

        #  Determine whether x is the first or second dimension of lons, lats. 

        xm1 = x[:,0].mean()
        xsd1 = ( x[:,0] - xm1 ).std()

        xm2 = x[0,:].mean()
        xsd2 = ( x[0,:] - xm2 ).std()

        if xsd1 > xsd2: 
            self.order = "xy"
        else: 
            self.order = "yx"

        #  Quantify x and y coordinate grid. 

        if self.order == "xy": 
            self.x = x.mean(axis=1)
            self.y = y.mean(axis=0)
        else: 
            self.x = x.mean(axis=0)
            self.y = y.mean(axis=1)

        #  Parameterize the grid. 

        self.xmin = self.x[0]
        self.xmax = self.x[-1]
        self.dx = ( self.x[1:] - self.x[:-1] ).mean()

        self.ymin = self.y[0]
        self.ymax = self.y[-1]
        self.dy = ( self.y[1:] - self.y[:-1] ).mean()

        #  Compute the interpolators. 

        flons = blons.flatten()
        flats = blats.flatten()

        x, y = self.p.lonlat2xy( flons, flats )

        #  Screen for input coordinates within range of grid. 

        goodx = np.logical_and( x >= self.xmin, x < self.xmax )
        goody = np.logical_and( y >= self.ymin, y < self.ymax )
        self.good = np.logical_and( goodx, goody )

        #  Interpolation indices and fractions. 

        self.ix = np.int16( ( x - self.xmin ) / self.dx )
        self.tx = ( x - self.xmin - self.ix * self.dx ) / self.dx
        self.iy = np.int16( ( y - self.ymin ) / self.dy )
        self.ty = ( y - self.ymin - self.iy * self.dy ) / self.dy

        #  Done. 

        return

    def __call__( self, field ): 
        """Interpolate a function field. 

        field       ndarray of input variable to be interpolated

        The input field must have the same shape as determined at instantiated 
        as mlons and mlats."""

        if len(field.shape) != 2: 
            raise LambertConformalInterpolatorError( "InvalidArgument", 
                    "The input field must be two-dimensional" )

        if field.shape[0] != self.mlons.shape[0] or field.shape[1] != self.mlons.shape[1]: 
            raise LambertConformalInterpolatorError( "InvalidArgument", 
                    "The input field must be have the same shape as the field coordinate grid" )

        vals = np.zeros( self.shape, dtype=field.dtype ).flatten()
        fmask = np.zeros( self.shape, dtype=np.bool_ ).flatten()

        ix, iy = self.ix, self.iy
        tx, ty = self.tx, self.ty
        good = self.good
        field_mask = np.ma.getmask( field )

        if self.order == "xy": 
            vals[good] = field[ix[good],iy[good]] * (1-tx[good]) * (1-ty[good]) + \
                    field[ix[good]+1,iy[good]] * tx[good] * (1-ty[good]) + \
                    field[ix[good],iy[good]+1] * (1-tx[good]) * ty[good] + \
                    field[ix[good]+1,iy[good]+1] * tx[good] * ty[good] 
            fmask[good] = field_mask[ix[good],iy[good]] | \
                    field_mask[ix[good]+1,iy[good]] | \
                    field_mask[ix[good],iy[good]+1] | \
                    field_mask[ix[good]+1,iy[good]+1] 
        else: 
            vals[good] = field[iy[good],ix[good]] * (1-tx[good]) * (1-ty[good]) + \
                    field[iy[good]+1,ix[good]] * tx[good] * (1-ty[good]) + \
                    field[iy[good],ix[good]+1] * (1-tx[good]) * ty[good] + \
                    field[iy[good]+1,ix[good]+1] * tx[good] * ty[good] 
            fmask[good] = field_mask[iy[good],ix[good]] | \
                    field_mask[iy[good]+1,ix[good]] | \
                    field_mask[iy[good],ix[good]+1] | \
                    field_mask[iy[good]+1,ix[good]+1] 

        #  Mask bad values (outside of coordinate range). 

        vals = np.ma.masked_where( np.logical_not(good) | fmask, vals ).reshape( self.shape )

        #  Done. 

        return vals


class LambertConformalProjection(): 
    def __init__( self, central_meridian, standard_parallels, reference_latitude ): 
        """Establish the parameters of a Lambert Conformal Conic Projection.  The 
        arguments are 

        central_meridian        A float, the central meridian (longitude) of the projects [degrees east]
        standard_parallels      A 2-element tuple/list/ndarray of floats, the standard parallels (latitudes) 
                                    of the projection [degrees north]
        reference_latitude      A float, the reference parallel (latitude) of the projection [degrees north]
        """

        self.central_meridian = np.deg2rad( central_meridian )
        self.standard_parallels = np.deg2rad( np.array( standard_parallels ) )
        self.reference_latitude = np.deg2rad( reference_latitude )
        self.radius = 6378.137e3        # meters

        self.kwargs = { 
                 'central_longitude': np.rad2deg( self.central_meridian ), 
                 'standard_parallels': np.rad2deg( self.standard_parallels ), 
                 'central_latitude': np.rad2deg( self.reference_latitude ) 
                 }

        #  Check inputs. 

        if self.central_meridian.size != 1: 
            raise LambertConformalProjectionError( "InvalidArgument", "central_meridian should contain only one value" )
        if self.standard_parallels.size != 2: 
            raise LambertConformalProjectionError( "InvalidArgument", "standard_parallels should contain two values" )
        if self.central_meridian.size != 1: 
            raise LambertConformalProjectionError( "InvalidArgument", "reference_latitude should contain only one value" )

        #  Derived quantities. 

        if self.standard_parallels[0] == self.standard_parallels[1]: 
            self.n = np.sin( self.standard_parallels[0] )
        else: 
            self.n = np.log( np.cos(self.standard_parallels[0]) / np.cos(self.standard_parallels[1]) ) \
                    / np.log( np.tan(np.pi/4+self.standard_parallels[1]/2) / np.tan(np.pi/4+self.standard_parallels[0]/2) ) 

        tan0 = np.tan( np.pi/4 + self.reference_latitude/2 )
        tan1 = np.tan( np.pi/4 + self.standard_parallels[0]/2 )
        tan2 = np.tan( np.pi/4 + self.standard_parallels[1]/2 )

        self.F = np.cos(self.standard_parallels[0]) * np.exp( self.n * np.log( tan1 ) ) / self.n
        self.rho0 = self.radius * self.F * np.exp( -self.n * np.log( tan0 ) )

    def lonlat2xy( self, lons, lats ): 
        """Transform longitude, latitude coordinates to x, y coordinates. 

        lons        np.ndarray of longitudes [degrees]; can be one- or two-dimensional to match shape of lats
        lats        np.ndarray of latitudes [degrees]; can be one- or two-dimensional to match shape of lons

        A 2-tuple of x, y is returned, with x and y have the same shape as lons and lats. 
        """

        #  Convert to radians. 

        rlons = np.deg2rad( lons )
        rlats = np.deg2rad( lats )

        #  Check arguments. 

        slons, slats = rlons.shape, rlats.shape

        if len(slons) != len(slats): 
            raise LambertConformalProjectionError( "InvalidArguments", "lons and lats arguments must have the same number of dimensions" )

        for n,m in zip(slons,slats): 
            if n != m: 
                raise LambertConformalProjectionError( "InvalidArguments", "lons and lats arguments must have the same dimension" )

        #  Computations. 

        tan = np.tan( np.pi/4 + rlats/2 )
        rho = self.radius * self.F * np.exp( -self.n * np.log( tan ) )

        dlons = rlons - self.central_meridian
        dlons = np.arctan2( np.sin(dlons), np.cos(dlons) )

        x = rho * np.sin( self.n * dlons )
        y = self.rho0 - rho * np.cos( self.n * dlons )

        #  Done. 

        return x, y

    def xy2lonlat( self, x, y ): 
        """Transform x, y coordinates to longitude, latitude coordinates. Exact conjugate of lonlat2xy. 

        x           np.ndarray of x [m]; can be one- or two-dimensional to match shape of y
        y           np.ndarray of y [m]; can be one- or two-dimensional to match shape of x

        A 2-tuple of lons, lats is returned, with lons and lats have the same shape as x and y. 
        """

        #  Check arguments. 

        sx, sy = x.shape, y.shape

        if sx.size != sy.size: 
            raise LambertConformalProjectionError( "InvalidArguments", "x and y arguments must have the same number of dimensions" )

        for i in range(sx.size): 
            if sx[i] != sy[i]: 
                raise LambertConformalProjectionError( "InvalidArguments", "x and y arguments must have the same dimension" )

        #  Compute lons (in radians) and rho. 

        rlons = self.central_meridian + np.arctan2( x, self.rho0 - y ) / self.n
        rho = np.sqrt( x**2 + (self.rho0-y)**2 )

        #  Compute lats (in radians). 

        rlats = 2 * np.arctan( np.exp( np.log( self.radius * self.F / rho ) / self.n ) ) - np.pi/2 

        #  Convert to degrees. 

        lons, lats = np.rad2deg( rlons ), np.rad2deg( rlats )

        #  Done. 

        return lons, lats 

