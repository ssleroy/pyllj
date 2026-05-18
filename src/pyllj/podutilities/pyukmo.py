import importlib
from scipy.io import readsav 
import numpy as np
from matplotlib.colors import LinearSegmentedColormap 


class UKMOcolorMaps(): 
    """Create matplotlib colormaps based on the UKMO color maps, which are available 
    in an IDL save format."""

    def __init__( self ): 

        files = importlib.resources.files( 'pyllj.podutilities' )
        with importlib.resources.as_file( files.joinpath( "ukmo_color_scales.sav" ) ) as path: 
            self.color_scales = readsav( path )['color_scales']
        shape = self.color_scales.shape
        self.nscales = shape[0]
        self.npoints = shape[2]


    def get_cmap( self, iscale, gamma=1.0 ): 
        """Create a matplotlib cmap for a particular color scale defined by the 
        UKMO in ukmo_color_scales.sav."""

        if iscale < 0 or iscale >= self.nscales: 
            print( f'The requested scale must be an integer between 0 and {self.nscales-1}.' )
            return

        cs = self.color_scales[iscale,:,:]

        #  Define color dictionary. 

        colors = [ 'red', 'green', 'blue' ]
        cdict = { color: [] for color in colors } 

        xanchors = np.arange( self.npoints ) / ( self.npoints - 1 )
        for ix, xanchor in enumerate( xanchors ): 
            for icolor, color in enumerate( colors ): 
                val = cs[icolor,ix] / 255.0
                cdict[color].append( np.array( [ xanchor, val, val ] ) )

        #  Create matplotlib colormap. 

        cmap_name = f'ukmo{iscale:02d}' 
        # print( f'Creating colormap "{cmap_name}".' )

        ret = LinearSegmentedColormap( cmap_name, cdict, N=255, gamma=gamma )

        return ret


if __name__ == "__main__": 
    ucs = UKMOcolorMaps()
    r = ucs.get_cmap( 0 )
    pass

