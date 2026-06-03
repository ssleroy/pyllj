from .libpod import WindField
import os 
from netCDF4 import Dataset 
import numpy as np 
from pyllj.podutilities.libpod import get_metricpath
import matplotlib.pyplot as plt
from matplotlib import ticker
import warnings

#  Pyplot settings. 

axeslinewidth = 0.5 
plt.rcParams.update( {
  'font.family': "Times New Roman", 
  'font.size': 8,
  'font.weight': "normal",
  'text.usetex': True,
  'xtick.major.width': axeslinewidth,
  'xtick.minor.width': axeslinewidth, 
  'ytick.major.width': axeslinewidth, 
  'ytick.minor.width': axeslinewidth, 
  'axes.linewidth': axeslinewidth } ) 

warnings.filterwarnings('ignore')


################################################################################
#  Define a class that contains all information and data relevant to a delta-
#  isohypsic wind-barb plot. 
################################################################################

class WindBarbs(): 
    """Compute the wind barbs for one of the models (or sondes). A dictionary is 
    returned containing the related WindField instance for the model and the 
    lengths of the u and v wind components decomposed by month, hour, and height 
    level above the surface."""

    def __init__( self, source:str=None, sourcename:str=None, uwnd=None, vwnd=None, 
                 hours=None, months=None, levels=None ): 

        """Create an instance of WindBarbs."""

        self.source = source
        self.sourcename = sourcename
        self.uwnd = uwnd
        self.vwnd = vwnd
        self.hours = hours
        self.months = months
        self.levels = levels

        return

    def __add__( self, other ): 

        #  Find common months. 

        imonths1, imonths2 = [], []
        for imonth1, month1 in enumerate( self.months.tolist() ): 
            i = np.argwhere( other.months == month1 )
            if i.size == 1: 
                imonths1.append( imonth1 )
                imonths2.append( int( i[0] ) )
        imonths1 = np.array( imonths1 )
        imonths2 = np.array( imonths2 )
        months = self.months[imonths1]

        #  find common hours. 

        ihours1, ihours2 = [], []
        for ihour1, hour1 in enumerate( self.hours.tolist() ): 
            i = np.argwhere( other.hours == hour1 )
            if i.size == 1: 
                ihours1.append( ihour1 )
                ihours2.append( int( i[0] ) )
        ihours1 = np.array( ihours1 )
        ihours2 = np.array( ihours2 )
        hours = self.hours[ihours1]

        #  Do the subtraction. 

        x = ( self.uwnd[imonths1,:,:] )[:,ihours1,:]
        y = ( other.uwnd[imonths2,:,:] )[:,ihours2,:]
        uwnd = x + y

        x = ( self.vwnd[imonths1,:,:] )[:,ihours1,:]
        y = ( other.vwnd[imonths2,:,:] )[:,ihours2,:]
        vwnd = x + y

        sourcename = self.sourcename + r'$+$' + other.sourcename

        ret = WindBarbs( sourcename=sourcename, uwnd=uwnd, vwnd=vwnd, hours=hours, months=months, levels=self.levels )
        return ret

    def __sub__( self, other ): 

        #  Find common months. 

        imonths1, imonths2 = [], []
        for imonth1, month1 in enumerate( self.months.tolist() ): 
            i = np.argwhere( other.months == month1 ).squeeze()
            if i.size == 1: 
                imonths1.append( imonth1 )
                imonths2.append( i ) 
        imonths1 = np.array( imonths1 )
        imonths2 = np.array( imonths2 )
        months = self.months[imonths1]

        #  find common hours. 

        ihours1, ihours2 = [], []
        for ihour1, hour1 in enumerate( self.hours.tolist() ): 
            i = np.argwhere( other.hours == hour1 ).squeeze()
            if i.size == 1: 
                ihours1.append( ihour1 )
                ihours2.append( i )
        ihours1 = np.array( ihours1 )
        ihours2 = np.array( ihours2 )
        hours = self.hours[ihours1]

        #  Do the subtraction. 

        x = ( self.uwnd[imonths1,:,:] )[:,ihours1,:]
        y = ( other.uwnd[imonths2,:,:] )[:,ihours2,:]
        uwnd = x - y

        x = ( self.vwnd[imonths1,:,:] )[:,ihours1,:]
        y = ( other.vwnd[imonths2,:,:] )[:,ihours2,:]
        vwnd = x - y

        sourcename = self.sourcename + r'$-$' + other.sourcename

        ret = WindBarbs( sourcename=sourcename, uwnd=uwnd, vwnd=vwnd, hours=hours, months=months, levels=self.levels )
        return ret

    def __mul__( self, f ): 

        ret = WindBarbs( sourcename=self.sourcename, 
                        uwnd=self.uwnd*f, vwnd=self.vwnd*f, 
                        hours=self.hours, months=self.months, levels=self.levels )
        return ret

    def __div__( self, f ): 

        ret = WindBarbs( sourcename=self.sourcename, 
                        uwnd=self.uwnd/f, vwnd=self.vwnd/f, 
                        hours=self.hours, months=self.months, levels=self.levels )
        return ret


################################################################################
#  Compute a wind-barb analysis. It essentially averages over a region or 
#  selects a sonde location. 
################################################################################

def compute_windbarbs( source:str, sourcelabel:str=None, region:str=None, sonde:str=None ): 
    """Do the regional averaging for the computation of wind barbs in the delta-
    ishypsic analysis.

    Arguments
    =========
    source          The name of a reanalysis ("narr","merra2","era5"), a 
                    sonde, a path to the root directory of model output, or 
                    a specific isohypses climatology file.

    sourcelabel     The official label to be associated with the instance. 
                    By default, it is the uppercase reanalysis name or the 
                    lowest level directory in the modelroot path. 

    region          The name of the region over which to average the delta-
                    isohypsic analysis. You cannot select both a region and 
                    a sonde. 

    sonde           The name of the sonde at which to select a single profile 
                    of isohypsic analysis. YOu cannot select botha region 
                    and a sonde."""

    if region is not None and sonde is not None: 
        print( 'compute_windbarbs: cannot interpolate at a sonde site and average over an area' )
        return None

    #  Create a WindField instance. 

    analysisfile = get_metricpath( "isohypses", source )

    if analysisfile is None: 

        #  Atmospheric model/sonde output. 

        if os.path.isdir( os.path.join( source, "isohypses" ) ): 
            analysisfile = os.path.join( source, "isohypses", "isohypses.nc" )

            if sourcelabel is None: 
                ss = source.split( "/" )
                if ss[-1] != "": 
                    sourcename = ss[-1]
                else: 
                    sourcename = ss[-2]
            else: 
                sourcename = sourcelabel

        elif os.path.isfile( source ): 
            analysisfile = source

            if sourcelabel is None: 
                print( 'No sourcelabel provided. setting to blank.' )
                sourcename = ""
            else: 
                sourcename = sourcelabel

    else: 

        #  Reanalysis output. 
        if sourcelabel is None: 
            sourcename = source.upper()
        else: 
            sourcename = sourcelabel

    WF = WindField( analysisfile, region=region, sonde=sonde, scale=150 )

    #  Wind barbs for region, annual cycle, diurnal cycle

    d = Dataset( analysisfile, 'r' )
    hours = d.variables['hour'][:]
    months = d.variables['month'][:]
    ncp = hours.size
    uwnd = np.ma.zeros( ( 12, ncp, WF.nz ), np.float32 )
    vwnd = np.ma.zeros( ( 12, ncp, WF.nz ), np.float32 )

    if WF.mask is None: 
        uwnd = d.variables['uwnd'][:]
        vwnd = d.variables['vwnd'][:]

    else: 
        for imonth in range(12): 
            for ihour in range(ncp): 
                uwnd[imonth,ihour,:] = ( d.variables['uwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()
                vwnd[imonth,ihour,:] = ( d.variables['vwnd'][imonth,ihour,:,:,:] * WF.mask ).reshape( (WF.nz,WF.nx*WF.ny) ).sum(axis=1) / WF.mask.sum()

    d.close()

    ret = { 'source': source, 'sourcename': sourcename, 'uwnd': uwnd, 'vwnd': vwnd, 
           'hours': hours, 'months': months, 'levels': WF.levels }

    return ret


def windbarbs_ax( fig, pos, wind_barbs, imonths, xticks=True, yticks=True, 
            scale=12, scale_length=None ): 
    """Generate a wind-barb axes. 

    Arguments
    =========

    fig             A pyplot figure instance

    pos             A 4-element list or ndarray defining the position of the 
                    axes within a figure. 

    wind_barbs      An instance of WindBarbs

    imonths         A numpy ndarray containing the months over which to 
                    average; if a scalar, then only one month is selected 
                    for the axes/plot. 

    xticks          Set to true if x tick labels and x axis title are to 
                    be generated

    yticks          Set to true if y tick labels and y axis title are to 
                    be generated

    scale           How many m/s of wind wind speed projects to 1 km in the 
                    y-axis of the plot. 

    scale_length    Define if a scale length vector should be plotted; it 
                    should be the reference speed in m/s."""

    meta = { 'scale': scale, 'scale_units': "y" }
    cmap = plt.get_cmap( 'gist_ncar' )

    ax = fig.add_axes( pos )

    ax.set_xlim( -3, 24 )
    xts = np.arange(0,24,3) 
    ax.set_xticks( xts )

    if xticks: 
        ax.set_xticklabels( [ f'{int(x):02d}' for x in xts ] ) 
    else: 
        ax.set_xticklabels( [] )

    ax.set_ylim( 0, 3 )
    yts = np.arange(0,4,1,dtype=np.int32)
    ax.set_yticks( yts )
    ax.yaxis.set_minor_locator( ticker.MultipleLocator(0.2) )
    if yticks: 
        ax.set_yticklabels( yts )
    else: 
        ax.set_yticklabels( [] )

    for ihour in range( wind_barbs.hours.size ): 
        hour = wind_barbs.hours[ihour]
        x = np.repeat( hour, wind_barbs.levels.size )
        y = wind_barbs.levels/1000
        uwnd = wind_barbs.uwnd[imonths,ihour,:].squeeze()
        vwnd = wind_barbs.vwnd[imonths,ihour,:].squeeze()

        if not isinstance(imonths,int): 
            if imonths.size > 1: 
                uwnd = uwnd.mean(axis=0)
                vwnd = vwnd.mean(axis=0)

        ax.quiver( x, y, uwnd, vwnd, color=cmap( (hour+1.5)/24 ), **meta )

    if scale_length is not None: 
        ax.quiver( 22, 0.2, 0.0, scale_length, color="#000000", **meta )
        ax.text( 22, 0.25+scale_length/scale, f'{int(scale_length)} m/s', rotation=90, rotation_mode="anchor", ha="left", va="center" )

    return ax

