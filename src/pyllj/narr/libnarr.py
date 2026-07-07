import os
import requests 
from datetime import datetime, timedelta, timezone
from netCDF4 import Dataset
import numpy as np
from tqdm import tqdm
from time import time
from scipy.interpolate import interp1d, CubicHermiteSpline
from ..libutils import RetClass 
from ..parameters import Rearth, Rideal, gravity, muvap, mudry, default_dataroot, aws_region, bucket

#  AWS and storage path settings. 

downloads_subdir = "NARR/downloads" 

cycle_time = 3      #  hours
fill_float = -1.0e20
epoch = datetime( year=1980, month=1, day=1 )

#  The NARR data server. 

narr_remote = "https://downloads.psl.noaa.gov/Datasets/NARR"

#  Exception handling. 

class Error( Exception ): 
    pass

class libnarrError( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment 


################################################################################
#  Define a class useful for dealing with NARR data files. 
################################################################################

class NARRfile(): 
    """A class to download, remove, and provide a local path to a North American 
    Regional Reanalysis (NARR) data file."""

    def __init__( self, var:str, time:datetime=None, dataroot:str=default_dataroot ): 

        self.localpath = None
        self.remotepath = None
        self.success = True
        self.message = None
        self.comment = None
        self.dataroot = dataroot
        downloadsroot = os.path.join( dataroot, downloads_subdir )

        upperair_vars = [ "air", "hgt", "omega", "shum", "uwnd", "vwnd" ]
        surface_vars = [ "acpcp", "pres.sfc", "hgt.sfc", "air.2m", "shum.2m", "uwnd.10m", "vwnd.10m" ]

        if var in upperair_vars: 

            basename = f'{var}.{time.year:4d}{time.month:02d}.nc'
            self.localpath = os.path.join( downloadsroot, "pressure", basename )
            self.remotepath = "/".join( [ narr_remote, "pressure", basename ] )

        elif var in surface_vars: 

            if var in [ "hgt.sfc" ]: 
                basename = f'{var}.nc'
                self.localpath = os.path.join( downloadsroot, "time_invariant", basename )
                self.remotepath = "/".join( [ narr_remote, "time_invariant", basename ] )
            else: 
                basename = f'{var}.{time.year:4d}.nc'
                self.localpath = os.path.join( downloadsroot, "monolevel", basename )
                self.remotepath = "/".join( [ narr_remote, "monolevel", basename ] )

        else: 

            self.success = False
            self.message = "UnrecognizedVariable"
            self.comment = f'Variable {var} is not recognized; recognized variables are ' + \
                    ', '.join( upperair_vars + surface_vars ) + '.' 

        #  Check if the file already exists, and, if it exists, if it is a valid 
        #  NetCDF file. If the file doesn't exist or is not a NetCDF file, then 
        #  self.success = False and a download is attempted. 

        if os.path.isfile( self.localpath ): 
            try: 
                d = Dataset( self.localpath, 'r' )
                d.close()
                self.success = True
            except: 
                self.success = False
        else: 
            self.success = False 

        #  Attempt download. 

        if not self.success : 

            print( f'Downloading {self.remotepath}' )
            self.success, ntries = False, 0

            #  Attempt download 10 times. 

            while not self.success and ntries < 10: 
                ntries += 1

                #  Test for validity of download. 

                try: 
                    resp = requests.get( self.remotepath, stream=True ) 
                    resp.raise_for_status()

                    os.makedirs( os.path.dirname( self.localpath ), exist_ok=True )
                    f = open( self.localpath, 'wb' ) 
                    for chunk in resp.iter_content( chunk_size=4194304 ):   # 4 MB chunks
                        f.write( chunk )

                    d = Dataset( self.localpath, 'r' )
                    d.close()
                    self.success = True

                except: 
                    self.success = False
                    continue

            if self.success: 
                self.comment = f"Downloaded {self.localpath} after {ntries} tries"
            else: 
                self.message = "RemoteFileUnavailable"
                self.comment = f"Unable to download remote file {self.remotepath}."
                raise libnarrError( self.message, self.comment )

    def open( self ): 
        """Open the NetCDF file and return a netCDF4.Dataset object."""

        ret = Dataset( self.localpath, 'r' )
        return ret

    def remove( self ): 
        os.unlink( self.localpath )


