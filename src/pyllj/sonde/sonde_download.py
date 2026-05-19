import os
import numpy as np
from datetime import datetime, timezone, timedelta
from ..parameters import default_dataroot
import argparse
import re
import urllib.request
import html2text
import time, random
import requests

def download_single_sonde(sonde_ID, sonde_path, dtime, session, delay_time=1):

    """ 
    Download radiosonde data from U-Wyoming database.
    Takes in sonde_ID, directory path, and datetime object.
    Returns whether the download was successful or not. 
    """

    error_text = ''
    
    # If datafile does not exist for that station and date, download it. 
    if not os.path.isfile(sonde_path):
    
        #if not os.path.isdir(f'{path}'):
        #    os.makedirs(f'{path}')
            
        sonde_url = (
            f'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&'
            f'YEAR={dtime.year}&MONTH={dtime.month:02}&FROM={dtime.day:02}{dtime.hour:02}&'
            f'TO={dtime.day:02}{dtime.hour:02}&STNM={sonde_ID}'
        )
            
    
        # Try to sucessfully download up to 10 times. 
        # Borrowed from Stephen's libnarr.py
        success = False
        ntries = 0
        while not success and ntries < 10: 
            ntries += 1
            try: 

                # Give a reasonable-looking User-Agent to avoid UWyoming
                #... database assuming request is coming from a bot. 
                user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0"

                # Scrape web. 
                response  = session.get(sonde_url, headers={"User-Agent": user_agent})

                # Check for bad responses. 
                #########################################
                if response.status_code == 503:
                    # UWyoming is rate-limiting web scraping. Slow down. 
                    time.sleep(delay_time)
                    # Increase delay time with each 503 error, but never let it exceed a minute. 
                    delay_time = min(delay*2, 60)
                    print("Website rate-limiting scrping (503 error). Slowing down...")
                    continue
                if response.status_code == 404:
                    # Site is blocking requests. 
                    print(f"Site is blocking requests. Web scraping ceased at {dt}.")
                    break
                if f"Can\'t get " in response.text:
                    # Data not avaible for that date. Log in error log. 
                    error_text = f"{dtime} {sonde_url}"
                    print("Logging an error")
                    return success, error_text
                
                # Remove unnecessary html code. 
                trim_idx = response.text.index('Description of the')
                trimmed_content = response.text[:trim_idx]
    
                with open(sonde_path, 'w') as f:
                    f.write(trimmed_content)
    
                success = True
                
            except Exception as exc: 
                if ntries==10:
                    print(f"Failing out with exception {exc}")
                continue
        if not success:
            # TODO: need to take this out of this list of datetimes to analyze.
            print(f'Unable to download data for {dtime} at {sonde_url}.  Skipping...')

        # Regardless of outcome, return.     
        return success, error_text
    else:
        # File already exists, return download successful. 
        print("File already exists.")
        return True, error_text

def download_sonde(station, dayrange, dataroot=default_dataroot):

    s = requests.session()
    
    # Get date range.
    start_date, end_date = dayrange.split(":")               
    dtimes = np.arange(datetime.fromisoformat(f"{start_date}T00"),datetime.fromisoformat(f"{end_date}T12"), 
                       timedelta(hours=12)).astype(datetime)
    dtimes = [dt.replace(tzinfo=timezone.utc) for dt in dtimes]
    
    
    # Create log of inaccessable sonde dates.
    error_log = f"UPDATED_error_log_{station}-{start_date}:{end_date}.txt"
    error_arr = []

    
    for dt in dtimes:
        # Read data from the sonde txt file. 
        sonde_path = f'{dataroot}/{dt.year}/{station}-{dt.year}{dt.month:02}{dt.day:02}{dt.hour:02}.txt'
        if not os.path.isfile(sonde_path):
            # Attempt to download file.
            print(f"Downloading {dt}...")
            download_result, error_message =  download_single_sonde(station, sonde_path, dt, s)
            if not download_result:
                # Unable to download sonde data for that date. Log date and URL. 
                error_arr.append(error_message)
        # Add jitter to avoid rate limit so UWyoming database
        #.... does not think request is coming from a bot.
        time.sleep(0.5 + random.random() * 1.5)
    # Generate log. 
    np.savetxt(error_log, error_arr, delimiter=" ", newline = "\n", fmt="%s")
    
    return error_log

def main():
    """ 
    Download rawinsonde data for a given station between given dates.
    Args: 
        station ID (string of 5 numbers)
        start_date (YYYY-MM-DD format)
        end_date (YYYY-MM-DD format)
        
    """


    parser = argparse.ArgumentParser( prog="sonde_download", description=
            """Download rawinsonde data for a given station in a given daterange. The text files are donwloaded to 
            DATAROOT/{station ID}. DATAROOT is defined using the
            --dataroot option.""" )
    
    parser.add_argument( "station", type=str, help="""ID of rawinsonde station. Must be 
    string of 5 ints.""" )
    
    parser.add_argument( "dayrange", type=str, help="""The range of days over which to compute, 
    format "YYYY-MM-DD:YYYY-MM-DD", and the range is inclusive.""" )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", type=str,
            default=default_dataroot,
            help="""The root directory where the LLJ data files are stored and where
                results will be written. """ + f'The default is {default_dataroot}; the '+ \
                'downloaded files will be saved to a subdirectory with the station ID name.' )

    parser.add_argument( "--pdb", dest="pdb", default=False, action="store_true", 
            help="Use this option to enter the Python line debugger")

    #  Parse. 

    args = parser.parse_args()

    if args.pdb: 
        import pdb
        pdb.set_trace()

    m = re.search( r'^(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})$', args.dayrange )
    if m: 
        station = args.station
        dayrange = args.dayrange
        err_log = download_sonde(station, dayrange, dataroot=args.dataroot)
        print(f"Download complete. Inaccessible files logged in {err_log}.") 
    else: 
        print( 'The daterange argument must have format "YYYY-MM-DD:YYYY-MM-DD".' )
    

if __name__ == "__main__": 
    main()
    pass

