import numpy as np
import xarray as xr
import pandas as pd
from stompy.io.local import cdec
import stompy.model.hydro_model as hm

# original csv has 2007-09-29 to 2017-04-30, and is identical
# to the data from CDEC after unit and sign conversion
# forcing-data/Barker_Pumping_Plant.csv


##

class BarkerPumpsBC(hm.FlowBC):
    station_id="BKS"
    sensor=70 # daily, discharge, pumping
    pad=np.timedelta64(2,'D')
    def src_data(self):
        ds=self.fetch_for_period(self.data_start,self.data_stop)
        return ds['Q']
    def fetch_for_period(self,period_start,period_stop):
        """
        Download or load from cache, take care of any filtering, unit conversion, etc.
        Returns a dataset with a 'Q' variable, and with time as UTC
        """
        ds=cdec.cdec_dataset("BKS",period_start,period_stop,70,
                             cache_dir='cache',duration='D')
        # cfs=>m3s, and 'pumping' sign to inflow sign
        ds['Q']=-0.028316847 * ds.sensor0070
        ds['Q'].attrs['units']='m3 s-1'
        return ds

if 0:
    # writes bc_barker.html in current directory
    barker=BarkerPumpsBC(name='barker')
    barker.data_start=np.datetime64("2018-05-01")
    barker.data_stop=np.datetime64("2018-08-01")
    barker.write_bokeh()
