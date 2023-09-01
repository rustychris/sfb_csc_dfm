import os
import xarray as xr
from stompy.model import data_comparison
from stompy import memoize, utils
from stompy.spatial import interp_4d, proj_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

underway_dir="../data/usgs_underway"
cruise_fns=["2018-08-23_Sample_data_ver.2.0.csv",
            "2018-09-05_Sample_data_ver.2.0.csv",
            "2018-09-13_Sample_data_ver.2.0.csv",
            "2018-10-04_Sample_data_ver.2.0.csv",
            "Delta Water Quality Mapping July 2018 High Resolution.csv",
            "Delta Water Quality Mapping May 2018 High Resolution.csv",
            "Delta Water Quality Mapping October 2018 High Resolution.csv",
           ]

def extrap_evaluate(pred, ref, label='Predicted', plot=True):
    # How does straight up diffusion do?
    if plot:
        fig,(ax,ax_txt)=plt.subplots(1,2,figsize=(8,5), width_ratios=[1,0.33])
        ax.plot( ref, pred, 'k.',ms=1.0, alpha=0.4)
        ax.set_xlabel('Underway (FNU)')
        ax.set_ylabel(label)
    else:
        fig=None
        
    metrics = data_comparison.calc_metrics(pred, ref)
    text=f"""
    Bias:   {metrics['bias']: .3f}
    R:      {metrics['r']: .3f}
    Murphy: {metrics['murphy']: .3f}
    Amp:    {metrics['amp']: .3f}"""
    if plot:
        ax_txt.axis('off')
        ax_txt.text(0.02,0.99,text,transform=ax_txt.transAxes,va='top',fontfamily='monospace')
    print(label)
    print(text)
    print()
    return fig


def mon_dataset(analyte):
    ds=xr.open_dataset(f"../data/usgs_nwis/{analyte}-2018-04-01-2018-11-01.nc")
    ll=np.c_[ds.lon.values, ds.lat.values]
    utm=proj_utils.mapper('WGS84','EPSG:26910')(ll)
    ds['x']=ds.lon.dims,utm[:,0]
    ds['y']=ds.lat.dims,utm[:,1]
    return ds


# Diffusion:

class Predictor:
    grid=None
    mon_ds=None
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        
    # def predict_map(self,t): SUBCLASS
    
    time_quant=np.timedelta64(1,'h')
    def predict_underway_from_map(self,boat_df):
        """
        Bin data by time interval, interpolate between successive
        predict_map output.
        """
        # Approach:
        # Quantize time (say, hourly), create map output for quantized time,
        # linearly interpolate. 
        data_dt = self.mon_ds.time.values[1] - self.mon_ds.time.values[0]
        assert self.time_quant>=data_dt
        
        binned = utils.floor_dt64(boat_df.time.values, self.time_quant)
        
        # Process each group
        t_last=None # cache last map since we can reuse most of the time.
        pred_map_last=None
        
        results=np.full(len(boat_df), np.nan, dtype=np.float64)
        
        boat_xy=boat_df[['x','y']].values
        
        for t_quant,idxs in utils.progress(utils.enumerate_groups(binned)):
            t0=t_quant
            t1=t0+self.time_quant
            # print(f"Processing {t0} -- {t1}")
            if t0==t_last:
                pred_map_t0=pred_map_last
            else:
                pred_map_t0=self.predict_map(t0)
            pred_map_t1=self.predict_map(t1)
            
            sample_time=boat_df['time'].values[idxs]
            frac=(sample_time - t0)/self.time_quant
            cells=[self.grid.select_cells_nearest(boat_xy[i]) for i in idxs]
            val0=pred_map_t0[cells]
            val1=pred_map_t1[cells]
            
            results[idxs] = (1-frac)*val0 + frac*val1 
            
        return results
    
    def predict_underway(self,boat_df):
        """
        boat_df: data frame with x,y,time
        Generate predictions for each row.
        """
        return self.predict_underway_from_map(boat_df)


class PreparedPredictor(Predictor):
    prepared=None
    alpha=1e-5
    K_factor=None # defaults to 1.0 everywhere. [Nedges]
        
    def get_prepared(self):
        if self.prepared is None:
            # K_factor is conceptually a scaling applied to the diffusion coefficient,
            # but that's equivalent to passing edge depth here.
            self.prepared=interp_4d.PreparedExtrapolation(self.grid, alpha=self.alpha, edge_depth=self.K_factor)
        return self.prepared
    
class DiffusionPredictor(PreparedPredictor):
    mon_field='turbidity'
    
    @memoize.imemoize()
    def predict_map(self,t):
        snap = self.mon_ds.sel(time=t,method='nearest')
        valid = np.isfinite(snap[self.mon_field].values)
        # Triggering a strange xarray/netcdf issue.
        for trojan in ['station_nm','site_no']:
            if trojan in snap:
                snap=snap.drop(trojan)
        snap_v = snap.isel(site=valid)
        
        df=snap_v.to_dataframe() #[ 'x','y','turbidity']
        df=df[ ['x','y',self.mon_field] ]
        df['weight']=1.0

        predicted=self.get_prepared().process(samples=df, value_col=self.mon_field)
            
        return predicted

        
class AdvDiffPredictor(PreparedPredictor):
    scal='turb'
    map_ds=None
    min_weight=0.0 # weights below this truncate to 0. mostly just to test getting back to diffusion.
    weight_exp=1.0 # exponent applied to weight.
    
    def __init__(self,**kw):
        super().__init__(**kw)
        if self.map_ds is not None and self.grid is None:
            self.grid = self.map_ds.grid # assumes it's a multiugrid.
            
    @memoize.imemoize()
    def predict_map(self,t):
        snap = self.map_ds.sel(time=t,method='nearest')
        wt    =snap[f'mesh2d_{self.scal}_wt'].values.clip(0)
        wt = wt**self.weight_exp
        
        truncate=wt<self.min_weight
        wt_obs=snap[f'mesh2d_{self.scal}_wtobs'].values.clip(0)
        wt_obs[truncate]=0.0
        wt[truncate]=0.0
            
        result = self.get_prepared().process_raw(cell_weights=wt, cell_value_weights=wt_obs)
        return result


def load_cruise(cruise_fn, 
                obs_fields=[
                    ('turb',['Turb (FNU) (EXO) HR']),
                    # There is conductivity from both the EXO and TSG. For now
                    # just using EXO
                    # Unicode fail. These two strings are not the same!
                    ('cond',['Sp Cond (µS/cm) (EXO) HR', # this one uses micro sign
                             'Sp Cond (μS/cm) (EXO) HR', # this one is Greek small mu.
                             #'Cond (µS/cm) (TSG) HR',
                            ]),
                    ('chl',['fCHLA (µg/L) (EXO) HR', # micro sign
                            'fCHLA (μg/L) (EXO) HR'  # Greek small mu
                           ]),
               ]):
    cruise_df=pd.read_csv(os.path.join(underway_dir,cruise_fn),
                          parse_dates=['Timestamp (PST)'],low_memory=False)

    cruise_df['time'] = cruise_df['Timestamp (PST)'] + np.timedelta64(8,'h')
    # Get date from PST, not UTC.
    cruise_df['date'] = cruise_df['Timestamp (PST)'].dt.date
    for lat_name in ['Latitude (Decimal Degrees)','Latitude (Decimal Degrees, NAD83)']:
        if lat_name in cruise_df.columns: break
    else: assert False
        
    for lon_name in ['Longitute (Decimal Degrees)','Longitude (Decimal Degrees, NAD83)']:
        if lon_name in cruise_df.columns: break
    else: assert False

    for obs_field,data_cols in obs_fields:
        cruise_df[obs_field]=np.nan
        for fld in data_cols:
            #print(f'Checking for {repr(fld)}')
            if fld not in cruise_df.columns: continue
            cruise_df[obs_field]=cruise_df[obs_field].combine_first( cruise_df[fld] )
    
    # Valid samples have lat,lon and at least one of the observation fields
    valid_ll=cruise_df[lon_name].notnull() & cruise_df[lat_name].notnull()
    valid_flds=np.zeros(len(valid_ll),bool)
    valid_flds[:]=False
    
    for obs_field,data_columns in obs_fields:
        valid_flds=valid_flds | cruise_df[obs_field].notnull()
    
    valid=valid_flds & valid_ll
    cruise_df=cruise_df[valid].copy()

    ll=cruise_df[ [lon_name, lat_name]].values
    
    xy=proj_utils.mapper('WGS84','EPSG:26910')(ll)
    bad_ll=~np.isfinite(xy[:,0])
    if np.any(bad_ll):
        print("Bad lon-lat in %s"%cruise_fn)
        print(ll[bad_ll])
    
    cruise_df['x']=xy[:,0]
    cruise_df['y']=xy[:,1]
    cruise_df=cruise_df[ ['time','date','x','y']+[of[0] for of in obs_fields] ]
    cruise_df['src']=cruise_fn
    return cruise_df
    
def load_cruises():
    cruise_dfs = [load_cruise(cruise_fn) for cruise_fn in cruise_fns]
    return pd.concat(cruise_dfs) # 389k points
