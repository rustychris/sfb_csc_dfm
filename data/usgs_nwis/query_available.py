# Develop methods to scan USGS stations in a LL box for

# That would be a clean way to get the starting station list.
import os
from stompy.io.local import usgs_nwis
from io import StringIO

import six
import xarray as xr
import requests
from stompy.io import rdb

from stompy.grid import unstructured_grid

##  

cache_dir='cache'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

##

# Do it all in one go?
six.moves.reload_module(usgs_nwis)

parameter=63680
parameter_name='turbidity'

df=usgs_nwis.stations_period_of_record(
    [-122.219,-121.253,37.756, 38.612],
    parameter, cache_dir=cache_dir)

# How many are available for a 2019 summer?
period=[np.datetime64("2019-04-01"),np.datetime64("2019-08-01")]
df_period=df[ (df.begin_date<period[1]) & (df.end_date>period[0]) ]

# 32 sites

##

# How do these compare to the grid?
# A few are one/close to shore. Safe to push them all to nearest cell.

ll=df_period[ ['dec_long_va','dec_lat_va'] ].values.astype(np.float64)
from stompy.spatial import proj_utils
xy = proj_utils.mapper('WGS84','EPSG:26910')(ll)


grid=unstructured_grid.UnstructuredGrid.read_ugrid("../../grid/v00/grid-bathy-base.nc")

##
import matplotlib.pyplot as plt

fig,ax=plt.subplots(num=1,clear=1)

# grid.plot_edges(ax=ax,color='k',lw=0.5,alpha=0.6)
grid.contourf_node_values(grid.nodes['node_z_bed'], np.linspace(-20,2,32), cmap='turbo',
                          alpha=0.75, ax=ax, extend='both')

ax.plot(xy[:,0],xy[:,1],'ro')

for name,loc in zip(df_period['station_nm'].values, xy):
    ax.text(loc[0],loc[1],name)

ax.axis('equal')

##

# Query it all, pack it into a netcdf.
# This way we avoid the DNS issues on farm.

all_data=[] # datasets from each time series

for row in df_period.itertuples():
    print(row.station_nm)
    ds=usgs_nwis.nwis_dataset(station=int(row.site_no),
                              start_date=max(period[0],row.begin_date),
                              end_date=min(period[1],row.end_date),
                              products=[parm],
                              cache_dir=cache_dir)
    all_data.append(ds)
    

##

# combine datasets, along with lat/long in df_period
# to get a master dataset

def combine_site(grp):
    start_date=max(period[0],grp.begin_date.min())
    end_date  =min(period[1],grp.end_date.max())
    site_no=grp.site_no.values[0]
    
    ds=usgs_nwis.nwis_dataset(station=int(site_no),
                              start_date=start_date, end_date=end_date,
                              products=[parm],name_with_ts_code=True,
                              cache_dir=cache_dir)
    data_vars=list(ds.data_vars)
    if len(data_vars)==1:
        da=ds[data_vars[0]]
    else:
        das=[ ds[v] for v in ds.data_vars]

        # If we knew which was surface, this would be the place to use that information.
        # as is, order them by increasing median.
        da_medians = [da.median() for da in das]
        da_ordered=[das[i] for i in np.argsort(da_medians)]
        da=da_ordered[0]
        for da_sub in da_ordered[1:]:
            da=da.combine_first(da_sub)

    lat=float(grp.dec_lat_va.values[0])
    lon=float(grp.dec_long_va.values[0])
    
    da=da.assign_coords(lat=lat,lon=lon,site_no=site_no,station_nm=grp.station_nm.values[0],
                        parm_cd=grp.parm_cd.values[0]) 

    return da

all_das=df_period.groupby('site_no').apply(combine_site)

##

# aggregate all data to 15 minute time steps.
# In some cases, like Suisun Bay Mallard Island, there are two elevations of sensors.
# Take the first one and warn.
all_data_15=[]
for i,da in enumerate(all_das):
    print(f"Processing {da.station_nm.item()}")

    da15=da.resample({'time':'15min'}).mean()
    ds15=xr.Dataset()
    ds15['turbidity']=da15
    all_data_15.append(ds15)

##
from stompy import utils

ds=xr.concat(all_data_15,dim='site')
fn=f"{parameter_name}-{utils.strftime(period[0],'%Y-%m-%d')}-{utils.strftime(period[1],'%Y-%m-%d')}.nc"
ds.to_netcdf(fn)

##

fig,ax=plt.subplots(num=1,clear=1)
img=ax.imshow(ds.turbidity.values,aspect='auto',cmap='turbo',interpolation='nearest')
img.set_clim([0,80])

