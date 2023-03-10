import os
import sys
import glob
import xarray as xr
import numpy as np
import pandas as pd
import shutil

import stompy.model.delft.dflow_model as dfm
import stompy.model.hydro_model as hm
from stompy import utils
from stompy.grid import unstructured_grid
from stompy.spatial import field
from stompy.io import dss

import local_config
import dem_cell_node_bathy
import barker_data

import rough_regions as rr

cache_dir=os.path.join(local_config.model_dir,'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

ft_to_m=0.3048

here=os.path.dirname(__file__)

class SfbCsc(local_config.LocalConfig,dfm.DFlowModel):
    """
    Base model domain
    """
    run_start=np.datetime64("2014-10-01") - np.timedelta64(150,'D')
    run_stop =np.datetime64("2014-10-01") + np.timedelta64(2,'D')
    
    run_dir=None
    extra_scripts=[] # additional scripts to copy into run directory

    salinity=False
    temperature=False
    nlayers=0 # 0 is 2D, 1 is 3D with a single layer
    scenario='base' # no scenarios at this point.
    
    dredge_depth=-1.0 
    
    projection='EPSG:26910'

    # DSS cache is particularly important because dssvue doesn't run in all places, so we probably want to
    # share the cached data across machines
    dss_cache_dir=os.path.join(os.path.dirname(__file__),"../data/dss_cache")
    gates_dss=os.path.join(os.path.dirname(__file__),"../data/dsm2/dsm2_2022.01_historical_update/timeseries/gates-v8.dss")
    hist_dss =os.path.join(os.path.dirname(__file__),"../data/dsm2/dsm2_2022.01_historical_update/timeseries/hist.dss")

    ###### CSC code
    
    # Forcing data is fetched as UTC, and adjusted according to this offset
    utc_offset=np.timedelta64(0,'h') # run model in UTC

    src_grid_fn=os.path.join(local_config.model_dir,'../grid/v00/grid.nc')
    bathy_version="base"

    wind = False
    dwaq = False
    dcd = True
    ptm_output = False
    
    # DCD nodes are matched to deep grid cells within this distance, otherwise
    # assumed to fall outside domain.
    # HERE: re-evaluate with the new grid
    dcd_node_tol=400 # (m) 500 includes one bad node at the DCC.  400 is good for the current grid.

    def configure(self):
        """
        Called when configuring a new run
        """
        super().configure()
        
        self.configure_global() # DFM-specific things

        self.set_grid_and_features()
        self.set_bcs()
        self.setup_structures() # 1 call
        self.setup_monitoring()
        
    def configure_global(self):
        """
        global, DFM-specific settings
        """
        # 6 is maybe better for getting good edges
        self.mdu['geometry','BedlevType']=6

        # fail out when it goes unstable.
        self.mdu['numerics','MinTimestepBreak']=0.01
        
        if self.ptm_output:
            self.mdu['output','WaqInterval']=1800 
            self.mdu['output','MapInterval']=1800
        else:
            self.mdu['output','MapInterval']=3600
            
        self.mdu['output','RstInterval']=86400

        if self.salinity:
            self.mdu['physics','Salinity']=1
            self.mdu['physics','InitialSalinity']=32.0
        else:
            self.mdu['physics','Salinity']=0
        if self.temperature:
            self.mdu['physics','Temperature']=1
            self.mdu['physics','InitialTemperature']=18.0 # rough pull from plots
        else:            
            self.mdu['physics','Temperature']=0
            
        if self.salinity or self.temperature:
            self.mdu['physics','Idensform']=2 # UNESCO
            self.mdu['numerics','TurbulenceModel']=3 # 0: breaks, 1: constant,  3: k-eps
            # These are values from Pescadero. Could be larger for SFE.
            self.mdu['physics','Dicoww']=1e-8
            self.mdu['physics','Vicoww']=1e-6 # used to be 1e-7, but isn't 1e-6 more appropriate?
        else:
            self.mdu['physics','Idensform']=0 # no density effects

        self.config_layers()

        
    def config_layers(self):
        """
        Handle layer-related config, separated into its own method to
        make it easier to specialize in subclasses.
        Now called for 2D and 3D alike
        """
        self.mdu['geometry','Kmx']=self.nlayers # number of layers
        self.mdu['geometry','LayerType']=1 # sigma

            
    def set_grid_and_features(self):
        grid_dir=os.path.dirname(self.src_grid_fn)
        # Register linear and point feature shapefiles
        self.add_gazetteer(os.path.join(os.path.dirname(self.src_grid_fn),
                                        "point-features.geojson"))
        self.add_gazetteer(os.path.join(os.path.dirname(self.src_grid_fn),
                                        "line-features.geojson"))
        self.add_gazetteer(os.path.join(os.path.dirname(self.src_grid_fn),
                                        "polygon-features.geojson"))
        self.add_gazetteer(os.path.join(os.path.dirname(self.src_grid_fn),
                                        "roughness-features.geojson"))
        # old way used shapefiles
        # self.add_gazetteer(os.path.join(grid_dir,"line_features.shp"))
        # self.add_gazetteer(os.path.join(grid_dir,"point_features.shp"))
        # self.add_gazetteer(os.path.join(grid_dir,"polygon_features.shp"))
        
        bathy_grid_fn=self.src_grid_fn.replace(".nc",f"-bathy-{self.bathy_version}.nc")
        assert bathy_grid_fn!=self.src_grid_fn

        if utils.is_stale(bathy_grid_fn,[self.src_grid_fn]+self.bathy_sources(self.bathy_version)):
            self.create_grid_with_bathy(self.src_grid_fn,bathy_grid_fn,self.bathy_version)

        g=unstructured_grid.UnstructuredGrid.read_ugrid(bathy_grid_fn)
        super().set_grid(g)

    def bathy_sources(self, bathy_version):
        """
        Return a list of paths, suitable for passing to MultiRasterField.
        (i.e. they will be stacked based on resolution, with no blending)
        Layers will not be reprojected, so should already match self.projection.

        This list is used both to set the bathymetry on the grid and to determine whether
        the grid bathymetry is outdated.
        """
        dwr_bathy_dir=os.path.join(local_config.common_data_dir,'bathy_dwr/gtiff')
        layers=[
            # 10m for the entire estuary.
            os.path.join(dwr_bathy_dir,'dem_bay_delta_10m_20201207.tif'),
            os.path.join(dwr_bathy_dir,'dem_calaveras_rvr_2m_20120902.tif'),
            os.path.join(dwr_bathy_dir,'dem_ccfb_south_delta_san_joaquin_rvr_2m_20200625.tif'),
            os.path.join(dwr_bathy_dir,'dem_columbia_cut_2m_20120911.tif'),
            os.path.join(dwr_bathy_dir,'dem_false_rvr_piper_sl_fisherman_cut_2m_20171109.tif'),
            os.path.join(dwr_bathy_dir,'dem_montezuma_sl_2m_20200909.tif'),
            os.path.join(dwr_bathy_dir,'dem_north_delta_2m_20201130.tif'),
            os.path.join(dwr_bathy_dir,'dem_sac_rvr_decker_isl_2m_20150914.tif'),
            os.path.join(dwr_bathy_dir,'dem_turner_cut_2m_20120907.tif'),
            os.path.join(dwr_bathy_dir,'dem_yolo_2m_20200505.tif'),
            os.path.join(dwr_bathy_dir,'dem_san_joaquin_bradford_2m_20151204_clip/dem_san_joaquin_bradford_2m_20151204_clip.tif'),
        ]

        # These are from the CSC model:
        if bathy_version=='pre_calhoun':
            layers.append( os.path.join(local_config.csc_bathy_dir,"merged-20190530-pre_calhoun.tif") )
        elif bathy_version=='base':    
            layers.append( os.path.join(local_config.csc_bathy_dir,"merged_2m-20190122.tif") )
            
        return layers
        
    def create_grid_with_bathy(self,src_grid_fn,dst_grid_fn,bathy_version):
        """ 
        Load grid without bathy, add bathymery. Depends on run_start in order to choose
        the right bathymetry.
        bathy_version: specify the version of the bathymetry.
        """
        bathy_sources=self.bathy_sources(bathy_version)
        
        g=unstructured_grid.UnstructuredGrid.from_ugrid(src_grid_fn)
        dem=field.MultiRasterField(bathy_sources)
        node_depths=dem_cell_node_bathy.dem_to_cell_node_bathy(dem,g)
        g.add_node_field('depth',node_depths,on_exists='overwrite')
        g.write_ugrid(dst_grid_fn,overwrite=True)
            
        return g

    def load_default_mdu(self):
        self.load_mdu(os.path.join(local_config.model_dir,'template.mdu'))

    # CSC model originally had special handling of Calhoun Cut to
    # revert the bathy when running periods before the "restoration"
    # (2014-11-01).
    # For simplicity, always use the post-restoration bathy. Makes almost
    # no difference unless right next to it. 

    def set_bcs(self):
        self.set_barker_bc()
        self.set_ulatis_bc()
        self.set_campbell_bc()
        self.set_ocean_bc()
        self.set_lisbon_bc()
        self.set_sac_bc()

        self.set_cvp_bc()
        self.set_swp_bc()

        if self.wind: # not including wind right now
            # Try file pass through for forcing data:
            windxy=self.read_tim('forcing-data/windxy.tim',columns=['wind_x','wind_y'])
            windxy['wind_xy']=('time','xy'),np.c_[ windxy['wind_x'].values, windxy['wind_y'].values]
            self.add_WindBC(wind=windxy['wind_xy'])

        self.set_roughness()

        if self.salinity:
            for tbc in tidal_bcs:
                # May not work with Threemile, originally just used Decker.
                self.add_bcs(hm.NwisScalarBC(station=tbc.station, parent=tbc, scalar='salinity'))

        if self.dwaq:
            self.setup_delwaq()
            # advect zero-order nitrate production coefficient from Sac River
            # self.add_bcs(dfm.DelwaqScalarBC(parent=sac, scalar='ZNit', value=1))
            self.add_bcs(dfm.DelwaqScalarBC(parent=sac, scalar='RcNit', value=1))

            sea_bcs=[dfm.DelwaqScalarBC(scalar='sea',parent=bc,value=1.0)
                     for bc in tidal_bcs]
            self.add_bcs(sea_bcs)

            nonsac_bcs=[dfm.DelwaqScalarBC(scalar='nonsac',parent=bc,value=1.0)
                        for bc in nonsac]
            self.add_bcs(nonsac_bcs)

        if self.dcd:
            self.setup_dcd()

    def set_sac_bc(self):
        # moving Sac flows upstream and removing tides.
        sac=hm.NwisFlowBC(name="SacramentoRiver",station=11447650,
                          pad=np.timedelta64(5,'D'),cache_dir=self.cache_dir,
                          dredge_depth=self.dredge_depth,
                          filters=[hm.LowpassGodin(),
                                   hm.Lag(np.timedelta64(-2*3600,'s'))])
        self.add_bcs(sac)
        
    def set_barker_bc(self):
        # check_bspp.py has the code that converted original tim to csv.
        bc=barker_data.BarkerPumpsBC(name='Barker_Pumping_Plant',dredge_depth=self.dredge_depth)
        self.add_bcs(bc)
        
    def set_ulatis_bc(self):
        # Ulatis inflow
        # Pretty sure that this data comes in as PST
        ulatis_ds=xr.open_dataset(os.path.join(here,"../data/lindsey/ulatis_hwy113.nc")).copy()
        # So convert to UTC
        ulatis_ds.time.values[:] += np.timedelta64(8,'h') # convert to UTC
        ulatis_ds['flow']=ulatis_ds.flow_cfs*0.02832
        ulatis_ds['flow'].attrs['units']='m3 s-1'
        ulatis_ds=self.pad_with_zero(ulatis_ds)
        
        bc=hm.FlowBC(name='Ulatis',flow=ulatis_ds.flow,dredge_depth=self.dredge_depth)
        self.add_bcs(bc)
        
    def set_campbell_bc(self):
        # Campbell Lake
        campbell_ds=xr.open_dataset(os.path.join(here,"../data/lindsey/campbell_lake.nc"))
        campbell_ds['flow']=campbell_ds.flow_cfs*0.02832
        campbell_ds['flow'].attrs['units']='m3 s-1'
        # Likewise, fairly sure this should be converted PST -> UTC
        campbell_ds.time.values[:] += np.timedelta64(8,'h')
        
        campbell_ds=self.pad_with_zero(campbell_ds)
        bc=hm.FlowBC(name='Campbell',flow=campbell_ds.flow,dredge_depth=self.dredge_depth)
        self.add_bcs(bc)

    def set_lisbon_bc(self):
        # LIS
        # WDL?
        # The grid extends north of Lisbon weir for a good distance.
        # Assuming that Lisbon weir flow can be plumbed in on the upstream
        # end of the toe drain canal
        # This is WDL stations B91560Q

        # sensor_no=12213&end=10/05/2021+09:43&geom=small&interval=2&cookies=cdec01
        
        lisbon_bc=hm.CdecFlowBC(name='lis',station="LIS",pad=np.timedelta64(5,'D'),
                                default=0.0,cache_dir=self.cache_dir,
                                dredge_depth=self.dredge_depth,
                                filters=[hm.FillGaps(),
                                         hm.Lowpass(cutoff_hours=1.0)])
        self.add_bcs(lisbon_bc)
        
        
    def set_ocean_bc(self):
        tidal_bc = hm.NOAAStageBC(name='ocean',station=9414290,cache_dir=cache_dir,
                                  filters=[hm.FillTidal(),
                                           hm.Lowpass(cutoff_hours=1.0)])
        self.add_bcs(tidal_bc)
            
    def set_swp_bc(self):
        # CHSWP003 from hist.dss
        # /FILL+CHAN/CHSWP003/FLOW-EXPORT//1DAY/${HISTFLOWVERSION}/
        self.log.warning("SWP not plumbed")

        # cache_by_full_path=False to make it easier to share cached data across machines.
        SWP_data=dss.read_records(self.hist_dss,
                                  "/FILL+CHAN/CHSWP003/FLOW-EXPORT//1DAY/DWR-DMS-202112/",
                                  cache_dir=self.dss_cache_dir,cache_by_full_path=False)
        # => DataFrame with time (dt64) and value (float) columns, no index.
        # times appear to be PST.
        # values (according to DSS-vue) are cfs, all positive.
        SWP_data.time.values[...] += self.utc_offset

        # Feature name is 'swp'
        
        self.log.warning("SWP not plumbed")

        # cfs
        da=xr.Dataset.from_dataframe(SWP_data.set_index('time'))['value']
        da.values[:] *= -0.3048**3 # to m3/s as export
        bc=hm.FlowBC(name='swp',flow=da,dredge_depth=self.dredge_depth)
        self.add_bcs(bc)

    def set_cvp_bc(self):
        # CHDMC004 from hist.dss (Delta-Mendota Canal, via Bill Jones Pumping Plant)
        CVP_data=dss.read_records(self.hist_dss,
                                  "/FILL+CHAN/CHDMC004/FLOW-EXPORT//1DAY/DWR-DMS-202112/",
                                  cache_dir=self.dss_cache_dir,cache_by_full_path=False)
        # => DataFrame with time (dt64) and value (float) columns, no index.
        # times appear to be PST.
        # values (according to DSS-vue) are cfs, all positive.
        CVP_data.time.values[...] += self.utc_offset

        # feature name is "cvp"
        # cfs
        da=xr.Dataset.from_dataframe(CVP_data.set_index('time'))['value']
        da.values[:] *= -0.3048**3 # to m3/s as export
        bc=hm.FlowBC(name='cvp',flow=da,dredge_depth=self.dredge_depth)
        self.add_bcs(bc)
        
    # Other exports/diversions:
    #  ccc - contra costa canal?
    #  cccoldr
    #  CCWP intake on Victoria Canal
    # Other sources
    #  stockton effluent
    #

    def pad_with_zero(self,ds,var='flow',pad=np.timedelta64(1,'D')):
        """
        Extend time to the padded extent of hte run, and extend the given variable with 
        zeros. Modifies ds in place! Lazy code!
        """
        if self.run_stop+pad>ds.time[-1]:
            self.log.warning("Will extend flow with 0 flow! (data end %s)"%(ds.time.values[-1]))
            # with xarray, easier to just overwrite the last sample.  lazy lazy.
            ds.time.values[-1] = self.run_stop+pad
            ds[var].values[-1] = 0.0
        if self.run_start-pad<ds.time[0]:
            self.log.warning("Will prepend flow with 0 flow! (data starts %s)"%(ds.time.values[0]))
            # with xarray, easier to just overwrite the last sample.  lazy lazy.
            ds.time.values[0] = self.run_start - pad
            ds[var].values[0] = 0.0
        return ds

    dcd_omits_evaporation=True
    
    def setup_dcd(self):
        # Update to use the CSV files from Eli - more recent dates, and evaporation
        # is not included.
        self.setup_dcd_no_evap()
        # self.setup_dcd_standard()

    def setup_dcd_no_evap(self):
        dcd_dir=os.path.join("../../data/dwr/schism/source_sink_dated")

        # Relevant files from DWR:
        # Daily csv. Columns are datetime then per-node
        #   msource_dated_v20210930.csv
        #     2000-01-01 through 2026-01-01
        #     values are 9.5-ish to 24ish. some -9999. Constant per time. Higher in summer.
        #     water temperature? evap in mm/day?
        #   source_202108_leach1_data.csv
        #   sink_202108_leach1_smcd1.3_data.csv
        #     elsewhere have concluded that these are CFS
        #     
        #   source_sink_dcd_delta_suisun.yaml
        #     {delta,suisun}_{src,sink}_\d+
        #     has many node indices but not all.

        # These are in CFS (as concluded in hybrid/data/dwr/schism/source_sink_dated/to_netcdf.py)
        # columns are 'datetime' (daily time step), and string node number
        df_sink  =pd.read_csv(os.path.join(dcd_dir,'sink_202108_leach1_smcd1.3_data.csv'),
                              parse_dates=['datetime'])
        df_source=pd.read_csv(os.path.join(dcd_dir,'source_202108_leach1_data.csv'),
                              parse_dates=['datetime'])
        
        # Subset around the time of the simulaiton
        pad=np.timedelta64(10,'D')
        start_i,stop_i=np.searchsorted(df_sink.datetime.values, [self.run_start-pad,self.run_stop+pad])
        df_sink=df_sink.iloc[start_i:stop_i,:]
        
        start_i,stop_i=np.searchsorted(df_source.datetime.values, [self.run_start-pad,self.run_stop+pad])
        df_source=df_source.iloc[start_i:stop_i,:]

        # handle the combination in pandas
        df_sink['type']='sink'
        df_source['type']='source'
        dfA=df_sink.set_index(['type','datetime'])
        dfB=df_source.set_index(['type','datetime'])
        dfA.columns=dfA.columns.astype(int) # '123' => 123
        dfA.columns.name='node'
        dfB.columns=dfB.columns.astype(int) # ..
        dfB.columns.name='node'
        
        df=pd.concat([dfA,dfB],axis=0)
        assert np.diff(df.columns.values).min() > 0,"Hmm - that was supposed to end up sorted"
        df=df.stack(0).unstack(0) # make nodes another row index, and source/sink will become columns
        ds=xr.Dataset.from_dataframe(df) # and now we get datetime and node as coordinates, and source/sink as variables.
        
        import yaml

        with open(os.path.join(dcd_dir,"source_sink_dcd_delta_suisun.yaml"),"rt") as fp:
            utm=yaml.safe_load(fp) # raises yaml.YAMLError

        # utm: {'sinks': {'{delta,suisun}_sink_\d+':{x,y}, ... },
        #       'sources':{'{delta,suisun}_src_\d+':{x,y}, ... },
        #
        # len(delta_sink_nodes) => 256
        # len(suisun_sink_nodes) => 75
        # lists of nodes are identical between sources and sinks
        # and there is no overlap between delta and suisun
        # assuming that coordinates are also identical between source and sinks
        node_xy=[ (int(s.split('_')[2]),utm['sources'][s]) for s in utm['sources'].keys()]
        nodes,xy=[np.array(x) for x in zip(*node_xy)]
        order=np.argsort(nodes)
        nodes=nodes[order]
        xy=xy[order]

        utm_ds=xr.Dataset()
        utm_ds['node'] = ('node',), nodes
        utm_ds['node_x'] = ('node',), xy[:,0]
        utm_ds['node_y'] = ('node',), xy[:,1]

        ds=ds.rename({'sink':'div_flow','source':'drain_flow','datetime':'time'})
        ds['seep_flow'] = ('time','node'), np.zeros( (ds.dims['time'],ds.dims['node']))

        # and splice in utm...
        ds['node_x']=utm_ds['node_x']
        ds['node_y']=utm_ds['node_y']
        
        self.setup_dcd_common(ds)
    
    def setup_dcd_standard(self):
        dcd_fn=os.path.join(here,"../bcs/dcd/dcd-1922_2016.nc")
        if not os.path.exists(dcd_fn):
            raise Exception("DCD is enabled, but did not find data file %s"%dcd_fn)
        ds_full=xr.open_dataset(dcd_fn)

        # Subset around the time of the simulaiton
        pad=np.timedelta64(10,'D')
        start_i,stop_i=np.searchsorted(ds_full.time, [self.run_start-pad,self.run_stop+pad])
        ds=ds_full.isel(time=slice(start_i,stop_i))

        self.setup_dcd_common(ds)

    def setup_dcd_common(self,ds):
        valid_dcd_nodes=( (np.isfinite(ds.seep_flow).any(dim='time')).values |
                          (np.isfinite(ds.drain_flow).any(dim='time')).values |
                          (np.isfinite(ds.div_flow).any(dim='time')).values )
        # Index array with nodes having valid data
        valid_dcd_nodes=np.nonzero(valid_dcd_nodes)[0]

        pairs=[]
        bad_pairs=[] # for debugging, keep track of the nodes that fall outside the domain, too
        z_cell=self.grid.interp_node_to_cell(self.grid.nodes['node_z_bed'])

        cc=self.grid.cells_centroid() # use centroid as center is not guaranteed to be inside cell

        for n in valid_dcd_nodes:
            dsm_x=np.r_[ ds.node_x.values[n], ds.node_y.values[n]]
            c_near=self.grid.select_cells_nearest(dsm_x)

            # And then find the deepest cell nearby, subject
            # to distance from DSM point
            c_nbrs=[c_near]
            for _ in range(5):
                c_nbrs=np.unique([ nbr
                                   for c in c_nbrs
                                   for nbr in self.grid.cell_to_cells(c)
                                   if nbr>=0 and utils.dist(dsm_x,cc[nbr])<self.dcd_node_tol])
            if len(c_nbrs)==0:
                bad_pairs.append( [n, dsm_x, cc[c_near]] )
            else:
                match=c_nbrs[ np.argmin(z_cell[c_nbrs]) ]
                pairs.append( [n, dsm_x, cc[match]] )
                
        # For each good matchup, add a source/sink.
        for n,dsm_x,dfm_x in pairs:
            # positive into the domain
            Q=(   ds['drain_flow'].isel(node=n)
                - ds['seep_flow'].isel(node=n)
                - ds['div_flow'].isel(node=n)
            )
            
            if not np.all(np.isfinite(Q)): # ,"Need to fill some nans, I guess":
                import pdb
                pdb.set_trace()
            bc=hm.SourceSinkBC(name="DCD%04d"%n,flow=Q,geom=dfm_x)
            self.add_bcs([bc])
            if self.dwaq:
                self.add_bcs(dfm.DelwaqScalarBC(parent=bc,scalar='drain',value=1.0))
        
    def set_roughness(self):
        # These are setting that came out of the optimization
        settings=dict(
            cache=0.04,
            dws=0.025,
            elk=0.015,
            fpt_to_dcc=0.030,
            lindsey=0.04,
            miner=0.035,
            rio_vista=0.025,
            sac_below_ges=0.0225,
            steamboat=0.025,
            toe=0.0375,
            upper_sac=0.0175)
        
        xyn=rr.settings_to_roughness_xyz(self,settings)
        # Turn that into a DataArray
        da=xr.DataArray( xyn[:,2],dims=['location'],name='n' )
        da=da.assign_coords(x=xr.DataArray(xyn[:,0],dims='location'),
                            y=xr.DataArray(xyn[:,1],dims='location'))
        da.attrs['long_name']='Manning n'
        rough_bc=hm.RoughnessBC(data_array=da)
        self.add_bcs(rough_bc)

    def setup_delwaq(self):
        pass # configure in subclass or mixin

    @property
    def ccs_pre_restoration(self):
        return self.run_start < np.datetime64("2014-11-01")

    def setup_structures(self):
        # Culvert at CCS
        # rough accounting of restoration
        if self.ccs_pre_restoration:
            self.add_ccs_culvert()

        self.add_dcc()

    def add_dcc(self):
        DCC_data=dss.read_records(self.gates_dss,
                                  "/HIST+GATE/RSAC128/POS//IR-YEAR/DWR-DMS-DSM2/",
                                  cache_dir=self.dss_cache_dir,cache_by_full_path=False)

        # => DataFrame with time (dt64) and value (float) columns, no index.
        # times appear to be PST.
        # Value is 0,1,2, corresponding to how many gates are open.
        DCC_data.time.values[...] += self.utc_offset

        da=xr.Dataset.from_dataframe(DCC_data.set_index('time'))['value']
        width=40*da

        # 
        self.add_Structure(name='dcc',
                           type='gate',
                           door_height=20, # no overtopping?
                           lower_edge_level=-10, # should be deep enough to hit the bed.
                           opening_width=da,
                           sill_level=-20,
                           horizontal_opening_direction = 'symmetric')
        
    def add_ccs_culvert(self):
        # name => id
        # polylinefile is filled in from shapefile and name

        # these structure parameters get an amplitude ratio at CCS of 1.241
        # the shape is not great, and suggests that as the stage gets close to 1.7
        # or so, wetted area expands rapidly. That is probably a shortcoming of the
        # current bathymetry.
        self.add_Structure(name='ccs_breach',
                           type='gate',
                           door_height=15, # no overtopping?
                           lower_edge_level=0.9,
                           # in release 1.5.2, this needs to be nonzero.  Could use
                           # width of 0.0 in some older versions, but no longer.
                           opening_width=0.1,
                           sill_level=0.8,
                           horizontal_opening_direction = 'symmetric')
        # these are really just closed
        for levee_name in ['ccs_west','ccs_east']:
            self.add_Structure(name=levee_name,
                               type='gate',
                               door_height=15,
                               lower_edge_level=3.5,
                               opening_width=0.0,
                               sill_level=3.5,
                               horizontal_opening_direction = 'symmetric')

    def setup_monitoring(self):
        # -- Extract locations for sections and monitor points
        mon_sections=self.match_gazetteer(monitor=1,geom_type='LineString')
        mon_points  =self.match_gazetteer(geom_type='Point')
        self.add_monitor_sections(mon_sections)
        self.add_monitor_points(mon_points)

    def write(self):
        super().write()
        
        # gen_polygons.gen_polygons(self.run_dir)
        self.write_bc_plots()

        scripts=['local_config.py'] + self.extra_scripts
        
        try:
            scripts.append(__file__)
        except NameError:
            scripts.append("sfb_csc.py")

        # could be smarter on restarts
        script_dir=os.path.join(self.run_dir,"scripts")
        if not os.path.exists(script_dir):
            os.makedirs(script_dir)
            
        for script in scripts:
            shutil.copyfile(script,
                            os.path.join( os.path.join(script_dir,
                                                       os.path.basename(script) ) ))
        
    def write_bc_plots(self):
        # experimental saving of BC data to html plots
        for bc in self.bcs:
            bc.write_bokeh(path=self.run_dir)

    @classmethod
    def main(cls,argv):
        import argparse

        parser = argparse.ArgumentParser(description='Run SFE model with CSC focus.')

        parser.add_argument('-n','--num-cores',help='Number of cores',
                            default=local_config.LocalConfig.num_procs, type=int)

        parser.add_argument('--mdu',help='Path to MDU file for restarts')

        parser.add_argument('-s','--scenario',help='Select scenario (scen1,scen2,scen2)',
                            default=cls.scenario)

        parser.add_argument('-p','--period',help='Select run period. np.datetime64 parseable, separated by colon',
                            default='2016-06-01:2016-06-10',type=str)

        parser.add_argument('-r','--run-dir',help='override default run_dir',
                            default=None,type=str)

        parser.add_argument('-l','--layers',help='Number of layers',default=0,type=int)

        parser.add_argument('--temperature',help="Enable temperature",action='store_true')
        parser.add_argument('--salinity',help="Enable salinity",action='store_true')

        # Get the MPI flavor to know how to identify rank and start the tasks
        parser.add_argument("-m", "--mpi", help="Enable MPI flavor",default=local_config.LocalConfig.mpi_flavor)

        parser.add_argument("--resume",help="Resume a run from last restart time, or a YYYY-MM-DDTHH:MM:SS timestamp if given",
                            const='last',default=None,nargs='?')

        args = parser.parse_args(argv)

        cls.driver_main(args)

    @classmethod
    def driver_main(cls,args):
        kwargs=dict(scenario=args.scenario,
                    num_procs=args.num_cores,
                    nlayers=args.layers,
                    salinity=args.salinity,
                    temperature=args.temperature)
        
        run_dir="data"

        t_start,t_stop = args.period.split(':')
        
        # even for resume, use the same logic here, then come back to make changes
        if t_start: # can be empty for restarts.
            kwargs['run_start']=np.datetime64(t_start)
        kwargs['run_stop']=np.datetime64(t_stop)

        run_dir += "_" + utils.to_datetime(kwargs['run_start']).strftime('%Y%m%d')
        
        if kwargs['nlayers']!=0:
            run_dir+=f"_l{kwargs['nlayers']}"

        if kwargs['scenario']!=cls.scenario:
            run_dir+=f"_{kwargs['scenario']}"

        if args.run_dir is not None:
            run_dir=args.run_dir

        # Choose a unique run dir by scanning through, select the next suffix
        # beyond the largest one already there
        last_suffix_found=-1
        def fn_suffixed(suffix): 
            if suffix==0:
                test_dir=run_dir
            else:
                test_dir=run_dir+f"-v{suffix:03d}"
            return test_dir
        max_runs=100
        for suffix in range(max_runs):
            test_dir=fn_suffixed(suffix)
            if os.path.exists(test_dir):
                last_suffix_found=suffix
        if last_suffix_found < max_runs-1:
            suffix=last_suffix_found+1
            kwargs['run_dir']=fn_suffixed(suffix)
        else:
            raise Exception("Failed to find unique run dir (%s)"%test_dir)

        if args.resume is not None:
            old_model=cls.load(args.mdu)
            deep=True

            parent_dir=os.path.dirname(args.mdu)
            import re
            m=re.search('_r([0-9][0-9])$',parent_dir)
            if m:
                # Not trying for perfection -- just handle the common case of chained
                # restarts
                parent_restart=int(m.group(1))
                parent_dir=parent_dir[:m.start()] # drops the _rNN
            else:
                parent_restart=-1

            for suffidx in range(parent_restart+1,10):
                suffix=f"_r{suffidx:02d}"
                run_dir=parent_dir+suffix
                if not os.path.exists(run_dir):
                    break
            else:
                # no real limit, but probably a sign of a bug.
                raise Exception("Too many restarts - ran out of suffixes")
            model=old_model.create_restart(deep=True,**kwargs)
            model.set_run_dir(run_dir,mode='noclobber')

            # selectively pull in kwargs
            # ignore terrain z_max z_min scenario num_procs, nlayers_3d, flow_regime
            #    run_start salinity temperature slr
            #  include run_stop
            # For now, run_start defaults to last restart file.
            if args.resume!='last':
                # to choose something else, set model.run_start and call model.set_restart_file()
                # is it enough to call model.update_config()? yes
                # but who is calling update_config? HydroModel.write() calls it.
                model.run_start=np.datetime64(args.resume)

        else:
            model=cls(**kwargs)


        model.write() # this is now calling DFlowModel.write() for restarts

        # be careful with restarts
        script_dir=model.run_dir

        shutil.copyfile(__file__,os.path.join(script_dir,os.path.basename(__file__)))
        shutil.copyfile("local_config.py",os.path.join(script_dir,"local_config.py"))
        with open(os.path.join(script_dir,"cmdline"),'wt') as fp:
            fp.write(str(args))
            fp.write("\n")
            fp.write(" ".join(sys.argv))

        # Recent DFM has problems reading cached data -- leads to freeze.
        # assuming this is safe even for restarts
        for f in glob.glob(os.path.join(model.run_dir,'*.cache')):
            os.unlink(f)
        if 'SLURM_JOB_ID' in os.environ:
            with open(os.path.join(script_dir,'job_id'),'wt') as fp:
                fp.write(f"{os.environ['SLURM_JOB_ID']}\n")

        model.partition()

        try:
            print(model.run_dir)
            if model.num_procs<=1:
                nthreads=8
            else:
                nthreads=1
            model.run_simulation(threads=nthreads)
        except subprocess.CalledProcessError as exc:
            print(exc.output.decode())
            raise

if __name__=='__main__':
    SfbCsc.main(sys.argv[1:])
# SfbCsc.main("script ".split())

