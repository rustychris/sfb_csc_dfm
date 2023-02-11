import os
import xarray as xr
import numpy as np
import pandas as pd

import stompy.model.delft.dflow_model as dfm
import stompy.model.hydro_model as hm
import local_config
import dem_cell_node_bathy

import rough_regions as rr

cache_dir=os.path.join(local_config.model_dir,'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

ft_to_m=0.3048

class SfbCsc(local_config.LocalConfig,dfm.DFlowModel):
    """
    Base model domain
    """
    run_start=np.datetime64("2014-10-01") - np.timedelta64(150,'D')
    run_stop =np.datetime64("2014-10-01") + np.timedelta64(2,'D')
    
    run_dir=None

    salinity=False
    temperature=False
    projection='EPSG:26910'

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

    def __init__(self,*a,**kw):
        # HERE: check on most recent Pescadero setup -- would like to make it possible
        # to use this class both for loading existing runs, and configuring new runs.
        # that means moving some of this to a config() method.
        super().__init__(*a,**kw)
        
        self.set_grid_and_features()

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
            
        self.set_bcs()
        self.set_structures() # 1 call
        self.set_monitoring()
                    
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
        
        bathy_grid_fn=self.src_grid_fn.replace(".nc","-bathy-{self.bathy_version}.nc")
        assert bathy_grid_fn!=self.src_grid_fn

        if utils.is_stale(bathy_grid_fn,[self.src_grid_fn]+self.bathy_sources(self.bathy_version)):
            self.create_grid_with_bathy(self.src_grid_fn,bathy_grid_fn,self.bathy_version)

        g=unstructured_grid.UnstructuredGrid(bathy_grid_fn)
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
        elif bathy_version='base':    
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
        dredge_depth=-1

        self.set_barker_bc()
        self.set_ulatis_bc()
        self.set_campbell_bc()
        self.set_ocean_bc()
        self.set_lisbon_bc()
        self.set_sac_bc()

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
                          dredge_depth=dredge_depth,
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
        ulatis_ds=xr.open_dataset(os.path.join(here,"../bcs/ulatis/ulatis_hwy113.nc")).copy()
        # So convert to UTC
        ulatis_ds.time.values[:] += np.timedelta64(8,'h') # convert to UTC
        ulatis_ds['flow']=ulatis_ds.flow_cfs*0.02832
        ulatis_ds['flow'].attrs['units']='m3 s-1'
        ulatis_ds=self.pad_with_zero(ulatis_ds)
        
        bc=hm.FlowBC(name='Ulatis',flow=ulatis_ds.flow,dredge_depth=self.dredge_depth)
        self.add_bcs(ulatis)
        
    def set_campbell_bc(self):
        # Campbell Lake
        campbell_ds=xr.open_dataset(os.path.join(here,"../bcs/ulatis/campbell_lake.nc"))
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
                                dredge_depth=dredge_depth,
                                filters=[hm.FillGaps(),
                                         hm.Lowpass(cutoff_hours=1.0)])
        self.add_bcs(lisbon_bc)
        
        
    def set_ocean_bc(self):
        tidal_bc = hm.NOAAStageBC(name='ocean',station=9414290,cache_dir=cache_dir,
                                  filters=[hm.FillTidal(),
                                           hm.Lowpass(cutoff_hours=1.0)])
        self.add_bcs(tidal_bc)
            

    def pad_with_zero(self,ds,var='flow',pad=np.timedelta64(1,'D')):
        """
        Extend time to the padded extent of hte run, and extend the given variable with 
        zeros. Modifies ds in place! Lazy code!
        """
        if self.run_stop+pad>ds.time[-1]:
            log.warning("Will extend flow with 0 flow! (data end %s)"%(ds.time.values[-1]))
            # with xarray, easier to just overwrite the last sample.  lazy lazy.
            ds.time.values[-1] = self.run_stop+pad
            ds[var].values[-1] = 0.0
        if self.run_start-pad<ds.time[0]:
            log.warning("Will prepend flow with 0 flow! (data starts %s)"%(ds.time.values[0]))
            # with xarray, easier to just overwrite the last sample.  lazy lazy.
            ds.time.values[0] = self.run_start - pad
            ds[var].values[0] = 0.0
        return ds
    
    def setup_dcd(self):
        dcd_fn=os.path.join(here,"../bcs/dcd/dcd-1922_2016.nc")
        if not os.path.exists(dcd_fn):
            raise Exception("DCD is enabled, but did not find data file %s"%dcd_fn)
        ds_full=xr.open_dataset(dcd_fn)

        # Subset around the time of the simulaiton
        pad=np.timedelta64(10,'D')
        start_i,stop_i=np.searchsorted(ds_full.time, [self.run_start-pad,self.run_stop+pad])
        ds=ds_full.isel(time=slice(start_i,stop_i))
        
        valid_dcd_nodes=( (np.isfinite(ds.seep_flow).any(dim='time')).values |
                          (np.isfinite(ds.drain_flow).any(dim='time')).values |
                          (np.isfinite(ds.div_flow).any(dim='time')).values )
        # Index array with nodes having valid data
        valid_dcd_nodes=np.nonzero(valid_dcd_nodes)[0]

        pairs=[]
        bad_pairs=[] # for debugging, keep track of the nodes that fall outside the domain, too
        z_cell=self.grid.interp_node_to_cell(self.grid.nodes['node_z_bed'])

        cc=self.grid.cells_center()

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
    
    def setup_structures(self):
        # Culvert at CCS
        # rough accounting of restoration
        if self.ccs_pre_restoration:
            self.add_ccs_culvert()
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
        
        gen_polygons.gen_polygons(self.run_dir)
        self.write_bc_plots()

        try:
            script=__file__
        except NameError:
            script=None
        if script:
            shutil.copyfile(script,
                            os.path.join( os.path.join(self.run_dir,
                                                       os.path.basename(script) ) ))
        
    def write_bc_plots(self):
        # experimental saving of BC data to html plots
        for bc in self.bcs:
            bc.write_bokeh(path=self.run_dir)

    #############################################################################
    # Pescadero code 
    
    def __init__(self,*a,**k):
        super().__init__(*a,**k)

        # No horizontal viscosity or diffusion
        self.mdu['physics','Vicouv']=0.0
        self.mdu['physics','Dicouv']=0.0

        # Initial bedlevtype was 3: at nodes, face levels mean of node values
        # then 4, at nodes, face levels min. of node values
        # That led to salt mass being introduced erroneously
        # 6 avoided the salt issues, but was not as stable
        # 5 is a good tradeoff, and also allows the bed adjustment above to be simple
        self.mdu['geometry','BedLevType']=5
        
        self.mdu['output','StatsInterval']=300 # stat output every 5 minutes?
        self.mdu['output','MapInterval']=12*3600 # 12h.
        self.mdu['output','RstInterval']=4*86400 # 4days
        self.mdu['output','HisInterval']=900 # 15 minutes
        self.mdu['output','MapFormat']=4 # ugrid output format 1= older, 4= Ugrid

        self.mdu['numerics','MinTimestepBreak']=0.001

        self.mdu['physics','UnifFrictCoef']=0.023 # just standard value.

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

        self.mdu['output','Wrimap_waterlevel_s0']=0 # no need for last step's water level
            
        self.set_grid_and_features()
        self.set_bcs()
        self.add_monitoring()
        self.add_structures()
        self.set_friction()

        if self.salinity or self.temperature:
            self.mdu['physics','Idensform']=2 # UNESCO
            # 10 sigma layers yielded nan at wetting front, and no tidal variability.
            # 2D works fine -- seems to bring in the mouth geometry okay.
            # Must be called after grid is set
            self.config_layers()
        else:
            self.mdu['physics','Idensform']=0 # no density effects
        
    def config_layers(self):
        """
        Handle layer-related config, separated into its own method to
        make it easier to specialize in subclasses.
        Only called for 3D runs.
        """
        self.mdu['geometry','Kmx']=self.nlayers_3d # number of layers
        self.mdu['geometry','LayerType']=2 # all z layers
        self.mdu['geometry','ZlayBot']=self.z_min
        self.mdu['geometry','ZlayTop']=self.z_max
        
        # Adjust node elevations to avoid being just below interfaces
        # This may not be necessary.
        z_node=self.grid.nodes['node_z_bed'] # positive up
        kmx=self.nlayers_3d
        z_interfaces=np.linspace(self.z_min,self.z_max,kmx+1)

        write_stretch=False 
        if self.deep_bed_layer:
            grid_z_min=self.grid.nodes['node_z_bed'].min()
            if z_interfaces[0]>grid_z_min:
                self.log.info("Bottom interface moved from %.3f to %.3f to match deepest node of grid"%
                              (z_interfaces[0], grid_z_min))
                z_interfaces[0]=grid_z_min
                self.mdu['geometry','ZlayBot']=grid_z_min
                write_stretch=True
            
        dz_bed=z_interfaces[ np.searchsorted(z_interfaces,z_node).clip(0,kmx)] - z_node
        thresh=min(0.05,0.2*np.median(np.diff(z_interfaces)))
        # will deepen these nodes.  Could push them up or down depending
        # on which is closer, but generally we end up lacking conveyance to
        # err on the side of deepening
        adjust=np.where((dz_bed>0)&(dz_bed<thresh),
                        thresh-dz_bed, 0)
        self.grid.nodes['node_z_bed']-=adjust

        if write_stretch:
            # Based on this post:
            # https://oss.deltares.nl/web/delft3dfm/general1/-/message_boards/message/1865851
            self.mdu['geometry','StretchType']=1
            cumul=100*(z_interfaces-z_interfaces[0])/(z_interfaces[-1] - z_interfaces[0])
            # round to 0 decimals to be completely sure the sum is exact.
            # hopefully 1 decimal is okay. gives more even layers when getting up to 25+ layers.
            cumul=np.round(cumul,1)
            fracs=np.diff(cumul)
            # something like 10 10 10 10 10 10 10 10 10 10
            # Best not to make this too long.  100 layers with %.4f is too long for the
            # default partitioning script to handle, and this gets truncated.
            self.mdu['geometry','stretchCoef']=" ".join(["%.1f"%frac for frac in fracs])

        # These *might* help in 3D...
        # On RH laptop they cause 2D runs to fail during startup.
        self.mdu['time','AutoTimestep']=3 # 5=bad. 4 okay but slower, seems no better than 3.
        self.mdu['time','AutoTimestepNoStruct']=1 # had been 0
        
    def set_bcs(self):
        raise Exception("set_bcs() must be overridden in subclass")

    def add_monitor_transects(self,features,dx=None):
        """
        Add a sampled transect. dx=None will eventually just pull each
        cell along the line.  otherwise sample at even distance dx.
        """
        # Sample each cell intersecting the given feature
        assert dx is not None,"Not ready for adaptive transect resolution"
        for feat in features:
            pnts=np.array(feat['geom'])
            pnts=linestring_utils.resample_linearring(pnts,dx,closed_ring=False)
            self.log.info("Resampling leads to %d points for %s"%(len(pnts),feat['name']))
            # punt with length of the name -- not sure if DFM is okay with >20 characters
            pnts_and_names=[ dict(geom=geometry.Point(pnt),name="%s_%04d"%(feat['name'][:13],i))
                             for i,pnt in enumerate(pnts) ]
            self.add_monitor_points(pnts_and_names)

    def add_monitoring(self):
        self.add_monitor_points(self.match_gazetteer(geom_type='Point',type='monitor'))
        # Bad choice of naming. features labeled 'transect' are for cross-sections.
        # Features labeled 'section' are for sampled transects
        self.add_monitor_sections(self.match_gazetteer(geom_type='LineString',type='transect'))
        self.add_monitor_transects(self.match_gazetteer(geom_type='LineString',type='section'),
                                   dx=5.0)




# Do I need levee locations? That's part of the CSC model, I think.
