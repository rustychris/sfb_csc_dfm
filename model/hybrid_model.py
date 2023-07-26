"""
1. Duplicate run of sfb_csc Done
2. Configure DWAQ tracers
   - need fresh compile that includes dwaq. Done.
   - set weight and tracers at BCs. Done for constant.
   - weight, weight*ssc tracers with longer delay. Done, not tested.
   - tau exposure with decay
3. Run it via BMI
   - processes are problematic
   - how to specify process library? in theory, set_var('processlibrary','path')
     but some numpy/ctypes machinery is missing.
   - does bmi expose dwaq tracers? Appears the answer is no.
   - options:
       a. Patch DFM, maybe just to expose all DWAQ tracers as a matrix.
       b. Use discharges, including a forcing tracer that dissipates immediately
          and triggers erasure of the existing state. Not clear exactly how this
          would work.
       c. Move the integration from DWAQ into python, do it all via BMI. Not clear
          whether BMI exposes enough state to do this. Could use temperature? and
          or some sediment classes?
          I think it needs to be a variable with shape ndkx, indicating 3D state.
          vextcum: no. This is cumulative inflow volume from qextreal.
          rnveg, diaveg, Cdvegsp: rnveg = rho veg == density
            -- these need vegetation (javeg) turned on. Maybe setting Cdvegsp=0 would be enough?
               Could probably check the vegetation workshop examples for hints about controlling friction.
            -- but are these actually advected?
          zws: no. This is z elevation of interfaces (w-points) at cell centers.


4. Set tracers during run, internal to domain
5. Extract correlates at the same time.

"""
import logging
logging.basicConfig(level=logging.INFO)

import sys
import sfb_csc
import stompy.model.delft.waq_scenario as waq
import stompy.model.delft.io as dio
from stompy.spatial import proj_utils
import stompy.model.hydro_model as hm
from stompy.grid import unstructured_grid
import numpy as np
import custom_process as custom
import os
import time
import xarray as xr


class HybridModel(sfb_csc.SfbCsc):
    dt=900.0 # update interval, in seconds. Used on class, not instance.
    
    # Invocation machinery
    @classmethod
    def arg_parser(cls):
        parser=super().arg_parser()

        parser.add_argument('-b','--bmi',action='store_true',
                            help='run as a BMI task')
        # --mdu already exists

        # See also pescadero/model/run_production_bmi.py
        # with bmi, mdu must also be set.
        return parser

    @classmethod
    def driver_main(cls,args):
        if args.bmi:
            assert args.mdu is not None
            cls.bmi_main(args)
        else:
            super().driver_main(args)

    def run_simulation(self,threads=1):
        """
        BMI version, start simulation. 
        Ignores threads, extra_args
        """
        num_procs=self.num_procs

        # May need to get smarter if there is further subclassing
        real_cmd=['python',__file__,'--bmi']
        options=[]
        
        options += ['--mdu',self.mdu.filename]

        if num_procs>1:
            real_cmd = real_cmd + ["--mpi=%s"%self.mpi_flavor] + options
            return self.mpirun(real_cmd)
        else:
            real_cmd= real_cmd + options
            self.log.info("Running command: %s"%(" ".join(real_cmd)))
            subprocess.call(real_cmd)

    @classmethod
    def bmi_main(cls,args):
        model=cls.load(args.mdu)
        model.run_bmi_simulation(args)
        
    def run_bmi_simulation(self,args):
        mdu=self.mdu

        if args.mpi is None:
            print("args.mpi is None")
            rank=0
        elif args.mpi in ['mpiexec','mpich','intel','slurm']:
            rank=int(os.environ['PMI_RANK'])
        else:
            raise Exception("Don't know how to find out rank")

        self.rank=rank

        #log_fn=os.path.join(os.path.dirname(args.mdu),f'log-{rank}')
        logging.warning("Top of task_main")
        logging.info('This message should go to the log file')
        logging.debug(f"PID {os.getpid()}  rank {rank}")

        import local_config
        import bmi.wrapper
        from numpy.ctypeslib import ndpointer  # nd arrays
        from ctypes import (
            # Types
            c_double, c_int, c_char_p, c_bool, c_char, c_float, c_void_p,
            # Complex types
            # ARRAY, Structure,
            # Making strings
            # Pointering
            POINTER, byref, # CFUNCTYPE,
            # Loading
            # cdll
        )

        for k in os.environ:
            if ('SLURM' in k) or ('MPI' in k):
                logging.debug(f"{k} => {os.environ[k]}")

        logging.info(f"[rank {rank}] about to open engine")

        sim=bmi.wrapper.BMIWrapper(engine=os.path.join(local_config.dfm_root,
                                                       "lib/libdflowfm.so"))
        logging.info(f"[rank {rank}] done with open engine")
        self.sim=sim
        self.rank=rank

        if rank==0:
            logging.info("dir(sim)")
            logging.info(str(dir(sim)))

        # Pescadero run_production_bmi.py has a lot more going on. See that for
        # hints.
        
        # runs don't always start at the reference time
        tstart_min=float(mdu['time','tstart'])/60
        is_restart = (mdu['restart','RestartFile'] not in [None,""])

        # if rank==0:
        #     SETUP for global input data
            
        # dfm will figure out the per-rank file
        logging.info(f"[{rank}] about to initialize")
        sim.initialize(args.mdu) # changes working directory to where mdu is.
        logging.info(f"[{rank}] initialized")


        self.init_observations()
        
        # if rank==0:
        #     READ initial history file if needed

        reported_dir_sim=False # TMP
        
        # TASK TIME LOOP
        t_calc=0.0
        t_bmi=0.0
        t_last=time.time()
        while sim.get_current_time()<sim.get_end_time(): 
            t_now=sim.get_current_time()

            # Do the work...
            t_now_dt64=self.ref_date + np.timedelta64(1,'s') * t_now
            self.update_tracers(t_now_dt64, self.dt)

            if rank==0 and not reported_dir_sim: # TMP
                reported_dir_sim=True
                logging.info("dir(sim)")
                logging.info(str(dir(sim)))

                for var_i in range(sim.get_var_count()):
                    logging.info(f"var {var_i}: {sim.get_var_name(var_i)}")
                
            t_bmi+=time.time() - t_last
            t_last=time.time()
            logging.info(f'taking a step dt={self.dt}')
            sim.update(self.dt)
            logging.info('Back from step')
            t_calc+=time.time() - t_last
            t_last=time.time()

            # Running via BMI will not fail out when the time step gets too short, but it will
            # return back to here without going as far as we requested.
            t_post=sim.get_current_time()
            if t_post<t_now+0.75*self.dt:
                logging.error("Looks like run has stalled out.")
                logging.error(f"Expected a step from {t_now} to {t_now+dt} but only got to {t_post}")
                logging.error("Will break out")
                break

            if rank==0:
                logging.info(f"t_bmi: {t_bmi}   t_calc: {t_calc}")

        sim.finalize()

    def load_observations(self):
        obs_fn=os.path.join( os.path.dirname(__file__), "../data/usgs_nwis/turbidity-2019-04-01-2019-08-01.nc")
        ds=xr.open_dataset(obs_fn)

        ll=np.c_[ds['lon'].values, ds['lat'].values]
        utm=proj_utils.mapper('WGS84','EPSG:26910')(ll)
        ds['x']=ds['lon'].dims,utm[:,0]
        ds['y']=ds['lat'].dims,utm[:,1]
        return ds
    
    def init_observations(self):
        self.obs_ds=self.load_observations()

        # Figure out our local grid, too.
        # While we could load the partitioned net file, DFM may have renumbered the grid
        # so it's safer to load from BMI data.
        # there are ..
        #  netelemnode dim 2
        #  flowelemnode, flowelemnbs, flowelemnlns  dim 2
        #  flowelemcontour_{x,y}  dim 2

        # or.... iglobal_s: "global flow node numbers"
        # get_var documentations says this should correspond to "original unpartitioned flow node numbers"
        # Note that it includes boundary "elements"

        # # For debugging: log local grid to a netcdf so we can figure this out visually
        # sim=self.sim
        # ds_log=xr.Dataset()
        # self.log.warning("Read netelemnode")
        # ds_log['netelemnode']=('nump1d2d','net_elem_max_nodes'), sim.get_var('netelemnode')
        # self.log.warning("Read flowelemnode")
        # ds_log["flowelemnode"]=('ndx2d','flow_elem_max_nodes'),sim.get_var('flowelemnode')
        # # Adjacency, presumably cell:cell
        # self.log.warning("Read flowelemnbs")
        # ds_log["flowelemnbs"]=('ndx','flow_elem_max_nbs'), sim.get_var('flowelemnbs')
        # # Adjacency, cell:face
        # self.log.warning("Read flowelemlns")
        # ds_log["flowelemlns"]=('ndx','flow_elem_max_nbs'), sim.get_var('flowelemlns')
        # #self.log.warning("Read flowelemcontour_x")
        # #ds_log["flowelemcontour_x"]=('ndx','flow_elem_max_contour'),sim.get_var('flowelemcontour_x')
        # #self.log.warning("Read flowelemcontour_y") # Triggers bug. array already allocated
        # #ds_log["flowelemcontour_y"]=('ndx','flow_elem_max_contour'),sim.get_var('flowelemcontour_y')
        # self.log.warning("Read iglobal_s")
        # ds_log["iglobal_s"]=('ndx',),sim.get_var('iglobal_s')
        # self.log.warning("Read ndxi")
        # ds_log["ndxi"] = (),sim.get_var('ndxi') # 2D+1D flowcells
        # self.log.warning("Read ndx1db")
        # ds_log["ndx1db"] =(),sim.get_var('ndx1db') # flow nodes incl. 1D bnds. 2D+1D+1D bnds
        # self.log.warning("Read lnxi")
        # ds_log["lnxi"]=(),sim.get_var('lnxi') # flow links, 1D and 2D
        # self.log.warning("Read xk")
        # ds_log["xk"]=('numk'),sim.get_var('xk') # node coordinates?
        # ds_log["yk"]=('numk'),sim.get_var('yk') # node coordinates?
        # self.log.warning("Read xz")
        # ds_log["xz"]=("ndx"),sim.get_var('xz') # cell centers
        # ds_log["yz"]=("ndx"),sim.get_var('yz') # cell centers
        # ds_log.to_netcdf(f"bmi-grid-{self.rank:04d}.nc")
        
        # the "interpreted" global grid. This should match up with iglobal_s-1 in the partitioned
        # grids.
        # Note that with BMI, by the time this is executed we have changed
        # PWD to run_dir
        self.g_int = unstructured_grid.UnstructuredGrid.read_dfm(
            os.path.join( 'DFM_interpreted_idomain_' + self.mdu['geometry','NetFile']))

        # Make it zero-based.  This maps local cells, 0-based, to global cells, but it includes
        # boundary cells.
        self.iglobal_s = self.sim.get_var('iglobal_s') - 1
        ndxi = self.sim.get_var('ndxi') # number of 1D and 2D flow elements.
        self.ilocal_to_global = self.iglobal_s[:ndxi] # just flow elements.
        #self.log.warning("iglobal_s0 ranges from %d to %d"%(self.iglobal_s.min(), self.iglobal_s.max()))
        #self.log.warning("g_int has %d cells"%(self.g_int.Ncells()))

        assert self.ilocal_to_global.max() < self.g_int.Ncells()
        
        # Make the reverse:
        self.iglobal_to_local = np.zeros(self.g_int.Ncells(), np.int32)
        self.iglobal_to_local[:] = -1 # not mapped
        for c_local,c_global in enumerate(self.ilocal_to_global):
            self.iglobal_to_local[c_global]=c_local

        ds=self.obs_ds
        site_stencils=np.zeros(ds.dims['site'],dtype=object)
        for site_i in range(ds.dims['site']):
            xy=np.r_[ ds['x'].isel(site=site_i),
                      ds['y'].isel(site=site_i) ]
            # prescreened these to be sure sites actually lined up with
            # the grid. No need to be overly cautious here
            cell=self.g_int.select_cells_nearest(xy)
            weights=np.zeros(self.g_int.Ncells(),np.float64)
            # Could expand/diffuse on the global grid right now...
            weights[cell]=1.0
            nonzero_cells=np.nonzero(weights>0.01)[0] # or some threshold.
            nonzero_weights=weights[nonzero_cells]
            # These are still on the global grid
            local_cells=self.iglobal_to_local[nonzero_cells] # some of these will be negative
            valid=local_cells>=0
            local_cells=local_cells[valid]
            local_weights=nonzero_weights[valid]
            
            stencil=np.zeros(len(local_cells),dtype=[ ('cell',np.int32), ('weight',np.float64) ])
            stencil['cell'][:]=local_cells
            stencil['weight'][:]=local_weights
            site_stencils[site_i]=stencil
        # This was failing. Trying now with more explicit construction
        ds['stencil']=('site',),site_stencils
        
    obs_field='turbidity'
    def update_observations(self,t,wt,wt_obs):
        """
        t: current time as datetime64
        """
        ds=self.obs_ds
        # First cut has a global time array
        if t<ds.time.values[0]: return
        if t>ds.time.values[-1]: return

        t_idx=np.searchsorted(ds.time.values, t)
        
        for site_i in range(ds.dims['site']):
            val = ds[self.obs_field].isel(time=t_idx,site=site_i).item()
            if np.isnan(val): continue
            # without the item(), it failed because 'weight' wasn't understood.
            stencil=ds['stencil'].isel(site=site_i).item() # ?

            # Currently only handles weight=1 case
            # Having some global/local friction here.
            # Pretty sure stencil is coming in global.
            wt[stencil['cell']]=stencil['weight']
            wt_obs[stencil['cell']] = stencil['weight'] * val
        
    def update_tracers(self, t, dt_s):
        """
        t: current time as np.datetime64
        dt_s: time step between calls to update_tracer, in seconds
        """
        sim=self.sim
        # Decay for wt, wt_obs
        weight_decay_time = 3*3600.
        decay0_time=3600.
        decay1_time=86400.
        
        fac=np.exp(-dt_s/weight_decay_time)

        wt=sim.get_var('wt')
        wt_obs=sim.get_var('wt_obs')
        wt *= fac
        wt_obs *= fac
        
        self.update_observations(t,wt,wt_obs)

        sim.set_var('wt',wt)
        sim.set_var('wt_obs',wt_obs)

        # This is 2D!  Maybe it will be okay for a 2D run. But might have shape trouble.
        tau = sim.get_var('taus') # cell centre tau N/m2 {"location": "face", "shape": ["ndx"]}
        
        tauDecay0 = sim.get_var('tauDecay0')
        fac0=np.exp(-dt_s/decay0_time)
        tauDecay0[:] = tauDecay0*fac0 + (1-fac0)*tau
        sim.set_var('tauDecay0',tauDecay0)

        # Somehow this is getting the same results for decay1 and decay0
        tauDecay1 = sim.get_var('tauDecay1')
        fac1=np.exp(-dt_s/decay1_time)
        tauDecay1[:] = tauDecay1*fac1 + (1-fac1)*tau
        sim.set_var('tauDecay1',tauDecay1)
        
    # Model setup
    def configure(self):
        super().configure()
        self.add_hybrid_tracers()
        
    def add_hybrid_tracers(self):
        # Will tracers default to 0 initial condition and BC value?
        
        # BCs will already in place when this is called 
        # (sfb_csc...set_bcs(), called from sfb_csc...configure())
        self.tracers += ['wt_obs','wt','tauDecay0','tauDecay1']
        
        # For initial test, tag Sac inflow
        sac=self.scalar_parent_bc("SacramentoRiver")
        sac_wt_obs = hm.ScalarBC(parent=sac,scalar='wt_obs',value=10.0)
        sac_wt = hm.ScalarBC(parent=sac,scalar='wt',value=1.0)
        sac_tauDecay0 = hm.ScalarBC(parent=sac,scalar='tauDecay0',value=0.0)
        sac_tauDecay1 = hm.ScalarBC(parent=sac,scalar='tauDecay1',value=0.0)
        self.add_bcs([sac_wt,sac_wt_obs,sac_tauDecay0,sac_tauDecay1])


        
if __name__=='__main__':
    HybridModel.main(sys.argv[1:])

