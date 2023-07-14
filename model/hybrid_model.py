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
import stompy.model.hydro_model as hm
import numpy as np
import custom_process as custom
import os
import time


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
        if args.mpi is None:
            print("args.mpi is None")
            rank=0
        elif args.mpi in ['mpiexec','mpich','intel','slurm']:
            rank=int(os.environ['PMI_RANK'])
        else:
            raise Exception("Don't know how to find out rank")

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

        if rank==0:
            logging.info("dir(sim)")
            logging.info(str(dir(sim)))


        mdu=dio.MDUFile(args.mdu)

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

        # if rank==0:
        #     READ initial history file if needed

        reported_dir_sim=False # TMP
        
        # TASK TIME LOOP
        t_calc=0.0
        t_bmi=0.0
        t_last=time.time()
        while sim.get_current_time()<sim.get_end_time(): 
            t_now=sim.get_current_time()

            # update...
            cls.update_tracers(sim,t_now,cls.dt)

            # HERE: setup serial run, pause here and check for availability
            # of dwaq tracers
            if rank==0 and not reported_dir_sim: # TMP
                reported_dir_sim=True
                logging.info("dir(sim)")
                logging.info(str(dir(sim)))

                for var_i in range(sim.get_var_count()):
                    logging.info(f"var {var_i}: {sim.get_var_name(var_i)}")
                
            t_bmi+=time.time() - t_last
            t_last=time.time()
            logging.info(f'taking a step dt={cls.dt}')
            sim.update(cls.dt)
            logging.info('Back from step')
            t_calc+=time.time() - t_last
            t_last=time.time()

            # Running via BMI will not fail out when the time step gets too short, but it will
            # return back to here without going as far as we requested.
            t_post=sim.get_current_time()
            if t_post<t_now+0.75*cls.dt:
                logging.error("Looks like run has stalled out.")
                logging.error(f"Expected a step from {t_now} to {t_now+dt} but only got to {t_post}")
                logging.error("Will break out")
                break

            if rank==0:
                logging.info(f"t_bmi: {t_bmi}   t_calc: {t_calc}")

        sim.finalize()

    @classmethod
    def update_tracers(cls, sim, t_now, dt_s):
        # Decay for wt, wt_obs
        weight_decay_time = 3*3600.
        decay0_time=3600.
        decay1_time=86400.
        
        fac=np.exp(-dt_s/weight_decay_time)

        for vname in ['wt','wt_obs']:
            data=sim.get_var(vname)
            data *= fac
            sim.set_var(vname,data)

        # This is 2D!  Maybe it will be okay for a 2D run. But might have shape trouble.
        tau = sim.get_var('taus') # cell centre tau N/m2 {"location": "face", "shape": ["ndx"]}
        
        tauDecay0 = sim.get_var('tauDecay0')
        fac0=np.exp(-dt_s/decay0_time)
        tauDecay0[:] = tauDecay0*fac + (1-fac)*tau
        sim.set_var('tauDecay0',tauDecay0)
        
        fac1=np.exp(-dt_s/decay1_time)
        tauDecay1 = sim.get_var('tauDecay1')
        tauDecay1[:] = tauDecay1*fac + (1-fac)*tau
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

