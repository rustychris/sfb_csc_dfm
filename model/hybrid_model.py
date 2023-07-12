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


class HybridModel(custom.CustomProcesses, sfb_csc.SfbCsc):
    dwaq=True # gets replaced by a WaqOnlineModel instance during configure()

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

        # DFM expects some commandline options, which for BMI are specific variables
        # First attempt DFM came back with '/'
        #  np.array(waq_proc_def)
        #proc_def = sim.get_var("processlibrary")
        #logging.info(f"existing proc_def is {proc_def}")
        # This also comes back with '/'
        # sim.set_var("processlibrary",np.array([local_config.LocalConfig.waq_proc_def]))
        # this yields ** ERROR  : Process library file does not exist: 0ñº;
        # sim.set_var("processlibrary",np.array(local_config.LocalConfig.waq_proc_def,dtype=object))

        # mimic what's going on inside wrapper:
        if 1:
            name = bmi.wrapper.create_string_buffer("processlibrary")
            type_ = bmi.wrapper.create_string_buffer(1024)
            sim.library.get_var_type.argtypes = [c_char_p, c_char_p]
            sim.library.get_var_type(name, type_)
            logging.info(f"get_var_type => {type_}")
            # type_: <bmi.wrapper.c_char_Array_1024 object at 0x1460b74d37c0>
            logging.info(f"get_var_type.value => {type_.value}")
            # .value => b''
            # return type_.value
            # 
        
        sim.set_var("processlibrary",np.array([local_config.LocalConfig.waq_proc_def]))

        dt=900.0 # update interval

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

            # HERE: setup serial run, pause here and check for availability
            # of dwaq tracers
            if rank==0 and not reported_dir_sim: # TMP
                reported_dir_sim=True
                logging.info("dir(sim)")
                logging.info(str(dir(sim)))

                
            t_bmi+=time.time() - t_last
            t_last=time.time()
            logging.info(f'taking a step dt={dt}')
            sim.update(dt)
            logging.info('Back from step')
            t_calc+=time.time() - t_last
            t_last=time.time()

            # Running via BMI will not fail out when the time step gets too short, but it will
            # return back to here without going as far as we requested.
            t_post=sim.get_current_time()
            if t_post<t_now+0.75*dt:
                logging.error("Looks like run has stalled out.")
                logging.error(f"Expected a step from {t_now} to {t_now+dt} but only got to {t_post}")
                logging.error("Will break out")
                break

            if rank==0:
                logging.info(f"t_bmi: {t_bmi}   t_calc: {t_calc}")

        sim.finalize()

    
    # Model setup
    def configure(self):
        super().configure()
        self.add_hybrid_tracers()
    def add_hybrid_tracers(self):
        self.dwaq.substances['wt_obs']=waq.Substance(initial=0.0)
        self.dwaq.substances['wt']=waq.Substance(initial=0.0)

        # These will get connected to dfm (if the hydro includes them)
        self.dwaq.parameters['tau']=0.0
        self.dwaq.parameters['VWind']=0.0
        self.dwaq.parameters['salinity']=0.0
        
        self.dwaq.substances['tauDecay1']=waq.Substance(initial=0.0)
        self.dwaq.substances['tauDecay2']=waq.Substance(initial=0.0)

        # Are the BCs already in place?
        # happens in sfb_csc set_bcs(), called from sfb_csc configure

        # For initial test, tag Sac inflow
        sac=self.scalar_parent_bc("SacramentoRiver")
        sac_wt_obs = hm.ScalarBC(parent=sac,scalar='wt_obs',value=10.0)
        sac_wt = hm.ScalarBC(parent=sac,scalar='wt',value=1.0)
        self.add_bcs([sac_wt,sac_wt_obs])

        # Straight exponential decay with configurable rate.
        self.custom_Decay(substance='wt_obs',rate=0.2)
        self.custom_Decay(substance='wt',rate=0.2)

        # Is it possible to use the nitrification process for exponential
        # filter on bed stress and/or wind exposure?
        
        # I'm looking for d tauDecay / dt = k*(tau-tauDecay)
        #                                 = k*tau - k*tauDecay

        # one way is to include two nitrification processes:
        #  custom_Decay(substance='tauDecay',rate=tauDecayRate)
        #  custom_CART(substance='Tau',age_substance='tauDecay',partial=tauDecayRate))

        # The CART code is basically this:
        #     d conc / dt = -conc_decay*partial * conc
        # d age_conc / dt =             partial * conc

        self.dwaq.substances['tauDecay1']=waq.Substance(initial=0.0)
        self.custom_ExpFilter(sub_in='tau',sub_out='tauDecay1',rate=16.0)

        self.dwaq.substances['tauDecay2']=waq.Substance(initial=0.0)
        self.custom_ExpFilter(sub_in='tau',sub_out='tauDecay2',rate=1.0)

        
if __name__=='__main__':
    # For testing, hardcode the settings
    #argv=["-n","1",
    #      "-p","2019-04-01:2019-04-02",
    #      "-l","0"]
    HybridModel.main(sys.argv[1:])

