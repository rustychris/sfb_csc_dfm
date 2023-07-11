"""
1. Duplicate run of sfb_csc Done
2. Configure DWAQ tracers
   - need fresh compile that includes dwaq. Done.
   - set weight and tracers at BCs. Done for constant.
   - weight, weight*ssc tracers with longer delay. Done, not tested.
   - tau exposure with decay - TROUBLE
     DFM is not picking up any of the linked constituents, according to .dia file.
       Is the message wrong, and it is finding them?
         -- no. Salinity is maybe picking up the 32ppt BC, but is *not* a copy of sa1.
         -- tau is all zero. The DFM tau is "taus".
       Is it reading the substance file at all?
  - probably the issue is that they need to be parameters, not substances.

3. Run it via BMI
4. Set tracers during run, internal to domain
5. Extract correlates at the same time.

"""
import sys
import sfb_csc
import stompy.model.delft.waq_scenario as waq
import stompy.model.hydro_model as hm
import custom_process as custom

class HybridModel(custom.CustomProcesses, sfb_csc.SfbCsc):
    dwaq=True # gets replaced by a WaqOnlineModel instance during configure()

    def configure(self):
        super().configure()
        self.add_hybrid_tracers()
    def add_hybrid_tracers(self):
        self.dwaq.substances['wt_obs']=waq.Substance(initial=0.0)
        self.dwaq.substances['wt']=waq.Substance(initial=0.0)

        # This *should* be enough to trigger DFM passing it to DWAQ
        # Tried with 'Tau' and the dia file suggested it was not connected.
        # HERE: regardless of tau vs Tau, dia file claims it's not connected.
        # Wut?
        # Does capitalization matter in the process??
        # Upgraded to 142431, still no joy.
        # trying with salinity, too.
        # Okay - I think these were supposed to be parameters.
        self.dwaq.parameters['tau']=0.0
        # self.dwaq.substances['tauflow']=waq.Substance(initial=0.0)
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

