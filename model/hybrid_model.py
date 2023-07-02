"""
1. Duplicate run of sfb_csc Done
2. Configure DWAQ tracers
   - need fresh compile that includes dwaq. Done.
   - set weight and tracers at BCs. Done for constant.
   - weight, weight*ssc tracers with longer delay
   - tau exposure with decay
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

        self.dwaq.substances['tauDecay']=waq.Substance(initial=0.0)

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
        self.dwaq.substances['tauDecay']=waq.Substance(initial=0.0)
        
        
if __name__=='__main__':
    # For testing, hardcode the settings
    #argv=["-n","1",
    #      "-p","2019-04-01:2019-04-02",
    #      "-l","0"]
    HybridModel.main(sys.argv[1:])

