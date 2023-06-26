"""
1. Duplicate run of sfb_csc
2. Configure DWAQ tracers
   - tau exposure with decay
   - weight, weight*ssc tracers with longer delay
   - set weight and tracers at BCs.
3. Run it via BMI
4. Set tracers during run, internal to domain
5. Extract correlates at the same time.

"""
import sfb_csc

class HybridModel(sfb_csc.SfbCsc):
    pass


#if __name__=='__main__':
# For testing, hardcode the settings
argv=["-n","1",
      "-p","2019-04-01:2019-04-02",
      "-l","0"]
sfb_csc.SfbCsc.main(argv)

