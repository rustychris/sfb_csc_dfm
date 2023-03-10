import os
import six
import numpy as np
import sfb_csc
import rough_regions
six.moves.reload_module(rough_regions)
six.moves.reload_module(sfb_csc)

import shutil

model=sfb_csc.SfbCsc(run_start=np.datetime64("2019-04-01"),
                     run_stop =np.datetime64("2019-04-10"),
                     run_dir="data_testrun000",
                     extra_scripts=[__file__])

model.write()


