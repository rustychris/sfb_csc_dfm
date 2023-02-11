class AgeMixin:
    def setup_delwaq(self):
        """
        Set up Delwaq model to run with Dflow. Currently used to calculate
        age of water using nitrification process
        """
        super().setup_delwaq()
        self.dwaq.substances['NO3']=self.dwaq.Sub(initial=0.0)
        self.dwaq.substances['RcNit']=self.dwaq.Sub(initial=0.0)
        if self.dcd:
            self.dwaq.substances['drain']=self.dwaq.Sub(initial=0.0) # Tag for Ag returns
        self.dwaq.substances['nonsac']=self.dwaq.Sub(initial=0.0) # Tag for non-Sac returns
        self.dwaq.substances['sea']=self.dwaq.Sub(initial=0.0) # Tag for water coming from seaward BCs

        self.dwaq.parameters['TcNit']=1  # no temp. dependence
        self.dwaq.parameters['NH4']=1 # inexhaustible constant ammonium supply

        # by default, nitrification process uses
        # pragmatic kinetics forumulation (SWVnNit = 0)
        self.dwaq.add_process(name='Nitrif_NH4')
