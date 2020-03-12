# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import numpy as np
from scipy.stats import norm

class BO():
    """Bayesian Optimization"""

    def __init__(self, GPR, X, y, x, xi = 0.1):
        self.GPR    = GPR
        self.X      = X
        self.y      = y
        self.x      = x
        self.xi     = xi
    ################################################################################
    # EXPECTED IMPROVEMENT(EI) ACQUISATION FUNCTION
    ################################################################################
    # x : values you want the Expected Imp. to be calculated [mxd]
    # X : observations                                       [nxd]
    # y : observed targets                                   [nx1]
    # xi: trade-off term for exploration and exploitation, default = 0 -> balanced
    ################################################################################
    def Expected_Improvement(self):
        # Posteriror mean and std. for GPR
        mean, std = self.GPR.inference(self.x,return_std=True);
        std = std.reshape(-1,1)
        # Best value of the target so far...
        best_samp = np.max(self.y)
        # Z calculation for normal distribution functions
        imp = mean - best_samp - self.xi;
        Z   = imp / std
        # Expected Improvement calculation
        EI  =  imp * norm.cdf(Z) + std * norm.pdf(Z);
        EI[std==0.] = 0.
        return EI;
    ################################################################################
    # PROBABILITY OF IMPROVEMENT(PI) ACQUISATION FUNCTION
    ################################################################################
    # x : values you want the Expected Imp. to be calculated [mxd]
    # X : observations                                       [nxd]
    # y : observed targets                                   [nx1]
    # xi: trade-off term for exploration and exploitation, default = 0 -> balanced
    ################################################################################
    def Probability_Improvement(self):
        # Posteriror mean and std. for GPR
        mean, std = self.GPR.inference(self.x,return_std=True);
        std = std.reshape(-1,1)
        # Best value of the target so far...
        best_samp = np.max(GPR.inference(X))
        # Z calculation for normal distribution functions
        imp = mean - best_samp - self.xi;
        Z   = imp / std
        # Probability of Improvement calculation
        PI  = norm.cdf(Z)
        return PI;
    ################################################################################
    # New Point Proposal Based on selected Acquisation Function:
    ################################################################################
    def prop(self):
        EI = self.Expected_Improvement( )
        np.amax(EI)
        i,j = np.unravel_index(EI.argmax(), EI.shape)
        return self.x[i,j]
