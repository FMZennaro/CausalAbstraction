
import numpy as np
from scipy.spatial import distance

from src.utils import jsd
import src.evaluationsets as es



class SCMMappingEvaluator():
    def __init__(self,A):
        self.A = A

class AbstractionEvaluator(SCMMappingEvaluator):
    def __init__(self,A):
        super().__init__(A)
        
class CategoricalAbstractionEvaluator(AbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):
        if metric is None:
            self.metric = jsd
        else:
            self.metric = metric
        
        if pinv is None:
            self.pinv = np.linalg.pinv
        else:
            self.pinv = pinv
        
        if aggerror is None:
            self.aggerror = np.max
        else:
            self.aggerror = aggerror
        
        if aggoverall is None:
            self.aggoverall = np.max
        else:
            self.aggoverall = aggoverall
               
        super().__init__(A)
    
    def compute_diagram(self, pair, verbose=False):
        # Get nodes in the abstracted model
        M1_sources = pair[0]
        M1_targets = pair[1]
        if verbose: print('\nM1: {0} -> {1}'.format(M1_sources,M1_targets))
        
        # Get nodes in the base model
        M0_sources = self.A.invert_a(M1_sources)
        M0_targets = self.A.invert_a(M1_targets)
        if verbose: print('M0: {0} -> {1}'.format(M0_sources,M0_targets))

        # Compute the high-level mechanisms
        if verbose: print('Args: {}'.format(self.A.M1,M1_sources,M1_targets))
        M1_cond_TS_val = self.A.compute_mechanisms(self.A.M1,M1_sources,M1_targets)
        if verbose: print('M1 mechanism shape: {}'.format(M1_cond_TS_val.shape))
        if verbose: print('M1 mechanism: {}'.format(M1_cond_TS_val))

        # Compute the low-level mechanisms
        M0_cond_TS_val = self.A.compute_mechanisms(self.A.M0,M0_sources,M0_targets)
        if verbose: print('M0 mechanism shape: {}'.format(M0_cond_TS_val.shape))

        # Compute the alpha for sources
        alpha_S = self.A.compute_abstractions(M1_sources)
        if verbose: print('Alpha_s shape: {}'.format(alpha_S.shape))

        # Compute the alpha for targets
        alpha_T = self.A.compute_abstractions(M1_targets)
        if verbose: print('Alpha_t shape: {}'.format(alpha_T.shape))
            
        return M0_cond_TS_val, M1_cond_TS_val, alpha_S, alpha_T
    
    def compute_error(self,pair, verbose=False):
        M0_cond_TS_val, M1_cond_TS_val, alpha_S, alpha_T = self.compute_diagram(pair,verbose)
        
        lowerpath,upperpath = self.compute_paths(M0_cond_TS_val, M1_cond_TS_val, alpha_S, alpha_T)
        
        distances = []
        for do in range(lowerpath.shape[1]):
            distances.append( self.metric(lowerpath[:,do],upperpath[:,do]) )
        if verbose: print('\nDistances: {0}'.format(distances))

        error = self.aggerror(distances)
        if verbose: print('\nError: {0}'.format(error))
        return error
    
    def compute_overall_error(self,J, verbose=False):        
        errors = []
        for pair in J:
            error = self.compute_error(pair,verbose)
            errors.append(error)
            if verbose: print('\nError of pair {0}: {1}'.format(pair,error))
        
        overallerrror = self.aggoverall(errors)
        if verbose: print('\nOverall error: {0}'.format(overallerrror))
        return overallerrror
    
class ICEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (nu@aA,aB@mu)
        super().__init__(A,metric,pinv,aggerror,aggoverall)
        
class IILEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (mu,self.pinv(aB)@nu@aA)
        super().__init__(A,metric,pinv,aggerror,aggoverall)
        
class ISILEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (nu,aB@mu@self.pinv(aA))
        super().__init__(A,metric,pinv,aggerror,aggoverall)
        
class ISCEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (self.pinv(aB)@nu,mu@self.pinv(aA))
        super().__init__(A,metric,pinv,aggerror,aggoverall)
    