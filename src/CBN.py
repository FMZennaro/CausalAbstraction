
import numpy as np

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

class CausalBayesianNetwork(BayesianNetwork):
    
    def __init__(self, ebunch=None, latents=set()):
        super().__init__(ebunch=ebunch, latents=latents)
    
    def compute_mechanisms(self,sources,targets):
        #Compute P(targets|do(sources)) as P(targets|sources) in M_do(sources)

        # Perform interventions in the abstracted model and setup the inference engine
        Mdo = self.do(sources)
        inference = VariableElimination(Mdo)

        #Compute P(targets|sources) as P(targets,sources)/P(sources)
        joint_TS = inference.query(targets+sources,show_progress=False)
        marginal_S = inference.query(sources,show_progress=False)
        cond_TS = joint_TS / marginal_S

        #Get the values of P
        cond_TS_val = cond_TS.values

        #Order the values according to targets and sources
        old_indexes = range(len(targets+sources))
        new_indexes = [(targets+sources).index(i) for i in joint_TS.variables]
        cond_TS_val = np.moveaxis(cond_TS_val, old_indexes, new_indexes)

        #Reshape the matrix as [targets x sources]
        cardinalities = self.get_cardinality()
        target_cards=[cardinalities[t] for t in targets]
        target_card = np.prod(target_cards)
        source_cards=[cardinalities[s] for s in sources]
        source_card = np.prod(source_cards)
        cond_TS_val = cond_TS_val.reshape(target_card,source_card)

        return cond_TS_val    
