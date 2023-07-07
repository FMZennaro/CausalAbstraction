import numpy as np
import networkx as nx
import itertools

import src.utils as ut
import src.examples.randomgenerators as rng

from pgmpy.inference import VariableElimination



class SCMMapping():
    def __init__(self,M0,M1):
        self.M0 = M0
        self.M1 = M1
            
            
class Abstraction(SCMMapping):
    
    ### INITIALIZER
    
    def __init__(self,M0,M1,R,a,alphas=None,deterministic=False):
        super().__init__(M0,M1)             
        
        self.deterministic = deterministic
        
        if not ut.is_list_contained_in_list(R,M0.nodes): raise ValueError("R contains illegal nodes")
        if not ut.is_list_contents_unique(R): raise ValueError("R contains duplicated nodes")
        self.R = R
        self.nR = len(R)
        
        if not ut.is_list_contained_in_list(list(a.keys()),M0.nodes): raise ValueError("Domain of a contains illegal nodes")
        if not ut.is_list_contained_in_list(list(a.values()),M1.nodes): raise ValueError("Codomain of a contains illegal nodes")
        if not ut.is_surjective(list(a.values()),M1.nodes): raise ValueError("a is not surjective")
        self.a = a
        
        if not alphas:
            alphas = rng.generate_random_alphas(M0,M1,a)
        
        if not ut.is_list_contained_in_list(list(alphas.keys()),M1.nodes): raise ValueError("Alphas contains functions defined on illegal nodes")
        if not ut.is_surjective(list(alphas.keys()),M1.nodes): raise ValueError("Alphas does not contain all the functions")
        self._are_alphas_cardinalities_correct(alphas)
        self._are_alphas_stochastic_and_deterministic(alphas,self.deterministic)
        self.alphas = alphas
        
            
    ### INITIALIZATION VERIFICATION FUNCTIONS    
    def _are_alphas_cardinalities_correct(self,alphas):
        for k in alphas.keys():
            card_domain,card_codomain = self.get_cardinalities_alpha(k)
            if card_domain != alphas[k].shape[1]: raise ValueError("Alpha_{0} domain is misspecified".format(k))
            if card_codomain != alphas[k].shape[0]: raise ValueError("Alpha_{0} codomain is misspecified {1} {2}".format(k,card_codomain,alphas[k].shape[0]))
                
    def _are_alphas_stochastic_and_deterministic(self,alphas,determinism):
        for k in alphas.keys():
            if not np.allclose(np.sum(alphas[k],axis=0),1): raise ValueError("Alpha_{0} is not stochastic".format(k))
            if determinism:
                if not len(np.where(alphas[k]==1)[0])==alphas[k].shape[1]: raise ValueError("Alpha_{0} is not deterministic".format(k))

                    
    ### ABSTRACTION PROPERTIES
    def is_varlevel_complete(self):
        return self.M0.number_of_nodes() == self.nR
    
    def is_varlevel_isomorphic(self):
        return self.nR == self.M1.number_of_nodes()
    
    def is_domlevel_isomorphic(self):
        for k in self.alphas.keys():
            print("Mapping alpha_{0}: {1}".format(k, alphas[k].shape[0]==alphas[k].shape[1]))
            
    def is_domlevel_deterministic(self):
        return self.deterministic                
                    
    
    ### UTILS    
    def copy(self):
        Acopy = Abstraction(self.M0, self.M1,
                            R=self.R.copy(), a=self.a.copy(), alphas=self.alphas.copy())
        return Acopy
    
    # list of strings -> string: ['X','Y'] -> 'A'
    def forward_a(self,V0s):
        V1 = self.a[V0s[0]]
        counterV1 = self.invert_a(V1)
        if len(V0s)>1:
            for V0 in V0s[1:]:
                if self.a[V0] != V1: raise ValueError("The nodes in the base model {0} maps to multiple different abstracted nodes".format(V0s))
        if set(V0s)!=set(counterV1): raise ValueError("The nodes in the base model {0} are a subset of nodes mapping of the counterimage {1} of {2}".format(V0s,counterV1,V1)) 
        return V1
    
    # list of lists of strings -> list of strings: [['X','Y'],['Z']] -> ['A','B']
    def forward_as(self,V0ss):
        V1s = []
        print(V0ss)
        for V0s in V0ss:
            V1s.append(self.forward_a(V0s))
        return V1s
    
    # string -> list of strings: 'A' -> ['X','Y']
    # list of one string -> list of strings: ['A'] -> ['X','Y']
    # list of multiple strings -> list of strings: ['A','B'] -> ['X','Y','Z']
    def invert_a(self,V1):
        if type(V1) is list:
            for V in V1:
                if not V in self.M1.nodes: raise ValueError("The node {0} is not in the abstracted nodes".format(V))
        elif not V1 in self.M1.nodes: raise ValueError("The node {0} is not in the abstracted nodes".format(V1))
        return ut.inverse_fx(self.a,V1)
    
    # list of strings -> list of lists of strings: ['A','B'] -> [['X','Y'],['Z']]
    def invert_as(self,V1s):
        V0s = []
        for V1 in V1s:
            V0s.append(self.invert_a(V1))
        return V0s
    
    def forward_alpha(self,alphakey,V0):
        self._are_alphas_stochastic_and_deterministic(self.alphas,True)
        return np.where(self.alphas[alphakey][v]==1)[0][0]
    
    def get_cardinalities_alpha(self,alphakey):
        return ut.get_cardinalities_Falpha(self.a,alphakey,self.M0.get_cardinality(),self.M1.get_cardinality())
            
    def compute_abstractions(self,alphakeys):
        alphas = [self.alphas[i] for i in alphakeys]
        return ut.tensorize_list(None,alphas)

    def compute_global_alpha(self):
        orderingM1 = list(self.M1.nodes)

        Alpha = self.alphas[orderingM1[0]]
        for i in range(1,len(orderingM1)):
            Alpha = ut.flat_tensor_product(Alpha,self.alphas[orderingM1[i]])

        notR = list(set(self.M0.nodes)-set(self.R))
        for nr in notR:
            Alpha = np.tile(Alpha,(1,self.M0.get_cardinality(nr)))

        orderingM0 = [self.invert_a(x) for x in orderingM1]
        orderingM0 = list(itertools.chain.from_iterable(orderingM0))
        orderingM0 = notR + orderingM0

        return Alpha, orderingM0, orderingM1 
    
    def compute_joints(self,verbose=False):
        Alpha, orderingM0, orderingM1 = self.compute_global_alpha()   
    
        inferM0 = VariableElimination(self.M0)
        joint_M0 = inferM0.query(orderingM0,show_progress=False)
        
        old_indexes = range(len(orderingM0))
        new_indexes = [(orderingM0).index(i) for i in joint_M0.variables]
        joint_M0 = np.moveaxis(joint_M0.values, old_indexes, new_indexes)
        joint_M0 = joint_M0.reshape((np.prod(joint_M0.shape),1))
        if verbose: print('M0 joint: {0}'.format(joint_M0))
        
        if (len(self.M1.nodes)==1 and self.M1.get_cardinality(orderingM1[0])==1):
            joint_M1 = self.M1.get_cpds(orderingM1[0])
        else:
            inferM1 = VariableElimination(self.M1)
            joint_M1 = inferM1.query(orderingM1,show_progress=False)
        
        old_indexes = range(len(orderingM1))
        new_indexes = [(orderingM1).index(i) for i in joint_M1.variables]
        joint_M1 = np.moveaxis(joint_M1.values, old_indexes, new_indexes)
        joint_M1 = joint_M1.reshape((np.prod(joint_M1.shape),1))
        if verbose: print('M1 joint: {0}'.format(joint_M1))
        
        return joint_M0,joint_M1
    
    def compute_inv_alpha(self,invalpha_algorithm=None, verbose=False):
        Alpha, orderingM0, orderingM1 = self.compute_global_alpha()
               
        if invalpha_algorithm is None:
            invalpha = ut.invert_matrix_max_entropy(Alpha)
        else:
            invalpha = invalpha_algorithm(Alpha)
        if verbose: print('Alpha^-1: {0}'.format(invalpha))
            
        return invalpha
    
    def compute_inverse_joint_M1(self, invalpha_algorithm=None, verbose=False):
        Alpha, orderingM0, orderingM1 = self.compute_global_alpha()
        joint_M0,joint_M1 = self.compute_joints(verbose=verbose)
        invalpha = self.compute_inv_alpha(invalpha_algorithm=invalpha_algorithm, verbose=verbose)
        inverse_joint_M1 = np.dot(invalpha,joint_M1)
            
        if verbose: print('Transformed M1 joint: {0}'.format(inverse_joint_M1))
            
        return inverse_joint_M1
    

    
    
    
    
        
    
    
    
    
    

class TauOmegaAbstraction(SCMMapping):
    def __init__(self,M0,M1,I0,I1,tau,omega):
        super().__init__(M0,M1)
        
        self.I0 = I0
        self.I1 = I1
        self.tau = tau
        
        if not ut.is_list_contained_in_list(list(omega.keys()),list(I0.keys())): raise ValueError("Domain of omega contains illegal elements")
        if not ut.is_list_contained_in_list(list(omega.values()),list(I1.keys())): raise ValueError("Codomain of omega contains illegal elements")
        if not ut.is_surjective(list(omega.values()),list(I1.keys())): raise ValueError("omega is not surjective")
        if not self._is_order_preserving(I0,I1,omega): raise ValueError("omega is not order-preserving")
        self.omega = omega
    
    def _is_order_preserving(self,I0,I1,omega):
        self.I0_poset = self._build_poset(I0)
        self.I1_poset = self._build_poset(I1)
        
        for e in self.I0_poset.edges():
            if not nx.has_path(self.I1_poset, omega[e[0]], omega[e[1]]): return False
        
        return True
    
    
    def _build_poset(self,I):        
        G0 = nx.DiGraph()
        G0.add_nodes_from(list(I.keys()))
        
        for k,v in I.items():    
            if len(v)==0:
                for k1 in I.keys():
                    if k1!=k: G0.add_edge(k,k1)

            else:
                for k1,v1 in I.items():
                    if k1!=k:
                        intervenednodeset_0 = list(I[k].keys())
                        intervenednodeset_1 = list(I[k1].keys())
                        
                        if ut.is_list_contained_in_list(intervenednodeset_0,intervenednodeset_1):
                            areintervenedvaluesequal = True
                            for i in intervenednodeset_0:
                                if I[k][i] != I[k1][i]:
                                    areintervenedvaluesequal = False

                            if areintervenedvaluesequal: 
                                G0.add_edge(k,k1)
        
        return G0