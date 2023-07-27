import itertools
import sys
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, Lock, Manager, Process
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from flyingsquid.label_model import LabelModel

# Conditions of agreements and disagreements among LFs
def c1(samples,i=0,j=1,k=2):
    vec = np.logical_and(samples[:,i]!=samples[:,j], samples[:,i]!=samples[:,k])    
    return vec

def c2(samples,i=0,j=1,k=2):
    vec = np.logical_and(samples[:,i]!=samples[:,j], samples[:,i]==samples[:,k])
    return vec

def c3(samples,i=0,j=1,k=2):
    vec = np.logical_and(samples[:,i]==samples[:,j], samples[:,i]!=samples[:,k])
    return vec

def c4(samples,i=0,j=1,k=2):
    vec = np.logical_and(samples[:,i]==samples[:,j], samples[:,i]==samples[:,k])
    return vec

# Exponential family distributions

# Proposed PGM
def exponential_family_complex(lam, y, theta_y, theta_lam_y_ind, theta_lam_y_cond, theta_lam_lam, n_ind=2):
    conds = np.asarray([int(c(np.expand_dims(lam,0))[0]) for c in [c1,c2,c3,c4]])
    return np.exp(np.dot(conds,theta_y) * y + np.dot(conds,np.dot(theta_lam_y_ind,lam[-n_ind:])) * y + np.dot(theta_lam_y_cond,conds) * lam[0] * y)
    
# Historical PGM
def exponential_family_simple(lam, y, theta_y, theta_lam_y, theta_lam_lam):
    return np.exp(theta_y * y + np.dot(np.squeeze(theta_lam_y), np.squeeze(lam)) * y)

# Data generating functions
def make_pmf(theta_y, theta_lam_y_ind, theta_lam_y_cond, theta_lam_lam, lst, comp=1):
    p = np.zeros(len(lst))
    for i in range(len(lst)):
        if comp: p[i] = exponential_family_complex(lst[i][:-1], lst[i][-1], theta_y, theta_lam_y_ind, theta_lam_y_cond, theta_lam_lam)
        else: p[i] = exponential_family_simple(lst[i][:-1], lst[i][-1], theta_y, theta_lam_y_ind, theta_lam_lam)
    return p/np.sum(p)

def make_cmf(pmf):
    return np.cumsum(pmf)

def sample_with_pmf(lst, pmf, n):
    choices = np.random.choice(len(lst), size=n, p=pmf)
    return np.array(lst)[choices]

def generate_data(n_train, n_test, theta_y, theta_lam_y_ind, theta_lam_y_cond, theta_lam_lam, m, v, comp=1):
    lst = list(map(list, itertools.product([-1, 1], repeat=v))) 
    pmf = make_pmf(theta_y, theta_lam_y_ind, theta_lam_y_cond, theta_lam_lam, lst, comp)
    cmf = make_cmf(pmf)
    sample_matrix = sample_with_pmf(lst, pmf, n_train)
    sample_matrix_test = sample_with_pmf(lst, pmf, n_test)
    return sample_matrix, sample_matrix_test, lst, pmf

# Helper functions for computing expected values (i.e. scaled accuracies)
def calc_exp(samples,entries,val=1):
    return 2 * np.mean(np.prod(samples[:,entries],1)==val) - 1

def calc_exp_cond(samples,entries,condition_vec,val=1):
    if np.sum(condition_vec)==0: return 0,0
    prob_condition = np.mean(condition_vec)
    conditioned_samples = samples[condition_vec,:]
    return prob_condition, (2 * np.mean(np.prod(conditioned_samples[:,entries],1)==val) - 1)

# Recover mean parameters with y from mean parameters with lambda only using triplet method
def solve_triplets(n01, n02, n12):
    a0 = np.sqrt(np.abs(n01*n02/n12))
    a1 = np.sqrt(np.abs(n01*n12/n02))
    a2 = np.sqrt(np.abs(n12*n02/n01))
    return a0, a1, a2

# Placeholder resolve signs func
def resolve_signs(n01, n02, n12, sign_2):
    sign_1 = np.sign(n12)/sign_2
    sign_0 = np.sign(n01)/sign_1
    return sign_0, sign_1, sign_2

# Functions for computing probabilities
def calc_p(samples, entries, val=1):
    return np.mean(np.prod(samples[:,entries],1)==val)

def calc_empirical_marginal(samples,entries,values):
    marginal_vec = np.ones(samples.shape[0])
    for entry, value in zip(entries,values): marginal_vec *= (samples[:,entry]==value)
    return np.mean(marginal_vec)

def calc_exp_from_marg(table,i,val=1):
    return 2 * np.sum(table[table[:,i]==val,-1]) - 1

def calc_exp_lam_i_Y_from_marg(table,i):
    return 2 * np.sum(table[table[:,i]==table[:,-2],-1]) - 1

def calc_exp_cond_Y_from_marg(table,i,condition_vec,val=1):
    if np.sum(condition_vec)==0: return 0, 0
    prob_condition = np.sum(table[condition_vec,-1])
    condition_table = table[condition_vec,:]
    condition_table[:,-1] = condition_table[:,-1]/np.sum(condition_table[:,-1])
    equal_vec = condition_table[:,i]==val
    return prob_condition, (2 * np.sum(condition_table[equal_vec,-1]) - 1)

def calc_exp_cond_lam_i_Y_from_marg(table,i,condition_vec):
    if np.sum(condition_vec)==0: return 0, 0
    prob_condition = np.sum(table[condition_vec,-1])
    condition_table = table[condition_vec,:]
    condition_table[:,-1] = condition_table[:,-1]/np.sum(condition_table[:,-1])
    equal_vec = condition_table[:,i]==condition_table[:,-2]
    return prob_condition, (2 * np.sum(condition_table[equal_vec,-1]) - 1)

def calc_exp_cond_lam_i_lam_j_lam_k_from_marg(table,i,j,k,condition_vec):
    if np.sum(condition_vec)==0: return 0, 0
    condition_table = table[condition_vec,:]
    condition_table[:,:-1] = condition_table[:,:-1] * 2 - 1
    values = condition_table[:,i]*condition_table[:,j]*condition_table[:,k]
    probs = condition_table[:,-1]
    return np.dot(values,probs)

class ConditionalModel:
                
    def __init__(self, theta, conds, seed=None):

        if seed is not None:
            np.random.seed(seed)
    
        self.G = MarkovModel()
        self.conds = conds

        for _, row in theta.iterrows():
            self._build_graph(row)

            if "unary" in row["notes"]:
                theta_ijk = row["value"]

                if "unary1" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"])], 
                                                      [2, 2, 2, 2], np.exp([0,0,0,-theta_ijk,-theta_ijk,0,0,0,
                                                                            0,0,0,theta_ijk,theta_ijk,0,0,0])))                     

                elif "unary2" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"])], 
                                                      [2, 2, 2, 2], np.exp([0,0,-theta_ijk,0,0,-theta_ijk,0,0,
                                                                            0,0,theta_ijk,0,0,theta_ijk,0,0])))

                elif "unary3" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"])], 
                                                      [2, 2, 2, 2], np.exp([0,-theta_ijk,0,0,0,0,-theta_ijk,0,
                                                                            0,theta_ijk,0,0,0,0,theta_ijk,0])))                

                elif "unary4" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"])], 
                                                      [2, 2, 2, 2],np.exp([-theta_ijk,0,0,0,0,0,0,-theta_ijk,
                                                                            theta_ijk,0,0,0,0,0,0,theta_ijk])))

            elif "pairwise" in row["notes"]:
                theta_ijk = row["value"]
                if "pairwise1" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"]), str(row["y"])], 
                                                      [2, 2, 2, 2, 2], np.exp([0,0,0,0,0,0,theta_ijk,-theta_ijk,theta_ijk,-theta_ijk,0,0,0,0,0,0,
                                                                               0,0,0,0,0,0,-theta_ijk,theta_ijk,-theta_ijk,theta_ijk,0,0,0,0,0,0])))
                
                elif "pairwise2" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"]), str(row["y"])], 
                                                      [2, 2, 2, 2, 2], np.exp([0,0,0,0,theta_ijk,-theta_ijk,0,0,0,0,theta_ijk,-theta_ijk,0,0,0,0,
                                                                               0,0,0,0,-theta_ijk,theta_ijk,0,0,0,0,-theta_ijk,theta_ijk,0,0,0,0])))
                elif "pairwise3" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"]), str(row["y"])], 
                                                      [2, 2, 2, 2, 2], np.exp([0,0, theta_ijk,-theta_ijk,0,0,0,0,0,0,0,0,theta_ijk,-theta_ijk,0,0,
                                                                               0,0, -theta_ijk,theta_ijk,0,0,0,0,0,0,0,0,-theta_ijk,theta_ijk,0,0])))
                elif "pairwise4" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["m"]), str(row["y"])], 
                                                      [2, 2, 2, 2, 2], np.exp([theta_ijk,-theta_ijk,0,0,0,0,0,0,0,0,0,0,0,0,theta_ijk,-theta_ijk,
                                                                               -theta_ijk,theta_ijk,0,0,0,0,0,0,0,0,0,0,0,0,-theta_ijk,theta_ijk])))
            else:
                theta_ijk = row["value"]
                if "lams1" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"])], 
                                                      [2, 2, 2], np.exp([0,0,0,-theta_ijk,theta_ijk,0,0,0])))        
                elif "lams2" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"])], 
                                                      [2, 2, 2], np.exp([0,0,theta_ijk,0,0,-theta_ijk,0,0])))   
                elif "lams3" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"])], 
                                                      [2, 2, 2], np.exp([0,theta_ijk,0,0,0,0,-theta_ijk,0])))        
                elif "lams4" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"])], 
                                                      [2, 2, 2], np.exp([-theta_ijk,0,0,0,0,0,0,theta_ijk])))   
                elif "triple1" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["y"])], 
                                                      [2, 2, 2, 2], np.exp([0,0,0,0,0,0,theta_ijk,-theta_ijk,-theta_ijk,theta_ijk,0,0,0,0,0,0])))                     
                elif "triple2" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["y"])], 
                                                      [2, 2, 2, 2], np.exp([0,0,0,0,theta_ijk,-theta_ijk,0,0,0,0,-theta_ijk,theta_ijk,0,0,0,0])))   
                elif "triple3" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["y"])], 
                                                      [2, 2, 2, 2], np.exp([0,0, theta_ijk,-theta_ijk,0,0,0,0,0,0,0,0,-theta_ijk,theta_ijk,0,0])))                     
                elif "triple4" in row["notes"]:
                    self.G.add_factors(DiscreteFactor([str(row["i"]), str(row["j"]), str(row["k"]), str(row["y"])], 
                                                      [2, 2, 2, 2], np.exp([theta_ijk,-theta_ijk,0,0,0,0,0,0,0,0,0,0,0,0,-theta_ijk,theta_ijk])))   
        
        self.G.check_model()
        self.infer = BeliefPropagation(self.G)
        self.infer.calibrate()
        
    def _build_graph(self, row):
        nodes = [str(r) for r in [row["i"],row["j"],row["k"],row["m"],row["y"]] if r != None]
        for node_id in nodes:
            if node_id not in self.G.nodes: self.G.add_node(node_id)
        for nodes in list(itertools.permutations(nodes, 2)):
            if (nodes[0],nodes[1]) not in self.G.edges: self.G.add_edge(nodes[0],nodes[1])
                
    def get_moments(self):

        p = len(list(self.G.nodes))
        full_dist = self.infer.query([str(x) for x in range(p)]).normalize(inplace=False)
        variable_order = [int(v) for v in full_dist.variables]
        init_dist_array = np.zeros((2**p,p+1))
        init_dist_array[:,:-1] = list(map(list, itertools.product([0, 1], repeat=p))) 
        full_dist_array = np.zeros((2**p,p+1))
        for placement in range(p):
            full_dist_array[:,placement] = init_dist_array[:,variable_order.index(placement)]
        full_dist_array[:,-1] = full_dist.values.flatten()

        cond_list = []
        for c, cond_func in enumerate(self.conds):
            cond_list += [cond_func(full_dist_array)]
            
        mom_dict = {}
        for c, p in zip(cond_list, ['unary1','unary2','unary3','unary4']): 
            prob, val = calc_exp_cond_Y_from_marg(full_dist_array,-2,c)
            mom_dict[(5,p)] = prob * val
        for i in [3,4]: 
            for c, p in zip(cond_list, ['pairwise1','pairwise2','pairwise3','pairwise4']): 
                prob, val = calc_exp_cond_lam_i_Y_from_marg(full_dist_array,i,c)
                mom_dict[(i,p)] = val * prob
        for c, t in zip(cond_list, ['triple1','triple2','triple3','triple4']):
            prob, val = calc_exp_cond_lam_i_Y_from_marg(full_dist_array,0,c)
            mom_dict[(0,1,2,t)] = val * prob
        return mom_dict

def mean_to_canonical(mean_parameter, parent_node_num, conds, theta=None, alpha=0.1, maxiter=200, accelerated=True, verbose=False):
    
    # initialization
    if theta is None:
        np.random.seed(seed=1)
        p_dep = [0,1,2]
        p_ind = [3,4]
        two_dep_notes = ['lams1','lams2','lams3','lams4']
        three_dep_notes = ['triple1','triple2','triple3','triple4']
        ind_notes = ['pairwise'+str(num) for num in np.arange(1,5)]*2
        un_notes = ['unary1','unary2','unary3','unary4']
        theta = pd.concat([
            pd.DataFrame([[parent_node_num,0,1,2,None,np.random.rand(1)[0],un_notes[x]] for x in range(len(un_notes))],columns=["i","j","k","m","y","value","notes"]),
            pd.DataFrame([[p_ind[int(x/4)],0,1,2,parent_node_num,np.random.rand(1)[0],ind_notes[x]] for x in range(len(ind_notes))],columns=["i","j","k","m","y","value","notes"]),
            pd.DataFrame([[0,1,2,None,parent_node_num,np.random.rand(1)[0],n] for n in three_dep_notes],columns=["i","j","k","m","y","value","notes"])],
        axis=0).reset_index(drop=True)

    error_thresh = 1e-4
    itr = 0
    error = 1000
    theta_list = []
    error_list = []
    while True:

        im = ConditionalModel(theta, conds)
        moment = im.get_moments()
        grad = []
        key_order = []
        for k in moment.keys():
            grad += [moment[k] - mean_parameter[k]]
            key_order += [k]
        grad = np.asarray(grad)
        
        error = np.linalg.norm(grad)
        error_list += [error]

        theta["value"] = theta["value"].values - alpha * grad
        theta_list.append(theta)
        # keep only the latest two elements
        if len(theta_list)>2:
            del theta_list[0]
            if accelerated:
                theta["value"] = theta_list[1]["value"] + (itr-2)/(itr-1) * (theta_list[1]["value"] - theta_list[0]["value"])

        if itr % 50 == 0 and verbose:
            print("Error:", error)
        itr = itr+1
        if error < error_thresh or itr == maxiter:
            break
            
    return im, theta, error_list

# Helper functions for computing final metrics
def binarize_conditionals(conds):
    result = np.ones_like(conds)
    result[conds<0.5] = -1
    return result

def compute_baselines(samples):
    indicator_samples = np.zeros_like(samples)
    indicator_samples[samples==1] = 1
    indicator_samples = np.sum(indicator_samples,1)
    mv = binarize_conditionals(indicator_samples/samples.shape[1])
    intersection = binarize_conditionals(1.0*(indicator_samples>0))
    union = binarize_conditionals(1.0*(indicator_samples==samples.shape[1]))
    return mv, intersection, union
    
# Baseline PGM to compare against---non segmentation version of weak supervision
def flying_squid(L_train, L_test, m, v, cb):
    
    triplet_model = LabelModel(
        m, v,
        [],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [],
#         allow_abstentions=False,
    )
    triplet_model.fit(
        L_train,
        class_balance=cb,
        verbose=True,
        solve_method = 'triplet_mean'
    )
        
    print('Done fitting with Flying Squid, now predicting')
    
    test_conditionals = triplet_model.predict_proba_marginalized(L_test)

    return triplet_model, test_conditionals

# Functions for computing disagreements among LFs---will use to choose which LFs to use in forming conditionals
def disagreement_count_by_col(samples,i=0,j=1,k=2):
    cond = 1.0*(samples[:,i]!=samples[:,j]) + 1.0*(samples[:,i]!=samples[:,k]) +1.0*(samples[:,j]!=samples[:,k])
    return cond

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
        
def init(l):
    '''
    This function is used by multiprocessing to lock before writing to files
    '''
    global lock
    lock = l
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def run_seg_label_model(L_train, L_dev, Y_dev, seed=1):
    """
    L_train: matrix of train set noisy labels, N x 5, e{-1, 1}
    L_dev: matrix of dev set noisy labels, M x 5, e{-1, 1}
    Y_dev: vector of dev set ground truth labels, M x 1, e{-1, 1}
    """
    
    ############ SET UP DATA MATRICES ##############
    
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)

    # Define nodes among which we check for disagreements --- leave these as defaults we'll rearrange LFs to max disagreement
    dep_nodes = [0,1,2] # nodes among which we check for disagreements
    ind_nodes = [3,4] # nodes we keep conditionally independent for triplet method
    all_nodes = dep_nodes + ind_nodes
    conds = [c1,c2,c3,c4]
    n_procs = 30 # This isn't actually doing anything
    n_conds = len(conds)
    lst = list(map(list, itertools.product([-1, 1], repeat=len(all_nodes)+1))) 

    # Set up multiprocessing
    l = Lock()

    # Choose which LFs we will check disagreements between
    sample_matrix_perm = np.ones((L_train.shape[0],6))*-1
    sample_matrix_perm[:,:-1] = L_train
    perms = list(itertools.combinations([0, 1, 2, 3, 4],3))
    max_disagreement = 0
    chosen_perm = 0
    for p in perms:
        not_p = [pp for pp in [0,1,2,3,4] if pp not in p]
        num_disagreement = np.sum(disagreement_count_by_col(sample_matrix_perm[:,:-1],p[0],p[1],p[2]))
        if num_disagreement>max_disagreement:
            max_disagreement = num_disagreement.copy()
            chosen_perm = p
    permind = [p for p in [0,1,2,3,4] if p not in chosen_perm]
    perm = [p for p in chosen_perm]+permind
    print('Ordering of LFs: ',perm)

    # Reformat data to max disagreements in conditions
    L_train = np.stack(([L_train[:,i] for i in perm]),-1)
    L_dev = np.stack(([L_dev[:,i] for i in perm]),-1)

    sample_matrix = np.ones((L_train.shape[0],6))*-1
    sample_matrix[:,:-1] = L_train
    
    sample_matrix_test = np.ones((L_dev.shape[0],6))*-1
    sample_matrix_test[:,:-1] = L_dev
    sample_matrix_test[:,-1] = Y_dev

    empirical_matrix = sample_matrix_test.copy()

    
    
    
    ############ COMPUTE MEAN PARAMETERS ##############
    print('Computing mean parameters...')
    # Store when conditions are true - needed for future computations
    cond_list = []
    empirical_cond_list = []
    cond_probs = []
    for c, cond_func in enumerate(conds):
        cond_list += [cond_func(sample_matrix)]
        empirical_cond_list += [cond_func(empirical_matrix)]
        cond_probs += [np.mean(cond_list[-1])]

    # Compute mean parameters
    empirical_exp_y = []
    for cond_vec in empirical_cond_list:
        prob, val = calc_exp_cond(empirical_matrix,[-1],cond_vec)
        empirical_exp_y += [prob*val] # Expected value of y

    def calc_exp_1O_with_parallel(ind,node,return_dict,empirical_matrix,sample_matrix,all_nodes):
        return_dict['empirical_exp_lam_i_Y',ind] = calc_exp(empirical_matrix,[node,-1])
        return_dict['exp_lam_i',ind] = calc_exp(sample_matrix,[node])
        for j, node1 in enumerate(all_nodes[ind+1:]):
            return_dict['exp_lam_i_lam_j',ind,j+ind] = calc_exp(sample_matrix,[node,node1])
        return

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i, node0 in enumerate(all_nodes):
        p = Process(target=calc_exp_1O_with_parallel, args=(i,node0,return_dict,empirical_matrix,sample_matrix,all_nodes))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()    

    empirical_exp_lam_i_Y = np.asarray([return_dict['empirical_exp_lam_i_Y',ind] for ind in range(len(all_nodes))]) # Expected value of lambda_i Y, observing Y
    exp_lam_i = np.asarray([return_dict['exp_lam_i',ind] for ind in range(len(all_nodes))]) # Expected value of lambda_i
    exp_lam_i_lam_j = np.zeros((len(all_nodes),len(all_nodes)-1)) # Expected value of lambda_i lambda_j
    for k in return_dict.keys():
        if k[0] == 'exp_lam_i_lam_j': exp_lam_i_lam_j[k[1],k[2]] = return_dict[k]

    def calc_exp_2O_with_parallel(cind,cond_vec,emp_cond_vec,return_dict,empirical_matrix,sample_matrix,ind_nodes,all_nodes):
        prob, val = calc_exp_cond(empirical_matrix,[0,-1],emp_cond_vec)
        return_dict['empirical_exp_cond_lam_0_Y',cind] = val * prob
        prob, val = calc_exp_cond(sample_matrix,[0,1,2],cond_vec)  
        return_dict['exp_cond_lam_0_lam_1_lam_2',cind] = val * prob
        for k, node0 in enumerate(ind_nodes):
            prob, val = calc_exp_cond(empirical_matrix,[node0,-1],emp_cond_vec)
            return_dict['empirical_exp_cond_lam_k_Y',cind,k] = prob * val
            prob, val = calc_exp_cond(sample_matrix,[0,node0],cond_vec)
            return_dict['exp_cond_lam_0_lam_k',cind,k] = prob * val
            return_dict['exp_cond_lam_i_lam_k_given_C',cind,k,0] = val
            for i, node1 in enumerate(all_nodes[1:]):
                if node1 != node0:
                    _, val = calc_exp_cond(sample_matrix,[node1,node0],cond_vec)
                    return_dict['exp_cond_lam_i_lam_k_given_C',cind,k,i+1] = val            
        return 

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for c, (condition_vec, empirical_condition_vec) in enumerate(zip(cond_list,empirical_cond_list)):
        p = Process(target=calc_exp_2O_with_parallel, args=(c,condition_vec,empirical_condition_vec,return_dict,empirical_matrix,sample_matrix,ind_nodes,all_nodes))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()  

    empirical_exp_cond_lam_0_Y = [return_dict['empirical_exp_cond_lam_0_Y',cind] for cind in range(len(conds))] # Expected value of 1{condition} lambda_0 Y, observing Y
    exp_cond_lam_0_lam_1_lam_2 = [return_dict['exp_cond_lam_0_lam_1_lam_2',cind] for cind in range(len(conds))] # Expected value of 1{condition} lambda_1 lambda_2 lambda_3
    empirical_exp_cond_lam_k_Y = np.zeros((len(conds),len(ind_nodes))) # Expected value of 1{condition} lambda_k Y, observing Y
    exp_cond_lam_0_lam_k = np.zeros((len(conds),len(ind_nodes))) # Expected value of 1{condition} lambda_0 lambda_k
    exp_cond_lam_i_lam_k_given_C = np.zeros((len(conds),len(ind_nodes),len(all_nodes))) #  Expected value of lambda_i lambda_k given 1{condition}
    for k in return_dict.keys():
        if k[0] == 'empirical_exp_cond_lam_k_Y': empirical_exp_cond_lam_k_Y[k[1],k[2]] = return_dict[k]
        elif k[0] == 'exp_cond_lam_0_lam_k': exp_cond_lam_0_lam_k[k[1],k[2]] = return_dict[k]
        elif k[0] == 'exp_cond_lam_i_lam_k_given_C': exp_cond_lam_i_lam_k_given_C[k[1],k[2],k[3]] = return_dict[k]

    def solve_trips(cind,ind,node,fin_sign,return_dict,exp_cond_lam_i_lam_k_given_C,cond_probs):
        if exp_cond_lam_i_lam_k_given_C[cind,0,ind]==0 or exp_cond_lam_i_lam_k_given_C[cind,1,ind]==0 or exp_cond_lam_i_lam_k_given_C[cind,0,-1]==0:
            return_dict['proposed_exp_cond_lam_ind_Y',cind,0,ind] = 0
            return_dict['proposed_exp_cond_lam_ind_Y',cind,1,ind] = 0
            return_dict['proposed_exp_cond_lam_dep_Y',cind,ind] = 0  
            return
        a_i_cond, a_k_cond, a_l_cond = solve_triplets(exp_cond_lam_i_lam_k_given_C[cind,0,ind],exp_cond_lam_i_lam_k_given_C[cind,1,ind],exp_cond_lam_i_lam_k_given_C[cind,0,-1])
        sign_a_i_cond, sign_a_k_cond, sign_a_l_cond = resolve_signs(exp_cond_lam_i_lam_k_given_C[cind,0,ind],exp_cond_lam_i_lam_k_given_C[cind,1,ind],exp_cond_lam_i_lam_k_given_C[cind,0,-1], fin_sign)
        return_dict['proposed_exp_cond_lam_ind_Y',cind,0,ind] = sign_a_k_cond * np.abs(a_k_cond) * cond_probs[cind]
        return_dict['proposed_exp_cond_lam_ind_Y',cind,1,ind] = sign_a_l_cond * np.abs(a_l_cond) * cond_probs[cind]
        return_dict['proposed_exp_cond_lam_dep_Y',cind,ind] = sign_a_i_cond * np.abs(a_i_cond) * cond_probs[cind]          
        return

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for c in range(len(cond_list)):
        gt_sign_final_node_cond = np.sign(empirical_exp_cond_lam_k_Y[c,-1]) 
        # TODO: fix sign resolution code
        for i, node0 in enumerate(dep_nodes):
            p = Process(target=solve_trips, args=(c,i,node0,gt_sign_final_node_cond,return_dict,exp_cond_lam_i_lam_k_given_C,cond_probs))
            jobs.append(p)
            p.start()
    for proc in jobs:
        proc.join()  

    proposed_exp_cond_lam_ind_Y = np.zeros((len(conds),len(ind_nodes),len(dep_nodes))) # Expected value of 1{condition} lambda_k Y, lambda_k cond. independent of other lambdas, according to new model
    proposed_exp_cond_lam_dep_Y = np.zeros((len(conds),len(dep_nodes))) # Expected value of 1{condition} lambda_k Y, lambda_k cond. independent of other lambdas, according to new model
    for k in return_dict.keys():
        if k[0] == 'proposed_exp_cond_lam_ind_Y': proposed_exp_cond_lam_ind_Y[k[1],k[2],k[3]] = return_dict[k]
        elif k[0] == 'proposed_exp_cond_lam_dep_Y': proposed_exp_cond_lam_dep_Y[k[1],k[2]] = return_dict[k]

    proposed_exp_cond_lam_0_Y = proposed_exp_cond_lam_dep_Y[:,0]
    print('Done with estimating mean parameters, starting backwards mapping...')

    
    

    #################### BACKWARDS MAPPING ####################
    # Backwards map to canonical parameters 
    parent_node_num = 5
    est_moment_matrix = {(parent_node_num,'unary1'): empirical_exp_y[0], 
                         (parent_node_num,'unary2'): empirical_exp_y[1], 
                         (parent_node_num,'unary3'): empirical_exp_y[2], 
                         (parent_node_num,'unary4'): empirical_exp_y[3], 
                         (3, 'pairwise1'): np.nanmean(proposed_exp_cond_lam_ind_Y[0,0,:]), 
                         (3, 'pairwise2'): np.nanmean(proposed_exp_cond_lam_ind_Y[1,0,:]), 
                         (3, 'pairwise3'): np.nanmean(proposed_exp_cond_lam_ind_Y[2,0,:]), 
                         (3, 'pairwise4'): np.nanmean(proposed_exp_cond_lam_ind_Y[3,0,:]), 
                         (4, 'pairwise1'): np.nanmean(proposed_exp_cond_lam_ind_Y[0,1,:]), 
                         (4, 'pairwise2'): np.nanmean(proposed_exp_cond_lam_ind_Y[1,1,:]), 
                         (4, 'pairwise3'): np.nanmean(proposed_exp_cond_lam_ind_Y[2,1,:]), 
                         (4, 'pairwise4'): np.nanmean(proposed_exp_cond_lam_ind_Y[3,1,:]), 
    #                      (0, 1, 2, 'lams1'): exp_cond_lam_0_lam_1_lam_2[0], (0, 1, 2, 'lams2'): exp_cond_lam_0_lam_1_lam_2[1], # Not estimating currently 
    #                      (0, 1, 2, 'lams3'): exp_cond_lam_0_lam_1_lam_2[2], (0, 1, 2, 'lams4'): exp_cond_lam_0_lam_1_lam_2[3], # Not estimating currently 
                         (0, 1, 2, 'triple1'): proposed_exp_cond_lam_0_Y[0], (0, 1, 2, 'triple2'): proposed_exp_cond_lam_0_Y[1], 
                         (0, 1, 2, 'triple3'): proposed_exp_cond_lam_0_Y[2], (0, 1, 2, 'triple4'): proposed_exp_cond_lam_0_Y[3]}
    ismod, theta_est, error = mean_to_canonical(est_moment_matrix, parent_node_num, conds, alpha=.05, maxiter=1000, accelerated=True)
    
    print('Done with backwards mapping, starting computing conditionals...')

    
    
    
    #################### MAKE ESTIMATED PMF ####################
    # Get distribution using estimated canonical params
    lst = list(map(list, itertools.product([-1, 1], repeat=len(all_nodes)+1))) 
    est_theta_y = np.asarray([theta_est['value'].values[:4]]).T
    est_theta_ind = np.vstack((theta_est['value'].values[4:8],theta_est['value'].values[8:12])).T
    est_theta_dep = theta_est['value'].values[12:]
    est_theta_lam = 0 #theta_est['value'].values[9:13] # Not estimating currently
    proposed_thetas = [est_theta_y,est_theta_ind,est_theta_dep,est_theta_lam]
    proposed_pmf = make_pmf(*proposed_thetas,lst,comp=1)

    
    
    
    #################### COMPUTE ESTIMATED CONDITIONAL PROBABILITIES ####################
    sample_dist = np.zeros(sample_matrix.shape[0]).astype('int')
    observed_indices = np.zeros(empirical_matrix.shape[0]).astype('int')
    sample_dist_test = np.zeros(sample_matrix_test.shape[0]).astype('int')
    for sample_ind, sample in enumerate(tqdm(lst)):
        if sample_ind % 2 == 1:
            sample_dist[np.prod([sample_matrix[:,i]==sample[i] for i in range(len(sample[:-1]))],0)==1] = sample_ind
            sample_dist_test[np.prod([sample_matrix_test[:,i]==sample[i] for i in range(len(sample[:-1]))],0)==1] = sample_ind
        observed_indices[np.prod([empirical_matrix[:,i]==sample[i] for i in range(len(sample))],0)==1] = sample_ind
    compl_sample_dist = sample_dist - 1
    compl_sample_dist_test = sample_dist_test - 1

    proposed_conditional = proposed_pmf[sample_dist]/(proposed_pmf[sample_dist]+proposed_pmf[compl_sample_dist])
    proposed_conditional_test = proposed_pmf[sample_dist_test]/(proposed_pmf[sample_dist_test]+proposed_pmf[compl_sample_dist_test])

    return proposed_thetas, proposed_pmf, proposed_conditional, proposed_conditional_test

    
