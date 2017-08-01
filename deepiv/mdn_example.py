#file to test out mdn functions
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #my created library

gitdir = '/home/luis/CausalML-project/'
np.random.seed(1923)


datadir = '/home/luis/CausalML-project/Data/'

settlers = pd.read_csv(datadir+'colonial_origins_data_missimp.csv')
#remove those missing either outcome, endog institution measure, or exog instrument
nomiss = (settlers['mi_avexpr']==0) & (settlers['mi_logpgp95']==0)
settlers = settlers.loc[nomiss,:]
num_obs = settlers.shape[0]

p = np.array(settlers['avexpr']) #the endogenous variable
p.shape = [p.shape[0],1]#make it 2D
z = np.array(settlers.loc[:,['logem4','mi_logem4']]) #the instrument
#z.shape = [z.shape[0],] 

all_covars=np.r_[1:3, 7:8,10:52, 54,58:84]
#feature sets of covariates we might consider
#x = np.array(settlers.iloc[:,17:38])
#x = np.array(settlers.iloc[:,10:13])
#x = np.array(settlers.iloc[:,10:46]) 
x = settlers.iloc[:,all_covars] #this should be all of them


covars =  np.concatenate((z, x), axis=1)
collin_vars = [] #indices of variables with no variation
#stdize all non-dummy variables to have mean 0 and SD 1
for v in range(covars.shape[1]):
    
    #remove variables with one unique value- they mess stuff up later
    if len(np.unique(covars[:,v].astype(np.float32)))==1:
        collin_vars.append(v)
        continue
    #skip normalizations for dummies (although I guess it doesn't really matter)
    is_dummy = (np.unique(covars[:,v].astype(np.float32))==np.array([0.,1.]))   
    if isinstance(is_dummy,bool):
        if is_dummy:
            continue
    else:
        if is_dummy.all():
            continue        
    covars[:,v] = (covars[:,v] - np.mean(covars[:,v]))/np.std(covars[:,v])


covars=np.delete(covars,collin_vars,axis=1)
[[W_in_final, B_in_final, W_out_final,B_out_final],[mixprobs,mixmeans,mixsds]] = mdn.fit_MDN(p,covars,learning_rate=0.01,num_nodes=15)
#mean_LL = mdn.cv_MDN(p,covars)

#graph those with nonmissing entries for both instrument and  policy variable
#nomiss_z = np.array(settlers['mi_logem4']==0)
#mdn.plot_mdn_sim(p[nomiss_z],z[nomiss_z,0],covars[nomiss_z,:], \
#    mixprobs[nomiss_z,:],mixmeans[nomiss_z,:],mixsds[nomiss_z,:],figdir = '/home/luis/CausalML-project/DeepIV/')

