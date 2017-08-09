import deepiv
import math
import mdn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
N=5000
np.random.seed(9)
t = np.random.rand(N)*10.
s = np.random.randint(1,8,size=N)
z = np.random.normal(size=N)
v = np.random.normal(size=N)
rho=.5
e = np.random.normal(rho*v,1-rho**2)
psi = 2 * ( (t-5)**4/600. + np.exp(-4*(t-5)**2) + t/10 - 2 )

p = 25 + psi * (z+3) +v
y = 100 + s*psi + (psi-2)*p + e

bounds = []
varnames = ['t','s','p','y']
for var in [t,s,z,p,y]:
	#print var.shape
	var.shape = (N,1)
	print [np.min(var),np.max(var)]
	bounds.append([np.min(var),np.max(var)])
	var[...] = (var - np.min(var) ) / (np.max(var)-np.min(var))

covars = np.concatenate([z,t,s],axis=1)
endog = p
outcome = y
#part 1; MDN

[weights,[mixprobs,mixmeans,mixsds]] = mdn.fit_MDN(endog,covars,
						num_components=10,
						deeplayer_nodes=[32,32,32],
						num_batches=10,num_epochs=100,plot_loss=False)

covars_ss = np.concatenate([p,t,s],axis=1)
#part 2; DeepIV Net
outcome_weights,div_layers = deepiv.train_second_stage_cont(y,covars_ss,
							mixprobs,mixmeans,mixsds,
							deeplayer_nodes=[32,8],p_index=0,
							learning_rate=1.,num_epochs=5)

for l in outcome_weights:
	print l,outcome_weights[l].shape

#create  frequentist coefficients
treat,inst = deepiv.predict_etas_cont(mixprobs,mixmeans,mixsds, \
                  covars_ss,outcome_weights,p_mean=0,p_sd=1, B=1000,p_index=0)
plt.scatter(treat.flatten(),inst.flatten(),alpha=.1)
plt.show()

#####
#treat and inst should be done on left-out data
beta,V_beta = deepiv.estimate_iv_coefs(y,treat,inst)


#finally, do a counterfactual
cf_obs = 100
p_cf=np.median(p)
s_cf=np.median(p)
cf = np.zeros(shape=(cf_obs,3))
cf[:,2] = s_cf
cf[:,0] = p_cf
cf[:,1] = np.linspace(0,1,cf_obs)
cf_etas = npdeepiv.predict_eta(outcome_weights,cf)
cf_etas = np.concatenate((np.ones([cf_obs,1]),cf_etas),axis=1)
h_cf = np.dot(cf_etas,beta)
#for i in range(len(h_cf)):
#	h_cf[i] = np.dot(cf[[i],:])
plt.plot(cf[:,1],h_cf)
plt.show()