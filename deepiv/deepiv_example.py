import deepiv
import math
import mdn
import numpy as np
import matplotlib.pyplot as plt
N=10000
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
outcome_weights = deepiv.train_second_stage_cont(y,covars_ss,
							mixprobs,mixmeans,mixsds,
							deeplayer_nodes=[10,10],p_index=0)

for l in outcome_weights:
	print l,outcome_weights[l].shape

