# DeepIV
Python library for implementing deep instrumental variables as described in Hartford et al.(2016) paper. (link: https://arxiv.org/abs/1612.09596). Implements it for both discrete and continuous variables in tensorflow. currently only compatible with one layer.

# Files / Code

* mdn.py: a library containing functions to estimate/run both the first stage distribution of deepIV as a mixture density network (MDN) or as a multinomial (MN) for the case of a discrete endogenous variable (in our setting, years of education)
* deepiv.py: a library containing functions to estimate/run the second stage of deepIV, given a distribution for each observation estimated w/ mdn.py. In particular, it contains functions for the loss function and gradient outlined in hartford et al., along with the training / CV functions for the 2nd stage. It also contains functions to perform frequentist counterfactuals, including estimating treatments/instruments, and the IV coefs.
