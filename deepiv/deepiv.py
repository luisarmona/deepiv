#Attempt to fit a deep IV model to an empirical dataset
#v0.1 - attempt to fit mixture density network a la Bishop (1994)

#heavily inspired by 
#http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #library for mixture density network estimation I made


#given set of features, predict our DNN conditional distribution for the first stage.
#useful for recovering parameters on new data
def predict_1stStage_cond_dist(features,W_features,B_features,W_hidden,B_hidden):
    hidden = np.tanh(np.dot(features,W_features) + B_features)
    output = np.dot(hidden,W_hidden) + B_hidden
    probs,means,sds = np.split(output,3,axis=1)
    sds = np.exp(sds)
    probs = np.exp(probs)/np.sum(np.exp(probs),axis=1)[:,np.newaxis]
    return probs,means,sds






#given estimated network parameters
#sample from the second stage network up to the final hidden layer (the "etas" in the paper)
#to get instruments and treatments given network parameters.
#p_index denotes location in features_2 of the policy variable
# B indicates the number of times we sample to calculate the expectation.
def predict_etas_cont(pi,mu,sigma, \
                      features_2,W_features_2,B_features_2,p_mean,p_sd, B=1000,p_index=0):
    num_obs = features_2.shape[0]
    num_etas = W_features_2.shape[1]
    #calculate predicted etas given observed p
    treatments =  np.tanh(np.dot(features_2,W_features_2) + B_features_2)
    #calculate predicted etas given 1st stage distribution
    instruments = np.zeros(shape=[num_obs,num_etas,B])
    temp_features_2 = features_2
    #sample from the policy fcn
    for j in range(B):
        distchoice =   (np.random.rand(num_obs,1)<=pi.cumsum(axis=1)).argmax(axis=1)
        p_samp= np.random.normal(loc=mu[np.arange(num_obs),distchoice],scale=sigma[np.arange(num_obs),distchoice])
        p_samp = (p_samp - p_mean)/p_sd
        temp_features_2[:,p_index] = p_samp 
        etas = np.tanh(np.dot(temp_features_2,W_features_2) + B_features_2)
        instruments[:,:,j] = etas

    instruments = np.mean(instruments,axis=2)

    return [treatments,instruments]

#given estimated network parameters
#sample from the second stage network (for use on left-out sample)
#to get instruments and treatments given network parameters
#p_index denotes location in features_2 of the policy variable
def predict_etas_discrete(P,features_2,W_features_2,B_features_2,p_range):
    num_obs = features_2.shape[0]
    num_etas = W_features_2.shape[1]
    #calculate predicted y given observed p (treatments)
    treatments = np.tanh(np.dot(features_2,W_features_2) + B_features_2)
    #calculate predicted y given 1st stage multinomial
    instruments = np.zeros(shape=[num_obs,num_etas])
    temp_features = features_2
    cats = range(P.shape[1]) # number of classes
    #sample from the policy fcn
    for c in cats:
        temp_p  = np.zeros(shape=[len(cats),1])
        temp_p[c] = 1
        temp_features[:,p_range] = temp_p.flatten()
        exp_eta = P[:,c][:,np.newaxis] * np.tanh(np.dot(temp_features,W_features_2) + B_features_2)
        instruments = instruments + exp_eta
    return [treatments,instruments]

#given treatments,instruments (as defined in DeepIV paper) and outcomes,
#from a left-out validation sample, calculate the treatment coefficients via 2SLS
#and the variance
def estimate_iv_coefs(y,treatment,instruments):
    num_validation_obs = y.shape[0]
    H =  np.concatenate((np.ones([num_validation_obs,1]), treatment), axis=1).astype(np.float32)
    H_hat =  np.concatenate((np.ones([num_validation_obs,1]), instruments), axis=1).astype(np.float32)

    H_hatxH_inv = np.linalg.inv(np.dot(H_hat.transpose(),H)).astype(np.float32)
    H_hatxH_hat_inv =  np.linalg.inv(np.dot(H_hat.transpose(),H_hat)).astype(np.float32)
    beta = np.dot( H_hatxH_inv, np.dot(H_hat.transpose(),y) ).astype(np.float32)
    #do the diagonal residual matrix factorization obs by obs
    res = (np.dot(H,beta) - y)**2
    D_res = np.zeros([H.shape[1],H.shape[1]])
    for i in range(num_validation_obs):
        #ind_transpose = res[i] * np.dot(H_hat[i,:].transpose(),H_hat[i,:])
        D_res = D_res + res[i] * np.dot(H_hat[i,:][:,np.newaxis],H_hat[i,:][:,np.newaxis].transpose() )
    #D_res = D_res/num_validation_obs
    #ind_transpose = res[i] * np.dot(H_hat[i,:][].transpose(),H_hat[i,:])
    #print ind_transpose.shape
    #D_res = np.zeros([num_validation_obs,num_validation_obs],dtype=np.float32)
    #res = (np.dot(H,beta) - y)**2
    #np.fill_diagonal(D_res,res)
    #V_beta = np.dot(np.dot(H_hat.transpose(),D_res),H_hat).astype(np.float32) #the inner filling of the sandwich
    V_beta  = np.dot(np.dot(H_hatxH_hat_inv,D_res ),H_hatxH_hat_inv).astype(np.float32) #full variance
    print "--------------------"
    print "Treatment Effects of Intercept+ NN output node(s):"
    print beta
    print "Variance of Average Treatment Effects:"
    print V_beta
    return beta,V_beta


#stdize variables that are not dummies to be stdizing mean 0 and sd 1,
#and remove all covariates that have no variation in the data
#so that all behave well when fed into DNN
def process_features(features):
    collin_vars = [] #indices of variables with no variation
    feature_means = [] #store means we normalize
    feature_sds = [] #store SDs we normalize
    for v in range(features.shape[1]):
        #remove variables with one unique value- they mess stuff up later
        if len(np.unique(features[:,v].astype(np.float32)))==1:
            collin_vars.append(v)
            continue
        #skip normalizations for dummies (although I guess it doesn't really matter)
        is_dummy = (np.unique(features[:,v].astype(np.float32))==np.array([0.,1.]))   
        if isinstance(is_dummy,bool):
            if is_dummy:
                feature_means.append(0) #for dummies do not transform
                feature_sds.append(1)
                continue
        else:
            if is_dummy.all():
                feature_means.append(0) #for dummies do not transform
                feature_sds.append(1)
                continue  
        feature_means.append(np.mean(features[:,v])) #for dummies do not transform
        feature_sds.append(np.std(features[:,v]))    
        features[:,v] = (features[:,v] - np.mean(features[:,v]))/np.std(features[:,v])
    return [features,feature_means,feature_sds,collin_vars]



#estimate loss function (for validation training)
#args: test_y is true outcome data,
#outcome dnn is the DNN trained to predict y, given inputs
#session is the current tensorflow session being used
#features2 is the set of 2nd stage features,
#probs/means/sds1 are the first stage cond. distro parameters,
#B is the number of simulations per obs
#p_index is the column index of the policy variable we simulate in the feature matrix
def secondstage_loss_cont(outcome,outcome_dnn,inputs,session,features2,probs1,means1,sds1,p_mean,p_sd,B=1000,p_index=0):
    mc_outcomes = np.zeros(shape = (outcome.shape[0],B))
    mc_policy = (mdn.sim_mdn(probs1,means1,sds1,B=B) - p_mean)/p_sd
    temp_features = features2
    for b in range(B):
        temp_features[:,p_index] = mc_policy[:,b]
        mc_outcomes[:,b] = session.run(outcome_dnn,feed_dict={inputs: temp_features.astype(np.float32)}).flatten()
    pred_y_hat = np.mean(mc_outcomes,axis=1)
    pred_y_hat.shape = [pred_y_hat.shape[0],1]
    return np.mean((pred_y_hat - outcome)**2.)

#fcn for gradient for SGD for 1st stage MDNs
#args: outcome is the real data, 
#features2 are second stage features
#p_index is the location of policy variable in the feature matrix
#pi/mu/sigam1 are conditional distro of each obs
#outcome_dnn is the output layer of 2nd stage dnn fcn
#grad_fcn calculates the gradients of the NN in tensorflow
#B is number of simulations for the gradient
#session is cur tf session
#currently accepts just one observation 
def ind_secondstage_loss_gradient_cont(outcome,features2,pi1,mu1,sigma1, \
        outcome_dnn,inputs,grad_fcn,session,p_mean,p_sd,p_index=0):
    #correct one obs issue w/ array instead of mat
    #print pi1.shape
    p1 = (mdn.sim_mdn(pi1,mu1,sigma1,B=1) - p_mean)/p_sd
    #print pi1.shape
    p2 = (mdn.sim_mdn(pi1,mu1,sigma1,B=1) - p_mean)/p_sd
    tempfeat_1 = features2
    tempfeat_2 = features2
    tempfeat_1[:,p_index] = p1
    tempfeat_2[:,p_index] = p2
    #print"-----"
    pred_outcome = session.run(outcome_dnn,feed_dict={inputs: tempfeat_1.astype(np.float32)})
    grad = session.run(grad_fcn,feed_dict={inputs: tempfeat_2.astype(np.float32)})
    #print(grad)
    multiplier = -2.* (outcome - pred_outcome)
    newgrad=[]
    for g in range(len(grad)):
        newgrad.append(multiplier*grad[g])
    return newgrad

#estimate loss function (for validation training)
#args: outcome is true outcome data,
#outcome_dnn is the DNN trained to predict y, given inputs
#session is the current tensorflow session being used
#features2 is the set of 2nd stage features,
#probs/means/sds1 are the first stage cond. distro parameters,
#B is the number of simulations per obs
#p_index is the column index of the policy variable we simulate in the feature matrix
def secondstage_loss_discrete(outcome,features2,P,p_range,outcome_dnn,inputs,session):
    exp_outcomes = np.zeros(shape = (outcome.shape[0],1)) #E[h|1st stage]
    temp_features = features2
    cats = range(P.shape[1])
    for c in cats:
        temp_p  = np.zeros(shape=[len(cats),1])
        temp_p[c] = 1
        temp_features[:,p_range] = temp_p.flatten()
        exp_outcomes[:,0]= P[:,c]*session.run(outcome_dnn,feed_dict={inputs: temp_features.astype(np.float32)}).flatten()
    return np.mean((exp_outcomes - outcome)**2.)


#the analogue to the function above, but done for discrete endogoneous variables
#which are assumed to be fed into the outcome NN via indicators for each category
#p-range defines the range of column indices that 
def ind_secondstage_loss_gradient_discrete(outcome,features2, P,p_range,
        outcome_dnn,inputs,grad_fcn,session):
    tempfeat = features2
    #print"-----"
    pred_outcome=0
    cats = range(P.shape[1])
    for c in cats:
        temp_p  = np.zeros(shape=[len(cats),1])
        temp_p[c] = 1
        tempfeat[:,p_range] = temp_p.flatten()
        pred_outcome = P[:,c]*session.run(outcome_dnn,feed_dict={inputs: tempfeat.astype(np.float32)})
        temp_grad = session.run(grad_fcn,feed_dict={inputs: tempfeat.astype(np.float32)})
        temp_grad = [P[:,c]*g for g in temp_grad]
        if c>0:
            new_grad = [ (ng + tg) for ng,tg in zip(new_grad,temp_grad)]
        else:
            new_grad = temp_grad
    #print(grad)
    multiplier = -2.* (outcome - pred_outcome)
    new_grad = [ multiplier*ng for ng in new_grad]
    return new_grad



#workflow:
#need to estimate the 2nd stage loss function; do this by, 
#initialize dnn to random one including all features besides the instruments.
#in loop:
# 1.sampling 1 obs per epoch
# 2.sampling to policy outcomes per obs
# 3.calculating gradient of DNN w.r.t. this obs via tf
# 4.step in that direction

# 5. to evaluate the loss for the CV step, across all obs, draw from the conditional distro
#    a lot of times, calc outcome dnn for each, use this to estimate the integral, then subtract from truth squared
#
#rinse and repeat a million times or whatever
# and from each drawing a policy variable; then calculating the gradient of the DNN of 2nd stage on each 
# obs.

#args:
#y: outcome
#p: the endogenous policy variable
#features_second: the covariates for the second stage (should be 1st column is p, the rest are x controls)
#if this must be changed, additional arg is available (p_index denotes col to train)
#pi,mu,sigma: the rows of each individual's 1st stage distribution of the endogenous variable (expressed as mix of normals)
#num_nodes: the number of nodes in the hidden layer
def train_second_stage_cont(y,features_second,pi,mu,sigma,num_nodes,p_mean,p_sd,seed=None,p_index=0,learning_rate=0.001):
    if seed!=None:
        np.random.seed(seed)  
    else:
        seed=9 #for the tf calls     

    num_inputs = features_second.shape[1] #the number of input features
    num_output = 1 # output layer (currently just one since outcome is one variable)
    num_obs = y.shape[0]


    #initialize weights and biases for input->hidden layer
    W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_in')
    b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_in')
    #initialize weights and biases for hidden->output layer
    W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_out')
    b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_out')
    #instantiate data vars
    inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
    outcome = tf.placeholder(dtype=tf.float32, shape=[None,1], name="outcome")
    #define the function for the hidden layer
    #use canonical tanh function for intermed, simple linear combo for final layer
    hidden_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
    outcome_layer = tf.matmul(hidden_layer,W_output) + b_output
    #the gradients of the output layer w.r.t. network parameters
    nn_gradients = tf.gradients(outcome_layer, [W_input, b_input,W_output,b_output]) #the gradients of the DNN w.r.t. parameters
    #placeholders for gradients I pass from numpy
    g_W_in = tf.placeholder(dtype=tf.float32, shape=W_input.get_shape(), name="g_W_in")
    g_b_in = tf.placeholder(dtype=tf.float32, shape=b_input.get_shape(), name="g_b_in")
    g_W_out = tf.placeholder(dtype=tf.float32, shape=W_output.get_shape(), name="g_W_out")
    g_b_out = tf.placeholder(dtype=tf.float32, shape=b_output.get_shape(), name="g_b_out")
    #the gradient-parameter pairs for gradient computation/application
    grad_var_pairs = zip([g_W_in,g_b_in,g_W_out,g_b_out],[W_input,b_input,W_output,b_output])

    #the optimizer
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #initialize tensorflow
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())

    #break up validation/training data
    validation_losses=[]
    validation_indices = np.random.choice(num_obs,num_obs/5)
    train_indices = np.ones(len(y), np.bool)
    train_indices[validation_indices]=0
    y_validation = y[validation_indices]
    features_validation = features_second[validation_indices,:]
    y_train = y[train_indices]
    features_train  = features_second[train_indices,:]
    num_train_obs = sum(train_indices)
    print "training..."
    num_iters = 10000
    for i in range(num_iters):
        if i%100==0:
            print "     iteration: " + str(i)
        #extract observation features for SGD
        g_ind=np.random.choice(num_train_obs,1)[0]
        obs_feat= features_second[train_indices,:][g_ind,:]
        obs_y = y[train_indices][g_ind]
        pi_i = pi[train_indices,:][g_ind,:]
        mu_i = mu[train_indices,:][g_ind,:]
        sd_i = sigma[train_indices,:][g_ind,:]
        #reshape everything so treated as 2d
        for v in [obs_y, obs_feat, pi_i, mu_i ,sd_i]:
            v.shape = [1,len(v)]

        stoch_grad = ind_secondstage_loss_gradient_cont(obs_y,obs_feat,pi_i,mu_i,sd_i,outcome_layer,inputs,nn_gradients,s,p_mean,p_sd)
        grad_dict={}
        grad_index=0
        for theta in [g_W_in,g_b_in,g_W_out,g_b_out]:
            grad_dict[theta]=stoch_grad[grad_index]
            grad_index+=1
        s.run(trainer.apply_gradients(grad_var_pairs),feed_dict=grad_dict)
        #the gradients of the output layer w.r.t. network parameters
        if i%10==0:
            loss=secondstage_loss_cont(y[validation_indices],outcome_layer,inputs,s,\
                features_second[validation_indices,:], \
                pi[validation_indices,:], \
                mu[validation_indices,:], \
                sigma[validation_indices,:], \
                p_mean,p_sd,B=1000)
            validation_losses.append(loss)
            if len(validation_losses) > 5:
                if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)])< validation_losses[len(validation_losses)-1]:
                    print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                    break
    plt.plot(range(len(validation_losses)),validation_losses)
    plt.show()
    #recover parameters and return them
    W_in_final = s.run(W_input)
    B_in_final = s.run(b_input)
    W_out_final = s.run(W_output)
    B_out_final = s.run(b_output)
    s.close()
    return [W_in_final, B_in_final, W_out_final,B_out_final]

def cv_second_stage_cont(y,features_second,pi,mu,sigma,num_nodes,p_mean,p_sd,seed=None,p_index=0,folds=5,learning_rate=0.001):
    if seed!=None:
        np.random.seed(seed)  
    else:
        seed=9 #for the tf calls     
    #some test code below
    num_inputs = features_second.shape[1] #the number of input features
    num_output = 1 # output layer (currently just one since outcome is one variable)
    num_obs = y.shape[0]

    #create folds
    rng_orders = np.argsort(np.random.uniform(size=num_obs))
    foldgroups = np.zeros(num_obs)
    for k in range(1,folds+1):
        group_obs = (rng_orders >= (k-1)*num_obs/folds) & (rng_orders <(k)*num_obs/folds)
        foldgroups[group_obs]=k
    #train / test for each fold
    test_losses=[]
    for k in range(1,folds+1):
        #print 'fold=' +str(k)
        #split up train/test samples
        y_train = y[foldgroups!=k]
        features_train = features_second[foldgroups!=k,:]
        pi_train = pi[foldgroups!=k,:]
        mu_train = mu[foldgroups!=k,:]
        sigma_train=sigma[foldgroups!=k,:]

        y_test = y[foldgroups==k]
        features_test = features_second[foldgroups==k,:]
        pi_test = pi[foldgroups==k,:]
        mu_test = mu[foldgroups==k,:]
        sigma_test=sigma[foldgroups==k,:]

        num_obs_train = y_train.shape[0]


        #initialize weights and biases for input->hidden layer
        W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_in')
        b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_in')
        #initialize weights and biases for hidden->output layer
        W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_out')
        b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_out')
        #instantiate data vars
        inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
        outcome = tf.placeholder(dtype=tf.float32, shape=[None,1], name="outcome")
        #define the function for the hidden layer
        #use canonical tanh function for intermed, simple linear combo for final layer
        hidden_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
        outcome_layer = tf.matmul(hidden_layer,W_output) + b_output
        #the gradients of the output layer w.r.t. network parameters
        nn_gradients = tf.gradients(outcome_layer, [W_input, b_input,W_output,b_output]) #the gradients of the DNN w.r.t. parameters
        #placeholders for gradients I pass from numpy
        g_W_in = tf.placeholder(dtype=tf.float32, shape=W_input.get_shape(), name="g_W_in")
        g_b_in = tf.placeholder(dtype=tf.float32, shape=b_input.get_shape(), name="g_b_in")
        g_W_out = tf.placeholder(dtype=tf.float32, shape=W_output.get_shape(), name="g_W_out")
        g_b_out = tf.placeholder(dtype=tf.float32, shape=b_output.get_shape(), name="g_b_out")
        #the gradient-parameter pairs for gradient computation/application
        grad_var_pairs = zip([g_W_in,g_b_in,g_W_out,g_b_out],[W_input,b_input,W_output,b_output])

        #the optimizer
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #initialize tensorflow
        s = tf.InteractiveSession()
        s.run(tf.global_variables_initializer())

        #split training further into validation and training data
        validation_losses=[]
        validation_indices = np.random.choice(num_obs_train,num_obs_train/5)
        train_indices = np.ones(len(y_train), np.bool)
        train_indices[validation_indices]=0
        
        y_validation = y_train[validation_indices]
        features_validation = features_train[validation_indices,:]
        pi_validation = pi_train[validation_indices,:]
        mu_validation = mu_train[validation_indices,:]
        sigma_validation=sigma_train[validation_indices,:]
        
        y_train = y_train[train_indices]
        features_train  = features_train[train_indices,:]
        pi_train = pi_train[train_indices,:]
        mu_train = mu_train[train_indices,:]
        sigma_train=sigma_train[train_indices,:]
        num_train_obs = sum(train_indices)

        #print "training..."
        num_iters = 10000
        for i in range(num_iters):
            #if i%100==0:
            # print "     iteration: " + str(i)
            #extract observation features for SGD
            g_ind=np.random.choice(num_train_obs,1)[0]
            obs_feat= features_train[g_ind,:]
            obs_y = y_train[g_ind]
            pi_i = pi_train[g_ind,:]
            mu_i = mu_train[g_ind,:]
            sd_i = sigma_train[g_ind,:]
            #reshape everything so treated as 2d
            for v in [obs_y, obs_feat, pi_i, mu_i ,sd_i]:
                v.shape = [1,len(v)]
            stoch_grad = ind_secondstage_loss_gradient_cont(obs_y,obs_feat,pi_i,mu_i,sd_i,outcome_layer,inputs,nn_gradients,s,p_mean,p_sd)
            grad_dict={}
            grad_index=0
            for theta in [g_W_in,g_b_in,g_W_out,g_b_out]:
                grad_dict[theta]=stoch_grad[grad_index]
                grad_index+=1
            s.run(trainer.apply_gradients(grad_var_pairs),feed_dict=grad_dict)
            #the gradients of the output layer w.r.t. network parameters
            if i%10==0:
                loss=secondstage_loss_cont(y_validation,outcome_layer,inputs,s,\
                    features_validation, \
                    pi_validation, \
                    mu_validation, \
                    sigma_validation, \
                    p_mean,p_sd,B=1000)
                validation_losses.append(loss)
                if len(validation_losses) > 5:
                    if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)]) < validation_losses[len(validation_losses)-1]:
                        print "--------------------------"
                        print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                        break
        test_loss = secondstage_loss_cont(y_test,outcome_layer,inputs,s,\
                    features_test, \
                    pi_test, \
                    mu_test, \
                    sigma_test, \
                    p_mean,p_sd,B=1000)
        test_losses.append(test_loss)
        s.close()
        print "completed fold " + str(k) + ' for n=' + str(num_nodes)
        print "--------------"

    return test_losses

def cv_mp_second_stage_cont(y,features_second,pi,mu,sigma,num_nodes,p_mean,p_sd,seed,p_index,folds,learning_rate,filename):
    test_losses = cv_second_stage_cont(y,features_second,pi,mu,sigma,num_nodes,p_mean,p_sd,seed,p_index,folds,learning_rate)
    meanerr = np.mean(test_losses)
    sderr = np.std(test_losses)
    f = open(filename,'w')
    f.write('node,mean,se \n')
    f.write( str(num_nodes) + ',' +str(meanerr) + ',' + str(sderr) )
    f.close()
    print "**********************"
    print "node=" + str(num_nodes) + "; mean test err=" + str(meanerr) + "; sd test err=" + str(sderr)
    print "***********************"
    return [num_nodes,test_losses]



#args:
#y: outcome
#p: the endogenous policy variable (assumed to be a slice of column dummies for each class)
#features_second: the covariates for the second stage (should be 1st set is p, the rest are x controls)
#if this must be changed, additional arg is available (p_index denotes col to train)
#P: the matrix of predicted probabilities that depend on x plus instruments
#p_range: the location the the p dummies for each class
#num_nodes: the number of nodes in the hidden layer
def train_second_stage_discrete(y,features_second,P,p_range,num_nodes,learning_rate=0.001,seed=None,):
    if seed!=None:
        np.random.seed(seed)  
    else:
        seed=9 #for the tf calls     

    num_inputs = features_second.shape[1] #the number of input features
    num_output = 1 # output layer (currently just one since outcome is one variable)
    num_obs = y.shape[0]


    #initialize weights and biases for input->hidden layer
    W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_in')
    b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_in')
    #initialize weights and biases for hidden->output layer
    W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_out')
    b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_out')
    #instantiate data vars
    inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
    outcome = tf.placeholder(dtype=tf.float32, shape=[None,1], name="outcome")
    #define the function for the hidden layer
    #use canonical tanh function for intermed, simple linear combo for final layer
    hidden_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
    outcome_layer = tf.matmul(hidden_layer,W_output) + b_output
    #the gradients of the output layer w.r.t. network parameters
    nn_gradients = tf.gradients(outcome_layer, [W_input, b_input,W_output,b_output]) #the gradients of the DNN w.r.t. parameters
    #placeholders for gradients I pass from numpy
    g_W_in = tf.placeholder(dtype=tf.float32, shape=W_input.get_shape(), name="g_W_in")
    g_b_in = tf.placeholder(dtype=tf.float32, shape=b_input.get_shape(), name="g_b_in")
    g_W_out = tf.placeholder(dtype=tf.float32, shape=W_output.get_shape(), name="g_W_out")
    g_b_out = tf.placeholder(dtype=tf.float32, shape=b_output.get_shape(), name="g_b_out")
    #the gradient-parameter pairs for gradient computation/application
    grad_var_pairs = zip([g_W_in,g_b_in,g_W_out,g_b_out],[W_input,b_input,W_output,b_output])

    #the optimizer
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #initialize tensorflow
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())

    #break up validation/training data
    validation_losses=[]
    validation_indices = np.random.choice(num_obs,num_obs/5)
    train_indices = np.ones(len(y), np.bool)
    train_indices[validation_indices]=0
    y_validation = y[validation_indices]
    features_validation = features_second[validation_indices,:]
    y_train = y[train_indices]
    features_train  = features_second[train_indices,:]
    num_train_obs = sum(train_indices)
    print "training..."
    num_iters = 10000
    for i in range(num_iters):
        if i%100==0:
            print "     iteration: " + str(i)
        #extract observation features for SGD
        g_ind=np.random.choice(num_train_obs,1)[0]
        obs_feat= features_second[train_indices,:][g_ind,:]
        obs_y = y[train_indices][g_ind]
        P_i = P[train_indices,:][g_ind,:]
        #reshape everything so treated as 2d
        for v in [obs_y, obs_feat, P_i]:
            v.shape = [1,len(v)]

        stoch_grad = ind_secondstage_loss_gradient_discrete(obs_y,obs_feat,P_i,p_range,outcome_layer,inputs,nn_gradients,s)
        grad_dict={}
        grad_index=0
        for theta in [g_W_in,g_b_in,g_W_out,g_b_out]:
            grad_dict[theta]=stoch_grad[grad_index]
            grad_index+=1
        s.run(trainer.apply_gradients(grad_var_pairs),feed_dict=grad_dict)
        #the gradients of the output layer w.r.t. network parameters
        if i%10==0:
            loss=secondstage_loss_discrete(y[validation_indices], \
                features_second[validation_indices,:], \
                P[validation_indices,:],p_range,\
                outcome_layer,inputs,s)
            validation_losses.append(loss)
            if len(validation_losses) > 5:
                if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)])< validation_losses[len(validation_losses)-1]:
                    print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                    break
    plt.plot(range(len(validation_losses)),validation_losses)
    plt.show()
    #recover parameters and return them
    W_in_final = s.run(W_input)
    B_in_final = s.run(b_input)
    W_out_final = s.run(W_output)
    B_out_final = s.run(b_output)
    s.close()
    return [W_in_final, B_in_final, W_out_final,B_out_final]


def cv_second_stage_discrete(y,features_second,P,p_range,num_nodes,seed=None,folds=5,learning_rate=0.001):
    if seed!=None:
        np.random.seed(seed)  
    else:
        seed=9 #for the tf calls     
    #some test code below
    num_inputs = features_second.shape[1] #the number of input features
    num_output = 1 # output layer (currently just one since outcome is one variable)
    num_obs = y.shape[0]

    #create folds
    rng_orders = np.argsort(np.random.uniform(size=num_obs))
    foldgroups = np.zeros(num_obs)
    for k in range(1,folds+1):
        group_obs = (rng_orders >= (k-1)*num_obs/folds) & (rng_orders <(k)*num_obs/folds)
        foldgroups[group_obs]=k
    #train / test for each fold
    test_losses=[]
    for k in range(1,folds+1):
        #print 'fold=' +str(k)
        #split up train/test samples
        y_train = y[foldgroups!=k]
        features_train = features_second[foldgroups!=k,:]
        P_train = P[foldgroups!=k,:]

        y_test = y[foldgroups==k]
        features_test = features_second[foldgroups==k,:]
        P_test = P[foldgroups==k,:]
        num_obs_train = y_train.shape[0]


        #initialize weights and biases for input->hidden layer
        W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_in')
        b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_in')
        #initialize weights and biases for hidden->output layer
        W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_out')
        b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_out')
        #instantiate data vars
        inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
        outcome = tf.placeholder(dtype=tf.float32, shape=[None,1], name="outcome")
        #define the function for the hidden layer
        #use canonical tanh function for intermed, simple linear combo for final layer
        hidden_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
        outcome_layer = tf.matmul(hidden_layer,W_output) + b_output
        #the gradients of the output layer w.r.t. network parameters
        nn_gradients = tf.gradients(outcome_layer, [W_input, b_input,W_output,b_output]) #the gradients of the DNN w.r.t. parameters
        #placeholders for gradients I pass from numpy
        g_W_in = tf.placeholder(dtype=tf.float32, shape=W_input.get_shape(), name="g_W_in")
        g_b_in = tf.placeholder(dtype=tf.float32, shape=b_input.get_shape(), name="g_b_in")
        g_W_out = tf.placeholder(dtype=tf.float32, shape=W_output.get_shape(), name="g_W_out")
        g_b_out = tf.placeholder(dtype=tf.float32, shape=b_output.get_shape(), name="g_b_out")
        #the gradient-parameter pairs for gradient computation/application
        grad_var_pairs = zip([g_W_in,g_b_in,g_W_out,g_b_out],[W_input,b_input,W_output,b_output])

        #the optimizer
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #initialize tensorflow
        s = tf.InteractiveSession()
        s.run(tf.global_variables_initializer())

        #split training further into validation and training data
        validation_losses=[]
        validation_indices = np.random.choice(num_obs_train,num_obs_train/5)
        train_indices = np.ones(len(y_train), np.bool)
        train_indices[validation_indices]=0
        
        y_validation = y_train[validation_indices]
        features_validation = features_train[validation_indices,:]
        P_validation = P_train[validation_indices,:]
        
        y_train = y_train[train_indices]
        features_train  = features_train[train_indices,:]
        P_train = P_train[train_indices,:]

        num_train_obs = sum(train_indices)

        #print "training..."
        num_iters = 10000
        tol=1e-4
        for i in range(num_iters):
            #if i%100==0:
            # print "     iteration: " + str(i)
            #extract observation features for SGD
            g_ind=np.random.choice(num_train_obs,1)[0]
            obs_feat= features_train[g_ind,:]
            obs_y = y_train[g_ind]
            P_i = P_train[g_ind,:]
            #reshape everything so treated as 2d
            for v in [obs_y, obs_feat, P_i]:
                v.shape = [1,len(v)]
            stoch_grad = ind_secondstage_loss_gradient_discrete(obs_y,obs_feat,P_i,p_range,outcome_layer,inputs,nn_gradients,s)
            grad_dict={}
            grad_index=0
            for theta in [g_W_in,g_b_in,g_W_out,g_b_out]:
                grad_dict[theta]=stoch_grad[grad_index]
                grad_index+=1
            s.run(trainer.apply_gradients(grad_var_pairs),feed_dict=grad_dict)
            #the gradients of the output layer w.r.t. network parameters
            if i%10==0:
                loss=secondstage_loss_discrete(y[validation_indices], \
                features_second[validation_indices,:], \
                P[validation_indices,:],p_range,\
                outcome_layer,inputs,s)
                validation_losses.append(loss)
                if len(validation_losses) > 5:
                    if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)]) < validation_losses[len(validation_losses)-1]:
                        print "--------------------------"
                        print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                        break
        test_loss = secondstage_loss_discrete(y_test, \
                features_test, \
                P_test,p_range,\
                outcome_layer,inputs,s)
        test_losses.append(test_loss)
        s.close()
        print "completed fold " + str(k) + ' for n=' + str(num_nodes)
        print "--------------"

    return test_losses

def cv_mp_second_stage_discrete(y,features_second,P,p_range,num_nodes,seed,learning_rate,filename):
    test_losses = cv_second_stage_discrete(y,features_second,P,p_range,num_nodes,seed,folds,learning_rate)
    meanerr = np.mean(test_losses)
    sderr = np.std(test_losses)
    f = open(filename,'w')
    f.write('node,mean,se \n')
    f.write( str(num_nodes) + ',' +str(meanerr) + ',' + str(sderr) )
    f.close()
    print "**********************"
    print "node=" + str(num_nodes) + "; mean test err=" + str(meanerr) + "; sd test err=" + str(sderr)
    print "***********************"
    return [num_nodes,test_losses]