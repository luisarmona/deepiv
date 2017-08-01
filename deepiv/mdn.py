#file that stores mdn functions
#heavily inspired by 
#http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt



#TRAIN a MDN
#values: p = the endogenous policy variable (currently only allowed to be 1D)
#covars: the inputs to the NN
#num_components: how many normals to use
#num_nodes: how many nodes in hidden layer
#learning_rate: how quickly to descend the gradient
def get_params(output,num_components):
    mixprobs = tf.placeholder(dtype=tf.float32, shape=[None,num_components], name="mixprobs")
    mixmeans = tf.placeholder(dtype=tf.float32, shape=[None,num_components], name="mixmeans")
    mixsds = tf.placeholder(dtype=tf.float32, shape=[None,num_components], name="mixsds")
    mixprobs,mixmeans,mixsds = tf.split(output,3,axis=1)
    mixprobs = tf.nn.softmax(mixprobs,dim=1) #normalized to be between 0 and 1 and sum to 1
    mixsds = tf.exp(mixsds) #so it's always positive
    return mixprobs,mixmeans,mixsds

#fcn for getting Lhood of a univariate normal 
#change it for now to return to log likelihood
def tf_normal(y,mean,sd):
    lhood = -tf.square(tf.subtract(y,mean)) #-(y-u)^2
    lhood = tf.divide(lhood,tf.multiply(2.,tf.square(sd)))
    #lhood = tf.multiply(tf.exp(lhood),tf.divide(1./math.sqrt(2.*math.pi),sd))
    llhood = tf.add(lhood,-tf.log(tf.square(sd)))
    return llhood


#now the actual mixture l-lhood
def tf_mixlhood(probs,means,sds,y):
    lhood = tf.exp(tf_normal(y,means,sds))
    lhood = tf.reduce_sum(tf.multiply(probs,lhood),axis=1,keep_dims=True)
    return tf.reduce_mean(-tf.log(lhood))



#the multinomial likelihood
#assumes P is a NxJ matrix, y is an N-by-J Matrix
#where J is the number of classes (i.e. number of points of support)
def tf_MNlhood(P,y):
    lhood = tf.multiply(y,tf.log(P))
    return tf.reduce_mean(-lhood)




#fit a MDN and return the NN parameters of interest (plus the estimated dist params)
def fit_MDN(p,covars,num_components=3,num_nodes=10,learning_rate=0.001,seed=None,num_batches=10,plot_loss=True):
    np.random.seed(seed)
    if seed==None:
        seed=9 #for TF calls
    num_inputs = covars.shape[1] #the number of input features
    num_output = num_components*3
    num_obs = p.shape[0]
    #initialize weights and biases for input->hidden layer
    W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))
    b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))

    #initialize weights and biases for hidden->output layer
    W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))
    b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))

    #instantiate data vars
    inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
    endog = tf.placeholder(dtype=tf.float32, shape=[None,1], name="endog")
    #define the function for the hidden layer
    #use canonical tanh function for intermed, simple linear combo for final layer
    #(note it will be further processed)
    intermed_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
    output_layer = tf.matmul(intermed_layer,W_output) + b_output
    #transform the final layer into probabilities, means, and variances
    mixprobs,mixmeans,mixsds=get_params(output_layer,num_components)
    #define the loss function- here the log likelihood of the mixture given parameters

    #now try to fit the NN against the Loss fcn
    loss = tf_mixlhood(mixprobs, mixmeans, mixsds, endog)
    #trainer = tf.train.AdamOptimizer().minimize(loss)
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())

    print "training..."
    num_iters = 10000 #the number of gradient descents
    validation_losses=[]
    losses=[]
    validation_indices = np.random.choice(num_obs,num_obs/5)
    train_indices = np.ones(len(p), np.bool)
    train_indices[validation_indices]=0
    num_train_obs=sum(train_indices)
    p_validation = p[validation_indices]
    covars_validation = covars[validation_indices,:]
    p_train = p[train_indices]
    covars_train = covars[train_indices,:]
    if num_batches=='all':
        num_batches=num_train_obs
    for i in range(num_iters):
        if i%100==0:
            print i
        if i%num_batches==0:
            #redo the batch order
            batchrank= np.argsort(np.random.uniform(size=num_train_obs))
        #assign the batch obs
        batch = (batchrank >= (i%num_batches)*num_train_obs/num_batches) & (batchrank <((i%num_batches)+1)*num_train_obs/num_batches)
        #SGD
        s.run(trainer,feed_dict={inputs: covars_train[batch,:].astype(np.float32), endog: p_train[batch,:].astype(np.float32)})
        #s.run(trainer,feed_dict={inputs: covars_train.astype(np.float32), endog: p_train.astype(np.float32)})
        #losses.append(s.run(loss,feed_dict={inputs: covars_train.astype(np.float32), endog: p_train.astype(np.float32)}))
        if i%10==0:
            validation_losses.append(s.run(loss,feed_dict={inputs: covars_validation.astype(np.float32), endog: p_validation.astype(np.float32)}))
            if len(validation_losses) > 5:
                #if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)])< validation_losses[len(validation_losses)-1]:
                if np.mean(validation_losses[len(validation_losses)-2])< validation_losses[len(validation_losses)-1]:
                    print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                    break
    if plot_loss:
        print validation_losses
        plt.plot(range(len(validation_losses)),validation_losses)
        plt.savefig('opt_loss.pdf')
        plt.show()
    #having trained the NN, recover parameters, and return them
    W_in_final = s.run(W_input)
    B_in_final = s.run(b_input)
    W_out_final = s.run(W_output)
    B_out_final = s.run(b_output)
    mixprobs,mixmeans,mixsds=s.run(get_params(output_layer,num_components),feed_dict={inputs:covars.astype(np.float32)})
    s.close()
    return [[W_in_final, B_in_final, W_out_final,B_out_final],[mixprobs,mixmeans,mixsds]]

##################################################################
#d k-fold CV to get an average LL on test data from the training
def cv_MDN(p,covars, \
    num_components=3,num_nodes=10,learning_rate=0.001,folds=5,seed=None,num_batches=10):
    if seed != None:
        np.random.seed(seed)
    else:
        seed=9 #for the tf calls
    #set high level primtiives
    num_obs = p.shape[0]
    num_inputs = covars.shape[1] #the number of input features
    num_output = num_components*3

    #assign to groups
    rng_orders = np.argsort(np.random.uniform(size=num_obs))
    foldgroups = np.zeros(num_obs)
    for k in range(1,folds+1):
        group_obs = (rng_orders >= (k-1)*num_obs/folds) & (rng_orders <(k)*num_obs/folds)
        foldgroups[group_obs]=k
    #train / test for each fold
    test_LL=[]
    for k in range(1,folds+1):
        print 'fold=' +str(k)
        #split up train/test samples
        p_train = p[foldgroups!=k]
        covars_train = covars[foldgroups!=k,:]
        p_test = p[foldgroups==k]
        covars_test = covars[foldgroups==k,:]
        num_train_obs = p_train.shape[0]
        #split train data further for validation set
        #to evaluate when to stop gradient descent
        validation_indices = np.random.choice(num_train_obs,num_train_obs/5)
        train_indices = np.ones(len(p_train), np.bool)
        train_indices[validation_indices]=0
        p_validation = p_train[validation_indices]
        covars_validation = covars_train[validation_indices,:]
        p_train = p_train[train_indices]
        covars_train = covars_train[train_indices,:]
        num_train_obs=sum(train_indices)
        #initialize weights and biases for input->hidden layer
        W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.01,maxval=.01,dtype=tf.float32,seed=seed))
        b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.01,maxval=.01,dtype=tf.float32,seed=seed))

        #initialize weights and biases for hidden->output layer
        W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.01,maxval=.01,dtype=tf.float32,seed=seed))
        b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.01,maxval=.01,dtype=tf.float32,seed=seed))

        #instantiate data vars
        inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
        endog = tf.placeholder(dtype=tf.float32, shape=[None,1], name="endog")
        #define the functions for the layers
        intermed_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
        output_layer = tf.matmul(intermed_layer,W_output) + b_output
        #transform the final layer into probabilities, means, and variances
        mixprobs,mixmeans,mixsds=get_params(output_layer,num_components)
        loss = tf_mixlhood(mixprobs, mixmeans, mixsds, endog)
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        s = tf.InteractiveSession()
        s.run(tf.global_variables_initializer())
        print "training..."
        num_iters = 10000 #the number of gradient descents
        validation_losses=[]
        if num_batches=='all':
            num_batches=num_train_obs
        for i in range(num_iters):
            if i%num_batches==0:
                #redo the batch order
                batchrank= np.argsort(np.random.uniform(size=num_train_obs))
            #assign the batch obs
            batch = (batchrank >= (i%num_batches)*num_train_obs/num_batches) & (batchrank <((i%num_batches)+1)*num_train_obs/num_batches)
            #SGD
            s.run(trainer,feed_dict={inputs: covars_train[batch,:].astype(np.float32), endog: p_train[batch,:].astype(np.float32)})
            if i%10==0:
                validation_losses.append(s.run(loss,feed_dict={inputs: covars_validation.astype(np.float32), endog: p_validation.astype(np.float32)}))
            if len(validation_losses) > 5:
                if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)])< validation_losses[len(validation_losses)-1]:
                    #if np.mean(validation_losses[len(validation_losses)-2])< validation_losses[len(validation_losses)-1]:
                    print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                    break
        fold_ll = s.run(loss,feed_dict={inputs: covars_test.astype(np.float32), endog: p_test.astype(np.float32)})
        print fold_ll
        test_LL.append(fold_ll)
        s.close()
        if np.isnan(fold_ll):
            print "exiting CV due to NaN in test data"
            return float('Inf')
    return test_LL




#fit a Multinomial model and return the NN parameters of interest (plus the estimated dist params)
#num_batches: either a number for #of batches or the keyword "all"  for one obs SGD
def fit_MN(p,covars,num_nodes=10,learning_rate=0.001,seed=None,plot_loss=True,num_batches=10):
    if seed != None:
        np.random.seed(seed)
    elif seed==None:
        seed=9 #for TF calls

    #convert p (factor to dummies)
    p_ind=pd.Series(p[:,0]).astype('category')
    p_ind = np.array(pd.get_dummies(p_ind))

    num_inputs = covars.shape[1] #the number of input features
    num_output = p_ind.shape[1]
    num_obs = p_ind.shape[0]
    #initialize weights and biases for input->hidden layer
    W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))
    b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))

    #initialize weights and biases for hidden->output layer
    W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))
    b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))

    #instantiate data vars
    inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
    endog = tf.placeholder(dtype=tf.float32, shape=[None,num_output], name="endog")
    #define the function for the hidden layer
    #use canonical tanh function for intermed, simple linear combo for final layer
    #(note it will be further processed)
    intermed_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
    output_layer = tf.matmul(intermed_layer,W_output) + b_output
    #define the loss function- here the log likelihood of the mixture given parameters

    #now try to fit the NN against the Loss fcn
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=endog))
    #trainer = tf.train.AdamOptimizer().minimize(loss)
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())

    print "training..."
    num_iters = 10000 #the number of gradient descents
    validation_losses=[]
    losses=[]
    validation_indices = np.random.choice(num_obs,num_obs/5)
    train_indices = np.ones(len(p), np.bool)
    train_indices[validation_indices]=0
    validation_indices= ~train_indices
    p_validation = p_ind[validation_indices,:]
    covars_validation = covars[validation_indices,:]
    p_train = p_ind[train_indices,:]
    covars_train = covars[train_indices,:]
    num_train_obs = p_train.shape[0]
    if num_batches=='all':
        num_batches=num_train_obs
    for i in range(num_iters):
        if i%num_batches==0:
            #redo the batch order
            print i
            batchrank= np.argsort(np.random.uniform(size=num_train_obs))
        #assign the batch obs
        batch = (batchrank >= (i%num_batches)*num_train_obs/num_batches) & (batchrank <((i%num_batches)+1)*num_train_obs/num_batches)
        #SGD
        s.run(trainer,feed_dict={inputs: covars_train[batch,:].astype(np.float32), endog: p_train[batch,:].astype(np.float32)})
        #losses.append(s.run(loss,feed_dict={inputs: covars_train.astype(np.float32), endog: p_train.astype(np.float32)}))
        if i%10==0:
            validation_losses.append(s.run(loss,feed_dict={inputs: covars_validation.astype(np.float32), endog: p_validation.astype(np.float32)}))
            if len(validation_losses) > 5:
                if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)])< validation_losses[len(validation_losses)-1]:
                #if np.mean(validation_losses[len(validation_losses)-2])< validation_losses[len(validation_losses)-1]:
                    print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                    break
    if plot_loss:
        print validation_losses
        plt.plot(range(len(validation_losses)),validation_losses)
        plt.savefig('opt_loss.pdf')
        plt.show()
    #having trained the NN, recover parameters, and return them
    W_in_final = s.run(W_input)
    B_in_final = s.run(b_input)
    W_out_final = s.run(W_output)
    B_out_final = s.run(b_output)
    P = s.run(tf.nn.softmax(output_layer),feed_dict={inputs:covars.astype(np.float32)})
    s.close()
    return [[W_in_final, B_in_final, W_out_final,B_out_final],P]


#d k-fold CV to get an average LL on test data from the training set (for the multinomial model)
def cv_MN(p,covars, num_nodes=10,learning_rate=0.001,folds=5,seed=None,num_batches=1):
    if seed != None:
        np.random.seed(seed)
    else:
        seed=9 #for the tf calls
    #set high level primtiives
    p_ind=pd.Series(p[:,0]).astype('category')
    p_ind = np.array(pd.get_dummies(p_ind))
    num_obs = p.shape[0]
    num_inputs = covars.shape[1] #the number of input features
    num_output = p_ind.shape[1]
    #assign to groups
    rng_orders = np.argsort(np.random.uniform(size=num_obs))
    foldgroups = np.zeros(num_obs)
    for k in range(1,folds+1):
        group_obs = (rng_orders >= (k-1)*num_obs/folds) & (rng_orders <(k)*num_obs/folds)
        foldgroups[group_obs]=k
    #train / test for each fold
    test_LL=[]
    for k in range(1,folds+1):
        print 'fold=' +str(k)
        #split up train/test samples
        p_train = p_ind[foldgroups!=k,:]
        covars_train = covars[foldgroups!=k,:]
        p_test = p_ind[foldgroups==k,:]
        covars_test = covars[foldgroups==k,:]
        num_obs_train = p_train.shape[0]
        #split train data further for validation set
        #to evaluate when to stop gradient descent
        validation_indices = np.random.choice(num_obs_train,num_obs_train/5)
        train_indices = np.ones(len(p_train), np.bool)
        train_indices[validation_indices]=0
        p_validation = p_train[validation_indices]
        covars_validation = covars_train[validation_indices,:]
        p_train = p_train[train_indices,:]
        covars_train = covars_train[train_indices,:]
        num_train_obs = p_train.shape[0]
        #initialize weights and biases for input->hidden layer
        W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))
        b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))

        #initialize weights and biases for hidden->output layer
        W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))
        b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed))

        #instantiate data vars
        inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
        endog = tf.placeholder(dtype=tf.float32, shape=[None,num_output], name="endog")
        #define the functions for the layers
        intermed_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
        output_layer = tf.matmul(intermed_layer,W_output) + b_output
        #transform the final layer into probabilities, means, and variances
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=endog))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        s = tf.InteractiveSession()
        s.run(tf.global_variables_initializer())
        print "training..."
        num_iters = 10000 #the number of gradient descents
        validation_losses=[]
        if num_batches=='all':
            num_batches=num_train_obs
        for i in range(num_iters):
            if i%num_batches==0:
                #redo the batch order
                print i
                batchrank= np.argsort(np.random.uniform(size=num_train_obs))
            #assign the batch obs
            batch = (batchrank >= (i%num_batches)*num_train_obs/num_batches) & (batchrank <((i%num_batches)+1)*num_train_obs/num_batches)
            #SGD
            s.run(trainer,feed_dict={inputs: covars_train[batch,:].astype(np.float32), endog: p_train[batch,:].astype(np.float32)})
            #s.run(trainer,feed_dict={inputs: covars_train.astype(np.float32), endog: p_train.astype(np.float32)})
            if i%100==0:
                validation_losses.append(s.run(loss,feed_dict={inputs: covars_validation.astype(np.float32), endog: p_validation.astype(np.float32)}))
                if len(validation_losses) > 5:
                    #if np.mean(validation_losses[len(validation_losses)-2])< validation_losses[len(validation_losses)-1]:
                    if np.mean(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)])< validation_losses[len(validation_losses)-1]:
                        print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                        break
        fold_ll = s.run(loss,feed_dict={inputs: covars_test.astype(np.float32), endog: p_test.astype(np.float32)})
        print fold_ll
        test_LL.append(fold_ll)
        s.close()
        if np.isnan(fold_ll):
            print "exiting CV due to NaN in test data"
            return float('Inf')
    return test_LL

#simulate from a multinomial; assumes classes 
#return nxB matrix of samples , where n is num obs, and B is num sims
def sim_mn(cats,P,B=10, seed=None):
    if seed!=None:
        np.random.seed(seed)
    num_obs = P.shape[0]
    samples = np.zeros(shape=[num_obs,B])
    for j in range(B):
        catchoice =   (np.random.rand(num_obs,1)<=P.cumsum(axis=1)).argmax(axis=1)
        samples[:,j] = cats[catchoice]
    return samples
 
######
#sample from mixture density network given NN output parameters
#return nxB matrix of samples , where n is num obs, and B is num sims

def sim_mdn(mixprobs,mixmeans,mixsds,B=10, seed=None):
    if seed!=None:
        np.random.seed(seed)
    num_obs = mixprobs.shape[0]
    samples = np.zeros(shape=[num_obs,B])
    for j in range(B):
        distchoice =   (np.random.rand(num_obs,1)<=mixprobs.cumsum(axis=1)).argmax(axis=1)
        samples[:,j] = np.random.normal(loc=mixmeans[np.arange(num_obs),distchoice],scale=mixsds[np.arange(num_obs),distchoice])
    return samples
 
######
#look at how we did by sampling from the distro
#and plotting some data
def plot_mdn_sim(p,z,covars,mixprobs,mixmeans,mixsds,B=10,figdir='', seed=1992,bins=None):
    np.random.seed(seed)
    num_obs = p.shape[0]
    p_samples = np.zeros(shape=[num_obs,B])
    z_samples = np.zeros(shape=[num_obs,B])
    for j in range(B):
        distchoice =   (np.random.rand(num_obs,1)<=mixprobs.cumsum(axis=1)).argmax(axis=1)
        z_samples[:,j] = z
        p_samples[:,j] = np.random.normal(loc=mixmeans[np.arange(num_obs),distchoice],scale=mixsds[np.arange(num_obs),distchoice])

    #a simple histogram
    if bins==None:
        plt.hist(p,color='b',normed=True,alpha=.3,label='Actual Data')
        plt.hist(p_samples.flatten(),color='r',normed=True,alpha=.3,label='Simulated Data')
    else:
        plt.hist(p,color='b',normed=True,alpha=.3,bins=bins,label='Actual Data')
        plt.hist(p_samples.flatten(),color='r',normed=True,alpha=.3,bins=bins,label='Simulated Data')        
    plt.xlabel('Endogenous Variable')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(figdir + 'hist_endog.pdf')
    plt.show()
    #a scatterplot of the instrument vs the policy variable
    plt.scatter(z_samples.flatten(),p_samples.flatten(),color='r', alpha=.1, label='Simulated Data')
    plt.scatter(z,p,color='b',alpha=1,label='Actual Data')
    plt.legend()
    plt.xlabel('Instrument Variable')
    plt.ylabel('Endogenous Variable')
    plt.savefig(figdir + 'endogVinstrument.pdf')
    plt.show()