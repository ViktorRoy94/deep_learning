
import mxnet as mx
import logging
import numpy as np
import time			  

class MultiIter(mx.io.DataIter):  
    def __init__(self, iter_list):  
        self.iters = iter_list   
        self.batch_size = 1  
    def next(self):  
        batches = [i.next() for i in self.iters]  
        return mx.io.DataBatch(data=[t.asnumpy().reshape(3*28*28) for t in batches[0].data], label= [t.asnumpy().reshape(3*28*28) for t in batches[1].label], pad=0)  
    def reset(self):  
        for i in self.iters:  
            i.reset()  
    @property  
    def provide_data(self):  
        return [t for t in self.iters[0].provide_data] 
    @property  
    def provide_label(self):  
        return [t for t in self.iters[1].provide_label] 


print("Load data from files ...")
Xtrain = np.load("X_train.npy")
Xtest = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
print("Done")

X_train = np.reshape(Xtrain, (Xtrain.shape[0], 3*28*28))
X_test = np.reshape(Xtest, (Xtest.shape[0], 3*28*28))

logging.getLogger().setLevel(logging.DEBUG) 
		
#X_train = np.random.sample((2,100))
#X_test = np.random.sample((2,100))

print X_train.shape
print X_test.shape

train_iter1 = mx.io.NDArrayIter(data = X_train,
                               label = X_train)	
train_iter2 = mx.io.NDArrayIter(data = X_train,
                               label = X_train)								   
val_iter1 = mx.io.NDArrayIter(X_test,
                             X_test)
val_iter2 = mx.io.NDArrayIter(X_test,
                             X_test)
							 
train = MultiIter([train_iter1, train_iter2])

test = MultiIter([val_iter1, val_iter2])

#for batch in train:
	#print batch.data[0]
	#print batch.label[0].shape
	#print batch.data[0].asnumpy().reshape(5)
	#print batch.label[0].asnumpy()
	#print batch.data[0].shape	
	
data = mx.sym.var('data')

fc1 = mx.symbol.FullyConnected(data=data, num_hidden=300)
encoder = mx.sym.Activation(data=fc1, act_type='tanh', name = 'tanh')

fc2 = mx.symbol.FullyConnected(data=encoder, num_hidden=3*28*28)
decoder = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

#Train
autoencoder_model = mx.mod.Module(symbol = decoder, context = mx.cpu())
autoencoder_model.fit(train,  # train data
                 eval_data = test,  # validation data
                 optimizer = 'sgd',  # use SGD to train
                 optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
                 eval_metric = 'ce',  # report cross-entropy during training
                 batch_end_callback = mx.callback.Speedometer(1000, 1000), # output progress
                 num_epoch = 20)  # train for at most 10 dataset passes

							  
ce = mx.metric.CrossEntropy()
autoencoder_model.score(test, ce)
print(ce)


