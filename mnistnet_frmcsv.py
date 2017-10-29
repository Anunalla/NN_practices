# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:26:21 2017

@author: kanmani
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import random


trainimages=pd.read_csv("mnist_train.csv",header=None,delimiter=',').values
testimages=pd.read_csv("mnist_test.csv",header=None,delimiter=',').values

#%%
tf.reset_default_graph()
print("I reset graph")
n_inputs =28*28
n_hidden1=300
n_hidden2=100
n_outputs = 10
learning_rate=0.01
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
y=tf.placeholder(tf.int64,shape=(None),name='y')                     
trainy=trainimages[:,0]
testy=testimages[:,0]

trainimages2=np.copy(trainimages)

#==============================================================================
# trainimages=trainimages[:,1:]
# testimages=testimages[:,1:]
#==============================================================================

train_num=len(trainimages)
def next_batch_shuffle():
    index=random.randint(0,len(trainimages2)-batch_size)
    random.shuffle(trainimages2)
    Xbatch=trainimages2[index:index+batch_size,1:]
    ybatch=trainimages2[index:index+batch_size,0]    
    return Xbatch,ybatch

with tf.name_scope("dnn"):
    hidden1=tf.contrib.layers.fully_connected(X,n_hidden1,scope="hidden1")
    hidden2=tf.contrib.layers.fully_connected(hidden1,n_hidden2,scope="hidden2")
    logits=tf.contrib.layers.fully_connected(hidden2,n_outputs,scope="outputs",activation_fn=None)
with tf.name_scope("loss"):
    xentropy =tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss =tf.reduce_mean(xentropy,name="loss")
    loss_summary=tf.summary.scalar("log_loss",loss)
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    
init =tf.global_variables_initializer()
saver =tf.train.Saver()

#==============================================================================
# from tensorflow.examples.tutorials.mnist import input_data
# mnist=input_data.read_data_sets("/tmp/data/")
#==============================================================================

n_epochs=400
batch_size=50
print("graph construction over i start train and test")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        random.shuffle(trainimages2)
        for iteration in range(train_num//batch_size):
            X_batch = trainimages2[iteration:iteration+batch_size,1:]
            y_batch =trainimages2[iteration:iteration+batch_size,0]
            if iteration%100==0: 
                print(X_batch.shape,y_batch.shape)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        print("iteration over")
        acc_train=accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_test=accuracy.eval(feed_dict={X:testimages[:,1:],y:testimages[:,0]})
        
        print(epoch, "Train accuracy:", acc_train,"Test accuracy:",acc_test)
    save_path=saver.save(sess,"./my_model_final.ckpt")
    

    