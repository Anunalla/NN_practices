# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:22:20 2017

@author: kanmani
"""

""" Some people tried to use TextLineReader for the assignment 1
but seem to have problems getting it work, so here is a short 
script demonstrating the use of CSV reader on the heart dataset.
Note that the heart dataset is originally in txt so I first
converted it to csv to take advantage of the already laid out columns.
You can download heart.csv in the data folder.
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import tensorflow as tf
import pandas as pd

DATA_PATH = 'I:\Documents\GitHub\mnist_train.csv'
trainimages=pd.read_csv(DATA_PATH,header=None,delimiter=',').values
train_num=len(trainimages)
n_epochs=400
BATCH_SIZE = 50
N_FEATURES = 785
n_inputs =28*28
n_hidden1=300
n_hidden2=100
n_outputs = 10
learning_rate=0.01
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
y=tf.placeholder(tf.int64,shape=(None),name='y')  

def batch_generator(filenames):
    """ filenames is the list of files you want to read from. 
    In this case, it contains only heart.csv
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=0) # skip the first line in the file
    _, value = reader.read(filename_queue)

    # record_defaults are the default values in case some of our columns are empty
    # This is also to tell tensorflow the format of our data (the type of the decode result)
    # for this dataset, out of 9 feature columns, 
    # 8 of them are floats (some are integers, but to make our features homogenous, 
    # we consider them floats), and 1 is string (at position 5)
    # the last column corresponds to the lable is an integer

    record_defaults = [[0] for _ in range(N_FEATURES)]

    # read in the 10 rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults) 

    # convert the 5th column (present/absent) to the binary value 0 and 1
    #content[4] = tf.cond(tf.equal(content[4], tf.constant('Present')), lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    # pack all 9 features into a tensor
    features = tf.stack(content[1:N_FEATURES])

    # assign the last column to label
    label = content[0]

    # minimum number elements in the queue after a dequeue, used to ensure 
    # that the samples are sufficiently mixed
    # I think 10 times the BATCH_SIZE is sufficient
    min_after_dequeue = 10 * BATCH_SIZE

    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE,capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch

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

def generate_batches(data_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        features, labels = sess.run([data_batch, label_batch])
        print(features)
        print(labels)
        coord.request_stop()
        coord.join(threads)

def main():
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        data_batch, label_batch = batch_generator([DATA_PATH])
        #sess.run(init)
        for epoch in range(n_epochs):
            for iteration in range(train_num//BATCH_SIZE):
                
                X_batch, y_batch = sess.run([data_batch, label_batch])
                print(X_batch)
                print(y_batch)
                if iteration%100==0: 
                    print(X_batch.shape,y_batch.shape)
                #sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
            print("iteration over")
            #acc_train=accuracy.eval(feed_dict={X:X_batch,y:y_batch})
            #acc_test=accuracy.eval(feed_dict={X:testimages[:,1:],y:testimages[:,0]})
        
            print(epoch, "Train accuracy:", acc_train)#,"Test accuracy:",acc_test)
        save_path=saver.save(sess,"./my_model_final.ckpt")
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()