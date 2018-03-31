from SimpleTask import SimpleGridTask
import numpy as np, numpy.random as npr, random as r, SimpleTask
from TransportTask import TransportTask
from NavTask import NavigationTask
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import time
from SeqData import SeqData

class ForwardModel():
    def __init__(self, 
                obs_space, 
                input_space,
                n_hidden=100
                ):
        self.n_hidden=n_hidden
        self.act_space=input_space-obs_space
        self.obs_space=obs_space
        #Placeholders 
        self.input = tf.placeholder("float", [None, input_space])
        self.truevalue = tf.placeholder("float", [None, obs_space])
        self.pred=self.build_graph(self.input)
        self.saver = tf.train.Saver()
        
        
    def loss_function(self,batch_size,env):
        accTotal=0
        cost=0
        for i in range(0,batch_size):
            predVecs = env.deconcatenateOneHotStateVector(self.pred[i,:])
            labelVecs = env.deconcatenateOneHotStateVector(self.truevalue[i,:])
            for pv,lv in zip(predVecs,labelVecs):
                cost += tf.nn.softmax_cross_entropy_with_logits(logits=pv, labels=lv)
                accTotal += tf.cast(tf.equal(tf.argmax(pv,axis=0), tf.argmax(lv,axis=0)), tf.float32)
        return cost,accTotal
    
    def build_graph(self,inputVec, reuse=None):
        with tf.variable_scope("forward-model", reuse=reuse):
            hidden = slim.fully_connected(inputVec, self.n_hidden, biases_initializer=None, activation_fn=tf.nn.relu)
            hidden2 = slim.fully_connected(hidden, self.n_hidden, biases_initializer=None, activation_fn=tf.nn.relu)
            return slim.fully_connected(hidden2,self.obs_space, activation_fn=None, biases_initializer=None)
        
    def predict(self, x):
        sess= tf.get_default_session()
        #x.shape = (1,n_steps, n_input)
        return sess.run([self.pred], {self.input:x})

    def load_model(self,model_file_name):
        sess= tf.get_default_session()
        self.saver.restore(sess, model_file_name)

    def train(self,trainset,testset,training_steps,batch_size,env,learning_rate,display_step, model_file_name="FWR_model_"+time.strftime("%Y%m%d-%H%M%S")):
        sess= tf.get_default_session()
        print('Entering loss func')
        cost,accTotal = self.loss_function(batch_size,env)
        print('Defining optimizer')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        self.accuracy = accTotal / (batch_size * trainset.env.stateSubVectors) #tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Initialize the variables (i.e. assign their default value)
        print('Running TF initializer')
        init = tf.global_variables_initializer()
        sess.run(init)
        noise_sigma = 0.3
        print('Entering train loop')
        for step in range(1, training_steps + 1):
            batch_x, batch_y = trainset.next_batch_nonseq(batch_size)
            npbx = np.array( batch_x )
            npbxs = npbx.shape
            noise = noise_sigma * np.random.randn( npbxs[0], npbxs[1] )
            batch_x += noise
            sess.run(self.optimizer, feed_dict={self.input: batch_x, self.truevalue: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([self.accuracy, cost], feed_dict={self.input: batch_x, self.truevalue: batch_y})
                print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")
        # Calculate accuracy
        test_data, test_label = testset.next_batch_nonseq(5000) 
        acc=sess.run(self.accuracy, feed_dict={self.input: test_data, self.truevalue: test_label})
        print("Testing Accuracy:",acc)
        save_path= self.saver.save(sess, "./"+model_file_name+".ckpt")
        print("Model Saved")
        return acc


def main():
    print('Reading Data')
    s = 'navigation' #'navigation'
    trainf, validf = s+"-data-train-small.pickle", s+"-data-test-small.pickle"
    train, test   = SeqData(trainf), SeqData(validf)
    
    # classType = NavigationTask if s == 'navigation' else TransportTask
    print(train.env.stateSubVectors)
    print('Defining Model')
    # Parameters
    learning_rate = 0.0002
    training_steps = 15000 #2000 # 10000
    batch_size = 64 #256 #128
    display_step = 200
    # Network Parameters
    n_hidden = 200 #128 #5*train.lenOfInput # hidden layer num of features
    len_state = train.lenOfState # linear sequence or not
    len_input = train.lenOfInput


    fake_input= np.reshape(test.data[5],[1,10,-1])
    fake_state = fake_input[0][0][0:len_state]
    fake_action = fake_input[0][0][len_state:]
    print('Initializing FM')
    with tf.Graph().as_default(), tf.Session() as sess:
        fm=ForwardModel(len_state,len_input, n_hidden)
        print('FM initialized')
        fm.train(train,test,training_steps,batch_size,train.env,learning_rate,display_step,"abcd")

def inference():
    print('Reading Data')
    s = 'navigation' #'navigation'
    trainf, validf = s+"-data-train-small.pickle", s+"-data-test-small.pickle"
    train, test   = SeqData(trainf), SeqData(validf)
    # classType = NavigationTask if s == 'navigation' else TransportTask
    print(train.env.stateSubVectors)
    print('Defining Model')
    # Parameters
    learning_rate = 0.01
    training_steps = 5000 #2000 # 10000
    batch_size = 64 #256 #128
    display_step = 200
    # Network Parameters
    seq_max_len = 10 # Sequence max length
    n_hidden = 100 #128 #5*train.lenOfInput # hidden layer num of features
    len_state = train.lenOfState # linear sequence or not
    len_input = train.lenOfInput


    fake_input= np.reshape(test.data[5],[1,10,-1])
    fake_state = fake_input[0][0][0:len_state]
    fake_action = fake_input[0][0][len_state:]
    print(fake_action)
    
    print('action:',np.argmax(fake_action))
    print('state:',[np.argmax(k) for k in train.env.deconcatenateOneHotStateVector(fake_state)])
    print(fake_input)



    with tf.Graph().as_default(), tf.Session() as sess:

        fm=ForwardModel(len_state,len_input, n_hidden)

        fm.load_model('abcd.ckpt')
        fake_output, state_out=fm.predict(fake_input)
        fake_output = train.env.deconcatenateOneHotStateVector(fake_output[0])
        fake_output= [np.argmax(i) for i in fake_output]
        print(fake_output)

################################################################################################################
if __name__ == '__main__':
    main()
