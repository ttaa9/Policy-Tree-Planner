from SimpleTask import SimpleGridTask
import numpy as np, numpy.random as npr, random as r, SimpleTask
from TransportTask import TransportTask
from NavTask import NavigationTask
import tensorflow as tf
import os
import time

class SeqData():
    def __init__(self,dataFile):
        import pickle
        with open(dataFile,'rb') as inFile:
            print('Reading',dataFile)
            env,data = pickle.load(inFile)
        inputs,labels,lengths = SimpleGridTask.convertDataSetIntoSeqToLabelSet(data, maxSeqLen=10)
        self.lenOfAction = env.numActions
        self.lenOfInput = len(inputs[0][0]) # len of state-action concatenation
        self.lenOfState = self.lenOfInput - self.lenOfAction
        self.data,self.labels,self.seqlen = inputs,labels,lengths
        self.batch_id = 0
        self.env = env
        self.datalen = len(self.data)
        print("Statelen:", self.lenOfState)
        print("Actionlen:", self.lenOfAction)
        print("Inputlen:", self.lenOfInput)
        print('\tBuilt')

    def next(self, batch_size, random=True):
        if random:
            rints = r.sample( range(0,self.datalen), batch_size )
            batch_data = [ self.data[ri] for ri in rints ]
            batch_labels = [ self.labels[ri] for ri in rints ]
            batch_seqlen = [ self.seqlen[ri] for ri in rints ]
        else:
            """ Return a batch of data. When dataset end is reached, start over."""
            if self.batch_id == len(self.data):
                self.batch_id = 0
            endind = min(self.batch_id + batch_size, len(self.data))
            batch_data = (self.data[self.batch_id:endind])
            batch_labels = (self.labels[self.batch_id:endind])
            batch_seqlen = (self.seqlen[self.batch_id:endind])
            self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


class ForwardModel():
    def __init__(self, 
                obs_space, 
                input_space, 
                max_seq_len,
                n_hidden=100
                ):
        
        self.n_hidden=n_hidden
        
        #Placeholders 
        self.input = tf.placeholder("float", [None, max_seq_len, input_space])
        self.truevalue = tf.placeholder("float", [None, obs_space])
        self.seqlen = tf.placeholder(tf.int32, [None])

        self.max_seq_len= max_seq_len
        
        # Define weights
        self.weights = { 'out': tf.Variable(tf.random_normal([n_hidden, obs_space])) }
        self.biases = { 'out': tf.Variable(tf.random_normal([obs_space])) }

        x = self.input 
        
        # Define a lstm cell with tensorflow
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
        
        self.state_size_c = self.lstm_cell.state_size.c
        self.state_size_h = self.lstm_cell.state_size.h
        
        self.c_in = tf.placeholder(tf.float32, [None,self.lstm_cell.state_size.c], name = 'c_in')
        self.h_in = tf.placeholder(tf.float32, [None,self.lstm_cell.state_size.h], name = 'h_in')
        
        state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in,self.h_in)
        
        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        self.pred, _ = self.dynamic_cell(x, self.seqlen, state_in)
         
        self.saver = tf.train.Saver()

    def dynamic_cell(self, x, seqlen, state_in, reuse=None):
        with tf.variable_scope("dynamic_cell", reuse=reuse):
            outputs, states = tf.nn.dynamic_rnn(self.lstm_cell, x, dtype=tf.float32, sequence_length=seqlen, initial_state=state_in)
            lstm_c, lstm_h = states
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            
            # Hack to build the indexing and retrieve the right output.
            batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            index = tf.range(0, batch_size) * self.max_seq_len + (seqlen - 1)
            # Indexing
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)
            # Linear activation, using outputs computed above
        
            return tf.matmul(outputs, self.weights['out']) + self.biases['out'], self.state_out
    
    def get_initial_features(self, batch_size):
        # Call this function to get reseted lstm memory cells
        c_init = np.zeros([batch_size,self.state_size_c], np.float32) 
        h_init = np.zeros([batch_size,self.state_size_h], np.float32)
        return tf.nn.rnn_cell.LSTMStateTuple(c_init, h_init)
    
    def predict(self, x, c, h):
        sess= tf.get_default_session()
        #x.shape = (1,n_steps, n_input)
        return sess.run([self.pred, self.state_out], {self.input:x, self.seqlen:[1], self.c_in: c, self.h_in:h})


    def load_model(self,model_file_name):

        sess= tf.get_default_session()
        self.saver.restore(sess, model_file_name)

    def train(self,trainset,testset,training_steps,batch_size,learning_rate,display_step, c, h, model_file_name="FWR_model_"+time.strftime("%Y%m%d-%H%M%S")):
        
        sess= tf.get_default_session()
        cost, accTotal = 0, 0
        
        for i in range(0,batch_size):
            predVecs = trainset.env.deconcatenateOneHotStateVector(self.pred[i,:])
            labelVecs = trainset.env.deconcatenateOneHotStateVector(self.truevalue[i,:])
            for pv,lv in zip(predVecs,labelVecs):
                cost += tf.nn.softmax_cross_entropy_with_logits(logits=pv, labels=lv)
                accTotal += tf.cast(tf.equal(tf.argmax(pv,axis=0), tf.argmax(lv,axis=0)), tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


        self.accuracy = accTotal / (batch_size * trainset.env.stateSubVectors) #tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(1, training_steps + 1):
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            # Run optimization op (backprop)
            #print("self.c_in",self.c_in)
            #print("self.h_in",self.h_in)
            #print("self.input",self.input)
            #print("batch_x",batch_x)
            #print("self.trueval",self.truevalue)
            #print("batch_y",batch_y)
            #print("self.seqlen",self.seqlen)
            #print("Batch_seq_len",batch_seqlen)
            #print('-') 
            sess.run(self.optimizer, feed_dict={self.input: batch_x, self.truevalue: batch_y,
                                           self.seqlen: batch_seqlen,  self.c_in: c, self.h_in:h})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([self.accuracy, cost], feed_dict={self.input: batch_x, self.truevalue: batch_y,
                                                    self.seqlen: batch_seqlen,  self.c_in: c, self.h_in:h})
                print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        
        print("Optimization Finished!")
        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        c,h = self.get_initial_features(len(test_data))
        acc=sess.run(self.accuracy, feed_dict={self.input: test_data, self.truevalue: test_label,
                                          self.seqlen: test_seqlen,  self.c_in: c, self.h_in:h})

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
    learning_rate = 0.01
    training_steps = 1000 #2000 # 10000
    batch_size = 128 #256 #128
    display_step = 200
    # Network Parameters
    seq_max_len = 10 # Sequence max length
    n_hidden = 100 #128 #5*train.lenOfInput # hidden layer num of features
    len_state = train.lenOfState # linear sequence or not
    len_input = train.lenOfInput


    fake_input= np.reshape(test.data[5],[1,10,-1])
    fake_state = fake_input[0][0][0:len_state]
    fake_action = fake_input[0][0][len_state:]

    with tf.Graph().as_default(), tf.Session() as sess:
        fm=ForwardModel(len_state,len_input, seq_max_len, n_hidden)
        c, h = fm.get_initial_features(batch_size)
        print(c, h)
        fm.train(train,test,training_steps,batch_size,learning_rate,display_step, c, h, "abcd")

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
    training_steps = 1000 #2000 # 10000
    batch_size = 128 #256 #128
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

        fm=ForwardModel(len_state,len_input, seq_max_len, n_hidden)
        c, h = fm.get_initial_features(1)

        fm.load_model('abcd.ckpt')
        fake_output, state_out=fm.predict(fake_input,c, h)
        c, h= state_out
        print(c.shape)
        fake_output = train.env.deconcatenateOneHotStateVector(fake_output[0])
        fake_output= [np.argmax(i) for i in fake_output]
        print(fake_output)

################################################################################################################
if __name__ == '__main__':
    main()
