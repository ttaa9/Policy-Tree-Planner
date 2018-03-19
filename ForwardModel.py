from SimpleTask import SimpleGridTask
import numpy as np, numpy.random as npr, random as r, SimpleTask
from TransportTask import TransportTask
from NavTask import NavigationTask
import tensorflow as tf

# Derived from: https://github.com/aymericdamien/TensorFlow-Examples/
class SeqData():
    def __init__(self,dataFile):
        import dill
        with open(dataFile,'rb') as inFile:
            print('Reading',dataFile)
            env,data = dill.load(inFile)
        inputs,labels,lengths = SimpleGridTask.convertDataSetIntoSeqToLabelSet(data, maxSeqLen=10)
        self.lenOfAction = env.numActions
        self.lenOfInput = len(inputs[0][0]) # len of state-action concatenation
        self.lenOfState = self.lenOfInput - self.lenOfAction
        self.data,self.labels,self.seqlen = inputs,labels,lengths
        self.batch_id = 0
        self.env = env
        self.datalen = len(self.data)
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


def main():
    print('Reading Data')
    s = 'navigation' #'transport'
    trainf, validf = s+"-data-train-small.dill", s+"-data-test-small.dill"
    train, valid   = SeqData(trainf), SeqData(validf)
    # classType = NavigationTask if s == 'navigation' else TransportTask
    #print(train.env.stateSubVectors)
    print('Defining Model')
    # Parameters
    learning_rate = 0.01
    training_steps = 1000 #2000 # 10000
    batch_size = 128 #256 #128
    display_step = 200
    # Network Parameters
    seq_max_len = 10 # Sequence max length
    n_hidden = 100 #128 #5*train.lenOfInput # hidden layer num of features
    n_classes = train.lenOfState # linear sequence or not
    print('\tN_Hidden =',n_hidden,'\n\tMax_Seq_Len =',seq_max_len,
        '\n\tInputLen =',train.lenOfInput)
    trainset = train #ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
    testset = valid #ToySequenceData(n_samples=500, max_seq_len=seq_max_len)
    # tf Graph input
    x = tf.placeholder("float", [None, seq_max_len, train.lenOfInput])
    y = tf.placeholder("float", [None, n_classes])
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])
    # Define weights
    weights = { 'out': tf.Variable(tf.random_normal([n_hidden, n_classes])) }
    biases = { 'out': tf.Variable(tf.random_normal([n_classes])) }
    # Definition for dynamic RNN
    def dynamicRNN(x, seqlen, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, seq_max_len, 1)
        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.
        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']
    # Creating dynamicRNN (manually) and running it on the sequence
    pred = dynamicRNN(x, seqlen, weights, biases)
    # Generate a saver
    saver = tf.train.Saver()
    # Define loss and optimizer
    cost, accTotal = 0, 0
    for i in range(0,batch_size):
        predVecs = train.env.deconcatenateOneHotStateVector(pred[i,:])
        labelVecs = train.env.deconcatenateOneHotStateVector(y[i,:])
        for pv,lv in zip(predVecs,labelVecs):
            cost += tf.nn.softmax_cross_entropy_with_logits(logits=pv, labels=lv)
            accTotal += tf.cast(tf.equal(tf.argmax(pv,axis=0), tf.argmax(lv,axis=0)), tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Evaluate model
    #correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = accTotal / (batch_size * train.env.stateSubVectors) #tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        # Training loop
        for step in range(1, training_steps + 1):
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                    seqlen: batch_seqlen})
                print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")
        # Save the variables to disk.
        save_path = saver.save(sess, "./ckpts/model.ckpt")
        print("Model saved in file: %s" % save_path)
        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                          seqlen: test_seqlen}))

################################################################################################################
if __name__ == '__main__':
    main()
