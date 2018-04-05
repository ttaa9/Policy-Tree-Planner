from SimpleTask import SimpleGridTask
import numpy as np, numpy.random as npr, random as r, SimpleTask
from TransportTask import TransportTask
from NavTask import NavigationTask


class SingDiscSeqData():
    def __init__(self,dataFile,env):
        import pickle
        with open(dataFile,'rb') as inFile:
            print('Reading',dataFile)
            data = pickle.load(inFile)
        # x_data = [ [ (s1a1), ..., ], ..., [...] ]
        # y_data = [ [ s1',...], ..., [...] ]
        # States are integers, need to be one-hot encoded
        self.x_data, self.y_data = data
        self.env = env
        self.stateSize = len(self.env.getSingularDiscreteState())
        self.actionLength = len(env.actions)
        self.inputSize = self.stateSize + self.actionLength
        self.lengths = [ len(x) for x in self.x_data ]
        self.dataLength = len(self.x_data)

    def _process_x(self,x_seq):
        npSeq = np.zeros( (len(x_seq),self.inputSize) )
        for i,s in enumerate(x_seq):
            indc = int(x_seq[i][0])
            #print(indc)
            npSeq[i,indc] = 1.0
            npSeq[i,-self.actionLength:] = x_seq[i][-self.actionLength:]
            #print('s',np.sum(npSeq[i,:]))
        return npSeq

    def _process_y(self,y_seq):
        npSeq = np.zeros( (len(y_seq),self.stateSize) )
        for i,s in enumerate(y_seq):
            npSeq[i,int(y_seq[i])] = 1.0
        return npSeq        

    # Returns a minibatch of sequences
    # i.e. [x,y] where m = batch_size
    #   x = [ s_1, ..., s_m ], s_i is seqlen_i x input_size, 
    #   y = [ ys_1, ... ys_m], where ys_i is seqlen_i x state_size
    def getRandomMinibatch(self, batch_size, oneHotLabels=False):
        rints = r.sample( range(0,self.dataLength), batch_size )
        batch_data = [ self._process_x( self.x_data[ri] ) for ri in rints ]
        if oneHotLabels:
            batch_labels = [ self._process_y( self.y_data[ri] ) for ri in rints ]
        else:
            batch_labels = [ [ int(k) for k in self.y_data[ri]] for ri in rints ]

        return batch_data, batch_labels


# Derived from: https://github.com/aymericdamien/TensorFlow-Examples/
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
        self.data, self.labels, self.seqlen = inputs,labels,lengths
        self.batch_id = 0
        self.env = env
        self.datalen = len(self.data)
        print('\tBuilt')

    def next(self, batch_size, random=True, nopad=False):
        if random:
            rints = r.sample( range(0,self.datalen), batch_size )
            batch_data = [ self.data[ri] for ri in rints ]
            batch_labels = [ self.labels[ri] for ri in rints ]
            batch_seqlen = [ self.seqlen[ri] for ri in rints ]
            if nopad:
                batch_data = [ bd[0:bs] for bd,bs in zip(batch_data,batch_seqlen) ]
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

    def next_batch_nonseq(self,batch_size):
        batch_data = []
        while len(batch_data) < batch_size:
            rint = r.sample( range(0,self.datalen), 1 )[0]
            slen = self.seqlen[rint]
            if slen >= 2:
                currData = self.data[rint]
                batch_data_index = r.sample(range(slen-1),1)[0]
                stateActionIn = currData[ batch_data_index ]
                stateOut = currData[ batch_data_index + 1 ][0:self.lenOfState]
                batch_data.append( [stateActionIn,stateOut] )
        return zip(*batch_data)

    def randomTrainingPair(self, padding=False):
        rint = r.sample(range(0,self.datalen), 1)[0]
        seq = self.data[rint]
        label = self.labels[rint]
        if not padding:
            slen = self.seqlen[rint]
            seq = seq[0:slen]
        return (seq,label)

    def unpaddedData(self):
        unpaddedData = [ a[0:s] for a,s in zip(self.data,self.seqlen) ]
        return unpaddedData






























