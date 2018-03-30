from SimpleTask import SimpleGridTask
import numpy as np, numpy.random as npr, random as r, SimpleTask
from TransportTask import TransportTask
from NavTask import NavigationTask


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
                batch_data.append( (stateActionIn,stateOut) )
            
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
