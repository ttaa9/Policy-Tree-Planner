import numpy as np, numpy.random as npr, random as r

class SimpleGridTask:

	# Helpers
    def _intToOneHot(self,i,size): a = np.zeros(size); a[i] = 1; return a
    def _oneHotToInt(self,h): return np.argmax(h)
    def _strToInt(self,s): return self.actionDict[s]

    # Action can be one-hot, int, or str; otherwise it returns None
    def _safePerformAction(self,action):
        if type(action) == int: return self.performAction(action)
        elif type(action) == str: return self.performAction( self._strToInt(action) )
        elif isinstance(action,(list,np.ndarray)): return self.performAction( self._oneHotToInt(action) )
        raise NotImplementedError( "Unknown action type" )

    # Interface
    def performAction( self, action ): raise NotImplementedError( "Required Method" )
    def getReward( self ): raise NotImplementedError( "Required Method" )
    def getStateRep( self ): raise NotImplementedError( "Required Method" )
    def display( self ): raise NotImplementedError( "Required Method" )

    # Assuming we are concatentating one-hot here, don't do anything by default
    def _convertHistoryStateToOneHot(self,history_state): return history_state

    # Constructor
    def __init__(self,track_history,hidden=[]):
        # If tracking history, save the states and actions as [s_0,a_0,s_1,a_1,s_2,...]
        self.track_history = track_history
        self.history = [ ]
        # Add initial state to history if desired
        if self.track_history:
            self.history.append(self.getStateRep())
            self.hiddenHistory = hidden

    # Returns the trajectory as a list of tuples (s_{i},a_{i},s_{i+1}) where a & s_j are one-hot
    def getHistoryAsTupleArray(self):
        if self.track_history:
            n,k,tupHist = (len(self.history) - 1) // 2, 0, []
            for i in range(0,n):
                a = self._intToOneHot(self.history[k+1],self.numActions)
                s1_1hot = self._convertHistoryStateToOneHot( self.history[k] ) # Ensure a state is one-hot format
                s2_1hot = self._convertHistoryStateToOneHot( self.history[k+2] )
                tupHist.append( (s1_1hot, a, s2_1hot) ) # S_i,a_i,S_{i+1}
                k += 2
            return tupHist
        else:
            print('No history present')
            return None

    def _stochasticAction(self,inAction):
        if self.isNoisy:
            action = npr.randint(0,self.numActions) if r.random() < self.stochasticity else inAction
        else:
            action = inAction
        return action


    # Static helper method to help construct training set
    # Returns (,labels,lengths)
    # Note: not seq2seq, only predicts final state (TODO in another function)
    def convertDataSetIntoSeqToLabelSet(dataset,maxSeqLen=None):
        # Each input is a sequence of concatenated state-actions pairs with variable length
        # e.g [ [(s_1,a_1),...,(s_k,a_k)], ..., [(s_1,a_1),...,(s_m,a_m)] ]
        # Note that each pair is several concatenated vectors in s_j, with an additional concatenated action vector a_j
        inputs = []
        # Each label is a state (at the end of the training)
        # e.g. [ s_k+1, ..., s_m+1 ]
        labels = []
        # Each number gives the length of the input sequence
        # e.g. [k, ..., m]
        lengths = []
        # Feature length (s_i + a_i)
        featlen = len( dataset[0][0][0] ) + len( dataset[0][0][1] )
        # Determine padding if not given
        if maxSeqLen is None:
            raise NotImplementedError( "Must give max seq len" ) # for now
        # For every generated trajectory, construct a training sequence
        for traj in dataset:
            currLen = len(traj)
            lengths.append(currLen)
            labels.append(traj[-1][-1]) # last state of last triplet
            currPairs = [ np.concatenate((t[0],t[1])) for t in traj ] # The input sequence
            if len(currPairs) < maxSeqLen:
                currPairs += ([ np.zeros(featlen) ]*( maxSeqLen - len(currPairs) ))
            inputs.append(currPairs)
        return (inputs,labels,lengths)

    def deconcatenateOneHotStateVector(self,vec):
        raise NotImplementedError( "Required Method" )
