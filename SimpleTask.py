import numpy as np

class SimpleGridTask:

	# Helpers
    def _intToOneHot(self,i): a = np.zeros(self.numActions); a[i] = 1; return a
    def _oneHotToInt(self,h): return np.argmax(h)
    def _strToInt(self,s): return self.actionDict[s]

    # Action can be one-hot, int, or str; otherwise it returns None
    def _safePerformAction(self,action):
        if type(action) == int: return self.performAction(action)
        elif type(action) == str: return self.performAction( self._strToInt(action) )
        elif isinstance(action,(list,np.ndarray)): return self.performAction( self._oneHotToInt(action) )
        raise NotImplementedError( "Unknown action type" )

    # Interface
    def performAction( self ): raise NotImplementedError( "Required Method" )
    def getReward( self ): raise NotImplementedError( "Required Method" )
    def getStateRep( self ): raise NotImplementedError( "Required Method" )
    def display( self ): raise NotImplementedError( "Required Method" )

    # Constructor
    def __init__(self,width,height,track_history):
    	# Grid size
        self.w, self.h = width, height
        # If tracking history, save the states and actions as [s_0,a_0,s_1,a_1,s_2,...]
        self.track_history = track_history
        self.history = [ ]

    # Returns the trajectory as a list of tuples (s_{i},a_{i},s_{i+1}) where a is one-hot
    def getHistoryAsTupleArray(self):
        if self.track_history:
            n,k,tupHist = (len(self.history) - 1) // 2, 0, []
            for i in range(0,n):
                a = self._intToOneHot(self.history[k+1])
                tupHist.append( (self.history[k],a,self.history[k+2]) ) # S_i,a_i,S_{i+1}
                k += 2
            return tupHist
        else:
            return None

