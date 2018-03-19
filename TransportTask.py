import numpy as np, numpy.random as npr, random as r, copy
from SimpleTask import SimpleGridTask

class TransportTask(SimpleGridTask):
    '''
    States are the locations of the agent and the objects.
    Actions are movements to each state + picking up objects.

    Note: multiple objects can be in one location.
    '''

    def __init__(self,numObjects=4,numLocations=8,objLocs=None,agentStartLocation=None,
            goalState=None,track_history=True,stochasticity=0.0,maxSteps=None):
        ## Checks
        if not objLocs is None and any(sobj >= numLocations for sobj in objLocs):
            raise ValueError('Untenable object locations',objLocs)
        elif not agentStartLocation is None and ((agentStartLocation < 0) or (agentStartLocation >= numLocations)):
            raise ValueError('Untenable agent location',agentStartLocation)
        ## Environment settings
        self.numLocations, self.numObjects = numLocations, numObjects
        ## Define the actions
        # There's one action per state (to move there) + 1 for picking up
        # No "Do_nothing" action because there already is one (move to the current location)
        self.actions = ['Move_to_'+str(i) for i in range(numLocations)] + ['Grab']
        self.numActions = len(self.actions)
        ## Define the current state
        # Tracks the current positions of the objects
        self.objectPositions = r.sample(range(0,numLocations),numObjects) if objLocs is None else objLocs
        # Which objects are held by the agent
        self.objHeldStatus = [False] * numObjects
        # Agent's current location (random if not specified)
        self.agentLocation = npr.randint(0,numLocations) if agentStartLocation is None else agentStartLocation
        ## Define the goal state
        # The goal state formatted as [agentLoc,[objectLocs]]
        if goalState is None:
            ## Generate random goal state. This is not trivial since the agent can only pick up; it cannot put them down.
            ## Thus, we assume some objects stay where they are, and the rest must all be in the same place (with the agent).
            # Sample a random number of the random objects
            whichObjsToMove = r.sample(range(0,numObjects), npr.randint(0,numObjects))
            fal = npr.randint(0,numLocations) # Final agent location
            finalObjLocs = [ ( fal if i in whichObjsToMove else loc ) for i,loc in enumerate(self.objectPositions) ]
            self.goalState = [ fal, list(finalObjLocs) ]
        else: self.goalState = goalState 
        # Call superclass constructor
        # Also track which objects are being held. Note that this is not an observed property to the agent. (?)
        super(TransportTask,self).__init__(track_history,hidden=[copy.copy(self.objHeldStatus)])
        # Noisy case
        self.isNoisy, self.stochasticity = stochasticity > 0.0, stochasticity
        # Helper function
        self._binarizeBools = lambda r: [ 1 if t else 0 for t in r ]
        # Store the max number of steps allowed for a generated trajectory, if it is given
        self.max_num_steps = maxSteps
        # 
        # TODO change this to include the goal state
        self.stateSubVectors = 2*self.numObjects + 1

    # Input: integer representing the action
    def performAction(self,inAction):
        # Handle stochasticity
        action = self._stochasticAction(inAction)
        # Grab action
        if action == self.numLocations:
            # For each object with same position as agent, make it held
            for i in range(0,self.numObjects):
                if self.objectPositions[i] == self.agentLocation:
                    self.objHeldStatus[i] = True
        # Movement action
        else:
            # Move agent
            self.agentLocation = action
            # For each held object, update its position when agent moves
            for i,held in enumerate(self.objHeldStatus):
                if held: self.objectPositions[i] = action
        # Record history
        if self.track_history:
            self.history.append(inAction) # Record input action, not actual one that occurred
            self.history.append(self.getStateRep())
            self.hiddenHistory.append(copy.copy(self.objHeldStatus))

    def display(self):
        print('Environment (NL = '+str(self.numLocations)+', NO = '+str(self.numObjects)+')')
        print('Current State\n\tAgent Location:',self.agentLocation,
            '\n\tObject Locations:',self.objectPositions,
            '\n\tObjects Held:',self._binarizeBools(self.objHeldStatus))
        print('Goal Position:',self.goalState)

    # One if all objects in right place; zero otherwise
    def getReward(self):
        return 1 if all( self.goalState[1][i] == self.objectPositions[i] for i in range(0,self.numObjects) ) else 0

    # State = {object positions & agent pos & goal positions}
    # Convert each position to one-hot and concatenate them
    # Note: goalState = [fal, finalObjLocs], where 
    #   fal -> integer for agent location
    #   finalObjLocs -> list of integers
    # Output format: array of numObjects+1 concatenated one-hot vectors, each of length numLocations
    def getStateRep(self):
        n,oh = self.numLocations, lambda i: self._intToOneHot(i,self.numLocations)
        g = [oh(loc) for loc in self.goalState[1]] + [oh(self.goalState[0])]
        a = [oh(loc) for loc in self.objectPositions] + [oh(self.agentLocation)]
        return np.array(a + g).flatten() # Appending goal state to current state = full state

    # Note: the one-hot form is actually multiple concatenated one-hot vectors
    def _convertOneHotToHistoryState(self,one_hot_state):
        q = np.reshape(one_hot_state,(2*self.numObjects + 1,self.numLocations))
        return [ self._oneHotToInt(u) for u in q ]

    # Static method: generates random data for forward model training
    def generateRandomTrajectories(num_trajectories,max_num_steps,numObjects=4,numLocations=8,
            verbose=False,noise_level=0,grab_obj_prob=0.5,grab_no_obj_prob=0.25,print_every=1):
        trajs = [] # Store output trajectories
        numObjsMoved = [] # For analysis purposes
        for traj in range(0,num_trajectories):
            if verbose and traj % print_every == 0: print("Starting traj",traj)
            # Generate env with random placement
            cenv = TransportTask(numObjects=numObjects,numLocations=numLocations,stochasticity=noise_level,maxSteps=max_num_steps)
            # Choose random number of actions to run in [1,max_steps]
            num_acs_to_run = npr.randint(1,max_num_steps)
            objPos_0 = copy.copy(cenv.objectPositions)
            if verbose and traj % print_every == 0: print("\tStart:",str([objPos_0,cenv.agentLocation])+", N_a:",num_acs_to_run)
            def randomGrabVsMove(prob):
                if r.random() <= prob: cenv.performAction(cenv.numActions-1) # Grab
                else:                  cenv.performAction( npr.randint(0,cenv.numActions-1) ) # Move
            for ac in range(0,num_acs_to_run):
                # Check whether there exists an object in the current location that is not held
                objHere = False
                for i,loc in enumerate(cenv.objectPositions):
                    if loc == cenv.agentLocation and not cenv.objHeldStatus[i]:
                        objHere = True
                        break
                # If so, pick it up with a certain probability
                if objHere: randomGrabVsMove(grab_obj_prob)    # Grab current object or move
                else:       randomGrabVsMove(grab_no_obj_prob) # Useless grab attempt or move
            numObjsMoved.append( sum( 0 if objPos_0[i]==cenv.objectPositions[i] else 1 for i in range(0,cenv.numObjects) ) )
            if verbose and traj % print_every == 0: print("\tEndObjPositions:",str(cenv.objectPositions)+", Num objects moved:",numObjsMoved[-1])
            # Save the history as a trajectory
            trajs.append( cenv.getHistoryAsTupleArray() )
        return trajs

    # Input: s_i as a concatenation of one-hot vectors
    # Returns: [v_1,...,v_n] where cat(v_1,...,v_n)
    # TODO change to allow for goal state in state
    def deconcatenateOneHotStateVector(self,vec):
        ntargs = 2*self.numObjects + 1 # + 1 for the agent
        return [ vec[c:c+self.numLocations] for c in range(0,ntargs,self.numLocations) ]

    # Note that which objects are held is unobserved
    def displayHistory(self):
        if not self.track_history: print("No history stored"); return
        i,k,n = 0,1,len(self.history)
        while i < n:
            si = self._convertOneHotToHistoryState( self.history[i] )
            hobs = str(self._binarizeBools( self.hiddenHistory[ k-1 ] ))
            print('State  '+str(k)+': ObjLocs =',str(si[0:self.numObjects])+", AgentLoc =",str(si[-1])+", ObjsHeld =",hobs)
            if i < n-1: print('Action '+str(k)+':',self.actions[self.history[i+1]].replace("_"," "))
            k += 1; i += 2

######################################################################################################

### Testing ###

def transport_main():
    print('In Main')
    env = TransportTask()

    if True:
        data = TransportTask.generateRandomTrajectories(20000,10,verbose=True,print_every=1000)
        toSave = [env,data]
        import dill, sys
        with open("transport-data-train-small.dill",'wb') as outFile:
            print('Saving')
            dill.dump(toSave,outFile)
        sys.exit(0)

    for i in range(0,100):
        j = np.random.randint( env.numActions )
        print('--')
        print(env.actions[j])
        env.performAction(j)
        env.display()
    env.displayHistory()
    thist = env.getHistoryAsTupleArray()
    print(thist[0])
    print('--')


if __name__ == '__main__':
    transport_main()
