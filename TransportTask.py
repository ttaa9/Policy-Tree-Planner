import numpy as np, numpy.random as npr, random as r
from SimpleTask import SimpleGridTask

class TransportTask(SimpleGridTask):
    '''
    States are the locations of the agent and the objects.
    Actions are movements to each state + picking up objects.

    Note: multiple objects can be in one location.
    '''

    def __init__(self,numObjects=4,numLocations=8,objLocs=None,agentStartLocation=None,goalState=None,track_history=True):
        ## Checks
        if not objLocs is None and any(sobj >= numLocations for sobj in objLocs):
            raise ValueError('Untenable object locations',objLocs)
        elif not agentStartLocation is None and ((agentStartLocation < 0) or (agentStartLocation >= numLocations)):
            raise ValueError('Untenable agent location',agentStartLocation)
        ## Environment settings
        self.numLocations, self.numObjects = numLocations, numObjects
        ## Define the actions
        # There's one action per state (to move there) + 1 for picking up
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
        super(TransportTask,self).__init__(track_history)

    # Input: integer representing the action
    def performAction(self,action):
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
        
    def display(self):
        print('Environment (NL = '+str(self.numLocations)+', NO = '+str(self.numObjects)+')')
        print('Current State\n\tAgent Location:',self.agentLocation,'\n\tObject Locations:',self.objectPositions)
        print('Goal Position:',self.goalState)

    # One if all objects in right place; zero otherwise
    def getReward( self ):
        return 1 if all( self.goalState[1][i] == self.objectPositions[i] for i in range(0,self.numObjects) ) else 0

    # State = {object positions & agent pos}
    # Convert each position to one-hot and concatenate them
    # can use a loss of sum of cross-entropies then (?)
    # Output format: array of numObjects+1 concatenated one-hot vectors, each of length numLocations
    def getStateRep( self ):
        n,oh = self.numLocations, lambda i: self._intToOneHot(i,self.numLocations)
        a = [oh(loc) for loc in self.objectPositions] + [oh(self.agentLocation)]
        return np.array(a).flatten()

    # Assuming we are concatentating one-hot here, don't do anything
    def _convertHistoryStateToOneHot( self, history_state ): return history_state

    # Static method: generates random data for forward model training
    def generateRandomTrajectories(num_trajectories,max_num_steps,width=15,height=15,verbose=False,noise_level=0):
        pass

def transport_main():
    env = TransportTask()
    env.display()
    print(env.getStateRep())

if __name__ == '__main__':
    transport_main()




# add stochasticity
# add history


