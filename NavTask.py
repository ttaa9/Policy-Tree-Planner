import numpy as np, numpy.random as npr, random as r, copy
from SimpleTask import SimpleGridTask
import pickle
class NavigationTask(SimpleGridTask):
    '''
    Implementation of a simple navigation task on a 2D discrete grid.
    Agents have position p=(x,y) and orientation o \in (N,E,S,W)

    Note that position is used as a tuple of integers until the one-hot version is needed.
    '''

    ### Static class variables ###
     # Set of possible actions
    numActions = 10
    actions = [
        "Do_nothing" ,
        "Face_north" ,
        "Face_east"  ,
        "Face_south" ,
        "Face_west"  ,
        "Move_1"     ,
        "Move_2"     ,
        "Move_3"     ,
        "Move_4"     ,
        "Move_5"
    ]
    # Actor Orientations
    oriens = ['N', 'E', 'S', 'W'] # 0,1,2,3


    # Constructor
    def __init__(self,width=15,height=15,agent_start_pos=[np.array([0,0]),'N'],goal_pos=None,
            track_history=True,stochasticity=0.0,maxSteps=None):
        # Grid size
        self.w, self.h = width, height
        # Convenience Dictionaries
        self.actionDict = { NavigationTask.actions[i] : i for i in range(0,NavigationTask.numActions) }
        self.oriensDict = { NavigationTask.oriens[i] : i for i in range(0,len(NavigationTask.oriens)) }
        # Target position/state
        self.goal_pos = np.array([self.w-1,self.h-1]) if goal_pos is None else goal_pos
        # Initial position and orientation of agent (i.e. the state of the system)
        self.agent_pos = np.array([0,0]) if agent_start_pos is None else copy.copy(agent_start_pos[0])
        self.agent_orientation = 0 if agent_start_pos is None else self.oriensDict[agent_start_pos[1]]
        # Add initial state to history if desired
        #if self.track_history: self.history.append(self.getStateRep())
        # Add stochasticity to the environment transitions (stoch value is prob of uniformly random action instead)
        self.stochasticity, self.isNoisy = stochasticity, stochasticity > 0.0
        # Call superclass constructor
        super(NavigationTask,self).__init__(track_history)
        # Store the max number of steps allowed for a generated trajectory, if it is given
        self.max_num_steps = maxSteps
        # Number of one-hot subvectors of the concatenated state vector
        # Agent pos + orien & goal pos
        self.stateSubVectors = 2 + 1 + 2       

    # Action should be an int
    def performAction(self,actionIn):
        # Add noise in stochastic case
        action = self._stochasticAction(actionIn)
        # Note: Do nothing if action == 0
        # Change facing
        if action >= 1 and action <= 4:
            self.agent_orientation = action - 1
        # Move character
        elif action > 4:
            numSteps = action - 4
            # Increment position
            if self.agent_orientation == 0:   self.agent_pos[1] += numSteps
            elif self.agent_orientation == 1: self.agent_pos[0] += numSteps
            elif self.agent_orientation == 2: self.agent_pos[1] -= numSteps
            else:                             self.agent_pos[0] -= numSteps
            # Truncate for validity
            if self.agent_pos[0] < 0:       self.agent_pos[0] = 0
            if self.agent_pos[1] < 0:       self.agent_pos[1] = 0
            if self.agent_pos[0] >= self.w: self.agent_pos[0] = self.w - 1
            if self.agent_pos[1] >= self.h: self.agent_pos[1] = self.h - 1
        # Track history if needed
        if self.track_history:
            self.history.append(actionIn) # Record input action, not actual one that occurred
            self.history.append(self.getStateRep(oneHotOutput=False))

    def getReward(self,distance_based=False):
        if distance_based:
            state = self.getStateRep(oneHotOutput=False)
            d = abs(state[0] - state[-2]) + abs(state[1] - state[-1])
            r = 1.0 /(1.0 + d) 
            return r
        atReward = self.goal_pos[0]==self.agent_pos[0] and self.goal_pos[1]==self.agent_pos[1]
        return 1 if atReward else 0

    # Note: this is not one-hot, so for fair comparison, we not want to use this for training directly
    def getStateRep(self,oneHotOutput=True):
        if oneHotOutput:
           noriens = len(self.oriens)
           inds = np.cumsum([0,self.w,self.h,noriens,self.w,self.h])
           p = np.zeros(2*self.w + 2*self.h + noriens)
           p[inds[0]:inds[1]] = self._intToOneHot(self.agent_pos[0],self.w)
           p[inds[1]:inds[2]] = self._intToOneHot(self.agent_pos[1],self.h)
           p[inds[2] + self.agent_orientation] = 1
           p[inds[3]:inds[4]] = self._intToOneHot(self.goal_pos[0],self.w)
           p[inds[4]:inds[5]] = self._intToOneHot(self.goal_pos[1],self.h)
        else:
           p = np.zeros(8) # pos_x, pos_y, one_hot_orien, goal_x, goal_y
           p[0:2] = copy.copy(self.agent_pos) # position as two integers in [0,w-1],[0,h-1] resp
           p[2 + self.agent_orientation] = 1 # orientation as one-hot
           p[-2:] = copy.copy(self.goal_pos) # goal pos as two integers 
        return p # Env state

    def display(self):
        print('Environment ('+str(self.w)+','+str(self.h)+')')
        print('Current State\n\tAgentPosition:',self.agent_pos,'\n\tAgentOrien:',self.oriens[self.agent_orientation])
        print('Goal Position:',self.goal_pos)

    def displayHistory(self):
        if not self.track_history: print("No history stored"); return
        i,k,n = 0,1,len(self.history)
        while i < n:
            si = self.history[i]
            print('State  '+str(k)+': P =',str(si[0:2])+", Orien = "+self.oriens[ self._oneHotToInt(si[2:]) ])
            if i < n-1: print('Action '+str(k)+':',self.actions[self.history[i+1]].replace("_"," "))
            k += 1; i += 2

    # Note: returns concatenated onehot vectors
    # There are 2 + 1 + 2 = 5 one-hot vecs of respective sizes (self.w +self.h) + numOriens + (self.w +self.h)
    def _convertHistoryStateToOneHot(self,h):
        px = self._intToOneHot(int(h[0]),self.w)
        py = self._intToOneHot(int(h[1]),self.h)
        gx = self._intToOneHot(int(h[-2]),self.w)
        gy = self._intToOneHot(int(h[-1]),self.h)
        orien = h[2:-2]
        return np.concatenate((px,py,orien,gx,gy))
        # i, j = int(history_state[0]), int(history_state[1])
        # k = self._oneHotToInt( history_state[2] )
        # a = np.zeros((self.w, self.h, self.numActions))
        # a[i,j,k] = 1
        # return a.flatten()

    def deconcatenateOneHotStateVector(self,vec):
        w,h,no = self.w,self.h,len(self.oriens)
        starts, incs = [0, w, w+h, w+h+no, 2*w+h+no], [w, h, no, w, h]
        return [ vec[s:s+inc] for s,inc in zip(starts,incs) ]

    #
    def _convertOneHotToHistoryState(self,one_hot_state_array):
        a = np.zeros(8) # p_x,p_y,orien,g_x,g_y
        incs = np.cumsum([0, self.w, self.h, self.numOriens, self.w, self.h])
        a[0] = self._oneHotToInt(one_hot_state_array[incs[0]:incs[1]])
        a[1] = self._oneHotToInt(one_hot_state_array[incs[1]:incs[2]])
        a[2] = one_hot_state_array[incs[2]:incs[3]]
        a[3] = self._oneHotToInt(one_hot_state_array[incs[3]:incs[4]])
        a[4] = self._oneHotToInt(one_hot_state_array[incs[4]:incs[5]])
        return a
        # i,j,k = list(zip(*np.where(one_hot_state_array == 1)))[0]
        # orien = self._intToOneHot(k,len(self.oriens)) # one-hot orientation
        # a = np.zeros(6)
        # a[0],a[1],a[2:] = i,j,orien
        # return a

    # Static method: generates random data for forward model training
    def generateRandomTrajectories(num_trajectories,max_num_steps,width=15,height=15,
            verbose=False,noise_level=0,print_every=1):
        trajs = []
        for traj in range(0,num_trajectories):
            if verbose and traj % print_every == 0: print("Starting traj",traj)
            # Generate env with random placement
            p_0 = np.array([npr.randint(0,width),npr.randint(0,height)])
            start_pos = [p_0, r.choice(NavigationTask.oriens)]
            cenv = NavigationTask(width=width,height=height,agent_start_pos=start_pos,goal_pos=None,
                track_history=True,stochasticity=noise_level,maxSteps=max_num_steps)
            # Choose random number of actions to run in [1,max_steps]
            num_acs_to_run = npr.randint(1,max_num_steps)
            if verbose and traj % print_every == 0: print("\tStart:",str(start_pos)+", N_a:",num_acs_to_run)
            for ac in range(0,num_acs_to_run):
                # Change direction vs move
                changeDir = r.random() >= 0.5
                if changeDir: # Change direction
                    newDir_action = npr.randint(0,len(NavigationTask.oriens)) + 1 # Add 1 to skip "do nothing"
                    cenv.performAction(newDir_action)
                else: # Walk
                    newMove_action = npr.randint(0,5) + 5 # 1 to 5 steps, + 5 to skip "change direction" & "do nothing"
                    cenv.performAction(newMove_action)
            # Save the history as a trajectory
            trajs.append( cenv.getHistoryAsTupleArray() )
        return trajs

######################################################################################################

### Testing ###

def navmain():
    env = NavigationTask() #(stochasticity=0.2)
    if False:
        for i in range(0,1000):
            j = np.random.randint( env.numActions )
            print('--')
            print(env.actions[j])
            env.performAction(j)
            env.display()
        env.displayHistory()
        thist = env.getHistoryAsTupleArray()
        print(thist[0:5])
        print('--')
        data = NavigationTask.generateRandomTrajectories(50,10,verbose=True)
    if True:
        data = NavigationTask.generateRandomTrajectories(20000,10,verbose=True,print_every=1000)
        toSave = [env,data]
        import pickle, sys

        with open("navigation-data-train-small.pickle",'wb') as outFile:
            print('Saving')
            pickle.dump(toSave,outFile)



if __name__ == '__main__':
    navmain()
