import numpy as np
from SimpleTask import SimpleGridTask

class NavigationTask(SimpleGridTask):

    # Constructor
    def __init__(self,width=15,height=15,agent_start_pos=[np.array([0,0]),'N'],goal_pos=None,track_history=True):
        # Call superclass constructor
        super(NavigationTask,self).__init__(width,height,track_history)
        # Target position/state
        self.goal_pos = np.array([self.w-1,self.h-1]) if goal_pos is None else goal_pos
        # Set of possible actions
        self.numActions = 10
        self.actions = [
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
        self.actionDict = { self.actions[i] : i for i in range(0,self.numActions) }
        # Actor Orientations
        self.oriens = ['N', 'E', 'S', 'W'] # 0,1,2,3
        self.oriensDict = { self.oriens[i] : i for i in range(0,len(self.oriens)) }
        # Initial position and orientation of agent (i.e. the state of the system)
        self.agent_pos = np.array([0,0]) if agent_start_pos is None else agent_start_pos[0]
        self.agent_orientation = 0 if agent_start_pos is None else self.oriensDict[agent_start_pos[1]]
        # Add initial state to history if desired
        if self.track_history: self.history.append(self.getStateRep())

    # Action should be an int
    def performAction(self,action):
        # Do nothing
        if action == 0: return 
        # Change facing
        if action >= 1 and action <= 4:
            self.agent_orientation = action - 1
        # Move character
        else:
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
            self.history.append(action)
            self.history.append(self.getStateRep())

    def getReward(self):
        atReward = self.goal_pos[0]==self.agent_pos[0] and self.goal_pos[1]==self.agent_pos[1]
        return 1 if atReward else 0

    def getStateRep(self):
        p = np.zeros(6) # pos_x, pos_y, one_hot_orien
        p[0:2] = self.agent_pos # position
        p[2 + self.agent_orientation] = 1 # orientation
        return p # Env state

    def display(self):
        print('Environment ('+str(self.w)+','+str(self.h)+')')
        print('Current State\n','\tAgentPosition:',self.agent_pos,'\n\tAgentOrien:',self.oriens[self.agent_orientation])
        print('Goal Position:',self.goal_pos)

    def displayHistory(self):
        if not self.track_history: print("No history stored")
        else:
            i,k,n = 0,1,len(self.history)
            while i < n:
                si = self.history[i]
                print('State  '+str(k)+': P =',str(si[0:2])+", Orien = "+self.oriens[ self._oneHotToInt(si[2:]) ])
                if i < n-1: print('Action '+str(k)+':',self.actions[self.history[i+1]].replace("_"," "))
                k += 1; i += 2

######################################################################################################

def navmain():
    env = NavigationTask()
    for i in range(0,1000):
        j = np.random.randint( env.numActions )
        print('--')
        print(env.actions[j])
        env.performAction(j)
        env.display()
    env.displayHistory()
    thist = env.getHistoryAsTupleArray()
    # print(thist)

if __name__ == '__main__':
    navmain()