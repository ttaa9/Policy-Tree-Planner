import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar
    
from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData
from LSTMFM2 import LSTMForwardModel
from GreedyValuePredictor import GreedyValuePredictor

import os, sys, pickle, numpy as np, numpy.random as npr, random as r

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape) #.cuda()
    return -avar(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return y #+ (y_hard - y).detach() 


def greedy_valueF(state):
    state = state.squeeze()
    #ForwardModel.printState(state)
    vx = torch.sum((state[0:15]-state[34:49]).pow(2))
    #print('vx',vx)
    vy = torch.sum((state[15:30]-state[49:64]).pow(2))
    #print('vy',vy)
    value = -( vx + vy ) 
    return value

def greedy_cont_valueF(state):
    state = state.squeeze()
    _,ix = state[0:15].max(0)
    _,gx = state[34:49].max(0)
    _,iy = state[15:30].max(0)
    _,gy = state[49:64].max(0)
    vx = torch.sum((ix - gx)*(ix - gx))
    vy = torch.sum((iy - gy)*(iy - gy))
    value = -( vx + vy ) 
    return value

def greedy_CE(state):
    state = state.squeeze()
    _, gx = state[34:49].max(0)
    _, gy = state[49:64].max(0)
    
    px = state[0:15]
    py = state[15:30]
    
    loss = torch.nn.CrossEntropyLoss()
    vx = loss(px.unsqueeze(dim=0), gx)
    vy  = loss(py.unsqueeze(dim=0), gy)
    return - (vx + vy)
    
def greedy_value_predictor(state):
    #state = state.squeeze()
    value = GreedyVP(state)
    return value.squeeze()

def generateTask(px,py,orien,gx,gy):
    direction = NavigationTask.oriens[orien]
    gs = np.array([gx, gy])
    env = NavigationTask(agent_start_pos=[np.array([px,py]), direction],goal_pos=gs)
    return env

class SimulationPolicy(nn.Module):
    def __init__(self,  env, layerSizes=[100,100]):
        super(SimulationPolicy, self).__init__()
        self.actionSize = len(env.actions)
        self.stateSize = len(env.getStateRep(oneHotOutput=True))
        self.env = env
        print("State Size: " , self.stateSize)
        print("Action Size: ", self.actionSize)
        
        # Input space: [Batch, observations], output:[Batch, action_space]
        self.layer1 = nn.Linear(self.stateSize, layerSizes[0])
        self.layer2 = nn.Linear(layerSizes[0], layerSizes[1])
        self.layer3 = nn.Linear(layerSizes[1], self.actionSize)
        
    def sample(self,state,temperature=2):
        output = F.relu( self.layer1(state) )
        output = F.relu( self.layer2(output) ) # F.sigmoid
        output = self.layer3(output)
        #print(output.shape)
        soft_output = F.softmax(output, dim=1)
        m = nn.LogSoftmax(dim=1)
        output = m(output)
        return gumbel_softmax(output, temperature), soft_output
    
    def forward(self, state):
        output = F.relu( self.layer1(state) )
        output = F.relu( self.layer2(output) ) # F.sigmoid
        output = self.layer3(output) 
        output = F.softmax(output,dim=1)
        return output
    
    def trainSad(self, forwardModel, GreedyVP, maxDepth = 7, treeBreadth = 2, niters=3000, lr = 0.0003):
        optimizer = optim.Adam(self.parameters(), lr = lr )
        for p in forwardModel.parameters(): p.requires_grad = False
        for p in GreedyVP.parameters(): p.requires_grad = False
    
        s0 = avar(torch.FloatTensor([self.env.getStateRep()]), requires_grad=False)
        for i in range(0,niters):
            tree = Tree(s0,forwardModel,self,greedy_valueF, self.env,maxDepth,treeBreadth)
            loss = tree.getLossFromLeaves()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 50 == 0: print('Loss',i,":",loss.data[0])
        
# POSSIBLE IDEA
# Dont just consider the leaves; consider all the nodes as possible leaves (consider all subpaths too)

class Node(object):
    
    def __init__(self, parent_node, state, action, hidden):
        self.parent = parent_node
        self.children = []
        self.state = state
        self.action = action
        self.hidden = hidden
        
    def addChild(self, child):
        self.children.append(child)
        
class Tree(object):
    
    def __init__(self, initialState, forwardModel, simPolicy, valueF, env,maxDepth=5, branchingFactor=3):
        self.simPolicy = simPolicy
        self.maxDepth, self.branchFactor = maxDepth, branchingFactor
        self.forwardModel = forwardModel
        self.valueF = valueF
        self.allStates = [initialState]
        self.allActions = []
        self.env = env
        self.forwardModel.reInitialize(1)
        parent = Node(None,initialState,None, self.forwardModel.hidden)
        self.tree_head = self.grow(parent,0,self.branchFactor)
        q, self.leaves = [ parent ], []
        while len(q) >= 1:
            currNode = q.pop()
            for child in currNode.children:
                if len( child.children ) == 0: self.leaves.append( child )
                else: q.append( child )
    
    def getPathFromLeaf(self,leafNumber):
        leaf = self.leaves[leafNumber]
        path = [leaf.state]
        actions = [leaf.action]
        currNode = leaf
        while not currNode.parent is None:
            #print(currNode.state)
            path.append(currNode.parent.state)
            if not currNode.parent.action is None:
                actions.append(currNode.parent.action)
            currNode = currNode.parent
        return (list(reversed(path)),list(reversed(actions)))
    
    def grow(self,node,d,b,verbose=False):
        if verbose: print('Grow depth: ',d)
        if verbose: self.env.printState(node.state[0].data.numpy())
        if d == self.maxDepth : return node
        for i in range(b):
            # Sample the current action
            hard_action, soft_a_s = self.simPolicy.sample(node.state)
            a_s =  [torch.squeeze(hard_action)]
            inital_state =  torch.squeeze(node.state)
            self.forwardModel.setHiddenState(node.hidden)
            current_state, _, current_hidden = self.forwardModel.forward(inital_state,a_s, 1)
            current_state = current_state.unsqueeze(dim=0)
            self.allStates.append(current_state)
            self.allActions.append(a_s)
            if verbose: print("int_state at depth",d)
            if verbose: self.env.printState(node.state[0].data.numpy())
            if verbose: print("a_s at depth ",d," and breath",i)
            if verbose: print("curr_state at depth",d)
            if verbose: self.env.printState(current_state[0].data.numpy())
            node.addChild( self.grow( Node(node, current_state, [soft_a_s], current_hidden), d+1, b) )
        return node
    
    def getBestPlan(self):
        bestInd, bestVal = 0, avar(torch.FloatTensor( [float('inf')])) #float('-inf')\n",
        for i, node in enumerate(self.leaves):
            currVal = self.valueF(node.state)
            if currVal.data.numpy() < bestVal.data.numpy():
                bestInd = i
                bestVal = currVal
        return self.getPathFromLeaf( bestInd )
    
    def getLossFromLeaves(self, lambda_h=-0.0):
        totalLosses = avar(torch.FloatTensor([0.0]))
        totalEntropy = avar(torch.FloatTensor([0.0]))
        for leaf in self.leaves:
            totalLosses += -self.valueF( leaf.state )
            totalEntropy += -torch.sum(leaf.action[0] * torch.log(leaf.action[0]))
        loss = totalLosses + lambda_h*totalEntropy
        return loss/len(self.leaves)
        
def main():
    f_model_name = 'LSTM_FM_1_99' 
    gvp_model_name = "greedy_value_predictor_3"

    numRepeats = 5
    tasks = [[5, generateTask(0,0,0,7,7)],
            [6, generateTask(0,0,0,10,12)],
            [7, generateTask(0,0,0,14,14)]]
    
    exampleEnv = NavigationTask()
    ForwardModel = LSTMForwardModel(74,64)
    ForwardModel.load_state_dict( torch.load(f_model_name) )
    GreedyVP = GreedyValuePredictor(exampleEnv)
    GreedyVP.load_state_dict( torch.load(gvp_model_name) )

    print("Running the tasks" )
    for i, task in enumerate(tasks):
        for j in range(numRepeats):
            task_state = task[1].getStateRep(oneHotOutput=False)
            px = int(task_state[0])
            py = int(task_state[1])
            orien = np.argmax(task_state[2:6])
            gx = int(task_state[-2])
            gy = int(task_state[-1])
            print("$$###############################")
            print("Repeat "+str(j)+" for "+ str(gx) + " , "+str(gy))
            #print('www',px,py,orien,gx,gy)
            cenv = generateTask(px,py,orien,gx,gy)
            SimPolicy = SimulationPolicy(cenv)
            SimPolicy.trainSad(ForwardModel, GreedyVP, maxDepth = task[0], niters=3000)

            s_0 = torch.unsqueeze(avar(torch.FloatTensor(cenv.getStateRep())), dim =0)
            tree = Tree(s_0,ForwardModel,SimPolicy,greedy_valueF,cenv,task[0],2)
            states, actions = tree.getBestPlan()

            for i in range(len(actions)):
                cenv.performAction( actions[i][0].data.numpy().argmax() )
            r = cenv.getReward()
            correct = (r==1)
        
            #print('Correct?',correct)
            if correct: 
                print('Correct final state',str(gx), str(gy))
                torch.save(SimPolicy.state_dict(), "SimPolicy_solve_"+ str(gx)  +"_"+ str(gy) + "_" + str(j))

############################
if __name__ == '__main__':
    main()
############################