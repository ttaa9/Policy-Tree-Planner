import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar
    
from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData
from LSTMFM2 import LSTMForwardModel


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
    #ForwardModel.printState(state)
    vx = torch.sum((ix - gx)*(ix - gx))
    #print('vx',vx)
    vy = torch.sum((iy - gy)*(iy - gy))
    #print('vy',vy)
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
#     print(vx,vy)
    return - (vx + vy)
    
def generateTask(px,py,orien,gx,gy):
    direction = NavigationTask.oriens[orien]
    gs = np.array([gx, gy])
    env = NavigationTask(agent_start_pos=[np.array([px,py]), direction],goal_pos=gs)
    return env

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
    
    def trainSad(self, forwardModel):
        
        optimizer = optim.Adam(self.parameters(), lr = 0.0005 )

        maxDepth = 5
        treeBreadth = 2
        for p in forwardModel.parameters(): p.requires_grad = False
#         p = npr.randint(0,15,2)
#         orien = npr.randint(0,4,1)
#         g = npr.randint(0,15,2)
        cenv = generateTask(0,0,0,7,7)
#       cenv = generateTask(p[0],p[1],orien,g[0],g[1])
        s0 = avar(torch.FloatTensor([self.env.getStateRep()]), requires_grad=False)
        for i in range(0,3000):
            tree = Tree(s0,forwardModel,self,greedy_valueF, self.env,maxDepth,treeBreadth)
            loss = tree.getLossFromLeaves()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 200 == 0: print('Loss',i,":",loss.data[0])
        
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
#         print('Generating growth')
        # Generate Tree
        self.forwardModel.reInitialize(1)
        parent = Node(None,initialState,None, self.forwardModel.hidden)
        self.tree_head = self.grow(parent,0,self.branchFactor)
        #self.tAllStates = tf.stack(self.allStates)
        # Get leaves
#         print('Getting leaves')
        q, self.leaves = [ parent ], []
        while len(q) >= 1:
            currNode = q.pop()
            for child in currNode.children:
                if len( child.children ) == 0: self.leaves.append( child )
                else: q.append( child )
        #print(self.leaves)
    
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
        #print(d)
        for i in range(b):
            # Sample the current action
            hard_action, soft_a_s = self.simPolicy.sample(node.state)
            a_s =  [torch.squeeze(hard_action)]
            #print(a_s)
            #concat_vec = torch.cat([node.state, a_s], 1)
            #print("concat_vec",concat_vec.data.numpy(),d)
            inital_state =  torch.squeeze(node.state)
            #print(inital_state.shape)
            #print(a_s[0].shape)
            #self.forwardModel.reInitialize(1)
            self.forwardModel.setHiddenState(node.hidden)
            current_state, _, current_hidden = self.forwardModel.forward(inital_state,a_s, 1)
            # Build the next subtre
            current_state = current_state.unsqueeze(dim=0)
            
            self.allStates.append(current_state)
            self.allActions.append(a_s)
            if verbose: print("int_state at depth",d)
            if verbose: self.env.printState(node.state[0].data.numpy())
            if verbose: print("a_s at depth ",d," and breath",i)
            #if verbose: self.env.printAction(a_s[0])
            #self.env.printAction(a_s[0])
            if verbose: print("curr_state at depth",d)
            if verbose: self.env.printState(current_state[0].data.numpy())
            node.addChild( self.grow( Node(node, current_state, [soft_a_s], current_hidden), d+1, b) )
        return node
    
    def getBestPlan(self):
        bestInd, bestVal = 0, avar(torch.FloatTensor( [float('-inf')])) #float('-inf')
        for i, leaf in enumerate(self.leaves):
            currVal = self.valueF(leaf.state)
            #print('State')
            #self.forwardModel.printState(leaf.state[0])
            #print('Value',currVal)
            if currVal.data.numpy() > bestVal.data.numpy():
                bestInd = i
                bestVal = currVal
        #print(bestVal)
        return self.getPathFromLeaf( bestInd )
    
    def getLossFromLeaves(self, lambda_h=0.001):
        totalLosses = avar(torch.FloatTensor([0.0]))
        #totalLosses = avar(torch.FloatTensor(len(self.leaves)))
        for i, leaf in enumerate(self.leaves):
            #totalLosses[i] = -self.valueF( leaf.state )
            totalLosses += -self.valueF( leaf.state ) + lambda_h * torch.sum(leaf.action[0] * torch.log(leaf.action[0]))
            #print(leaf.action[0].data.numpy().argmax(),-self.valueF( leaf.state ).data[0])
        return  totalLosses/len(self.leaves) #torch.min(totalLosses)
        

def main():
    f_model_name = 'LSTM_FM_1_99' 
    s = 'navigation' # 'transport'
    trainf, validf = s + "-data-train-small.pickle", s + "-data-test-small.pickle"
    print('Reading Data')
    train, valid = SeqData(trainf), SeqData(validf)
    exampleEnv = generateTask(0,0,0,14,14)
    ForwardModel = LSTMForwardModel(train.lenOfInput,train.lenOfState)
    ForwardModel.load_state_dict( torch.load(f_model_name) )
    SimPolicy = SimulationPolicy(exampleEnv)
    SimPolicy.trainSad(ForwardModel)
    s_0 = torch.unsqueeze(avar(torch.FloatTensor(exampleEnv.getStateRep())), dim =0)
    tree = Tree(s_0,ForwardModel,SimPolicy,greedy_cont_valueF,exampleEnv,5,2)
    states, actions = tree.getBestPlan()
    for i in range(len(actions)):
        print(actions[i][0].data.numpy().argmax())