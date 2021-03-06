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

##############################################
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

def gumbel_softmax_hard(logits, temperature):
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
    return y + (y_hard - y).detach() 

##############################################


def greedy_valueF(state):
    state = state.squeeze()
    vx = torch.sum((state[0:15]-state[34:49]).pow(2))
    vy = torch.sum((state[15:30]-state[49:64]).pow(2))
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
    
def generateTask(px,py,orien,gx,gy):
    direction = NavigationTask.oriens[orien]
    gs = np.array([gx, gy])
    env = NavigationTask(agent_start_pos=[np.array([px,py]), direction],goal_pos=gs)
    return env

class SimulationPolicy(nn.Module):
    def __init__(self,  env, layerSizes=[100,100], maxBranchingFactor=3):
        super(SimulationPolicy, self).__init__()
        self.actionSize = len(env.actions)
        self.stateSize = len(env.getStateRep(oneHotOutput=True))
        self.env = env
        self.maxBranchingFactor = maxBranchingFactor
        self.intvec = avar( torch.LongTensor(list(range(maxBranchingFactor + 1))) ).unsqueeze(0)
        #print("State Size: " , self.stateSize, "\nAction Size: ", self.actionSize)
        # Input space: [Batch, observations], output:[Batch, action_space]
        self.layer1 = nn.Linear(self.stateSize, layerSizes[0])
        self.layer2 = nn.Linear(layerSizes[0], layerSizes[1])
        self.layer3 = nn.Linear(layerSizes[1], self.actionSize)
        # Layer to sample branching factor
        self.intSamplingLayer = nn.Linear(layerSizes[1], self.maxBranchingFactor + 1)
        
    def sample(self,state,temperature=2,branching_temperature=1,verbose=False):
        # Compute action output
        output1 = F.relu( self.layer1(state) )
        output2 = F.relu( self.layer2(output1) ) # F.sigmoid
        output = self.layer3(output2)
        # Process action
        soft_output = F.softmax(output, dim=1)
        m = nn.LogSoftmax(dim=1)
        output = m(output)
        # Use Gumbel-Softmax 
        gumbeled_action, soft_action = gumbel_softmax(output, temperature), soft_output
        # Sample the branching factor
        b_ann = self.intSamplingLayer(output2)
        b_ann_logsoft = m(b_ann)
        if verbose: print('BAL',b_ann_logsoft)
        b_gumbeled_gsh = gumbel_softmax_hard(b_ann_logsoft, branching_temperature)#.type(torch.LongTensor)
        b_gumbeled = b_gumbeled_gsh.type(torch.LongTensor)
        if verbose: print('BG',b_gumbeled)
        branchingRateSample = torch.dot(self.intvec, b_gumbeled)
        if verbose: print('BRS',branchingRateSample)
        return gumbeled_action, soft_action, branchingRateSample

    
    def forward(self, state):
        output = F.relu( self.layer1(state) )
        output = F.relu( self.layer2(output) ) # F.sigmoid
        output = self.layer3(output) 
        output = F.softmax(output,dim=1)
        return output
    
    def trainSad(self, 
        taskEnv, 
        forwardModel, 
        printActions=False, 
        maxDepth=5, 
        treeBreadth=2, 
        eta_lr=0.0005,
        trainIters=500,
        alpha=0.5,
        lambda_h=-0.025,
        useHolder=False,
        holderp=-6.0, 
        useOnlyLeaves=False, 
        gamma=0.01,
        temperature=2,
        branching_temperature=1
        ):
        optimizer = optim.Adam(self.parameters(), lr = eta_lr )
        for p in forwardModel.parameters(): p.requires_grad = False
        s0 = avar(torch.FloatTensor([self.env.getStateRep()]), requires_grad=False)
        for i in range(0,trainIters):
            tree = Tree(s0, forwardModel, self,greedy_valueF, self.env, maxDepth, treeBreadth, temperature=temperature,branching_temperature=branching_temperature)
            loss = tree.getLossFromAllNodes(
                alpha=alpha, lambda_h=lambda_h, useHolder=useHolder, 
                holderp=holderp, useOnlyLeaves=useOnlyLeaves, gamma=gamma
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 50 == 0: 
                # print('Loss',i,":",loss.data[0])
                # print('NumTreeNodes:', len(tree.allNodes))
                if printActions:
                    plan = tree.getBestPlan()
                    # print(plan)
                    print( "\n".join([ "A" + str(qi) + ": " + 
                        ",".join([
                            ",".join(["%.3f" % q for q in qq]) for qq in a[0].data.numpy()
                        ]) for qi,a in enumerate(plan[1])
                    ] ) )
        
# POSSIBLE IDEA
# Dont just consider the leaves; consider all the nodes as possible leaves (consider all subpaths too)

class Node(object):
    
    def __init__(self, parent_node, state, action, sampledAction, hidden):
        self.parent = parent_node
        self.children = []
        self.state = state
        self.action = action
        self.hidden = hidden
        self.sampledAction = sampledAction
        self.branchingBreadth = None
        
    def addChild(self, child):
        self.children.append(child)
        
class Tree(object):
    
    def __init__(self, initialState, forwardModel, simPolicy, valueF, env, maxDepth=5, branchingFactor=3, temperature=2,branching_temperature=1):
        self.simPolicy = simPolicy
        self.maxDepth, self.branchFactor = maxDepth, branchingFactor
        self.forwardModel = forwardModel
        self.valueF = valueF
        self.allStates = [initialState]
        self.allActions = []
        self.allNodes = []
        self.env = env
        self.temperature = temperature
        self.branching_temperature = branching_temperature
        # Generate Tree
        self.forwardModel.reInitialize(1)
        parent = Node(None, initialState, None, None, self.forwardModel.hidden)
        self.allNodes.append(parent)
        self.tree_head = self.grow(parent,0,self.branchFactor)
        # Get leaves
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
            path.append(currNode.parent.state)
            if not currNode.parent.action is None:
                actions.append(currNode.parent.action)
            currNode = currNode.parent
        return (list(reversed(path)), list(reversed(actions)))


    def getPathFromNode(self,nodeNumber):
        node = self.allNodes[nodeNumber]
        path = [node.state]
        actions = [node.action]
        currNode = node
        while not currNode.parent is None:
            path.append(currNode.parent.state)
            if not currNode.parent.action is None:
                actions.append(currNode.parent.action)
            currNode = currNode.parent
        return (list(reversed(path)),list(reversed(actions)))

    
    def grow(self,node,d,b,verbose=False):
        if verbose: print('Grow depth: ',d)
        if verbose: self.env.printState(node.state[0].data.numpy())
        if d == self.maxDepth : return node
        if type(b) is int: b = avar(torch.LongTensor([b]))
        i = 0
        while (i < b.data).all():
            # Sample the current action
            hard_action, soft_a_s, new_branching_breadth = self.simPolicy.sample(node.state, temperature=self.temperature,branching_temperature=self.branching_temperature)
            a_s =  [torch.squeeze(hard_action)]
            inital_state =  torch.squeeze(node.state)
            self.forwardModel.setHiddenState(node.hidden)
            current_state, _, current_hidden = self.forwardModel.forward(inital_state,a_s, 1)
            # Build the next subtre
            current_state = current_state.unsqueeze(dim=0)
            self.allStates.append(current_state)
            self.allActions.append(a_s)
            if verbose:
                print("int_state at depth",d)
                self.env.printState(node.state[0].data.numpy())
                print("a_s at depth ",d," and breath",i)
                self.env.printAction(a_s[0])
                self.env.printAction(a_s[0])
                print("curr_state at depth",d)
                self.env.printState(current_state[0].data.numpy())
            childNode = Node(node, current_state, [soft_a_s], [hard_action], current_hidden)
            self.allNodes.append( childNode )
            childNode.branchingBreadth = new_branching_breadth
            node.addChild( self.grow( childNode, d + 1, new_branching_breadth) )
            i += 1
        return node
    
    #
    def getBestPlanFromLeaves(self):
        bestInd, bestVal = 0, avar(torch.FloatTensor( [float('-inf')])) #float('-inf')
        for i, leaf in enumerate(self.leaves):
            currVal = self.valueF(leaf.state)
            if currVal.data.numpy() > bestVal.data.numpy():
                bestInd = i
                bestVal = currVal
        return self.getPathFromLeaf( bestInd )

    def getBestPlan(self, useOnlyLeaves=False):
        bestInd, bestVal = 0, avar(torch.FloatTensor( [float('inf')])) #float('-inf')\n",
        targetNodes = self.allNodes
        if useOnlyLeaves: targetNodes = self.leaves
        for i, node in enumerate(targetNodes):
            currVal = node.loss 
            if currVal.data.numpy() < bestVal.data.numpy():
                bestInd = i
                bestVal = currVal
        return self.getPathFromNode( bestInd )
        
    def getLossFromAllNodes(self, alpha=0.5, lambda_h=-0.025, useHolder=False, holderp=-5.0, useOnlyLeaves=False, gamma=0.01):
        targetNodes = self.allNodes
        if useOnlyLeaves: targetNodes = self.leaves
        totalInverseValue = avar(torch.FloatTensor([0.0]))
        totalEntropy      = avar(torch.FloatTensor([0.0]))
        totalBranching    = avar(torch.FloatTensor([0.0]))
        if not useHolder: holderp = 1.0
        nNodes = len(targetNodes)
        for i, node in enumerate(targetNodes):
            if i == 0: 
                node.loss = avar(torch.FloatTensor( [float('inf')] ))
                continue
            if not node.branchingBreadth is None:
                totalBranching += node.branchingBreadth.type(torch.FloatTensor) # IGNORES PARENT TODO
            node.loss = -self.valueF( node.state )
            totalInverseValue += node.loss.pow( holderp )
            if not node.action is None:
                totalEntropy += -torch.sum(node.action[0] * torch.log(node.action[0]))
        # Penalize negative reward and entropy
        totalLosses = alpha * (totalInverseValue / nNodes).pow(1.0 / holderp) + lambda_h * totalEntropy
        # Penalize too many branches
        totalLosses += gamma * totalBranching / nNodes
        return totalLosses 

    # Call before getting best plan
    def measureLossAtTestTime(self, useOnlyLeaves=False):
        targetNodes = self.allNodes
        if useOnlyLeaves: targetNodes = self.leaves
        for i, node in enumerate(targetNodes):
            if i == 0: 
                node.loss = avar(torch.FloatTensor( [float('inf')]))
                continue
            node.loss = -self.valueF( node.state )


def main():

    alphas = [0.0,0.5, 1.0]                                  # Entropy strength
    etas = [0.003,0.0003,0.00003]        # Learning rate
    useOnlyLeaves = [True, False]                                       # Whether to use Gumbel-softmax
    temperatures = [0.02, 0.2 , 2, 10]                             # Temperature for Gumbel-softmax
    branching_temperatures = [0.02, 0.2 , 2, 10]
    gammas = [0, 0.01, 1.0]   # Noise strength on input
    lambda_hs = [-0.05,0,0.05]
    # alphas = [0.5, 1.0]                                  # Entropy strength
    # etas = [0.0003]        # Learning rate
    # useOnlyLeaves = [True, False]                                       # Whether to use Gumbel-softmax
    # temperatures = [0.2]                             # Temperature for Gumbel-softmax
    # branching_temperatures = [2]
    # gammas = [0]   # Noise strength on input
    # lambda_hs = [0]
    ########################################################

    ###### Settings ######
    niters = 1000
    numRepeats = 5
    ######################

    # Initialize forward Model
    f_model_name = 'LSTM_FM_1_99' 
    s = 'navigation' # 'transport'
    # Read training/validation data
    print('Starting')
    trainf, validf = s + "-data-train-small.pickle", s + "-data-test-small.pickle"
    #train, valid = SeqData(trainf), SeqData(validf)
    # Load forward model
    ForwardModel = LSTMForwardModel(74,64)
    ForwardModel.load_state_dict( torch.load(f_model_name) )    

    hyperparam_output=[]
    # Run tasks over all hyper-parameter settings
    N_p, cp = len(lambda_hs)*len(etas)*len(gammas)*len(temperatures)*len(branching_temperatures)*len(useOnlyLeaves) * len(alphas), 1
    for lambda_h in lambda_hs:
        for eta in etas:
            for gamma in gammas:
                for temperature in temperatures:
                    for branching_temperature in branching_temperatures:          
                        for useOnlyLeave in useOnlyLeaves:
                            for  alpha in alphas:
                                ps = str(cp) + '/' + str(N_p)   
                                acc,trials=runTests(eta,alpha,lambda_h,useOnlyLeave,gamma,temperature,branching_temperature,ForwardModel,niters,ps,numRepeats)
                                acc=acc/trials
                                hyperparam_output.append({'lambda_h':lambda_h,'eta':eta,'gamma':gamma,'temperature':temperature,'branching_temperature':branching_temperature,'useOnlyLeave': useOnlyLeave, 'alpha':alpha, 'ps':ps,'acc':acc,'trials':trials,'cp':cp})
                                if cp%10:
                                    with open('hyperparam_search_henaff.pickle', 'wb') as handle:
                                        pickle.dump(hyperparam_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                cp += 1
                                



# Function for running a single suite of tests (on one hyper-param set)
def runTests(eta,alpha,lambda_h,useOnlyLeaves,gamma,temperature,branching_temperature,ForwardModel,niters,cnum,numRepeats):
    # Define tasks
    tasks = [
        [1, generateTask(0,0,0,0,5)],
        [2, generateTask(0,0,0,0,7)],
        [3, generateTask(0,0,0,4,4)],
        [4, generateTask(0,0,0,4,8)],
        [5, generateTask(0,0,0,7,7)],
    ]
 
    # Display status
    wstring = cnum + ',lambda_h=' + str(lambda_h) + ',eta=' + str(eta) + ',sigma=' + str(gamma) + ',temperature=' + str(temperature) + ',branching_temperature=' + str(branching_temperature)  + ',useOnlyLeaves=' + str(useOnlyLeaves) + ',alpha=' + str(alpha) 
    # print(wstring)
    # For each tasks, repeated a few times, attempt to solve the problem
    score, tot = 0, 0
    for i, task in enumerate(tasks):
        #print(i)
        for _ in range(numRepeats):
            cenv = task[1]
            SimPolicy = SimulationPolicy(cenv)
            SimPolicy.trainSad(
                cenv, 
                ForwardModel, 
                printActions=False, 
                maxDepth=task[0], 
                treeBreadth=2, 
                eta_lr=eta,
                trainIters=niters,
                alpha=alpha,
                lambda_h=lambda_h,
                useHolder=True,
                holderp=-6.0, 
                useOnlyLeaves=useOnlyLeaves, 
                gamma=gamma,
                temperature=temperature,
                branching_temperature=branching_temperature
            )
            s_0 = torch.unsqueeze(avar(torch.FloatTensor(cenv.getStateRep())), dim=0)
            tree = Tree(s_0, ForwardModel, SimPolicy, greedy_valueF, cenv, maxDepth=task[0], branchingFactor=2, temperature=temperature,branching_temperature=branching_temperature)
            tree.measureLossAtTestTime()
            states, actions = tree.getBestPlan(useOnlyLeaves=useOnlyLeaves)
            # Check for correctness
            if actions[0] == None:
                tot += 1
                continue
            for i in range(len(actions)):
                a = actions[i][0].data.numpy().argmax()
                cenv.performAction( a )
            r = cenv.getReward()
            correct = (r==1)
            tot += 1
            if correct: score += 1
    wstring += ' -> Score:' + str(score) + '/' + str(tot)


    print(wstring)
    # Write output
    # if not fileToWriteTo is None:
    #     with open(fileToWriteTo,'a') as filehandle:
    #         filehandle.write( wstring + '\n' )

    return score, tot


############################
if __name__ == "__main__":
    main()
############################



