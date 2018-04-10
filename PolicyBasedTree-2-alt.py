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
    env = NavigationTask(agent_start_pos=[np.array([px,py]), direction], goal_pos=gs)
    return env

class SimulationPolicy(nn.Module):
    def __init__(self,  env, layerSizes=[100,100], maxBranchingFactor=10):
        super(SimulationPolicy, self).__init__()
        self.actionSize = len(env.actions)
        self.stateSize = len(env.getStateRep(oneHotOutput=True))
        self.env = env
        self.maxBranchingFactor = maxBranchingFactor
        self.intvec = avar( torch.LongTensor(list(range(maxBranchingFactor ))) ).unsqueeze(0)
        print("State Size: " , self.stateSize, "\nAction Size: ", self.actionSize)
        # Input space: [Batch, observations], output:[Batch, action_space]
        self.layer1 = nn.Linear(self.stateSize, layerSizes[0])
        self.layer2 = nn.Linear(layerSizes[0], layerSizes[1])
        self.layer3 = nn.Linear(layerSizes[1], self.actionSize)
        # Layer to sample branching factor
        #self.layers1 = nn.Linear(self.stateSize, layerSizes[0])
        self.layers2 = nn.Linear(layerSizes[0], layerSizes[1])
        self.intSamplingLayer = nn.Linear(layerSizes[1], self.maxBranchingFactor)
        
    def sample(self,state,temperature=0.5,branching_temperature=0.01,verbose=False):
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
        #outputs1 = F.relu( self.layers1(state) )
        outputs2 = F.sigmoid( self.layers2(output1) )
        b_ann = self.intSamplingLayer(outputs2)
        b_ann_logsoft = m(b_ann)
        if verbose: print('BAL',b_ann_logsoft)
        b_gumbeled_gsh = gumbel_softmax_hard(b_ann_logsoft, branching_temperature)#.type(torch.LongTensor)
        b_gumbeled = b_gumbeled_gsh.type(torch.LongTensor)
        if verbose: print('BG',b_gumbeled)
        branchingRateSample = torch.dot(self.intvec, b_gumbeled)
        if verbose: print('BRS',branchingRateSample)
        return gumbeled_action, soft_action, branchingRateSample + 1, b_ann ### <--- the + 1 forces it to be at least 1

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
        #treeBreadth=2, 
        eta_lr=0.0005,
        trainIters=500,
        alpha=0.5,
        lambda_h=-0.025,
        useHolder=False,
        holderp=-6.0, 
        useOnlyLeaves=False, 
        gamma=0.01,
        xi=0.01
        ):
        optimizer = optim.Adam(self.parameters(), lr = eta_lr )
        for p in forwardModel.parameters(): p.requires_grad = False
        s0 = avar(torch.FloatTensor([self.env.getStateRep()]), requires_grad=False)
        overallNumNodes = []
        accuracies = []
        numNodes = []
        accs = []
        def getAccOfPlan(plan):
            ss = self.env.getStateRep(oneHotOutput=False)
            px = ss[0]
            py = ss[1]
            orien = np.argmax(ss[2:6])
            gx = ss[-2]
            gy = ss[-1]
            envCopy = generateTask(px,py,orien,gx,gy)
            ###
            #print(plan[1])
            for act in plan[1]:
                aaa = act[0].data.numpy().argmax()
                envCopy.performAction(aaa)
            return envCopy.getReward()
        printEvery = 25
        for i in range(0,trainIters):
            
            tree = Tree(s0, forwardModel, self, greedy_valueF, self.env, maxDepth) #, treeBreadth)
            loss = tree.getLossFromAllNodes(
                alpha=alpha, lambda_h=lambda_h, useHolder=useHolder, 
                holderp=holderp, useOnlyLeaves=useOnlyLeaves, gamma=gamma, xi = xi
            )
            plan = tree.getBestPlan()
            #
            clen = len(tree.allNodes)
            cacc = getAccOfPlan(plan)
            numNodes.append( clen )
            accs.append( cacc )
            overallNumNodes.append( clen )
            accuracies.append( cacc )
            #
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % printEvery == 0:
                nns = np.array(numNodes)
                accsnp = np.array(accs)
                print('----------------')
                print('Local Loss',i,":",loss.data[0])
                print('NumNodes:', np.mean(nns), '('+str(np.std(nns))+')') #,len(nns)) 
                print('Accs:', np.mean(accs), '('+str(np.std(accs))+')') #,len(accs)) 
                #print('NumTreeNodes (avg over last %d):' % printEvery, avgNumNodes / float(printEvery))
                #avgNumNodes = 0.0
                if printActions:
                        # treeSizes = []
                        # nSolved = []
                        # for i in range(0,3):
                        #     tree = Tree(s0, forwardModel, self, greedy_valueF, self.env, maxDepth)
                        #     plan = tree.getBestPlan()
                        #     treeSizes.append( len(tree.allNodes) )
                        #     print(plan)
                        #     sys.exit(0)
                            #nSolved.append(  )
                    # print(plan)
                    print( "\n".join([ "A" + str(qi) + ": " + 
                        ",".join([
                            ",".join(["%.3f" % q for q in qq]) for qq in a[0].data.numpy()
                        ]) for qi,a in enumerate(plan[1])
                    ] ) )
                numNodes = []
                accs = []
        print("--- TRAINING END ---")
        print("NNs",overallNumNodes)
        print("Accs",accuracies)
        print("-----------")
        
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
    
    def __init__(self, initialState, forwardModel, simPolicy, valueF, env, maxDepth=5): #, branchingFactor=3):
        self.simPolicy = simPolicy
        self.maxDepth = maxDepth #, self.branchFactor = maxDepth, branchingFactor
        self.forwardModel = forwardModel
        self.valueF = valueF
        self.allStates = [initialState]
        self.allActions = []
        self.allNodes = []
        self.env = env
        # Generate Tree
        self.forwardModel.reInitialize(1)
        parent = Node(None, initialState, None, None, self.forwardModel.hidden)
        self.allNodes.append(parent)
        _, _, treeBreadth, sb = simPolicy.sample(initialState)
        parent.softBranching = sb
        parent.branchingBreadth = treeBreadth
        self.tree_head = self.grow(parent, 0, treeBreadth)
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
            hard_action, soft_a_s, new_branching_breadth, softBranching = self.simPolicy.sample(node.state)
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
            childNode.softBranching = F.softmax(softBranching, dim=1)
            node.addChild( self.grow( childNode, d + 1, new_branching_breadth) )
            i += 1
        return node

    def getBestPlan(self):
        bestInd, bestVal = 0, avar(torch.FloatTensor( [float('inf')])) #float('-inf')\n",
        currpath = None
        for i, node in enumerate(self.allNodes):
            if i == 0: continue
            currVal = node.loss 
            if currVal.data.numpy() < bestVal.data.numpy():
                putPath = self.getPathFromNode( i )
                bestInd = i
                bestVal = currVal
                currpath = putPath
            elif currVal.data.numpy() == bestVal.data.numpy():
                if (currpath is None) or (len(putPath) < len(currpath)):
                    putPath = self.getPathFromNode( i )
                    bestInd = i
                    bestVal = currVal
                    currpath = putPath
        return currpath
        
    def getLossFromAllNodes(self, alpha=0.5, lambda_h=-0.025, useHolder=False, holderp=-5.0, useOnlyLeaves=False, gamma=0.01, xi = 0.01):
        targetNodes = self.allNodes
        if useOnlyLeaves: targetNodes = self.leaves
        totalInverseValue = avar(torch.FloatTensor([0.0]))
        totalEntropy      = avar(torch.FloatTensor([0.0]))
        totalBranching    = avar(torch.FloatTensor([0.0]))
        totalEntropyB      = avar(torch.FloatTensor([0.0])) # For branching sampler
        if not useHolder: holderp = 1.0
        nNodes = len(targetNodes)
        mbf = avar(torch.FloatTensor( np.array(list(range(1,self.simPolicy.maxBranchingFactor+1))) ))
        for i, node in enumerate(targetNodes):
            if i == 0: 
                node.loss = avar(torch.FloatTensor( [float('inf')] ))
                continue
            if not node.branchingBreadth is None:
                expectedBranching = torch.sum( node.softBranching * mbf )
                totalBranching += expectedBranching
                # print('----')
                # print('Exp',expectedBranching)
                # print('TrueBB',node.branchingBreadth)
                # totalBranching += node.branchingBreadth.type(torch.FloatTensor) # IGNORES PARENT TODO
            node.loss = -self.valueF( node.state )
            totalInverseValue += node.loss.pow( holderp )
            if not node.action is None:
                totalEntropy += -torch.sum(node.action[0] * torch.log(node.action[0]))
            if not node.softBranching is None:
                totalEntropyB += -torch.sum(node.softBranching * torch.log(node.softBranching))
        # Penalize negative reward and entropy
        totalLosses = alpha * (totalInverseValue / nNodes).pow(1.0 / holderp) + lambda_h * totalEntropy / nNodes
        # Penalize too many branches
        totalLosses += gamma * totalBranching / nNodes
        # Penalize entropy in the branching sampler
        totalLosses += xi * totalEntropyB / nNodes
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

    ###
    runTraining = True
     
    ###

    f_model_name = 'LSTM_FM_1_99' 
    s = 'navigation' # 'transport'
    # Read training/validation data
    print('Reading Data')
    trainf, validf = s + "-data-train-small.pickle", s + "-data-test-small.pickle"
    train, valid = SeqData(trainf), SeqData(validf)
    # Load forward model
    ForwardModel = LSTMForwardModel(train.lenOfInput,train.lenOfState)
    ForwardModel.load_state_dict( torch.load(f_model_name) )
    # Initialize forward policy
    exampleEnv = generateTask(0,0,0,5,10) 
    SimPolicy = SimulationPolicy(exampleEnv)
    # Run training
    if runTraining:
        maxDepth = 4
        SimPolicy.trainSad(
            exampleEnv, 
            ForwardModel, 
            printActions=True, 
            maxDepth=maxDepth, 
            # treeBreadth=2, 
            eta_lr=0.0025,  #0.000375,
            trainIters=400,
            alpha=0.5,
            lambda_h=-0.025, #-0.0125, # negative = encourage entropy in actions
            useHolder=True,
            holderp=-1.0, 
            useOnlyLeaves=False, 
            gamma=0.00000125, #1.5
            xi=0.0000000125
        )
         
        # It has no effect anywhere else
        s_0 = torch.unsqueeze(avar(torch.FloatTensor(exampleEnv.getStateRep())), dim=0)
        tree = Tree(s_0, ForwardModel, SimPolicy, greedy_valueF, exampleEnv, maxDepth=maxDepth) #, branchingFactor=2)
        tree.measureLossAtTestTime()
        states, actions = tree.getBestPlan()
        print('Final Actions')
        for i in range(len(actions)):
            jq = actions[i][0].data.numpy().argmax()
            print('A'+str(i)+':',jq,NavigationTask.actions[jq])


############################
if __name__ == "__main__":
    main()
############################



