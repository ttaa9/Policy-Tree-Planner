import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar
    
from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData
from LSTMFM import LSTMForwardModel
import pdb
import os, sys, pickle, numpy as np, numpy.random as npr, random as r
import pickle

f_model_name = 'LSTM_FM_1_99'    
s = 'navigation' # 'transport'
trainf, validf = s + "-data-train-small.pickle", s + "-data-test-small.pickle"
print('Reading Data')
#train, valid = SeqData(trainf), SeqData(validf)
fm = LSTMForwardModel(74,64)
fm.load_state_dict( torch.load(f_model_name) )
fileToWriteTo='hyperparam_output'

class HenaffPlanner():

    def __init__(self,forward_model,env, maxNumActions=1,noiseSigma=0.0,startNoiseSigma=0.1,niters=200):
        # Parameters
        self.sigma = noiseSigma
        self.start_sigma = startNoiseSigma
        self.nacts = maxNumActions
        self.niters = niters
        # The forward model
        self.f = forward_model
        # Stop forward model from training
        for p in self.f.parameters(): p.requires_grad = False
        # Get shapes from forward model
        self.state_size = self.f.stateSize
        self.action_size = self.f.inputSize - self.f.stateSize
        self.env = env

    def generatePlan(self,
            start_state,         # The starting state of the agent
            eta=0.0003,            # The learning rate given to ADAM
            noiseSigma=None,     # Noise strength on inputs. Overwrites the default setting from the init
            niters=None,         # Number of optimization iterations. Overwrites the default setting from the init
            goal_state=None,     # Use to specify a goal state manually (instead of reading it from a given state)
            useCE=False,         # Specifies use of the cross-entropy loss, taken over subvectors of the state
            verbose=False,       # Specifies verbosity
            extraVerbose=False,  # Specifies extra verbosity
            useGumbel=True,      # Whether to use Gumbel-Softmax in the action sampling
            temp=0.01,           # The temperature of the Gumbel-Softmax method
            lambda_h=0.0,        # Specify the strength of entropy regularization (negative values encourage entropy)
            useIntDistance=False
        ):

        # Other settings
        useIntDistance = False # Note: does not apply if using CE
        useMSE_loss = (not useIntDistance) and (not useCE)
        if not noiseSigma is None: self.sigma = noiseSigma
        if not niters is None: self.niters = niters

        # Initialize random actions and optimizer
        x_t = avar( torch.randn(self.nacts, self.action_size) * self.start_sigma, requires_grad=True )
        optimizer = torch.optim.Adam( [x_t], lr=eta )
        # Choose loss function
        if useCE:
            lossf = nn.CrossEntropyLoss()
            if verbose: print('Using CE loss')
        elif useMSE_loss:
            if verbose: print('Using MSE loss')
            lossf = nn.MSELoss()
        else:
            if verbose: print('Using int distance loss')

        # Set goal state
        deconStartState = self.env.deconcatenateOneHotStateVector(start_state)
        if goal_state is None:
            gx, gy = avar(torch.FloatTensor(deconStartState[-2])), avar(torch.FloatTensor(deconStartState[-1]))
        else: print('Not yet implemented'); sys.exit(0)

        # Indices of start state position
        sindx = avar(torch.FloatTensor(deconStartState[0])).max(0)[1]
        sindy = avar(torch.FloatTensor(deconStartState[1])).max(0)[1]

        # Indices of goal state position
        indx, indy = gx.max(0)[1], gy.max(0)[1]

        # Start optimization loop
        for i in range(self.niters):
            # Generate soft action sequence
            epsilon = avar( torch.randn(self.nacts, self.action_size) * self.sigma )
            # Add noise to current action sequence
            y_t = x_t + epsilon
            # Softmax inputs to get current soft actions
            a_t = F.softmax( y_t, dim=1 )
            # Compute predicted state
            #currState = avar(torch.FloatTensor(start_state)).unsqueeze(0)
            # Loop over actions to obtain predicted state
            self.f.reInitialize(1)
            currState, intStates = self.f.forward(start_state, a_t, self.nacts)
            # Now have final (predicted) result of action sequence
            # Extract predicted position of current state
            pvx = currState[0:15] 
            pvy = currState[15:30] 
            # Compute loss
            if useCE: # Cross-entropy loss
                lossx = lossf(pvx.view(1,pvx.shape[0]), indx) 
                lossy = lossf(pvy.view(1,pvy.shape[0]), indy)
            elif useIntDistance: # Integer distance loss
                ints = avar( torch.FloatTensor( list(range(15)) ) )
                prx = torch.sum( ints * pvx )
                pry = torch.sum( ints * pvy )
                lossx = (1.0/15.0) * (prx - indx.data[0]).pow(2)
                lossy = (1.0/15.0) * (pry - indy.data[0]).pow(2) 
            else: # Using MSE loss via one-hot pos
                lossx = lossf(pvx, gx)
                lossy = lossf(pvy, gy)
            # Entropy penalty
            H = -torch.sum( torch.sum( a_t*torch.log(a_t) , dim = 1 ) )
            # Final loss function 
            loss = lossx + lossy + lambda_h * H
            # Print status
            if extraVerbose:
                a_inds = ",".join([ str(a.max(dim=0)[1].data[0]) for a in a_t  ]) 
                print(i,'->','Lx =',lossx.data[0],', Ly =',lossy.data[0],', H =',H.data[0],', TL =',loss.data[0],', A =',a_inds)
            # Clear the optimizer gradient
            optimizer.zero_grad()
            # Back-prop the errors to get the new gradients
            loss.backward(retain_graph=True)
            optimizer.step()
            # Print the predicted result at the current iteration
            if extraVerbose:
                print('Predicted End:',pvx.max(0)[1].data[0],pvy.max(0)[1].data[0])
            # Ensure the x_t gradients are cleared
            #x_t.grad.data.zero_()
        # Print and analyze the final plan, if desired
        if verbose:
            print('\nEnd\n')
            print('Actions')
            for k in range(0,self.nacts):
                action = F.softmax( x_t[k,:], dim=0 )
                print(action.max(0)[1].data[0],end=' -> ')
                print(NavigationTask.actions[action.max(0)[1].data[0]],end=' ')
                print(action.data)
            print('--')
            print('START ',sindx.data[0],sindy.data[0])
            print('TARGET END ',indx.data[0],indy.data[0])
            print('PREDICTED END',pvx.max(0)[1].data[0], pvy.max(0)[1].data[0])
            print('--')
        # Return the final action sequence 
        return [ x.max(0)[1].data[0] for x in x_t ]




# Build an env with the given INT inputs
def generateTask(px,py,orien,gx,gy):
    direction = NavigationTask.oriens[orien]
    gs = np.array([gx, gy])
    env = NavigationTask(agent_start_pos=[np.array([px,py]), direction],goal_pos=gs)
    return env

# Function for running a single suite of tests (on one hyper-param set)
env = NavigationTask()

def runTests(lh,eta,noiseLevel,ug,cnum,temp=None,distType=0,difficulty='Hard', tasks=None, verbose=False,extraVerbose=False):

    numRepeats=10
    niters=100

    # Define tasks
    if tasks== None:
        if difficulty=='Hard':
            print('\tin Task choice')
            tasks = [[7, generateTask(0,0,0,14,14)]] #[[5, generateTask(0,0,0,10,10)],[6, generateTask(0,0,0,10,12)],[7, generateTask(0,0,0,14,14)]]


        if difficulty=='Easy':

            tasks = [[3, generateTask(0,0,0,4,4)],
                [4, generateTask(0,0,0,4,8)]]

    print('Difficulty',difficulty)
    # Choose dist type
    if distType == 0:   useCE = False; intDist = False
    elif distType == 1: useCE = True;  intDist = False
    elif distType == 2: useCE = False; intDist = True 
    # Display status
    wstring = cnum + ',lambda_h=' + str(lh) + ',eta=' + str(eta) + ',sigma=' + str(noiseLevel) + ',dType=' + str(distType) + ',ug=' + str(ug)
    if ug: wstring += ',temp=' + str(temp) 
    print(wstring)
    # For each tasks, repeated a few times, attempt to solve the problem
    score, tot = 0, 0
    for i, task in enumerate(tasks):
        print('Task',i)
        print('Nactions',task[0],'; Start State',task[1].getStateRep(False))
        for _ in range(numRepeats):
            planner = HenaffPlanner(fm,env,maxNumActions=task[0])
            task_state = task[1].getStateRep(oneHotOutput=False)
            px = int(task_state[0])
            py = int(task_state[1])
            orien = np.argmax(task_state[2:6])
            gx = int(task_state[-2])
            gy = int(task_state[-1])
            print('www',px,py,orien,gx,gy)
            cenv = generateTask(px,py,orien,gx,gy)
            print(cenv.getStateRep(True)) 
            actions = planner.generatePlan(
                    cenv.getStateRep(oneHotOutput=True),
                    eta=eta,
                    noiseSigma=noiseLevel,
                    niters=niters,
                    goal_state=None,
                    useCE=True,
                    verbose=verbose,
                    extraVerbose=extraVerbose,
                    useGumbel=ug,
                    temp=temp,
                    lambda_h=lh,
                    useIntDistance=intDist )
            print('\tAs:',actions)
            # Check for correctness
            for a in actions: cenv.performAction( a )
            r = cenv.getReward()
            correct = (r==1)
            tot += 1
            print('Correct?',correct)
            if correct: score += 1
    wstring += ' -> Score:' + str(score) + '/' + str(tot)


    print(wstring)
    # Write output

    return score, tot





# Run tasks over all hyper-parameter settings

def hyperparam_search(lambda_hs=[0.0,-0.005, 0.005] ,
                    etas = [0.01,0.1,0.2,0.3,0.5, 1],
                    useGumbels = [True, False], 
                    temperatures = [0.02,0.1, 1, 10],
                    noiseSigmas = [0,0.01,0.1, 1.0,2.0],
                    niters = 200,
                    verbose = False,
                    extraVerbose = False, 
                    numRepeats = 10,
                    file_name_output = 'hyperparam_search_henaff.pickle',
                    distType = 1,
                    difficulty='Easy'):

    hyperparam_output=[]
    

    N_p, cp = len(lambda_hs)*len(etas)*len(noiseSigmas)*(1 + len(temperatures)), 1
    for lambda_h in lambda_hs:
        for eta in etas:
            for noiseLevel in noiseSigmas:
                for ug in useGumbels:          
                    for temp in temperatures:
                        ps = str(cp) + '/' + str(N_p)

                        if ug:    
                            acc,trials=runTests(lambda_h,eta,noiseLevel,ug,ps,temp,distType=distType,difficulty=difficulty)
                            acc=acc/trials
                            hyperparam_output.append({'lambda_h':lambda_h,'eta':eta,'noiseLevel':noiseLevel,'ug':ug,'ps':ps,'temp':temp,'distType':distType,'acc':acc,'trials':trials,'cp':cp,'difficulty':difficulty})
                        else: 
                            ps = str(cp) + '/' + str(N_p)
                            acc,trials=runTests(lambda_h,eta,noiseLevel,ug,ps,distType=distType,difficulty=difficulty)
                            acc=acc/trials
                            hyperparam_output.append({'lambda_h':lambda_h,'eta':eta,'noiseLevel':noiseLevel,'ug':ug,'ps':ps,'temp':None,'distType':distType,'acc':acc,'trials':trials,'cp':cp,'difficulty':difficulty})
        
                        cp += 1
                        if cp%10:
                            with open(file_name_output, 'wb') as handle:
                                pickle.dump(hyperparam_output, handle, protocol=pickle.HIGHEST_PROTOCOL)



