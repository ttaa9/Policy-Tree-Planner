import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar
    
from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData, SingDiscSeqData

import os, sys, pickle, numpy as np, numpy.random as npr, random as r

########################################################################################################
########################################################################################################

class ForwardModelLSTM_SD(nn.Module):

    def __init__(self, env, h_size=100, nlayers=1, lstmdropout=0.0):
        super(ForwardModelLSTM_SD, self).__init__()
        self.stateSize = len(env.getSingularDiscreteState())
        self.actionSize = len(env.actions)
        self.inputSize = self.stateSize + self.actionSize
        self.nlayers = nlayers
        # Input dimensions are (seq_len, batch, input_size)
        self.hdim = h_size
        # Linear (FC) layer for initial input embedding
        self.embed = nn.Linear(self.inputSize, self.hdim)
        # PT LSTM
        # Input: inval, (h,c)
        #  - inval: (seq_len, batch, input_size)
        #  - (h,c): each is (num_layers * num_directions, batch, hidden_size) 
        #           [num_directions = 2 for bidir_lstm]
        # Output: outval, (h_n,c_n)
        #  - outval: (seq_len, batch, hidden_size * num_directions)
        #  - h_n is the hidden state at the final timestep
        #  - c_n is the cell state at the final timestep
        self.lstm = nn.LSTM(input_size=self.hdim, hidden_size=self.hdim, num_layers=nlayers, dropout=lstmdropout)
        # Linear (FC) layer for final output transformation
        self.hiddenToState = nn.Linear(self.hdim, self.stateSize)

    # Pass hidden=None to reinitialize
    def step(self, inputVal, hidden=None):
        newIn = self.embed(inputVal.view(1, -1)).unsqueeze(1)
        output, hidden = self.lstm(newIn, hidden)
        output = F.softmax( self.hiddenToState(output.squeeze(1)), dim=1 )
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = avar(torch.zeros(steps, 1, self.stateSize))
        for i in range(steps):
            if force or i == 0:
                inputv = inputs[i]
            else:
                trueInput = inputs[i] # Even if not teacher forcing, still take true action
                inputv = torch.cat( [output,trueInput[-self.actionSize:].unsqueeze(0)], dim=1 )
            output, hidden = self.step(inputv, hidden)
            outputs[i] = output
        return outputs, hidden  

    # Get final state via outputs[-1]
    def runOnActionSequence(self,actions,hidden=None):
        steps = len(actions)
        outputs = avar(torch.zeros(steps, 1, self.stateSize)) # seqlen x batchlen x stateSize
        for i in range(steps):
            action = actions[i]
            inputv = torch.cat( [output, action.unsqueeze(0)], dim=1 )
            output, hidden = self.step(inputv, hidden)
            outputs[i] = output
        return outputs, hidden 

    def runTraining(self, trainSeq, validSeq, modelFilenameToSave=None,
            nEpochs=5000, epochLen=100, validateEvery=25, vbs=2500, noiseSigma=0.01,
            teacherForcingProbStart=0.9, teacherForcingProbEnd=0.0, eta_lr=0.001): # 0.001 ok
        print('--- Starting Training (nE=' + str(nEpochs) + ',eL=' + str(epochLen) + ') ---')
        optimizer = optim.Adam(self.parameters(), lr = eta_lr)
        lossf = nn.CrossEntropyLoss()
        ns, na, tenv = self.stateSize, self.actionSize, trainSeq.env
        teacherForcingProb = lambda t: teacherForcingProbStart*(1-t) + teacherForcingProbEnd*t
        for epoch in range(nEpochs):
            if epoch % validateEvery == 0: print('Epoch:', '%4d' % epoch, end='')
            train_x, train_y = trainSeq.getRandomMinibatch(epochLen)
            # TODO sigma noise for actions?!?!
            lossTotal = 0.0
            currTeacherProb = teacherForcingProb(float(epoch) / nEpochs)
            for seq_x, label_y in zip(train_x, train_y):
                seq_x = avar(torch.FloatTensor(seq_x), requires_grad=False)
                label_y = avar(torch.LongTensor(label_y), requires_grad=False)
                useTeacherForcing = r.random() < currTeacherProb
                outputs, hidden = self.forward(seq_x, None, useTeacherForcing)
                loss = lossf(outputs.squeeze(1), label_y)
                lossTotal += loss.data[0]
                # Back-propagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if epoch % validateEvery == 0:
                print(' (p_tf = ' + '%.2f' % (currTeacherProb) + ')',end='')
                print(" -> AvgLoss",'%.4f' % (lossTotal / epochLen),end=', ')
                print('Val Loss: ',end='')
                vx, vy = validSeq.getRandomMinibatch(vbs)
                lossc, acc = 0.0, 0.0
                for valid_x, valid_y in zip(vx, vy):
                    valid_x = avar(torch.FloatTensor(valid_x), requires_grad=False)
                    valid_y = avar(torch.LongTensor(valid_y), requires_grad=False)
                    teacherForced = False
                    outputs, hidden = self.forward(valid_x, None, teacherForced)
                    lossc += lossf(outputs.squeeze(1), valid_y).data[0]
                    finalOutput = outputs.squeeze(1)[-1]
                    outputInd = finalOutput.max(0)[1]
                    correct = outputInd.data[0] == valid_y[-1].data[0]
                    if correct: acc += 1.0
                print('%.4f' % (lossc / vbs) + ', Val Acc:', '%.4f' % (acc / float(vbs)) )
                # Save model
                if not modelFilenameToSave is None:
                    torch.save(self.state_dict(), modelFilenameToSave)

########################################################################################################

class GumbelSoftmax(object):
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape) #.cuda()
        return -avar(torch.log(-torch.log(U + eps) + eps))
    def gumbel_softmax_sample(logits, temperature):
        y = logits + GumbelSoftmax.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)
    def gumbel_softmax(logits, temperature):
        y = GumbelSoftmax.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

class Utils(object):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

########################################################################################################

class HenaffPlanner():

    def __init__(self,forward_model,maxNumActions=1,noiseSigma=0.0,startNoiseSigma=0.1,niters=200):
        # Parameters
        self.sigma = noiseSigma
        self.start_sigma = startNoiseSigma
        self.nacts = maxNumActions
        self.niters = niters
        # The forward model
        self.f = forward_model # lstm_sd
        # Stop forward model from training
        # for p in self.f.parameters(): p.requires_grad = False
        # Get shapes from forward model
        self.state_size = self.f.stateSize
        self.action_size = self.f.inputSize - self.f.stateSize

    def generatePlan(self,
            start_state,         # The starting state of the agent (one-hot singDisc)
            goal_state,          # The goal state of the agent as two ints (e.g. [gx,gy])
            eta=0.03,            # The learning rate given to ADAM
            noiseSigma=None,     # Noise strength on inputs. Overwrites the default setting from the init
            niters=None,         # Number of optimization iterations. Overwrites the default setting from the init
            useCE=False,         # Specifies use of the cross-entropy loss, taken over subvectors of the state
            verbose=False,       # Specifies verbosity
            extraVerbose=False,  # Specifies extra verbosity
            useGumbel=True,      # Whether to use Gumbel-Softmax in the action sampling
            temp=0.01,           # The temperature of the Gumbel-Softmax method
            lambda_h=0.0,        # Specify the strength of entropy regularization (negative values encourage entropy)
            holder_power=-5      # Holder exponent to use in the power mean
        ):

        # Other settings
        if not noiseSigma is None: self.sigma = noiseSigma
        if not niters is None: self.niters = niters

        # Initialize random actions and optimizer
        x_t = avar( torch.randn(self.nacts, self.action_size) * self.start_sigma, requires_grad=True )
        optimizer = torch.optim.Adam( [x_t], lr=eta )
        # Choose loss function
        lossf = nn.CrossEntropyLoss()    
        # Get goal state position
        gx_ind, gy_ind = avar(torch.LongTensor([goal_state[0]])), avar(torch.LongTensor([goal_state[1]]))
        g_statesn = [ self.f.env.singularDiscreteStateFromInts(goal_state[0],goal_state[1],ii) for ii in range(0,4) ]
        g_states = [ avar(torch.LongTensor( [int(np.argmax(gsn)) ])) for gsn in g_statesn ]
        # Indices of start state position
        sindx, sindy, sorien = self.f.env.singularDiscreteStateToInts(start_state)

        # Start optimization loop
        for i in range(self.niters):
            # Generate soft action sequence
            epsilon = avar( torch.randn(self.nacts, self.action_size) * self.sigma )
            # Add noise to current action sequence
            y_t = x_t + epsilon
            # Softmax inputs to get current soft actions
            a_t = F.softmax( y_t, dim=1 )
            # Compute predicted state
            currState = avar(torch.FloatTensor(start_state)).unsqueeze(0)
            
            # If using Gumbel, replace the soft actions with sampled ones
            if useGumbel:
                a_t_input = avar( torch.zeros(self.nacts, self.action_size) )
                for k in range(0,self.nacts):
                    logProbAction = torch.log( a_t[k,:] ) 
                    a_t_input[k,:] = GumbelSoftmax.gumbel_softmax(logProbAction, temp)
            else:
                a_t_input = a_t

            # Run through the lstm forward model
            outputs, hidden = self.f.runOnActionSequence(a_t_input,hidden=None)
            finalOutput = output[-1]

            # Now have final (predicted) result of action sequence
            # Compute the loss via the holder (power) mean of the CE losses
            lossce1 = lossf( finalOutput , g_states[0] )
            lossce2 = lossf( finalOutput , g_states[1] )
            lossce3 = lossf( finalOutput , g_states[2] )
            lossce4 = lossf( finalOutput , g_states[3] )
            lossce = (( lossce1.pow(holder_power) + 
                        lossce2.pow(holder_power) + 
                        lossce3.pow(holder_power) + 
                        lossce4.pow(holder_power) ) / 4.0 
                     ).pow(1.0 / holder_power)
            # Entropy penalty
            H = -torch.sum( torch.sum( a_t*torch.log(a_t) , dim = 1 ) )
            # Final loss function 
            loss = lossce + lambda_h * H           
            # Clear the optimizer gradient
            optimizer.zero_grad()
            # Back-prop the errors to get the new gradients
            loss.backward()
            optimizer.step()
            # Ensure the x_t gradients are cleared
            x_t.grad.data.zero_()
        # Return the final action sequence 
        return [ x.max(0)[1].data[0] for x in x_t ]


########################################################################################################

def main():

    ####### Settings #######
    preloadModel = True
    runTraining = True      # Task 1
    testHenaff = False       # Task 2
    ########################

    ############################ External Files ############################
    ts = "navigation-data-train-sequence-singularDiscrete.pickle"
    vs = "navigation-data-test-sequence-singularDiscrete.pickle"
    f_model_name_to_preload = 'forward-lstm-singDisc-1.pt'    
    f_model_name_to_save = 'forward-lstm-singDisc-2.pt'    
    ########################################################################

    # Define shell environment and empty forward model
    exampleEnv = NavigationTask()
    f = ForwardModelLSTM_SD(exampleEnv)

    # Preload the forward model, if wanted
    if preloadModel and not f_model_name_to_preload is None:
        f.load_state_dict( torch.load(f_model_name_to_preload) )

    # Run training if desired
    if runTraining:
        trainSeqs = SingDiscSeqData(ts, exampleEnv)
        validSeqs = SingDiscSeqData(vs, exampleEnv)
        f.runTraining(
            trainSeqs, 
            validSeqs, 
            modelFilenameToSave=f_model_name_to_save,
            nEpochs=5000, 
            epochLen=100, 
            validateEvery=50, 
            vbs=2500, 
            noiseSigma=0.01, # Note: does nothing
            teacherForcingProbStart=0.1, 
            teacherForcingProbEnd=0.0, 
            eta_lr=0.001
        )

    #
    if testHenaff:
        print('Environment states')
        start_px = 0
        start_py = 0
        start_orien = 0
        start_state = exampleEnv.singularDiscreteStateFromInts(start_px,start_py,start_orien)
        goal_state = [0,2]
        print('Building planner')
        planner = HenaffPlanner(f, maxNumActions=2)
        print('Starting generation')
        actions = planner.generatePlan(
            start_state,         # The starting state of the agent (one-hot singDisc)
            goal_state,          # The goal state of the agent as two ints (e.g. [gx,gy])
            eta=0.01,            # The learning rate given to ADAM
            noiseSigma=None,     # Noise strength on inputs. Overwrites the default setting from the init
            niters=None,         # Number of optimization iterations. Overwrites the default setting from the init
            useCE=False,         # Specifies use of the cross-entropy loss, taken over subvectors of the state
            verbose=False,       # Specifies verbosity
            extraVerbose=False,  # Specifies extra verbosity
            useGumbel=False,     # Whether to use Gumbel-Softmax in the action sampling
            temp=0.01,           # The temperature of the Gumbel-Softmax method
            lambda_h=0.0         # Specify the strength of entropy regularization (negative values encourage entropy)
        )
        print('START STATE:', start_px, start_py, start_orien)
        print('FINAL ACTIONS:', ", ".join([str(a)+' ('+NavigationTask.actions[a]+')' for a in actions]))
        print('GOAL STATE:', goal_state)
        newEnv = NavigationTask(
            agent_start_pos=[np.array([start_px,start_py]), NavigationTask.oriens[start_orien]],
            goal_pos=np.array(goal_state))
        for action in actions: newEnv.performAction(action)
        state = newEnv.getStateRep(oneHotOutput=False)
        pred_x = state[0]
        pred_y = state[1]
        pred_orien = NavigationTask.oriens[ np.argmax(state[2:6]) ]
        print('PREDICTED FINAL STATE:',pred_x,pred_y,pred_orien)


########################################################################################################
########################################################################################################
def mainOld():

    ####################################################
    useFFANN = True
    trainingFFANN = False # 1
    henaffHyperSearch = False # 4
    runHenaffFFANN = False # 5
    manualTest = False # 2
    ####################################################

    if len(sys.argv) > 1:
        if sys.argv[1] == '1': trainingFFANN = True
        if sys.argv[1] == '2': manualTest = True
        if sys.argv[1] == '3': autoTest = True
        if sys.argv[1] == '4': henaffHyperSearch = True
        if sys.argv[1] == '5': runHenaffFFANN = True

    if useFFANN:

        f_model_name = 'forward-ffann-singDisc-noisy-2.pt' 
        exampleEnv = NavigationTask()
        f = ForwardModelFFANN(exampleEnv)

        ################################################################################################################
        if trainingFFANN:
            ############
            ts = "navigation-data-train-single-singularDiscrete.pickle"
            vs = "navigation-data-test-single-singularDiscrete.pickle"
            preload_name = None
            saveName = 'forward-ffann-singDisc-noisy-3.pt'
            ############
            print('Reading Data')
            with open(ts,'rb') as inFile:
                print('\tReading',ts); trainSet = pickle.load(inFile)
            with open(vs,'rb') as inFile:
                print('\tReading',vs); validSet = pickle.load(inFile)
            if not preload_name is None:
                print('Loading from',f_model_name)
                f.load_state_dict( torch.load(f_model_name) )
            f.runTraining(trainSet,validSet,maxIters=50000,modelFilenameToSave=saveName,testEvery=100)

        if manualTest: # 2
            print('Loading from',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )
            print('Environment states')
            ###
            start_px = 7
            start_py = 9
            start_orien = 1
            action = [5,1,5]  
            ###
            cstate = avar(torch.FloatTensor( exampleEnv.singularDiscreteStateFromInts(start_px,start_py,start_orien) )).unsqueeze(0)
            for act in action:
                action1h = avar(torch.FloatTensor( exampleEnv._intToOneHot(act, 10) )).unsqueeze(0)
                inputVal = torch.cat([cstate, action1h], dim=1) 
                cstate = f.forward( inputVal )
            print(cstate)
            print( "sx,sy,sorien =", start_px,',',start_py,',',start_orien )
            print( "As =", ",".join([NavigationTask.actions[a] for a in action]) )
            print( "px,py,orien =", f.env.singularDiscreteStateToInts( cstate.squeeze(0).data.numpy() ))
            

        ################################################################################################################
        if runHenaffFFANN: # 5
            print('Loading from',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )
            print('Environment states')
            start_px = 0
            start_py = 0
            start_orien = 0
            start_state = exampleEnv.singularDiscreteStateFromInts(start_px,start_py,start_orien)
            goal_state = [0,2]
            print('Building planner')
            planner = HenaffPlanner(f, maxNumActions=2)
            print('Starting generation')
            actions = planner.generatePlan(
                start_state,         # The starting state of the agent (one-hot singDisc)
                goal_state,          # The goal state of the agent as two ints (e.g. [gx,gy])
                eta=0.01,            # The learning rate given to ADAM
                noiseSigma=None,     # Noise strength on inputs. Overwrites the default setting from the init
                niters=None,         # Number of optimization iterations. Overwrites the default setting from the init
                useCE=False,         # Specifies use of the cross-entropy loss, taken over subvectors of the state
                verbose=False,       # Specifies verbosity
                extraVerbose=False,  # Specifies extra verbosity
                useGumbel=False,      # Whether to use Gumbel-Softmax in the action sampling
                temp=0.01,           # The temperature of the Gumbel-Softmax method
                lambda_h=0.0         # Specify the strength of entropy regularization (negative values encourage entropy)
            )
            print('START STATE:', start_px, start_py, start_orien)
            print('FINAL ACTIONS:', ", ".join([str(a)+' ('+NavigationTask.actions[a]+')' for a in actions]))
            print('GOAL STATE:', goal_state)
            newEnv = NavigationTask(
                agent_start_pos=[np.array([start_px,start_py]),NavigationTask.oriens[start_orien]],
                goal_pos=np.array(goal_state))
            for action in actions: newEnv.performAction(action)
            state = newEnv.getStateRep(oneHotOutput=False)
            pred_x = state[0]
            pred_y = state[1]
            pred_orien = NavigationTask.oriens[ np.argmax(state[2:6]) ]
            print('PREDICTED FINAL STATE:',pred_x,pred_y,pred_orien)

        ################################################################################################################
        if henaffHyperSearch:
            print('Loading ',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )
            
            ##################### Hyper-params #####################
            # lambda_hs = [0.0,0.01,-0.01,0.05,-0.05,0.005,-0.005]            # Entropy strength
            # etas = [0.5,0.25,0.1,0.05,0.025,0.01,0.005,0.001,0.0005]        # Learning rate
            # useGumbels = [True,False]                                       # Whether to use Gumbel-softmax
            # temperatures = [0.1,0.01,0.001,1.0]                             # Temperature for Gumbel-softmax
            # noiseSigmas = [0.0,0.01,0.02,0.05,0.1,0.25,0.5,0.75,1.0,1.25]   # Noise strength on input
            ## Init try
            # lambda_hs = [0.0,0.005,-0.005]                                  # Entropy strength
            # etas = [0.5,0.25,0.1,0.05,0.025,0.01,0.005,0.001,0.0005]        # Learning rate
            # useGumbels = [True,False]                                       # Whether to use Gumbel-softmax
            # temperatures = [0.1,0.01,0.001]                             # Temperature for Gumbel-softmax
            # noiseSigmas = [0.0,0.05,0.1,0.5,1.0]   # Noise strength on input
            ## Only use ones with decent results
            lambda_hs = [0.0,-0.005]                                  # Entropy strength
            etas = [0.5,0.25,0.1,0.005]        # Learning rate
            useGumbels = [True,False]                                       # Whether to use Gumbel-softmax
            temperatures = [0.1,0.001]                             # Temperature for Gumbel-softmax
            noiseSigmas = [0.5,1.0]   # Noise strength on input
            ########################################################

            ###### Settings ######
            niters = 75
            verbose = False
            extraVerbose = False
            numRepeats = 10
            fileToWriteTo = 'hyper-param-results.txt' # Set to None to do no writing
            distType = 1 # 0 = MSE, 1 = CE, 2 = dist
            ######################

            # Build an env with the given INT inputs
            def generateTask(px,py,orien,gx,gy):
                direction = NavigationTask.oriens[orien]
                gs = np.array([gx, gy])
                env = NavigationTask(agent_start_pos=[np.array([px,py]), direction],goal_pos=gs)
                return env

            # Function for running a single suite of tests (on one hyper-param set)
            def runTests(lh,eta,noiseLevel,ug,cnum,temp=None,distType=0):
                # Define tasks
                tasks = [
                    [1, generateTask(0,0,0,0,2)],
                    [1, generateTask(5,5,1,8,5)],
                    [1, generateTask(3,2,2,3,0)],
                    [1, generateTask(9,9,3,7,9)],
                    [2, generateTask(0,0,0,0,6)],
                    [2, generateTask(0,0,0,0,8)],
                    [2, generateTask(2,3,0,2,8)],
                    [2, generateTask(0,0,0,0,10)],
                    [3, generateTask(1,1,0,2,2)]
                ]
                # Choose dist type
                if distType == 0:   useCE = False; intDist = False
                elif distType == 1: useCE = True;  intDist = False
                elif distType == 2: useCE = False; intDist = True 
                # Display status
                wstring = cnum + ',lambda_h=' + str(lh) + ',eta=' + str(eta) + ',sigma=' + str(noiseLevel) + ',dType=' + str(distType) + ',ug=' + str(ug)
                if ug: wstring += ',temp=' + str(temp) 
                # For each tasks, repeated a few times, attempt to solve the problem
                score, tot = 0, 0
                for i, task in enumerate(tasks):
                    #print(i)
                    for _ in range(numRepeats):
                        planner = HenaffPlanner(f,maxNumActions=task[0])
                        cenv = task[1]
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
                        # Check for correctness
                        for a in actions: cenv.performAction( a )
                        r = cenv.getReward()
                        correct = (r==1)
                        tot += 1
                        if correct: score += 1
                wstring += ' -> Score:' + str(score) + '/' + str(tot)
                print(wstring)
                # Write output
                if not fileToWriteTo is None:
                    with open(fileToWriteTo,'a') as filehandle:
                        filehandle.write( wstring + '\n' )

            # Run tasks over all hyper-parameter settings
            N_p, cp = len(lambda_hs)*len(etas)*len(noiseSigmas)*(1 + len(temperatures)), 1
            for lambda_h in lambda_hs:
                for eta in etas:
                    for noiseLevel in noiseSigmas:
                        for ug in useGumbels:
                            if ug:
                                for temp in temperatures: 
                                    ps = str(cp) + '/' + str(N_p)
                                    runTests(lambda_h,eta,noiseLevel,ug,ps,temp,distType=distType)
                                    cp += 1
                            else: 
                                ps = str(cp) + '/' + str(N_p)
                                runTests(lambda_h,eta,noiseLevel,ug,ps,distType=distType)
                                cp += 1

############################
if __name__ == '__main__':
    main()
############################

