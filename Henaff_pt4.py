import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar
    
from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData

import os, sys, pickle, numpy as np, numpy.random as npr, random as r

########################################################################################################
########################################################################################################

class ForwardModelFFANN(nn.Module):
    def __init__(self, env, layerSizes=[350,350], dropoutProb=0.0):
        super(ForwardModelFFANN, self).__init__()
        self.env = env
        self.actionSize = len( env.actions )
        self.stateSize = len( env.getSingularDiscreteState() )
        self.inputSize = self.actionSize + self.stateSize
        self.layer1 = nn.Linear(self.inputSize, layerSizes[0])
        self.dropout1 = nn.AlphaDropout(p=dropoutProb)
        self.layer2 = nn.Linear(layerSizes[0], layerSizes[1])
        self.dropout2 = nn.AlphaDropout(p=dropoutProb)
        self.layer3 = nn.Linear(layerSizes[1], self.stateSize)

    def forward(self, inputValue):
        output = F.relu( self.layer1(inputValue) )
        output = self.dropout1( output )
        output = F.relu( self.layer2(output) ) 
        output = self.dropout2( output )
        output = self.layer3(output) 
        return F.softmax(output,dim=1)

    def runTraining(self,trainSet,validSet,maxIters=50000,modelFilenameToSave=None,testEvery=50,minibatchSize=150):

        ### Settings ###
        eta_training = 0.0000125
        optimizer = optim.Adam(self.parameters(), lr = eta_training * minibatchSize)
        lossFunction = nn.CrossEntropyLoss()
        validationTestSize = 3000
        noiseType = 1

        ### Load Data ###
        # Note these are integers for states, to save memory, in numpy form
        # Conversion to one-hot torch type occurs when the random mini-batches are extracted
        train_x, train_y = trainSet
        test_x, test_y = validSet
        ntrain, ntest = len(train_x), len(test_x)

        ### Function to extract minibatches from data ###
        # Noise = 0 -> none, 1 -> uniform, 2 -> Gaussian+softmax
        # maxUniformNoiseLevel \in [0.0,0.0011)
        def getRandomMiniBatch(dsx,dsy,mbs,nmax,noiseType=1,
            maxUniformNoiseLevel=0.001,gaussianSigma=0.001):
            choices = npr.choice(nmax, size=mbs, replace=False) 
            xs, ys = dsx[choices], dsy[choices]
            newMB_x = np.zeros( (mbs, self.inputSize) )
            newMB_y = np.zeros( mbs )
            if noiseType == 0:
                for i in range(mbs):
                    jx, jy = int(xs[i,0]), int(ys[i,0]) 
                    newMB_x[i,jx] = 1.0
                    newMB_x[i,-10:] = xs[i,-10:]
                    newMB_y[i] = jy 
            elif noiseType == 1:
                if self.stateSize*maxUniformNoiseLevel > 1:
                    print('Noise level is untenable! Max =',1.0/self.stateSize)
                    sys.exit(0)
                for i in range(mbs):
                    unifNoisedState = npr.uniform(low=0.0,high=maxUniformNoiseLevel,size=self.stateSize)
                    jx, jy = int(xs[i,0]), int(ys[i,0])
                    spikeValue = 1.0 - np.sum(unifNoisedState) + unifNoisedState[ jx ]
                    newMB_x[i, 0:self.stateSize] = unifNoisedState
                    newMB_x[i, jx] = spikeValue
                    newMB_x[i, 0:self.stateSize] /= np.sum(newMB_x[i, 0:self.stateSize])
                    newMB_x[i, -10:] = xs[i,-10:]
                    newMB_y[i] = jy
                    # print(np.argmax(newMB_x[i,0:self.stateSize]),',',
                    #      np.sum(newMB_x[i, 0:self.stateSize]),',',
                    #      newMB_x[i,np.argmax(newMB_x[i,0:self.stateSize])])
            elif noiseType == 2:
                for i in range(mbs):
                    jx, jy = int(xs[i,0]), int(ys[i,0]) 
                    noise = gaussianSigma * npr.normal(size=self.stateSize)
                    newMB_x[i,jx] = 1.0
                    newMB_x[i,0:self.stateSize] += noise
                    newMB_x[i,0:self.stateSize] = Utils.softmax( newMB_x[i,0:self.stateSize] )
                    newMB_x[i,-10:] = xs[i,-10:]
                    newMB_y[i] = jy 
                    # print(newMB_x[i,0:self.stateSize])
                    # sys.exit(0)

            newMB_x = avar( torch.FloatTensor(newMB_x), requires_grad=False )
            newMB_y = avar( torch.LongTensor(newMB_y),  requires_grad=False )
            return newMB_x, newMB_y

        ### Training Loop ###
        for i in range(0,maxIters):
            #print(i)
            x, y = getRandomMiniBatch(train_x, train_y, minibatchSize, ntrain,
                                      noiseType=noiseType,maxUniformNoiseLevel=0.0001)
            y_hat = self.forward(x)
            loss = lossFunction(y_hat, y)
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            self.zero_grad()
            if i % testEvery == 0:
                self.eval() # Turn off dropout
                print('Epoch', str(i) + ': L_t =', '%.4f' % loss.data[0], end=', ')
                vx, vy = getRandomMiniBatch(test_x,test_y,validationTestSize,ntest,
                                            noiseType=noiseType,maxUniformNoiseLevel=0.0001)
                predv = self.forward(vx)
                lossv = lossFunction(predv, vy)
                print('L_v =', '%.4f' % lossv.data[0], end=', ')
                acc = self._accuracyBatch(vy,predv)
                print('Acc =', '%.4f' % acc)
                self.train() # Turn back on dropout
                # Save model
                if not modelFilenameToSave is None:
                    torch.save(self.state_dict(), modelFilenameToSave)

    def printState(self,s,pre=''):
        trueInds = self.env.singularDiscreteStateToInts(s)
        print(str(pre) + 'px:',    trueInds[0])
        print(str(pre) + 'py:',    trueInds[1])
        print(str(pre) + 'orien:', trueInds[2],'(',self.env.oriens[trueInds[2]],')')

    def printAction(self,a,pre=''):
        if type(a) is avar: ind = np.argmax(a.data.numpy())
        else: ind = np.argmax(a)
        print(str(pre)+'a:',self.env.actions[ind],'('+str(ind)+')')

    def _accuracyBatch(self,ylist,yhatlist):
        n, acc = ylist.data.shape[0], 0.0 
        for i in range(n):
            acc += self._accuracySingle(ylist[i], yhatlist[i])
        return acc / n

    def _accuracySingle(self,label,prediction):
        return 1.0 if int(label.data.numpy()) == int(np.argmax(prediction.data.numpy())) else 0.0

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
        self.f = forward_model
        # Stop forward model from training
        # for p in self.f.parameters(): p.requires_grad = False
        # Get shapes from forward model
        self.state_size = self.f.stateSize
        self.action_size = self.f.inputSize - self.f.stateSize

    def generatePlan(self,
            start_state,         # The starting state of the agent (one-hot singDisc)
            goal_state,          # The goal state of the agent as two ints
            eta=0.03,            # The learning rate given to ADAM
            noiseSigma=None,     # Noise strength on inputs. Overwrites the default setting from the init
            niters=None,         # Number of optimization iterations. Overwrites the default setting from the init
            useCE=False,         # Specifies use of the cross-entropy loss, taken over subvectors of the state
            verbose=False,       # Specifies verbosity
            extraVerbose=False,  # Specifies extra verbosity
            useGumbel=True,      # Whether to use Gumbel-Softmax in the action sampling
            temp=0.01,           # The temperature of the Gumbel-Softmax method
            lambda_h=0.0         # Specify the strength of entropy regularization (negative values encourage entropy)
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
        lossf = nn.CrossEntropyLoss()    

        # Set goal state
        deconStartState = self.f.env.deconcatenateOneHotStateVector(start_state)
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
            currState = avar(torch.FloatTensor(start_state)).unsqueeze(0)
            # Loop over actions to obtain predicted state
            for k in range(0,self.nacts):
                # Current action
                action = a_t[k,:]
                # Apply Gumbel Softmax, if desired
                if useGumbel:
                    logProbAction = torch.log( action ) 
                    action = GumbelSoftmax.gumbel_softmax(logProbAction, temp)
                # Get next input via last state and current action
                currInput = torch.cat([currState[0],action], 0)
                currInput = torch.unsqueeze(currInput,0)
                # Get next state (to be used as next input) from forward model
                currState = self.f.forward( currInput )
                # Print result of current action if needed
                if extraVerbose:
                    print('Action:',k)
                    self.f.printState(currInput[0,0:64])
                    self.f.printAction(currInput[0,-10:])
                    self.f.printState(currState[0])
                    print('--')
            # Now have final (predicted) result of action sequence
            # Extract predicted position of current state
            pvx = currState[0,0:15] 
            pvy = currState[0,15:30] 
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
            if verbose:
                a_inds = ",".join([ str(a.max(dim=0)[1].data[0]) for a in a_t  ]) 
                print(i,'->','Lx =',lossx.data[0],', Ly =',lossy.data[0],', H =',H.data[0],', TL =',loss.data[0],', A =',a_inds)
            # Clear the optimizer gradient
            optimizer.zero_grad()
            # Back-prop the errors to get the new gradients
            loss.backward()
            # Print the predicted result at the current iteration
            if extraVerbose:
                print('Predicted End:',pvx.max(0)[1].data[0],pvy.max(0)[1].data[0])
            # Ensure the x_t gradients are cleared
            x_t.grad.data.zero_()
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

 

# trained on 0.1 sigma
# intermixed non-noisy data
# Reached 99.7%
# Next trained on 0.2 sigma
# Get 1.0 on noiseless, 0.99 on noisy
# Trained more on set with additional action noise

########################################################################################################
########################################################################################################
def main():

    ####################################################
    useFFANN = True
    trainingFFANN = False # 1
    manualTest = False # 2
    autoTest = False # 3
    henaffHyperSearch = False # 4
    runHenaffFFANN = False # 5
    ####################################################

    if len(sys.argv) > 1:
        if sys.argv[1] == '1': trainingFFANN = True
        if sys.argv[1] == '2': manualTest = True
        if sys.argv[1] == '3': autoTest = True
        if sys.argv[1] == '4': henaffHyperSearch = True
        if sys.argv[1] == '5': runHenaffFFANN = True

    if useFFANN:

        f_model_name = 'forward-ffann-noisy-wan-1.pt' # 6 gets 99% on 0.1% noise
        exampleEnv = NavigationTask()
        f = ForwardModelFFANN(exampleEnv)

        ################################################################################################################
        if trainingFFANN:
            ############
            ts = "navigation-data-train-single-singularDiscrete.pickle"
            vs = "navigation-data-test-single-singularDiscrete.pickle"
            preload_name = None
            saveName = 'forward-ffann-singDisc-noisy-1.pt'
            ############
            print('Reading Data')
            with open(ts,'rb') as inFile:
                print('\tReading',ts); trainSet = pickle.load(inFile)
            with open(vs,'rb') as inFile:
                print('\tReading',vs); validSet = pickle.load(inFile)
            if not preload_name is None:
                print('Loading from',f_model_name)
                f.load_state_dict( torch.load(f_model_name) )
            f.runTraining(trainSet,validSet,maxIters=50000,modelFilenameToSave=saveName,testEvery=250)



            #f.train(trainSet,validSet,)
            #print('Saving to',saveName)
            #torch.save(f.state_dict(), saveName)

        ################################################################################################################
        elif manualTest:
            ###
            #f_model_name = 'forward-ffann-noisy6.pt'
            ###
            f.load_state_dict( torch.load(f_model_name) )
            start = np.zeros(74, dtype=np.float32)
            start[0+4] = 1
            start[15+6] = 1
            start[15+15+0] = 1
            start[15+15+4+8] = 1
            start[15+15+4+15+7] = 1
            start[15+15+4+15+15+4] = 1.0
            f.test(start)
            print('-----\n','Starting manualTest loop')
            for i in range(5):
                width, height = 15, 15
                p_0 = np.array([npr.randint(0,width),npr.randint(0,height)])
                start_pos = [p_0, r.choice(NavigationTask.oriens)]
                goal_pos = np.array([ npr.randint(0,width), npr.randint(0,height) ])
                checkEnv = NavigationTask(
                    width=width, height=height, agent_start_pos=start_pos, goal_pos=goal_pos,
                    track_history=True, stochasticity=0.0, maxSteps=10)
                s_0 = checkEnv.getStateRep()
                #a1, a2 = np.zeros(10), np.zeros(10)
                #a1[ npr.randint(0,10) ] = 1
                #a2[ npr.randint(0,10) ] = 1
                numActions = 3
                currState = avar( torch.FloatTensor(s_0).unsqueeze(0) )
                print('Start State')
                f.printState( currState[0] )
                actionSet = []
                for j in range(numActions):
                    action = np.zeros( 10 )
                    action[ npr.randint(0,10) ] = 1
                    action += npr.randn( 10 )*0.1
                    action = Utils.softmax( action )
                    print('\tSoft Noisy Action ',j,'=',action)
                    #### Apply Gumbel Softmax ####
                    temperature = 0.01
                    logProbAction = torch.log( avar(torch.FloatTensor(action)) ) 
                    actiong = GumbelSoftmax.gumbel_softmax(logProbAction, temperature)
                    ##############################
                    print('\tGumbel Action ',j,'=',actiong.data.numpy())
                    actionSet.append( actiong )
                    checkEnv.performAction( np.argmax(action) )
                    a = actiong  # avar( torch.FloatTensor(actiong) )
                    currState = f.forward( torch.cat([currState[0],a]).unsqueeze(0) )
                    print("Intermediate State",j)
                    f.printState( currState[0] )
                #checkEnv.performAction(np.argmax(a1))
                #checkEnv.performAction(np.argmax(a2))
                s_1 = checkEnv.getStateRep()
                #inval = np.concatenate( (s_0,a1) )
                #outval1 = f.forward( avar(torch.FloatTensor(inval).unsqueeze(0)) )
                #print(outval1.shape)
                #print(a2.shape)
                #inval2 = np.concatenate( (outval1[0].data.numpy(),a2) )
                #outval2 = f.forward( avar(torch.FloatTensor(inval2).unsqueeze(0)) )
                for action in actionSet:
                    f.printAction(action)
                print('Predicted')
                f.printState( currState[0] )
                print('Actual')
                s1 = avar( torch.FloatTensor( s_1 ).unsqueeze(0) )
                f.printState( s1[0] ) 
                print("Rough accuracy", torch.sum( (currState - s1).pow(2) ).data[0] )
                #print('Predicted',currState.data[0].numpy())
                #print('Actual',s_1)
                #outval1 = f.test(inval,s_1)
                print('----\n')
        
        if autoTest:
            print('Loading from',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )
            # TODO

        ################################################################################################################
        if runHenaffFFANN:
            print('Loading from',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )
            start = np.zeros(64)
            start[0] = 1
            start[15] = 1
            start[15+15] = 1
            start[15+15+4+0] = 1
            start[15+15+4+15+4] = 1
            print(f.env.deconcatenateOneHotStateVector(start))
            print('Building planner')
            planner = HenaffPlanner(f,maxNumActions=2)
            print('Starting generation')
            actions = planner.generatePlan(
                                start,
                                eta=0.1,
                                noiseSigma=0.5,
                                niters=500,
                                goal_state=None,
                                useCE=True,
                                verbose=True,
                                extraVerbose=False,
                                useGumbel=True,
                                temp=0.1,
                                lambda_h=-0.005,
                                useIntDistance=False
                                )
            print('FINAL ACTIONS:', actions)

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

