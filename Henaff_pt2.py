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
    def __init__(self, env, layerSizes=[800,600]):
        super(ForwardModelFFANN, self).__init__()
        self.env = env
        self.actionSize = len( env.actions )
        self.stateSize = len(env.getStateRep(oneHotOutput=True))
        self.inputSize = self.actionSize + self.stateSize
        self.layer1 = nn.Linear(self.inputSize, layerSizes[0])
        self.layer2 = nn.Linear(layerSizes[0], layerSizes[1])
        self.layer3 = nn.Linear(layerSizes[1], self.stateSize)

    def forward(self, inputValue):
        output = F.relu( self.layer1(inputValue) )
        output = F.relu( self.layer2(output) ) # F.sigmoid
        output = self.layer3(output) 
        
        #output1 = avar(torch.FloatTensor(output.shape), requires_grad=True)
        # output1[:,0:15]  = F.softmax(output[:,0:15],dim=1)
        # output1[:,15:30] = F.softmax(output[:,15:30],dim=1)
        # output1[:,30:34] = F.softmax(output[:,30:34],dim=1)
        # output1[:,34:49] = F.softmax(output[:,34:49],dim=1)
        # output1[:,49:64] = F.softmax(output[:,49:64],dim=1)

        v1 = F.softmax(output[:,0:15],dim=1)
        v2 = F.softmax(output[:,15:30],dim=1)
        v3 = F.softmax(output[:,30:34],dim=1)
        v4 = F.softmax(output[:,34:49],dim=1)
        v5 = F.softmax(output[:,49:64],dim=1)

        return torch.cat( ( v1,v2,v3,v4,v5 ), dim=1 ) #output1 #self._statewiseSoftmax( output )

    def noisify(self,data,noiseSigma,wantAdditionalActionNoise=True):
        ds = data.shape
        output = np.zeros(ds)
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        for i in range(ds[0]): 
            currIn = data[i,:]
            # Original action index
            ind_a = np.argmax(currIn[-10:])
            # Add noise to the state and action
            currIn = currIn + npr.randn(ds[1])*noiseSigma
            currIn[0:15]  = softmax(currIn[0:15])
            currIn[15:30] = softmax(currIn[15:30])
            currIn[30:34] = softmax(currIn[30:34])
            currIn[34:49] = softmax(currIn[34:49])
            currIn[49:64] = softmax(currIn[49:64])
            # Softmax the action part (since that will happen in Henaff)
            # Additional action noise that preserves maximal action
            if wantAdditionalActionNoise:
                newval = npr.uniform(0.01, 0.09999, 10) # in [0.02,0.1]
                offset = npr.uniform(0.01, 0.05, 1) # max in ~~~[0.11,0.15]
                currIn[-10:] = newval
                ind_a_new = np.argmax(currIn[-10:])
                currIn[ 64 + ind_a ] = np.max(newval) + offset
                #print(data[i,-10:])
                
#                secondSigma = 0.1
#                currIn[-10:] = currIn[-10:] + secondSigma*npr.randn(10)
#                ind_a_new = np.argmax(currIn[-10:])
#                if ind_a != ind_a_new:
#                if currIn[ ind_a + 64 ] < currIn[ ind_a_new + 64 ]
            a = softmax( currIn[-10:] )
            #print(currIn[-10:])
            #sys.exit(0)
            currIn[-10:] = a
            #if npr.uniform() > 0.99999: print(i,'->','Out:', list(zip(currIn, data[i,:])))
            output[i,:] = currIn
        return output

    def train(self,trainSet,validSet,minibatch_size=200,maxIters=30000,testEvery=250,noiseSigma=0.2,
            noisyDataSetTxLoc=None,f_model_name=None):
        optimizer = optim.Adam(self.parameters(), lr = 0.0000025 * minibatch_size)
        lossf = nn.MSELoss() # nn.L1Loss() # nn.MSELoss() 
        train_x, train_y = trainSet 
        np.set_printoptions(precision=3)
        
        if not noisyDataSetTxLoc is None and os.path.exists(noisyDataSetTxLoc):
            print('Loading noised data (Note this ignores any changes to sigma)')
            with open(noisyDataSetTxLoc,'rb') as fff:
                train_x_noisy = pickle.load(fff)
        else:
            print('Noisifying data')
            train_x_noisy = self.noisify(train_x,noiseSigma)
            if not noisyDataSetTxLoc is None:
                print('Saving noised data to',noisyDataSetTxLoc)
                with open(noisyDataSetTxLoc,'wb') as fff:
                    pickle.dump(train_x_noisy, fff)
        np.set_printoptions()
        train_x = avar( torch.FloatTensor(train_x), requires_grad=False)
        train_x_noisy = avar( torch.FloatTensor(train_x_noisy), requires_grad=False)
        train_y = avar( torch.FloatTensor(train_y), requires_grad=False)
        valid_x, valid_y = validSet 
        valid_x = avar( torch.FloatTensor(valid_x), requires_grad=False)
        valid_y = avar( torch.FloatTensor(valid_y), requires_grad=False)
        ntrain, nvalid = len(train_x), len(valid_x)
        def getRandomMiniBatch(dsx,dsy,mbs,nmax):
            choices = torch.LongTensor( np.random.choice(nmax, size=mbs, replace=False) )
            return dsx[choices], dsy[choices]
        print('Starting training')
        switchTime = 0
        noiselessProb = 0.1
        for i in range(0,maxIters):
            self.zero_grad()
            if i == switchTime: print('Changing to noisy dataset')
            train = train_x_noisy if i > switchTime and npr.uniform() > noiselessProb else train_x
            x, y = getRandomMiniBatch(train,train_y,minibatch_size,ntrain)
            y_hat = self.forward(x)
            loss = lossf(y_hat, y)
            loss.backward()
            optimizer.step()
            if i % testEvery == 0:
                print('Epoch', str(i) + ': L_t =', '%.4f' % loss.data[0], end=', ')
                vx, vy = getRandomMiniBatch(valid_x,valid_y,2000,nvalid)
                predv = self.forward(vx)
                lossv = lossf(predv, vy)
                print('L_v =','%.4f' % lossv.data[0],end=', ')
                acc = self._accuracyBatch(vy,predv)
                print("VACC (noiseless) =",'%.4f' % acc,end=', ')

                tx, ty = getRandomMiniBatch(train_x_noisy,train_y,2000,ntrain)
                predt = self.forward(tx)
                acctn = self._accuracyBatch(ty,predt)
                print("TACC (noisy) =",'%.4f' % acctn)

                if not f_model_name is None:
                    torch.save(self.state_dict(), f_model_name)

    def largeTest(self,datafile,additionalNoiseSigma=0.3):
        pass

    def test(self,x,y=None):
        if not type(x) is avar:
            x = avar( torch.FloatTensor(x) )
            if len(x.shape) <= 1:
                x = torch.unsqueeze(x,0)
        print('Input State')
        s_0 = x[0,0:-10]
        self.printState(s_0,'\t')
        print('Input Action')
        self.printAction(x[0,-10:],'\t')
        print('Predicted Final State')
        yhat = self.forward(x)[0,:]
        self.printState( yhat, '\t' )
        if not y is None:
            if not type(y) is avar:
                y = avar( torch.FloatTensor(y) )
            print('Actual Final State')
            self.printState(y,'\t')
            print('Acc: ', self._accuracySingle(y, yhat))

    def printState(self,s,pre='',p=2): 
        deconRes = self.env.deconcatenateOneHotStateVector(s)
        tt = lambda x: deconRes[x].data.numpy()
        np.set_printoptions(precision=p)
        ll = '%.'+str(p)+'f'
        qqp = lambda k: '['+",".join( map(lambda ss: ll % ss, tt(k)) )+']'
        print(str(pre)+'px:',    np.argmax(tt(0)), ' ', qqp(0) )
        print(str(pre)+'py:',    np.argmax(tt(1)), ' ', qqp(1)  )
        print(str(pre)+'orien:', np.argmax(tt(2)), ' ', qqp(2)  )
        print(str(pre)+'gx:',    np.argmax(tt(3)), ' ', qqp(3)  )
        print(str(pre)+'gy:',    np.argmax(tt(4)), ' ', qqp(4)  )
        np.set_printoptions()

    def printAction(self,a,pre=''):
        ind = np.argmax(a.data.numpy())
        print(str(pre)+'a:',self.env.actions[ind],'('+str(ind)+')')

    def _accuracyBatch(self,ylist,yhatlist):
        # print(ylist.data.shape)
        # print(yhatlist.data.shape)
        # print(ylist[0].shape)
        # sys.exit(0)
        n, acc = ylist.data.shape[0], 0.0 #float(len(ylist)), 0.0
#        for s,l in zip(seqs,labels): 
        for i in range(n):
            acc += self._accuracySingle(ylist[i], yhatlist[i])
        return acc / n

    # Accuracy averaged over subvecs
    def _accuracySingle(self,ys,yshat):
        #seq = torch.cat(seq).view(len(seq), 1, -1) # [seqlen x batchlen x hidden_size]
        prediction = yshat #self.forward(seq) # Only retrieves final time state
        label = ys
        predVec = self.env.deconcatenateOneHotStateVector(prediction)
        labelVec = self.env.deconcatenateOneHotStateVector(label)
        locAcc = 0.0
        for pv, lv in zip(predVec, labelVec):
            _, ind_pred = pv.max(0)
            _, ind_label = lv.max(0) #np.argmax(lv)
            #print('ip',ind_pred)
            #print('il',ind_label)
            locAcc += 1.0 if ind_pred.data[0] == ind_label.data[0] else 0.0
        return locAcc / len(predVec)
    
    def stateSoftmax(self,s):
        decon = self.env.deconcatenateOneHotStateVector(s)
        varr = []
        for v in decon:
            vs = F.softmax(v,dim=0)
            varr.append(vs)
        return torch.cat(varr)

########################################################################################################

class ForwardModelLSTM(nn.Module):
    ''' Currently has a bug. Probably. '''

    def __init__(self, inputSize, stateSize, h_size=100, nlayers=1):
        super(ForwardModelLSTM, self).__init__()
        # Input dimensions are (seq_len, batch, input_size)
        self.hdim, self.stateSize, self.nlayers, self.inputSize, self.actionSize = h_size, stateSize, nlayers, inputSize, inputSize - stateSize
        # The lstm: outputs (1) all the hidden states and (2) the most recent hidden state
        self.lstm = nn.LSTM(input_size=inputSize, hidden_size=self.hdim, num_layers=nlayers)
        # Linear (FC) layer for final output transformation
        self.hiddenToState = nn.Linear(self.hdim, stateSize)
        # Initialize LSTM state variables
        self.reInitialize()

    def reInitialize(self):
        # Size = (num_layers, minibatch_size, hidden_dim)
        self.hidden = (avar(torch.zeros(self.nlayers,1,self.hdim)), 
            avar(torch.zeros(self.nlayers,1,self.hdim)))

    # Only retrieves the last (final) result state
    def forward(self, stateSeq):
        lstm_out, self.hidden = self.lstm( stateSeq, self.hidden )
        return self.hiddenToState( lstm_out[-1,0,:] ) # Only run on last output

    def train(self, trainSeq, validSeq, nEpochs=800, epochLen=175, validateEvery=25, vbs=500, printEvery=5, noiseSigma=0.4):
        print('-- Starting Training (nE=' + str(nEpochs) + ',eL=' + str(epochLen) + ') --')
        optimizer = optim.Adam(self.parameters(), lr = 0.03 * epochLen / 150.0)
        ns, na, tenv = self.stateSize, self.actionSize, trainSeq.env
        for epoch in range(nEpochs):
            if epoch % printEvery == 0: print('Epoch:',epoch, end='')
            loss = 0.0
            self.zero_grad() # Zero out gradients
            for i in range(epochLen):
                self.reInitialize() # Reset LSTM hidden state
                seq,label = trainSeq.randomTrainingPair() # Current value
                seq = [ s + npr.randn(len(s))*noiseSigma for s in seq ]
                seq = [ avar(torch.from_numpy(s).float()) for s in seq] 
                seq = [ torch.cat([self.stateSoftmax(sa[0:ns], tenv), F.softmax(sa[ns:ns+na])]) for sa in seq ]
                seqn = torch.cat(seq).view(len(seq), 1, -1) # [seqlen x batchlen x featureLen]
                prediction = self.forward(seqn)#[-1,:]
                label = avar(torch.from_numpy(label).float())
                loss += self._lossFunction(prediction, label, env=tenv)
            loss.backward()
            optimizer.step()
            if epoch % printEvery == 0: print(" -> AvgLoss",str(loss.data[0] / epochLen))
            if epoch % validateEvery == 0:
                bdata,blabels,bseqlen = validSeq.next(vbs,nopad=True)
                acc1, _ = self._accuracyBatch(bdata,blabels,validSeq.env)
                bdata,blabels,bseqlen = trainSeq.next(vbs,nopad=True)
                acc2, _ = self._accuracyBatch(bdata,blabels,tenv)
                print('\tCurrent Training Acc (est) =', acc1)
                print('\tCurrent Validation Acc (est) =', acc2)
        # Check training & final validation accuracy
        print('----')
        nmax = 5000 # Num from total to check at the end
        totalTrainAcc,nt = self._accuracyBatch(trainSeq.unpaddedData()[0:nmax],trainSeq.labels[0:nmax],trainSeq.env)
        print('Final Train Acc ('+str(nt)+'):',totalTrainAcc)
        totalValidAcc,nv = self._accuracyBatch(validSeq.unpaddedData()[0:nmax],validSeq.labels[0:nmax],validSeq.env)
        print('Final Validation Acc ('+str(nv)+'):',totalValidAcc)

    def _accuracyBatch(self,seqs,labels,env):
        n, acc = float(len(seqs)), 0.0
        for s,l in zip(seqs,labels): acc += self._accuracySingle(s,l,env)
        return acc / n, int(n)

    # Accuracy averaged over subvecs
    def _accuracySingle(self,seq,label,env):
        seq = [avar(torch.from_numpy(s).float()) for s in seq] 
        seq = torch.cat(seq).view(len(seq), 1, -1) # [seqlen x batchlen x hidden_size]
        self.reInitialize() # Reset LSTM hidden state
        prediction = self.forward(seq) # Only retrieves final time state
        predVec = env.deconcatenateOneHotStateVector(prediction)
        labelVec = env.deconcatenateOneHotStateVector(label)
        locAcc = 0.0
        for pv, lv in zip(predVec, labelVec):
            _, ind_pred = pv.max(0)
            ind_label = np.argmax(lv)
            locAcc += 1.0 if ind_pred.data[0] == ind_label else 0.0
        return locAcc / len(predVec)

    def _lossFunction(self,outputs,targets,useMSE=False,env=None):
        if useMSE:
            loss = nn.MSELoss()
            return loss(outputs,targets)
        else: # Use Cross-entropy
            loss = nn.CrossEntropyLoss()
            cost = avar( torch.FloatTensor( [0] ) )
            predVec = env.deconcatenateOneHotStateVector(outputs)
            labelVec = env.deconcatenateOneHotStateVector(targets)
            for pv,lv in zip(predVec,labelVec):
                val,ind = lv.max(0)
                cost += loss(pv.view(1,len(pv)), ind)
            return cost / len(predVec)

    def stateSoftmax(self,s,env):
        decon = env.deconcatenateOneHotStateVector(s)
        varr = []
        for v in decon:
            vs = F.softmax(v,dim=0)
            varr.append(vs)
        return torch.cat(varr)

########################################################################################################
########################################################################################################

class HenaffPlanner():

    def __init__(self,forward_model,maxNumActions=1,noiseSigma=0.2,startNoiseSigma=0.1,niters=200):
        # Parameters
        self.sigma = noiseSigma
        self.start_sigma = startNoiseSigma
        self.nacts = maxNumActions
        self.niters = niters
        # Stop forward model from training
        self.f = forward_model
        #for p in self.f.parameters(): p.requires_grad = False
        # Get shapes from forward model
        self.state_size = self.f.stateSize
        self.action_size = self.f.inputSize - self.f.stateSize

    def generatePlan(self,start_state,eta=0.1,niters=None,goal_state=None,useCE=True):
        x_t = avar( torch.randn(self.nacts, self.action_size) * self.start_sigma, requires_grad=True )
        deconStartState = self.f.env.deconcatenateOneHotStateVector(start_state)
        if useCE:
            lossf = nn.CrossEntropyLoss()
        else:
            lossf = nn.MSELoss()
        if goal_state is None:
            gx, gy = avar(torch.FloatTensor(deconStartState[-2])), avar(torch.FloatTensor(deconStartState[-1]))
        else: 
            sys.exit(0)
        _, sindx = avar(torch.FloatTensor(deconStartState[0])).max(0)
        _, sindy = avar(torch.FloatTensor(deconStartState[1])).max(0)
        _, indx = gx.max(0)
        _, indy = gy.max(0)
        niters = self.niters if niters is None else niters
        for i in range(niters):
            # Generate soft action sequence
            epsilon = avar( torch.randn(self.nacts, self.action_size) * self.sigma )
            y_t = x_t + epsilon
            a_t = F.softmax( y_t, dim=1 )
            # Compute predicted state
            currState = avar(torch.FloatTensor(start_state))
            for k in range(0,self.nacts):
                action = a_t[k,:]
                #currState = self.f.stateSoftmax(currState)
                currInput = torch.cat([currState,action], 0)
                currInput = torch.unsqueeze(currInput,0)
                self.f.printState(currInput[0,0:64])
                self.f.printAction(currInput[0,-10:])
                currState = self.f.forward( currInput )
                self.f.printState(currState[0])
                print('--')
                #sys.exit(0)
                # currInput = currInput.view(1, 1, -1) # [seqlen x batchlen x feat_size]
            # Compute loss
            predFinal = self.f.env.deconcatenateOneHotStateVector( self.f.stateSoftmax(currState[0]) )
            pvx = predFinal[0]
            pvy = predFinal[1]
            #
            if useCE:
                lossx = lossf(pvx.view(1,len(pvx)), indx) 
                lossy = lossf(pvy.view(1,len(pvy)), indy)
            else:
                lossx = lossf(pvx, gx)
                lossy = lossf(pvy, gy)
            # Entropy penalty
            H = -torch.sum( torch.sum( a_t*torch.log(a_t) , dim = 1 ) )
            lambda_H = 0.1
            print('Entropy',H)
            #for i in range(0,self.nacts):
            #    H += a_t
            # Loss function
            loss = lossx + lossy + lambda_H * H
            #
            print(i, '-> L =', lossx.data[0],' + ',lossy.data[0])
            # print(indx.data[0],indy.data[0],end='  ###  ')
            # print( pvx.max(0)[1].data[0], pvy.max(0)[1].data[0] )
            # print('--')
            loss.backward()
            x_t.data -= eta * x_t.grad.data
            # print('g_t',x_t.grad.data)
            # print('x_t',x_t.data)
            print('Predicted End:',pvx.max(0)[1].data[0],pvy.max(0)[1].data[0])
            x_t.grad.data.zero_()

        print('\nEnd\n')
#        print(F.softmax( x_t, dim=1 ))
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

    def generatePlanOld(self,start_state,env,eta=0.05,niters=None):
        x_t = avar( torch.randn(self.nacts,self.action_size) * self.start_sigma, requires_grad=True )
        deconStartState = env.deconcatenateOneHotStateVector(start_state)
        lossf = nn.CrossEntropyLoss()
        gx, gy = avar(torch.FloatTensor(deconStartState[-2])), avar(torch.FloatTensor(deconStartState[-1]))
        _,sindx = avar(torch.FloatTensor(deconStartState[0])).max(0)
        _,sindy = avar(torch.FloatTensor(deconStartState[1])).max(0)
        _,indx = gx.max(0)
        _,indy = gy.max(0)
        niters = self.niters if niters is None else niters
        for i in range(niters):
            # Generate soft action sequence
            epsilon = avar( torch.randn(self.nacts, self.action_size) * self.sigma )
            y_t = x_t + epsilon
            a_t = F.softmax( y_t, dim=1 )
            # Compute predicted state
            self.f.reInitialize() # Reset LSTM hidden state
            currState = avar(torch.FloatTensor(start_state))
            for k in range(0,self.nacts):
                action = a_t[k,:]
                currState = self.f.stateSoftmax(currState,env)
                currInput = torch.cat([currState,action],0)
                currInput = currInput.view(1, 1, -1) # [seqlen x batchlen x feat_size]
                lstm_out, self.f.hidden = self.f.lstm( currInput, self.f.hidden )
                currState = self.f.hiddenToState( lstm_out[-1,0,:] ) # [seqlen x batchlen x hidden_size]
            # Compute loss
            predFinal = env.deconcatenateOneHotStateVector( self.f.stateSoftmax(currState,env) )
            pvx = predFinal[0]
            pvy = predFinal[1]
            #
            lossx = lossf(pvx.view(1,len(pvx)), indx) 
            lossy = lossf(pvy.view(1,len(pvy)), indy)
            loss = lossx + lossy
            #
            print(i, '-> L =', lossx.data[0],' + ',lossy.data[0])
            print(indx.data[0],indy.data[0],end='  ###  ')
            print( pvx.max(0)[1].data[0], pvy.max(0)[1].data[0] )
            print('--')
            loss.backward()
            x_t.data -= eta * x_t.grad.data
            print('g_t',x_t.grad.data)
            print('x_t',x_t.data)
            print('Predicted End:',pvx.max(0)[1].data[0],pvy.max(0)[1].data[0])
            x_t.grad.data.zero_()

        print('\nEnd\n')
        print(F.softmax( x_t, dim=1 ))
        for k in range(0,self.nacts):
            action = x_t[k,:]
            print(action.max(0)[1].data[0],end=' -> ')
            print(NavigationTask.actions[action.max(0)[1].data[0]])
        print('--')
        print('START ',sindx.data[0],sindy.data[0])
        print('TARGET END ',indx.data[0],indy.data[0])
        print('--')

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
    trainingLSTM = False
    overwrite = False
    runHenaff = False
    testFM = False
    ###
    useFFANN = True
    trainingFFANN = False
    manualTest = False
    autoTest = False
    runHenaffFFANN = True
    ####################################################
    if useFFANN:
        f_model_name = 'forward-ffann-noisy-wan-1.pt' # 6 gets 99% on 0.1% noise
        exampleEnv = NavigationTask()
        f = ForwardModelFFANN(exampleEnv)

        if trainingFFANN:
            ############
            ts = "navigation-data-train-single-small.pickle"
            vs = "navigation-data-test-single-small.pickle"
            tsx_noisy = "noisier-actNoise-navigation-data-single.pickle"
            preload_name = f_model_name
            saveName = 'forward-ffann-noisy-wan-2.pt'
            ############
            print('Reading Data')
            with open(ts,'rb') as inFile:
                print('\tReading',ts); trainSet = pickle.load(inFile)
            with open(vs,'rb') as inFile:
                print('\tReading',vs); validSet = pickle.load(inFile)
            if not preload_name is None:
                print('Loading from',f_model_name)
                f.load_state_dict( torch.load(f_model_name) )
            f.train(trainSet,validSet,noisyDataSetTxLoc=tsx_noisy,f_model_name=saveName)
            print('Saving to',saveName)
            torch.save(f.state_dict(), saveName)

        elif manualTest:
            ###
            f_model_name = 'forward-ffann-noisy6.pt'
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
            for i in range(10):
                width, height = 15, 15
                p_0 = np.array([npr.randint(0,width),npr.randint(0,height)])
                start_pos = [p_0, r.choice(NavigationTask.oriens)]
                goal_pos = np.array([ npr.randint(0,width), npr.randint(0,height) ])
                checkEnv = NavigationTask(
                    width=width, height=height, agent_start_pos=start_pos, goal_pos=goal_pos,
                    track_history=True, stochasticity=0.0, maxSteps=10)
                s_0 = checkEnv.getStateRep()
                a1, a2 = np.zeros(10), np.zeros(10)
                a1[ npr.randint(0,10) ] = 1
                a2[ npr.randint(0,10) ] = 1

                checkEnv.performAction(np.argmax(a1))
                s_1 = checkEnv.getStateRep()

                inval = np.concatenate( (s_0,a1) )
                outval1 = f.test(inval,s_1)
                print('----')
        if autoTest:
            print('Loading from',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )


        if runHenaffFFANN:
            print('Loading from',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )
            start = np.zeros(64)
            start[0] = 1
            start[15] = 1
            start[15+15] = 1
            start[15+15+4+0] = 1
            start[15+15+4+15+2] = 1
            print(f.env.deconcatenateOneHotStateVector(start))
            #sys.exit(0)
            print('Building planner')
            planner = HenaffPlanner(f,maxNumActions=1)
            print('Starting generation')
            planner.generatePlan(start,niters=500)

    else:
        f_model_name = 'forward-lstm-stochastic.pt'    
        s = 'navigation' # 'transport'
        trainf, validf = s + "-data-train-small.pickle", s + "-data-test-small.pickle"
        print('Reading Data')
        train, valid = SeqData(trainf), SeqData(validf)
        f = ForwardModelLSTM(train.lenOfInput,train.lenOfState)
        if trainingLSTM:
            if os.path.exists(f_model_name) and not overwrite:
                print('Loading from',f_model_name)
                f.load_state_dict( torch.load(f_model_name) )
            else:
                f.train(train,valid)
                print('Saving to',f_model_name)
                torch.save(f.state_dict(), f_model_name)
            print('Q-test')
            bdata, blabels, _ = valid.next(2000, nopad=True)
            acc1, _ = f._accuracyBatch(bdata,blabels,valid.env)
            print(acc1)
        if runHenaff:
            print('Loading from',f_model_name)
            f.load_state_dict( torch.load(f_model_name) )
    #        seq,label = train.randomTrainingPair()
    #        start = seq[0][0:64]
     #       start[63] = 0
     #       start[63-15] = 0
     #       start[15+15+4+5] = 1
     #       start[15+15+4+15+5] = 1
     #       start
            start = np.zeros(64)
            start[0] = 1
            start[15] = 1
            start[15+15] = 1
            start[15+15+4+0] = 1
            start[15+15+4+15+2] = 1
            print(train.env.deconcatenateOneHotStateVector(start))
            #sys.exit(0)
            print('Building planner')
            planner = HenaffPlanner(f)
            print('Starting generation')
            planner.generatePlan(start,train.env,niters=150)
        if testFM:
            f.load_state_dict( torch.load(f_model_name) )
            start = np.zeros(64)
            start[0+2] = 1
            start[15+3] = 1
            start[15+15+0] = 1
            start[15+15+4+5] = 1
            start[15+15+4+15+5] = 1
            action = np.zeros(10)
            deconRes = train.env.deconcatenateOneHotStateVector(start)
            print('Start state')
            print('px',    np.argmax(deconRes[0]) )
            print('py',    np.argmax(deconRes[1]) )
            print('orien', np.argmax(deconRes[2]) )
            print('gx',    np.argmax(deconRes[3]) )
            print('gy',    np.argmax(deconRes[4]) )
            action[5] = 1.0
            stateAction = [torch.cat([(torch.FloatTensor(start)), (torch.FloatTensor(action))])]
            #print('SA:',stateAction)
            #print('Start State')
            #printState( stateAction[0][0:-10], train.env )
            print('Action',NavigationTask.actions[np.argmax( action )])
            f.reInitialize()
            seq = avar(torch.cat(stateAction).view(len(stateAction), 1, -1)) # [seqlen x batchlen x hidden_size]
            result = f.forward(seq)
            print('PredState')
            printState( result, train.env )
            #deconRes = train.env.deconcatenateOneHotStateVector(result)
        

def printState(s,env): 
        deconRes = env.deconcatenateOneHotStateVector(s)
        print('px',    np.argmax(deconRes[0].data.numpy()) )
        print('py',    np.argmax(deconRes[1].data.numpy()) )
        print('orien', np.argmax(deconRes[2].data.numpy()) )
        print('gx',    np.argmax(deconRes[3].data.numpy()) )
        print('gy',    np.argmax(deconRes[4].data.numpy()) )

if __name__ == '__main__':
    main()


