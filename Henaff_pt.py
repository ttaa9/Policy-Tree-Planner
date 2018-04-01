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
    def __init__(self, env, layerSizes=[2560,256]):
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
        output = F.sigmoid( self.layer2(output) )
        output = self.layer3(output) 
        return output #self._statewiseSoftmax( output )

    # Assume s is [batchsize x statesize]

    #def _statewiseSoftmax(self,s):
        # for i in range(slen):
        #     decon = self.env.deconcatenateOneHotStateVector(s)
        #     varr = []
        #     for v in decon:
        #         vs = F.softmax(v,dim=0)
        #         varr.append(vs)
        #     return torch.cat(varr)

    def train(self,trainSet,validSet,minibatch_size=120,maxIters=4000,testEvery=150):
        optimizer = optim.Adam(self.parameters(), lr = 0.000002 * minibatch_size)
        lossf = nn.MSELoss() # nn.L1Loss() # nn.MSELoss() 
        train_x, train_y = trainSet 
        train_x = avar( torch.FloatTensor(train_x), requires_grad=False)
        train_y = avar( torch.FloatTensor(train_y), requires_grad=False)
        valid_x, valid_y = validSet 
        valid_x = avar( torch.FloatTensor(valid_x), requires_grad=False)
        valid_y = avar( torch.FloatTensor(valid_y), requires_grad=False)
        ntrain, nvalid = len(train_x), len(valid_x)
        def getRandomMiniBatch(dsx,dsy,mbs,nmax):
            choices = torch.LongTensor( np.random.choice(nmax, size=mbs, replace=False) )
            return dsx[choices], dsy[choices]
        print('Starting training')
        for i in range(0,maxIters):
            self.zero_grad()
            x, y = getRandomMiniBatch(train_x,train_y,minibatch_size,ntrain)
            y_hat = self.forward(x)
            loss = lossf(y_hat, y)
            #print(i,loss)
            loss.backward()
            optimizer.step()
            if i % testEvery == 0:
                print('Epoch', str(i) + ': L_t =', '%.4f' % loss.data[0], end=', ')
                vx, vy = getRandomMiniBatch(valid_x,valid_y,2000,nvalid)
                predv = self.forward(vx)
                lossv = lossf(predv, vy)
                print('L_v =','%.4f' % lossv.data[0],end=', ')
                acc = self._accuracyBatch(vy,predv)
                print("VACC =",'%.4f' % acc)

    def test(self,x,y=None):
        if not type(x) is avar:
            x = avar( torch.FloatTensor(x) )
        print('Input State')
        s_0 = x[0:-10]
        self.printState(s_0,'\t')
        print('Input Action')
        self.printAction(x[-10:],'\t')
        print('Predicted Final State')
        yhat = self.forward(x)
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

    def __init__(self,forward_model,maxNumActions=1,noiseSigma=0.005,startNoiseSigma=0.1,niters=200):
        # Parameters
        self.sigma = noiseSigma
        self.start_sigma = startNoiseSigma
        self.nacts = maxNumActions
        self.niters = niters
        # Stop forward model from training
        self.f = forward_model
        for p in self.f.parameters(): p.requires_grad = False
        # Get shapes from forward model
        self.state_size = self.f.stateSize
        self.action_size = self.f.inputSize - self.f.stateSize

    def generatePlan(self,start_state,env,eta=0.05,niters=None):
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
    ####################################################
    if useFFANN:
        f_model_name = 'forward-ffann-stochastic.pt'
        exampleEnv = NavigationTask()
        f = ForwardModelFFANN(exampleEnv)
        if trainingFFANN:
            ts = "navigation-data-train-single-small.pickle"
            vs = "navigation-data-test-single-small.pickle"
            print('Reading Data')
            with open(ts,'rb') as inFile:
                print('\tReading',ts); trainSet = pickle.load(inFile)
            with open(vs,'rb') as inFile:
                print('\tReading',vs); validSet = pickle.load(inFile)
            
            f.train(trainSet,validSet)
            print('Saving to',f_model_name)
            torch.save(f.state_dict(), f_model_name)
        else:
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
                a = np.zeros(10)
                a[ npr.randint(0,10) ] = 1
                inval = np.concatenate( (s_0,a) )
                checkEnv.performAction(np.argmax(a))
                s_1 = checkEnv.getStateRep()
                f.test(inval,s_1)
                print('----')
            

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


