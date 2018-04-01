import torch, torch.autograd as autograd, numpy as np, numpy.random as npr
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, os, sys
from torch.autograd import Variable as avar

from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData

########################################################################################################
########################################################################################################

class ForwardModel(nn.Module):

    def __init__(self, inputSize, stateSize, h_size=100, nlayers=1):
        super(ForwardModel, self).__init__()
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
        print('-- Starting Training (nE='+str(nEpochs)+',eL='+str(epochLen)+') --')
        optimizer = optim.Adam(self.parameters(), lr = 0.03 * epochLen / 150.0)
        ns, na, tenv = self.stateSize, self.actionSize, trainSeq.env
        for epoch in range(nEpochs):
            if epoch % printEvery == 0: print('Epoch:',epoch, end='')
            loss = 0.0
            self.zero_grad() # Zero out gradients
            for i in range(epochLen):
                self.reInitialize() # Reset LSTM hidden state
                seq,label = trainSeq.randomTrainingPair() # Current value
                #print(seq)
                
                seq = [ s + npr.randn(len(s))*noiseSigma for s in seq ]
                #print(seq)
                #self.stateSoftmax( 
                seq = [ avar(torch.from_numpy(s).float()) for s in seq] 

                seq = [ torch.cat([self.stateSoftmax(sa[0:ns], tenv), F.softmax(sa[ns:ns+na])]) for sa in seq ]
                # print(seq)
                # print(torch.sum(seq[0][0:15]))
                # print(torch.sum(seq[0][15:30]))
                # print(torch.sum(seq[0][30:34]))
                # print(torch.sum(seq[0][34:34+15]))
                # print(torch.sum(seq[0][34+15:34+15+15]))
                # print(torch.sum(seq[0][34+15+15:34+15+15+10]))                
                # sys.exit(0)
                seqn = torch.cat(seq).view(len(seq), 1, -1) # [seqlen x batchlen x featureLen]
                #print(seq)
                #sys.exit(0)
                # Add noise
                # epsilon = avar( torch.randn(len(seq), 1, seq.shape[-2]) * noiseSigma )
                #seqn = seq + epsilon
                prediction = self.forward(seqn)#[-1,:]
                # Compute loss
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
                #print(action)
                #print(sum(action))
                currState = self.f.stateSoftmax(currState,env)
                currInput = torch.cat([currState,action],0)
                currInput = currInput.view(1, 1, -1) # [seqlen x batchlen x feat_size]
                lstm_out, self.f.hidden = self.f.lstm( currInput, self.f.hidden )
                currState = self.f.hiddenToState( lstm_out[-1,0,:] ) # [seqlen x batchlen x hidden_size]
            # Compute loss
            predFinal = env.deconcatenateOneHotStateVector( self.f.stateSoftmax(currState,env) )
            # print('goal',gx,gy)
#            loss = ( predFinal[0] - gx ).pow(2).sum() + ( predFinal[1] - gy ).pow(2).sum()
            pvx = predFinal[0]
            pvy = predFinal[1]
            #
            lossx = lossf(pvx.view(1,len(pvx)), indx) 
            lossy = lossf(pvy.view(1,len(pvy)), indy)
            loss = lossx + lossy
            #
            #loss = (pvx.max(0)[1].type(torch.FloatTensor) - indx.type(torch.FloatTensor))**2 + (pvy.max(0)[1].type(torch.FloatTensor) - indy.type(torch.FloatTensor))**2
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
            #print(x_t)
            #seq,label = trainSeq.randomTrainingPair() # Current value
            #seq = [avar(torch.from_numpy(s).float()) for s in seq] 
            #seq = torch.cat(seq).view(len(seq), 1, -1) 
            #prediction = self.f(seq)
            ##
            #lstm_out, self.hidden = self.lstm( stateSeq, self.hidden )
            #return self.hiddenToState( lstm_out[-1,0,:] ) # Only run on last output
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
    training = False
    overwrite = False
    runHenaff = False
    testFM = True
    ####################################################
    f_model_name = 'forward-lstm-stochastic.pt'    
    s = 'navigation' # 'transport'
    trainf, validf = s + "-data-train-small.pickle", s + "-data-test-small.pickle"
    print('Reading Data')
    train, valid = SeqData(trainf), SeqData(validf)
    f = ForwardModel(train.lenOfInput,train.lenOfState)
    if training:
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


