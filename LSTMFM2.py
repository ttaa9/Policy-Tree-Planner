import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar

from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData

import os, sys, pickle, numpy as np, numpy.random as npr, random as r


#Pytorch LSTM input: Sequence * Batch * Input

class LSTMForwardModel(nn.Module):
    
    def __init__(self, inputSize, stateSize, h_size=400, nlayers=1):
        super(LSTMForwardModel, self).__init__()
        self.hdim, self.stateSize, self.nlayers, self.inputSize, self.actionSize = h_size, stateSize, nlayers, inputSize, inputSize - stateSize
        self.lstm = nn.LSTM(input_size=inputSize, hidden_size=self.hdim, num_layers=nlayers)
        self.hiddenToState = nn.Linear(self.hdim, stateSize)
        self.reInitialize(1)
        
    def reInitialize(self, batch_size):
        # Size = (num_layers, minibatch_size, hidden_dim)
        self.hidden = (avar(torch.zeros(self.nlayers,batch_size,self.hdim)), avar(torch.zeros(self.nlayers,batch_size,self.hdim)))
    
    def setHiddenState(self, hidden):
        self.hidden = hidden
    
    def forward(self, initial_state, actions, seqn):
        #initalState [1*1*state_size] actions[batch*noOfActions*Action_size] 
        #print(actions[0].shape)
        #print(seqn)
        int_states = []
        current_state = initial_state
        #print(current_state.shape)
        #print(torch.cat((current_state, actions[0]),0))
        for i in range(seqn):
            concat_vec = torch.cat((current_state, actions[i]),0).view(1,1,-1)
            lstm_out, self.hidden = self.lstm(concat_vec, self.hidden)
            output_state = self.hiddenToState(lstm_out[0,0,:])
            int_states.append(output_state)
            current_state = output_state
            
        return current_state, int_states, self.hidden
    
    def train(self, trainSeq, validSeq, nEpochs=1500, epochLen=500, validateEvery=20, vbs=500, printEvery=5, noiseSigma=0.4):
        optimizer = optim.Adam(self.parameters(), lr = 0.003)
        state_size, action_size, tenv = self.stateSize, self.actionSize, trainSeq.env
        for epoch in range(nEpochs):
            if epoch % printEvery == 0: print('Epoch:',epoch, end='')
            loss = 0.0
            self.zero_grad() # Zero out gradients
            for i in range(epochLen):
                self.reInitialize(1) # Reset LSTM hidden state
                seq,label = trainSeq.randomTrainingPair() # Current value
                actions = [ s[64:74]  for s in seq ]
                actions = [ avar(torch.from_numpy(s).float()) for s in actions] 
                intial_state = seq[0][0:64]
                seqn = len(seq)
                prediction, _ = self.forward(intial_state,actions,seqn) #[-1,:]
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
    
    def _lossFunction(self,outputs,targets,useMSE=True,env=None):
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
        
    def _accuracyBatch(self,seqs,labels,env):
        n, acc = float(len(seqs)), 0.0
        #print(len(seq))
        for s,l in zip(seqs,labels): acc += self._accuracySingle(s,l,env)
        return acc / n, int(n)

    # Accuracy averaged over subvecs
    def _accuracySingle(self,seq,label,env):
        seq = [avar(torch.from_numpy(s).float()) for s in seq] 
        seq = torch.cat(seq).view(len(seq), 1, -1) # [seqlen x batchlen x hidden_size]
        self.reInitialize(1) # Reset LSTM hidden state
        #print(seq.shape)
        actions = [ s[0][64:74]  for s in seq ]
        #actions = [ avar(torch.from_numpy(s).float()) for s in actions] 
        intial_state = seq[0][0][0:64].data.numpy()
        seqn = len(seq)
        prediction, _ = self.forward(intial_state,actions,seqn) #[-1,:]
        #prediction = self.forward(seq) # Only retrieves final time state
        predVec = env.deconcatenateOneHotStateVector(prediction)
        labelVec = env.deconcatenateOneHotStateVector(label)
        locAcc = 0.0
        for pv, lv in zip(predVec, labelVec):
            _, ind_pred = pv.max(0)
            ind_label = np.argmax(lv)
            locAcc += 1.0 if ind_pred.data[0] == ind_label else 0.0
        return locAcc / len(predVec)

def main():
    f_model_name = 'forward-lstm-stochastic.pt'    
    s = 'navigation' # 'transport'
    trainf, validf = s + "-data-train-small.pickle", s + "-data-test-small.pickle"
    print('Reading Data')
    train, valid = SeqData(trainf), SeqData(validf)
    fm = LSTMForwardModel(train.lenOfInput,train.lenOfState)
    fm.train(train, valid)
    torch.save(fm.state_dict(), f_model_name)

if __name__ == '__main__':
    main()