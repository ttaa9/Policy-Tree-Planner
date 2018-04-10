import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar
    
from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData



import os, sys, pickle, numpy as np, numpy.random as npr, random as r


class GreedyValuePredictor(nn.Module):
    def __init__(self,  env, layerSizes=[100,100]):
        super(GreedyValuePredictor, self).__init__()
        self.stateSize = len(env.getStateRep(oneHotOutput=True))
        self.env = env
        self.rewardSize = 1
        print("State Size: " , self.stateSize)        
        # Input space: [Batch, observations], output:[Batch, Reward]
        self.layer1 = nn.Linear(self.stateSize, layerSizes[0])
        self.layer2 = nn.Linear(layerSizes[0], layerSizes[1])
        self.layer3 = nn.Linear(layerSizes[1], self.rewardSize)
        
    def forward(self,state):
        output = F.relu( self.layer1(state) )
        output = F.relu( self.layer2(output) ) # F.sigmoid
        output = self.layer3(output)
        #print(output.shape)
        m = nn.Sigmoid()
        output = m(output)
        return output
    
    def train(self, trainSet, validSet, nEpochs=1500, batch_size=200, validateEvery=200, vbs=500, printEvery=200):
        optimizer = optim.Adam(self.parameters(), lr = 0.0003)
        state_size = self.stateSize
        lossFunction = nn.BCELoss()
        
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
        for epoch in range(nEpochs):
            if epoch % printEvery == 0: print('Epoch:',epoch, end='')
            loss = 0.0
            self.zero_grad() # Zero out gradients
            batch_x, batch_y = getRandomMiniBatch(train_x,train_y,batch_size,ntrain)
            prediction = self.forward(batch_x) #[-1,:]
            label = batch_y.unsqueeze(dim=1)
            #print(label.shape, prediction.shape)
            loss = lossFunction(prediction, label)
            loss.backward()
            optimizer.step()
            if epoch % printEvery == 0: print(" -> AvgLoss",str(loss.data[0]/ batch_size))
            if epoch % validateEvery == 0:
                batch_vx, batch_vy = getRandomMiniBatch(valid_x,valid_y,batch_size,nvalid)
                predv = self.forward(batch_vx) #[-1,:]
                vy = batch_vy.unsqueeze(dim=1)
                acc = self._accuracyBatch(vy,predv)
                print("VACC (noiseless) =",'%.4f' % acc,end=', ')
                print('/n')
                
    def _accuracyBatch(self,ylist,yhatlist):
        n, acc = ylist.data.shape[0], 0.0 
        for i in range(n):
            acc += self._accuracySingle(ylist[i], yhatlist[i])
        return acc / n

    # Accuracy averaged over subvecs
    def _accuracySingle(self,label,prediction):
        #print(label.data[0],prediction.data[0])
        if label.data[0] == 1.0:
            locAcc = 1.0 if prediction.data[0] > 0.5 else 0.0
        else:
            locAcc = 1.0 if prediction.data[0] < 0.5 else 0.0
        return locAcc 


def main():
    ts = "navigation-data-state_to_reward-train.pickle"
    vs = "navigation-data-state_to_reward-valid.pickle"
    ############
    print('Reading Data')
    with open(ts,'rb') as inFile:
        print('\tReading',ts); trainSet = pickle.load(inFile)
    with open(vs,'rb') as inFile:
        print('\tReading',vs); validSet = pickle.load(inFile)
    env = NavigationTask()
    greedyvp = GreedyValuePredictor(env)
    greedyvp.train( trainSet, validSet)
    def generateTask(px,py,orien,gx,gy):
        direction = NavigationTask.oriens[orien]
        gs = np.array([gx, gy])
        env = NavigationTask(agent_start_pos=[np.array([px,py]), direction],goal_pos=gs)
        return env
    env = generateTask(0,1,2,3,2)
    state = avar( torch.FloatTensor(env.getStateRep()), requires_grad=False).view(1,-1)
    print(state.shape)
    greedyvp.forward(state).data.numpy()
    torch.save(greedyvp.state_dict(), "greedy_value_predictor")

############################
if __name__ == '__main__':
    main()
############################