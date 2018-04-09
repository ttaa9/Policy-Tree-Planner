import pickle
import pdb

import torch, torch.autograd as autograd
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as avar
    
from SimpleTask import SimpleGridTask
from TransportTask import TransportTask
from NavTask import NavigationTask
from SeqData import SeqData
from LSTMFM import LSTMForwardModel

import os, sys, pickle, numpy as np, numpy.random as npr, random as r

from Henaffs_Method_2 import *


filename=['hyperparam_search_henaff_temp-0.02.pickle',
			'hyperparam_search_henaff_temp-0.1.pickle',
			'hyperparam_search_henaff_temp-10.pickle']

with open(filename[0],'rb') as f: 
	henaff_hyper_1=pickle.load(f)

with open(filename[1],'rb') as f: 
	henaff_hyper_2=pickle.load(f)

with open(filename[2],'rb') as f: 
	henaff_hyper_3=pickle.load(f)


henaff_hyper=henaff_hyper_1+henaff_hyper_2+henaff_hyper_3

sorted_hyper=sorted(henaff_hyper, key=lambda d: d['acc'], reverse=True)

top_hyper=list(filter(lambda d: d['acc']==1,sorted_hyper))


hyperparam_output=[]

difficulty='Hard'
for param in top_hyper:
	lh,eta,noiseLevel,ug,cnum,temp,distType=param['lambda_h'],param['eta'],param['noiseLevel'],param['ug'],param['ps'],param['temp'],param['distType']
	acc,trials=runTests(lh,eta,noiseLevel,ug,cnum,temp=temp,distType=distType,difficulty=difficulty)
	hyperparam_output.append({'lambda_h':lh,'eta':eta,'noiseLevel':noiseLevel,'ug':ug,'temp':temp,'distType':distType,'acc':acc,'trials':trials,'difficulty':difficulty})
#def runTests(lh,eta,noiseLevel,ug,cnum,temp=None,distType=0):


pdb.set_trace()

