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


filename=['hyperparam_search_henaff_temp-[0.02].pickle',
			'hyperparam_search_henaff_temp-[0.1].pickle',
			'hyperparam_search_henaff_temp-[1].pickle',
			'hyperparam_search_henaff_temp-[10].pickle']


#collects all the files from the server
henaff_hyper=[]
for name in filename:
	with open(name,'rb') as f: 
		henaff_hyper_temp=pickle.load(f)
	if len(henaff_hyper)==0:
		henaff_hyper=henaff_hyper_temp
	else:
		henaff_hyper=henaff_hyper+henaff_hyper_temp


gumble_score=[]
no_gumble_score=[]

for param in henaff_hyper:
	if param['ug']: 
		gumble_score.append(param['acc'])
	else:
		no_gumble_score.append(param['acc'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

fig, ax = plt.subplots()

score_gumble_mean=np.mean(np.array(gumble_score))
score_gumble_std=np.std(np.array(gumble_score))

score_no_gumble_mean=np.mean(np.array(no_gumble_score))
score_no_gumble_std=np.std(np.array(no_gumble_score))



x=['Gumble','No Gumble']
y=[score_gumble_mean,score_no_gumble_mean]
error=[score_gumble_std,score_no_gumble_std]
ax.bar(x, y, 0.35, color='r')
ax.errorbar(x, y, yerr=error, fmt='o')
ax.set_title('Accuracy of Henaffs Method using Gumble Softmax')
#get average for gumble softmax

#plt.show(block=False)


sorted_hyper=sorted(henaff_hyper, key=lambda d: d['acc'], reverse=True)

top_hyper=list(filter(lambda d: d['acc']==1,sorted_hyper))




pdb.set_trace()

