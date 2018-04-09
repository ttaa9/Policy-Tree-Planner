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


server_index=1
temperatures = [0.02,0.1, 1, 10]
temp=[temperatures[server_index]]

filename='hyperparam_search_henaff_temp-'+str(temp)+'.pickle'

hyperparam_search(lambda_hs=[0.0,-0.005, 0.005] ,
                    etas = [0.01,0.1,0.2,0.3,0.5, 1],
                    useGumbels = [True, False], 
                    temperatures = temp,
                    noiseSigmas = [0,0.01,0.1, 1.0,2.0],
                    niters = 200,
                    verbose = False,
                    extraVerbose = False, 
                    numRepeats = 10,
                    file_name_output = filename,
                    distType = 1,
                    difficulty='Hard')