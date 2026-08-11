import math
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import scipy as sp
import scipy.linalg

# torch should be imported at the beginning, reasons not clear

class SaveManager():
    def __init__(self,path,keep=10):
        self.path = path
        self.keep = keep
        self.savenum = 0

    def Ensemble_save(self,model):
        torch.save(model.state_dict(), self.path+'/model'+str(self.savenum%self.keep)+'.pt')
        self.savenum += 1

    def list_models(self):
        return [self.path+'/'+f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]