from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pdb
import logging
import os

import torch
import torch.nn as nn

import scipy
import scipy.io as sio
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
import tqdm

import condflows as FLs
import Chemical_Dynamics
import myutils

import numpy as np

import NFSDE


class NFSSDE(nn.Module):
	def __init__(self,config):
		super().__init__()
		## parameter set
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dat_config = config.dat_config
		self.dim          = self.eqn_config.dim
		self.fname        = self.net_config.fname
		self.flevel       = self.net_config.flevel
		self.d_RNN        = self.net_config.N_rec
		self.n_epochs     = self.net_config.N_epochs
		self.batch_size   = self.net_config.batch_size
		self.l_rate       = self.net_config.l_rate
		self.Test_mode    = self.net_config.Test_mode
		self.w_decay      = self.net_config.weight_decay
		self.l_rate_sch   = self.net_config.l_rate_config
		## test model
		self.test_model = self.test_model_choose(self.Test_mode)
		## build model
		self.build_model()

	def build_model(self):
		flowmodel = self.flow_choose(self.fname)
		flows = [flowmodel(dim=self.dim,config=self.net_config) for _ in range(self.flevel)]
		self.prior = torch.distributions.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
		self.flows = nn.ModuleList(flows)

		# print(self.get_n_params())

		# self.optimizer = torch.optim.Adam(self.flows.parameters(), lr=self.l_rate)
		self.optimizer = torch.optim.Adam(self.flows.parameters(), lr=self.l_rate,weight_decay=self.w_decay)
		self.lr_scheduler = self.lr_scheduler_choose(self.optimizer,self.l_rate_sch)

		if ('DiscretePred' in self.dat_config.keys()) and self.dat_config.DiscretePred:
			self.predict = self.predict_discrete
		else:
			self.predict = self.predict_regular

	def train(self, train_data, model_path, hist_path, Monitor, DatVes, predt_path):
		logtim = int(self.n_epochs/10)
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		logprobL, log_detL, lr = [],[],[]
		try:
			ThLikelihoodData = self.ThLikelihood(self.eqn_config.eqn_name,train_data[:,0],train_data[:,1],self.eqn_config.Delta)
		except:
			ThLikelihoodData = "None"
		# ThLikelihoodData = self.ThLikelihood(self.eqn_config.eqn_name,train_data[:,0],train_data[:,1],self.eqn_config.Delta)
		## saver
		SManager = myutils.SaveManager(path=Monitor.Ens_save_path)
		## train data
		est_drift  = torch.from_numpy(self.Model_drift.myevaluate(train_data[:,:self.dim])).to(torch.float32)
		est_driftset  = torch.utils.data.DataLoader(est_drift, batch_size=self.batch_size, shuffle=False)
		train_data = torch.from_numpy(train_data).to(torch.float32)
		train_dataset = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
		N_batch = int(train_data.shape[0]/self.batch_size) # check
		## train
		for epoch in range(self.n_epochs):
			for batch,train_x,est_drift_x in tqdm.tqdm(zip(np.arange(N_batch)+1,train_dataset,est_driftset), total=N_batch):
				self.optimizer.zero_grad() # survy
				z, prior_logprob, log_det = self.forward(train_x[:,:self.dim],train_x[:,self.dim:]-est_drift_x)
				logprob = prior_logprob + log_det
				loss = -torch.mean(prior_logprob + log_det)
				# loss = -torch.mean(prior_logprob)
				loss.backward()
				self.optimizer.step()
			self.lr_scheduler.step()
			print("Epoch: {} | Logprob: {} | True: {}".format(epoch, logprob.mean().data, ThLikelihoodData))
			# save model
			if (epoch + 1) % 500 == 0:
				torch.save(self.state_dict(), model_path+'model.pt')
			if (epoch + 1) % logtim == 0:
				logging.info('Epoch %d/%d has been reached'%(epoch+1,self.n_epochs))
			# monitor
			if Monitor.monitor_config.repdf_display['if']:
				Monitor.complete_condpdf(self,epoch)
			if Monitor.monitor_config.cond_mv['if']:
				Monitor.cond_meanvar(self,epoch)
			if Monitor.monitor_config.loss['if']:
				Monitor.Eva_loss(self,epoch,logprobL,log_detL,ThLikelihoodData)
			if Monitor.monitor_config.Evameanv['type']=="Normal":
				Monitor.Eva_meanv(self,epoch,DatVes,predt_path)
			# if Monitor.monitor_config.Evameanv['if']:
			# 	if Monitor.monitor_config.Evameanv['type']=="Normal":
			# 		Monitor.Eva_meanv(self,epoch,DatVes,predt_path)
			# 	elif Monitor.monitor_config.Evameanv['type']=="Multiple_last":
			# 		Monitor.Eva_meanv_Multiple_last(self,epoch,DatVes,predt_path)
			if Monitor.monitor_config.Ens_monitor['if']:
				Monitor.Ens_monitor(epoch,SManager,self,DatVes)
			if 'Best_monitor' in Monitor.monitor_config.keys() and Monitor.monitor_config.Best_monitor['if']:
				Monitor.Best_monitor(epoch,logprob.mean().item(),self,DatVes)
			# test model
			self.Test_last(DatVes, predt_path, epoch)
			logprobL.append(logprob.mean().item())
			log_detL.append(log_det.mean().item())
			lr.append(self.lr_scheduler._last_lr[0])
			
			losslist = {'Logprob':logprobL, 'LogDet': log_detL, 'LogprobTrue': ThLikelihoodData}
			if 'Best_monitor' in Monitor.monitor_config.keys() and Monitor.monitor_config.Best_monitor['if']:
				min_lossL = [] if 'min_lossL' not in locals() else min_lossL
				min_lossL.append(self.min_loss)
				losslist['best'] = min_lossL
				json.dump(losslist, open(hist_path, 'w'),indent=2)
		torch.save(self.state_dict(), model_path+'model.pt')

	def flow_choose(self,flow):
		if flow=='Planar':
			return FLs.Planar
		# elif flow=='Radial':
		# 	return FLs.Radial
		# elif flow=='RealNVP':
		# 	return FLs.RealNVP
		elif flow=='MAF':
			return FLs.MAFCond
		# elif flow=='ActNorm':
		# 	return FLs.ActNorm
		# elif flow=='OneByOneConv':
		# 	return FLs.OneByOneConv
		elif flow=='NSF_AR':
			return FLs.NSF_ARCond
		# elif flow=='NSF_CL':
		# 	return FLs.NSF_CL
		else:
			raise AttributeError('NFSSDE: No this type of flow')

	def forward(self, x0, x):
		bsz, _ = x.shape
		log_det = torch.zeros(bsz)
		for flow in self.flows:
			x, ld = flow.forward(x0, x)
			log_det += ld
		z, prior_logprob = x, self.prior.log_prob(x)
		return z, prior_logprob, log_det

	def inverse(self, x0, z):
		bsz, _ = z.shape
		log_det = torch.zeros(bsz)
		for flow in self.flows[::-1]:
			z, ld = flow.inverse(x0,z)
			log_det += ld
		x = z
		return x, log_det

	def predict_regular(self,x0):
		z = self.prior.sample((x0.shape[0],))
		if torch.is_tensor(x0):
			pass
		else:
			x0 = torch.from_numpy(x0).to(torch.float32)
		re, _ = self.inverse(x0,z)
		return self.Model_drift.myevaluate(x0)+re
		# return re

	def predict_discrete(self,x0):
		z = self.prior.sample((x0.shape[0],))
		if torch.is_tensor(x0):
			pass
		else:
			x0 = torch.from_numpy(x0).to(torch.float32)
		re, _ = self.inverse(x0,z)
		re = torch.from_numpy(self.Model_drift.myevaluate(x0)).to(torch.float32)+re
		return self.clamp_Z(re)

	def clamp_Z(self,x):
		re = x
		re[x < 0] = 0
		return torch.round(re)

	def Test_last(self, DatVes, predt_path, epoch):
		if epoch+1==self.n_epochs:
			logging.info('Test on Epoch %d'%(epoch+1))
			DatVes.test_mdat1model(self,predt_path)

	def ThLikelihood(self,name,x0,x1,Delta):
		if self.dim==1:
			if name=='Brownian Motion':
				m,s = 0*x0,np.sqrt(Delta)
			elif name=='Geometric Brownian Motion':
				m,s = self.eqn_config.mu*x0,self.eqn_config.sigma*x0
			elif name=='OU Process':
				m  = self.eqn_config.theta*(self.eqn_config.mu-x0)
				s  = np.sqrt(self.eqn_config.sigma**2)
			elif name=='Exp_diffusion':
				m   = -self.eqn_config.mu*x0
				s   = self.eqn_config.sigma*np.exp(-x0**2)
			elif name=='Trig_drift':
				m   = np.sin(2*self.eqn_config.k*np.pi*x0)
				s   = np.abs(self.eqn_config.sigma*np.cos(2*self.eqn_config.k*np.pi*x0))
			elif name=='Exp_OU':
				th,dt  = self.eqn_config.theta,Delta
				mu,sig = self.eqn_config.mu,self.eqn_config.sigma
				MU,SIG = (1-th*dt)*np.log(x0)+th*mu*dt,sig*np.sqrt(dt)
				m = -th*np.log(x0)+th*mu+sig**2/2
				s = np.sqrt((np.exp(SIG**2)-1)*np.exp(2*MU+SIG**2))
			elif name=='Double_well':
				m   = x0-x0**3
				s   = self.eqn_config.sigma
			elif name=='Exp_dis':
				m = self.eqn_config.theta*x0+self.eqn_config.sigma/np.sqrt(self.eqn_config.Delta)
				s = self.eqn_config.sigma
			else:
				print('The distribution %s is not supported'%(name))
			a0,b0 = x0+m*Delta,s*np.sqrt(Delta)
			logprob = -np.log(np.sqrt(2*np.pi)*np.abs(b0))-(x1-a0)**2/(2*b0**2)
			loglk = np.mean(logprob)
		elif self.dim==2:
			if name=='MdOU':
				Mean = np.array(x0)+np.array(x0)@np.array(self.eqn_config.mu)*Delta
				Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
				logprob = -np.log(np.sqrt((2*np.pi)**2*np.abs(np.linalg.det(Cov))))-np.einsum('ji,jk,ki->i',(x1-Mean).T,np.linalg.inv(Cov),(x1-Mean).T)/2
				loglk = np.mean(logprob)
			else:
				print('The distribution %s is not supported'%(name))
		return loglk

	# Ensemble related
	def test_model_choose(self, Test_mode):
		if Test_mode=='Normal':
			self.Testepoches = self.Last_epochs(20,10,self.n_epochs)
			return self.Test_last
		elif Test_mode=='Multiple_last':
			self.Testepoches = self.Last_epochs(20,10,self.n_epochs)
			return self.Test_Multiple_last
		else:
			raise AttributeError('test_model: Do not support %s type Test'%(Test_mode))

	def Test_Multiple_last(self, DatVes, predt_path, epoch):
		if epoch+1 in self.Testepoches:
			if (epoch+1)==self.Testepoches[0]:
				logging.info('Ensemble Test on Epoch %d'%(epoch+1))
				DatVes.test_mdat1model(self,predt_path,mode='w')
			else:
				logging.info('Ensemble Test on Epoch %d'%(epoch+1))
				DatVes.test_mdat1model(self,predt_path,mode='a')

	def Last_epochs(self,n1,n2,nepoch):
		# for nepoch epoches, return the ** # of epoch ** before last one (including last)
		step = int(nepoch/(n1*n2))
		# if nepoch too small, then degenerate to Test_last
		if step==0:
			return np.array((nepoch))
		initial = nepoch-n2*step
		return (np.arange(n2)+1)*step+initial

	def get_n_params(self):
		pp=0
		for p in list(self.flows.parameters()):
			nn=1
			for s in list(p.size()):
				nn = nn*s
			pp += nn
		return pp

	def read_Model(self,path):
		self.load_state_dict(torch.load(path))

	# Learning Rate related
	def lr_scheduler_choose(self,optimizer,lr_info):
		if lr_info['name']=='value':
			# In config, l_rate_config = {'name':'value'}
			return torch.optim.lr_scheduler.StepLR(optimizer, self.n_epochs, gamma=1.0)
		elif lr_info['name']=='Step':
			# In config, l_rate_config = {'name':'Step','step':,'gamma':}
			step_,gamma_ = lr_info['step'],lr_info['gamma']
			return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_, gamma=gamma_)
		elif lr_info['name']=='Cyclic':
			# In config, l_rate_config = {'name':'Cyclic','base':,'max':,'step':}
			base_,max_,step_,gamma_ = lr_info['base'],lr_info['max'],lr_info['step'],lr_info['gamma']
			return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_, max_lr=max_, step_size_up=step_,mode='exp_range',gamma=gamma_,cycle_momentum=False)
		elif lr_info['name']=='StepCyclic':
			# In config, l_rate_config = {'name':'Cyclic','base':,'max':,'step':}
			base_,max_,step_,gamma_,scal_,gstep_ = lr_info['base'],lr_info['max'],lr_info['step'],lr_info['gamma'],lr_info['scale'],lr_info['gstep']
			scheduler_l,ms = [],[]
			for i in range(int(self.n_epochs/gstep_)+1):
				scali = scal_/(i+1)
				scheduler_l.append(torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_*scali, max_lr=max_*scali, step_size_up=step_,mode='exp_range',gamma=gamma_,cycle_momentum=False))
				ms.append(gstep_*(i+1))
			return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=scheduler_l, milestones=ms[:-1])
		elif lr_info['name']=='ReduceOnPlateau':
			minr_,factor_,patience_ = lr_info['minr'],lr_info['factor'],lr_info['patience']
			return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor_, patience=patience_, min_lr=minr_)



class Monitor(NFSDE.Monitor):
	def readMultiplemodel(self,ckptmanager,model):
		# This function is designed for test for multiple models
		modellist = []
		modeldict = ckptmanager.list_models()
		for i in range(len(modeldict)):
			ModelX = self.NFModel(self.config)
			ModelX.Model_drift = model.Model_drift
			ModelX.load_state_dict(torch.load(modeldict[i]))
			# ModelX.eval()
			modellist.append(ModelX)
		return modellist

	def readmodel(self,path,model):
		# This function is designed for test for single models
		ModelX = self.NFModel(self.config)
		ModelX.Model_drift = model.Model_drift
		ModelX.load_state_dict(torch.load(path))
		return ModelX


class DataTran(NFSDE.DataTran):
	def train_hiddendata(self):
		if self.eqn_config.resmodel=='ResNetwM':
			ResModel = ResnetPDEwM.ResnetPDEwM
			with open(self.resconfig_path) as json_data_file:
				resmodel_config = json.load(json_data_file)
			resmodel_config = munch.munchify(resmodel_config)
			self.Model_drift = self.readmodelKeras(self.resmodel_path,ResModel,resmodel_config)
		elif self.eqn_config.resmodel=='Exact':
			if self.eqn_config.eqn_name=='Geometric Brownian Motion':
				self.Model_drift = GeoBrowian_exact_drift(self.eqn_config.mu,self.eqn_config.sigma,self.eqn_config.Delta)
			elif self.eqn_config.eqn_name=='Trig_drift':
				self.Model_drift = Trig_drift_exact_drift(self.eqn_config.k,self.eqn_config.sigma,self.eqn_config.Delta)
			elif self.eqn_config.eqn_name in ['MdOU','SO']:
				self.Model_drift = MdOU_exact_drift(self.eqn_config.mu,self.eqn_config.Delta)
			elif self.eqn_config.eqn_name=='LagvinSchlogl':
				self.Model_drift = Ex45LagvinSchlogl_exact_drift(self.eqn_config.s,self.eqn_config.Delta)
		elif self.eqn_config.resmodel=='ChemicalODE':
			self.Model_drift = Chemical_Dynamics.ChemicalDynamics(self.eqn_config)
			self.Model_drift.myevaluate = self.Model_drift.ODEsolverT(self.eqn_config.Delta)
		else:
			raise AttributeError('train_hiddendata::No this model')

class GeoBrowian_exact_drift:
	def __init__(self,mu,sigma,Delta):
		self.mu = mu
		self.sigma = sigma
		self.Delta = Delta
	def myevaluate(self,x):
		return x+self.Delta*self.mu*x
	def myevaluateincrem(self,x):
		return self.Delta*self.mu*x

class Trig_drift_exact_drift:
	def __init__(self,k,sigma,Delta):
		self.k = k
		self.sigma = sigma
		self.Delta = Delta
	def myevaluate(self,x):
		return x+self.Delta*torch.sin(2*np.pi*self.k*x)
	def myevaluateincrem(self,x):
		return self.Delta*torch.sin(2*np.pi*self.k*x)

class Ex45LagvinSchlogl_exact_drift:
	def __init__(self,s,Delta):
		self.s = s
		self.Delta = Delta
	def myevaluate(self,x):
		return x+self.Delta*(3e-2*x*(x-1)/2-1e-4*x*(x-1)*(x-2)/6+200-3.5*x)
	def myevaluateincrem(self,x):
		return self.Delta*(3e-2*x*(x-1)/2-1e-4*x*(x-1)*(x-2)/6+200-3.5*x)

class MdOU_exact_drift:
	def __init__(self,mu,Delta):
		self.mu = mu
		self.muT = np.array(mu).T
		self.Delta = Delta
	def myevaluate(self,x):
		return x+self.Delta*x@self.muT
	def myevaluateincrem(self,x):
		return self.Delta*x@self.muT

