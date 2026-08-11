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

import gen_condflows as FLs
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
		## postprocess
		self.postprocs_choose()
		## build model
		self.build_model()

	def build_model(self):
		flowmodel = self.flow_choose(self.fname)
		flows = [flowmodel(dimx0=self.dim,dimx=self.dim_r,config=self.net_config) for _ in range(self.flevel)]
		self.prior = torch.distributions.MultivariateNormal(torch.zeros(self.dim_r), torch.eye(self.dim_r))
		self.flows = nn.ModuleList(flows)

		self.optimizer = torch.optim.Adam(self.flows.parameters(), lr=self.l_rate, weight_decay=self.w_decay)
		self.lr_scheduler = self.lr_scheduler_choose(self.optimizer,self.l_rate_sch)

		if ('DiscretePred' in self.dat_config.keys()) and self.dat_config.DiscretePred:
			self.predict = self.predict_discrete
		else:
			self.predict = self.predict_discrete

	def train(self, train_data, model_path, hist_path, Monitor, DatVes, predt_path):
		logtim = int(self.n_epochs/10)
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		logprobL, log_detL, lr = [],[],[]
		try:
			ThLikelihoodData = self.ThLikelihood(self.eqn_config.eqn_name,train_data[:,0],train_data[:,1],self.eqn_config.Delta)
		except:
			ThLikelihoodData = "None"
		## saver
		SManager = myutils.SaveManager(path=Monitor.Ens_save_path)
		## train data
		train_data = self.train_data_transfer(train_data)
		train_data = torch.from_numpy(train_data).to(torch.float32)
		train_dataset = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
		N_batch = int(train_data.shape[0]/self.batch_size) # check
		## train
		for epoch in range(self.n_epochs):
			for batch,train_x in tqdm.tqdm(zip(np.arange(N_batch)+1,train_dataset), total=N_batch):
				self.optimizer.zero_grad() # survy
				z, prior_logprob, log_det = self.forward(train_x[:,:self.dim],train_x[:,self.dim:])
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
				Monitor.Eva_lr(self,epoch,lr)
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
			if Monitor.monitor_config.Ens_monitor['if']:
				Monitor.Ens_monitor(epoch,SManager,self,DatVes)
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

	def postprocs_choose(self):
		if self.eqn_config.eqn_name=='SSAVilar2002R':
			self.dim_r = 9-2
			self.predict_discrete = self.SSAVilar2002R_predict_discrete
			self.train_data_transfer = self.SSAVilar2002R_train_data_transfer
		elif self.eqn_config.eqn_name=='SSAautocatalytic':
			self.dim_r = self.dim-1
			self.predict_discrete = self.SSAautocatalytic_predict_discrete
			self.train_data_transfer = self.SSAautocatalytic_train_data_transfer

	def SSAVilar2002R_predict_discrete(self,x0):
		z = self.prior.sample((x0.shape[0],))
		if torch.is_tensor(x0):
			pass
		else:
			x0 = torch.from_numpy(x0).to(torch.float32)
		N1 = torch.sum(x0[:,[0,2]],axis=1)
		N2 = torch.sum(x0[:,[1,3]],axis=1)
		# xt = torch.cat((x0[:,:self.dim-1],N),axis=1)
		
		re, _ = self.inverse(x0,z)
		re = x0[:,[0,1,4,5,6,7,8]]+re
		re_ = self.SSAVilar2002R_clamp_Z(re,N1,N2)
		# should steady state here
		rest1 = N1-re_[:,0]
		rest2 = N2-re_[:,1]
		ref = torch.cat((re_[:,[0]],re_[:,[1]],rest1[:,None],rest2[:,None],re_[:,2:]),axis=1)
		return self.clamp_Z(ref)

	def SSAVilar2002R_train_data_transfer(self,data):
		data_new = np.zeros([data.shape[0],self.dim+self.dim_r])
		data_new[:,:self.dim] = data[:,:self.dim]
		data_new[:,self.dim:] = data[:,self.dim+np.array([0,1,4,5,6,7,8])]
		return data_new

	def SSAautocatalytic_predict_discrete(self,x0):
		z = self.prior.sample((x0.shape[0],))
		if torch.is_tensor(x0):
			pass
		else:
			x0 = torch.from_numpy(x0).to(torch.float32)
		N = torch.sum(x0,axis=1,keepdim=True)
		xt = torch.cat((x0[:,:self.dim-1],N),axis=1)
		re, _ = self.inverse(xt,z)
		re = x0[:,:self.dim-1]+re
		re_ = self.clamp_Z(re)
		rest = N-torch.sum(re_,axis=1,keepdim=True)
		ref = torch.cat((re_,rest),axis=1)
		return self.steady_state(x0,self.clamp_Z(ref))

	def SSAautocatalytic_train_data_transfer(self,data):
		data_new = np.zeros([data.shape[0],data.shape[1]-1])
		data_new[:,:self.dim-1] = data[:,:self.dim-1]
		data_new[:,self.dim-1]  = np.sum(data[:,:self.dim],axis=1)
		data_new[:,self.dim:]   = data[:,self.dim:2*self.dim-1]
		return data_new

	def SSAVilar2002R_clamp_Z(self,x,N1,N2):
		re = x
		re[x < 0] = 0
		re[x[:,0]>N1,0] = N1[x[:,0]>N1]
		re[x[:,1]>N2,1] = N2[x[:,1]>N2]
		return torch.round(re)

	def steady_state(self,x0,pred):
		re = pred
		_id = np.where((abs(x0[:,0])<1e-8)*(abs(x0[:,1])<1e-8))
		re[_id] = x0[_id]
		return re

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
	pass

class DataTran(NFSDE.DataTran):
	def train_hiddendata(self):
		# change learned data to increment
		if ('DiscretePred' in self.dat_config.keys()) and self.dat_config.DiscretePred:
			self.train_mat[:,self.dim:] = self.train_mat[:,self.dim:]-np.maximum(self.train_mat[:,:-self.dim],0)
		else:
			self.train_mat[:,self.dim:] = self.train_mat[:,self.dim:]-self.train_mat[:,:-self.dim]
