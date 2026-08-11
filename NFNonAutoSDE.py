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

import NonAutocondflows as FLs
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
		self.dim_para     = self.eqn_config.dim_para
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
		flows = [flowmodel(dim=self.dim,dim_para=self.dim_para,config=self.net_config) for _ in range(self.flevel)]
		self.prior = torch.distributions.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
		self.flows = nn.ModuleList(flows)

		self.optimizer = torch.optim.Adam(self.flows.parameters(), lr=self.l_rate,weight_decay=self.w_decay)
		self.lr_scheduler = self.lr_scheduler_choose(self.optimizer,self.l_rate_sch)

	def train(self, train_data, para_data, model_path, hist_path, Monitor, DatVes, predt_path):
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
		train_data = torch.from_numpy(train_data).to(torch.float32)
		train_dataset = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
		para_data  = torch.from_numpy(para_data).to(torch.float32)
		para_dataset  = torch.utils.data.DataLoader(para_data, batch_size=self.batch_size, shuffle=False) ### Note that
		N_batch = int(train_data.shape[0]/self.batch_size) # check
		## train
		for epoch in range(self.n_epochs):
			for batch,train_x,para_x in tqdm.tqdm(zip(np.arange(N_batch)+1,train_dataset,para_dataset), total=N_batch):
				self.optimizer.zero_grad() # survy
				z, prior_logprob, log_det = self.forward(train_x[:,:self.dim],para_x[:,:self.dim_para],train_x[:,self.dim:])
				logprob = prior_logprob + log_det
				loss = -torch.mean(prior_logprob + log_det)
				# loss = -torch.mean(prior_logprob)
				loss.backward()
				self.optimizer.step()
			# update learning rate
			if self.l_rate_sch['name']=='ReduceOnPlateau':
				### can be changed to global
				self.lr_scheduler.step(-logprob.mean().data)
			else:
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
			# test model
			self.Test_last(DatVes, predt_path, epoch)
			logprobL.append(logprob.mean().item())
			log_detL.append(log_det.mean().item())
			lr.append(self.lr_scheduler._last_lr[0])
		json.dump({'Logprob':logprobL, 'LogDet': log_detL, 'LogprobTrue': ThLikelihoodData}, open(hist_path, 'w'),indent=2)
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

	def forward(self, x0, para, x):
		bsz, _ = x.shape
		log_det = torch.zeros(bsz)
		for flow in self.flows:
			x, ld = flow.forward(x0, para, x)
			log_det += ld
		z, prior_logprob = x, self.prior.log_prob(x)
		return z, prior_logprob, log_det

	def inverse(self, x0, para, z):
		bsz, _ = z.shape
		log_det = torch.zeros(bsz)
		for flow in self.flows[::-1]:
			z, ld = flow.inverse(x0,para,z)
			log_det += ld
		x = z
		return x, log_det

	def predict(self,x0,para):
		z = self.prior.sample((x0.shape[0],))
		if torch.is_tensor(x0):
			pass
		else:
			x0 = torch.from_numpy(x0).to(torch.float32)
		if torch.is_tensor(para):
			pass
		else:
			para = torch.from_numpy(para).to(torch.float32)
		re, _ = self.inverse(x0,para,z)
		return re

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
	## Ensemble related
	def Eva_Ensemble(self,modellist,DatVes,epoch):
		N_T = (DatVes.test_data).shape[1]
		data_ = DatVes.datachoose((np.vstack(DatVes.test_data)).T, DatVes.dim, np.zeros([DatVes.test_data.shape[-1],1],dtype=int), 1)
		data_para = DatVes.datachoose((np.vstack(DatVes.test_para_data_re)).T, DatVes.dim_para, 0, N_T-1)
		Xs = np.tile(data_[:,:DatVes.dim],(len(modellist),1))
		data_para = np.tile(data_para,(len(modellist),1))
		pre = [Xs]
		for i in range(N_T-1):
			Xs = self.Mulmodel_Generate(modellist,Xs,data_para[:,DatVes.dim_para*i:DatVes.dim_para*(i+1)])
			pre += [Xs]
		pre = np.concatenate(pre, -1)
		pre_ = np.zeros([DatVes.dim,N_T,DatVes.N_pred*len(modellist)])
		for j in range(self.eqn_config.dim):
			pre_[j] = (pre[:,j::DatVes.dim]).T
		for i in range(min(DatVes.dim,10)):
			save_ = (self.Ens_evapath+'/'+str(epoch+1)+'M'+str(i+1)+'.pdf')
			fig,ax = self.Evaulation.plot_meanstd(DatVes.test_data[i].T,pre_[i].T,self.eqn_config.Delta,savepath=save_)
		# save_ = (self.Ens_evapath+'/'+str(epoch+1)+'M'+'.pdf')
		# fig,ax = self.Evaulation.plot_meanstdGeneralD(DatVes.test_data[:,-1,:],pre_,DatVes.dim,self.eqn_config.Delta,savepath=save_)
		plt.close()

	def Mulmodel_Generate(self,modellist,Xs,Xpara):
		Nmodel = len(modellist)
		modelid = np.random.randint(Nmodel, size=Xs.shape[0])
		Xre = np.zeros(Xs.shape)
		for j in range(Nmodel):
			_id = np.where(modelid==j)[0]
			Xre[_id] = modellist[j].predict(Xs[_id],Xpara[_id]).detach().numpy()
		return Xre


class DataTran(NFSDE.DataTran):
	def __init__(self,config,Monitor=None):
		# Note: d_RNN here denotes the time-length of data (diff with ResNetPDE)
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dat_config = config.dat_config
		self.d_RNN  = self.net_config.N_rec
		self.n_ea_traj       = self.dat_config.n_ea_traj
		self.train_data_path = self.dat_config.TrainData_dir
		self.test_data_path  = self.dat_config.TestData_dir
		self.N_pred          = self.dat_config.N_pred
		self.Monitor         = Monitor
		if ('pair_data' in self.dat_config.keys()) and (self.dat_config.pair_data):
			self.train_data_trans = self.train_data_trans_pair
		else:
			self.train_data_trans = self.train_data_trans_traj

	def read_traindata(self):
		# train data is assumed to be stored under key 'data' of matfile
		# train data in this function is in the form of [dim,n_of_time_step,n_of_tracjectory]
		try:
			self.train_data = (sio.loadmat(self.train_data_path))['data']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file.')
		try:
			self.train_para_data = (sio.loadmat(self.train_data_path))['para']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file, there is no parameter data.')
		self.dim, self.L_Nmax, self.N_long_traj = (self.train_data).shape
		self.n_train = self.n_ea_traj * self.N_long_traj
		self.dim_para = (self.train_para_data).shape[0]

	def read_testdata(self):
		try:
			self.test_data = (sio.loadmat(self.test_data_path))['data']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file.')
		try:
			self.test_para_data = (sio.loadmat(self.test_data_path))['para']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file.')
		self.dim = (self.test_data).shape[0]
		self.dim_para = (self.test_para_data).shape[0]
		self.test_para_data_re = np.repeat(self.test_para_data[:, :, np.newaxis], self.test_data.shape[-1], axis=2)

	def test_mdat1model(self,model,save_path,mode='w'):
		L_Nmax_Test = (self.test_data).shape[1]
		Nullstart = True if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if'] else False
		self.pred = self.test_tensordata(self.test_data,self.test_para_data_re,model,L_Nmax_Test,Nullstart)
		if mode=='w':
			sio.savemat(save_path,{'pred':self.pred})
		elif mode=='a':
			if os.path.exists(save_path):
				data_exist = (sio.loadmat(save_path))['pred']
				self.pred = np.concatenate([data_exist,self.pred],axis=-1)
			sio.savemat(save_path,{'pred':self.pred})

	def train_data_trans_traj(self,seed_):
		smaple_L_Nmax = self.L_Nmax-self.d_RNN
		# random setting
		np.random.seed(seed_)
		sample_init_L = np.random.randint(smaple_L_Nmax+1,size=(self.N_long_traj,self.n_ea_traj))
		temp_wu = np.random.permutation(self.n_train)
		if ('zeroinit' in self.dat_config.keys()) and self.dat_config.zeroinit:
			sample_init_L = np.zeros(sample_init_L.shape,dtype=int)
		# data merging
		data_ = (np.vstack(self.train_data)).T
		data_para = (np.vstack(self.train_para_data)).T
		# set train inputs and outputs
		train_mat = np.zeros((self.n_train, self.dim*self.d_RNN))
		paramt_train = np.zeros((self.n_train, self.dim_para*(self.d_RNN-1)))
		for i in range(self.n_ea_traj):
			train_mat[i*self.N_long_traj:(i+1)*self.N_long_traj] = self.datachoose(data_, self.dim, sample_init_L[:,[i]], self.d_RNN)
			paramt_train[i*self.N_long_traj:(i+1)*self.N_long_traj] = self.datachoose(data_para, self.dim_para, sample_init_L[:,[i]], self.d_RNN-1)
		self.train_mat  = train_mat[temp_wu,:]
		self.para_mat   = paramt_train[temp_wu,:]
		# monitor
		if self.Monitor.monitor_config.traindata_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.data2dhistogram(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],self.eqn_config.Delta,"Train_data"+str(i))
		if self.Monitor.monitor_config.traintransin_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.transprobinfo(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],"Train_data"+str(i))
		# # final data
		# self.train_mat  = np.concatenate((self.train_mat,self.para_mat),axis=-1)

	def train_data_trans_pair(self,seed_):
		self.N_base = self.dat_config.N_train_base
		# data merging
		data_ = (np.vstack(self.train_data)).T
		data_para = (np.vstack(self.train_para_data)).T
		# set train inputs and outputs
		train_mat = np.zeros((self.n_ea_traj*self.N_base, self.dim*self.d_RNN))
		paramt_train = np.zeros((self.n_ea_traj*self.N_base, self.dim_para*(self.d_RNN-1)))
		temp_wu = np.random.permutation(data_.shape[0])
		data_     = data_[temp_wu,:]
		data_para = data_para[temp_wu,:]
		for i in range(self.n_ea_traj):
			train_mat[i*self.N_base:(i+1)*self.N_base,:self.dim] = data_[i*self.N_base:(i+1)*self.N_base,:-1:2]
			train_mat[i*self.N_base:(i+1)*self.N_base,self.dim:] = data_[i*self.N_base:(i+1)*self.N_base,1::2]
			paramt_train[i*self.N_base:(i+1)*self.N_base] = data_para[i*self.N_base:(i+1)*self.N_base]
		self.train_mat  = train_mat
		self.para_mat   = paramt_train
		# monitor
		if self.Monitor.monitor_config.traindata_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.data2dhistogram(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],self.eqn_config.Delta,"Train_data"+str(i))
		if self.Monitor.monitor_config.traintransin_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.transprobinfo(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],"Train_data"+str(i))
		# # final data
		# self.train_mat  = np.concatenate((self.train_mat,self.para_mat),axis=-1)

	def test_singledata(self,test_data,para_data,model,N_T):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose(test_data, self.dim, 0, 1)
		pre = np.zeros(N_T*self.dim)
		pre[:self.dim] = data_
		for i in range(N_T-1):
			next_time = model.predict(np.array([pre[self.dim*i:self.dim*(i+1)]]),para_data[:,self.dim_para*i:self.dim_para*(i+1)])
			pre[self.dim*(i+1):self.dim*(i+2)] = next_time
		pre = (pre.reshape([N_T,self.dim])).T
		return pre

	def test_tensordata(self,test_data,para_data_re,model,N_T,Nullstart=False):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
		data_para = self.datachoose((np.vstack(para_data_re)).T, self.dim_para, 0, N_T-1)
		# data_para = self.datachoose(para_data, self.dim_para, 0, N_T-1)
		# Xs = torch.from_numpy(data_[:,:self.dim]).to(torch.float32)
		Xs = data_[:,:self.dim]
		if Nullstart:
			Xs = torch.zeros(Xs.shape)
		# pre = [Xs] 
		# for i in range(N_T-1):
		# 	with torch.no_grad():
		# 		Xs = model.predict(Xs,data_para[:,self.dim_para*i:self.dim_para*(i+1)])
		# 		pre += [Xs.detach().numpy()]
		# 	print(i)
		# pre = np.concatenate(pre, -1)
		# pre_ = np.zeros([self.dim,N_T,self.N_pred])
		pre = np.zeros([Xs.shape[0],Xs.shape[1]*N_T])
		pre[:,:self.dim] = Xs
		for i in range(N_T-1):
			with torch.no_grad():
				Xs = model.predict(Xs,data_para[:,self.dim_para*i:self.dim_para*(i+1)])
				pre[:,(i+1)*self.dim:(i+2)*self.dim] = Xs.detach().numpy()
		pre_ = np.zeros([self.dim,N_T,data_.shape[0]])
		for j in range(self.dim):
			pre_[j] = (pre[:,j::self.dim]).T
		return pre_

	# def test_tensordata(self,test_data,para_data_re,model,N_T,Nullstart=False):
	# 	# data is in the form of [dim*n_of_time_step]
	# 	# aranging as [dim1_tracj, dim2_tracj,...]
	# 	data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
	# 	data_para = self.datachoose((np.vstack(para_data_re)).T, self.dim_para, 0, N_T-1)
	# 	# data_para = self.datachoose(para_data, self.dim_para, 0, N_T-1)
	# 	# Xs = torch.from_numpy(data_[:,:self.dim]).to(torch.float32)
	# 	Xs = data_[:,:self.dim]
	# 	if Nullstart:
	# 		Xs = torch.zeros(Xs.shape)
	# 	pre = np.zeros([Xs.shape[0],Xs.shape[1]*N_T])
	# 	for i in range(N_T-1):
	# 		Xs = model.predict(Xs,data_para[:,self.dim_para*i:self.dim_para*(i+1)])
	# 		pre[:,i*self.dim:(i+1)*self.dim] = Xs.detach().numpy()
	# 	pre_ = np.zeros([self.dim,N_T,self.N_pred])
	# 	for j in range(self.dim):
	# 		pre_[j] = (pre[:,j::self.dim]).T
	# 	return pre_