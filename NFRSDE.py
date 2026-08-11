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

import Rcondflows as FLs
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
		self.d_RNN        = self.net_config.N_rec-1
		self.N_his        = self.net_config.N_his
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
		flows = [flowmodel(dim=self.dim,N_his=self.N_his,config=self.net_config) for _ in range(self.flevel)]
		self.prior = torch.distributions.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
		self.flows = nn.ModuleList(flows)

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
		## saver
		SManager = myutils.SaveManager(path=Monitor.Ens_save_path)
		## train data
		train_data = torch.from_numpy(train_data).to(torch.float32)
		train_dataset = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
		N_batch = int(train_data.shape[0]/self.batch_size) # check
		## train
		for epoch in range(self.n_epochs):
			for batch,train_x in tqdm.tqdm(zip(np.arange(N_batch)+1,train_dataset), total=N_batch):
				self.optimizer.zero_grad() # survy
				z, prior_logprob, log_det = self.forward(train_x[:,:(self.dim*self.N_his)],train_x[:,(self.dim*self.N_his):])
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

	def predict_regular(self,x0,seeds=None):
		if seeds is not None:
			torch.manual_seed(seeds)
		z = self.prior.sample((x0.shape[0],))
		if torch.is_tensor(x0):
			pass
		else:
			x0 = torch.from_numpy(x0).to(torch.float32)
		re, _ = self.inverse(x0,z)
		return re

	def predict_discrete(self,x0,seeds=None):
		if seeds is not None:
			torch.manual_seed(seeds)
		z = self.prior.sample((x0.shape[0],))
		if torch.is_tensor(x0):
			pass
		else:
			x0 = torch.from_numpy(x0).to(torch.float32)
		re, _ = self.inverse(x0,z)
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
	def complete_condpdf(self,model,epoch,best=False,enforce=False):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.repdf_display['times']))==0) or (epoch==0) or enforce:
			# logging.info('--------------Plotting final pdf on Epoch %d'%(epoch+1))
			## check if model list
			path = self.repdfpath if type(model) != list else self.Ens_repdfpath
			if best:
				path = self.Best_repdfpath

			if self.eqn_config.eqn_name in ['REx2_3DOssilator']:
				N = 10000
				dim = self.eqn_config.dim
				L_pre = self.net_config.N_his
				data_dic = sio.loadmat(self.monitor_config.repdf_display['path'])
				px,py = (data_dic['size'].astype('int')).flatten()
				if dim>=5:
					px = px*py
					py = 1

				fig, axes = plt.subplots(nrows=px, ncols=py*dim, figsize=(py*(dim+1)*2.5, px*2), constrained_layout=True, squeeze=False)
				for i in range(px):
					for j in range(py):
						ini = data_dic[str(i*py+j)+'_i'].flatten()
						condata = self.dat_trans(data_dic[str(i*py+j)+'_d'][:,:L_pre,:])
						dat_std = data_dic[str(i*py+j)+'_d'][:,L_pre,:].T
						dat_mod = (model.predict(condata)).detach().numpy()
						for k in range(dim):
							axes[i,j*dim+k].set_title("$X_s=$(%.1f,%.1f,%.1f), dim %d, ite %d"%(ini[0],ini[1],ini[2],k+1,epoch+1))
							axes[i,j*dim+k].hist(dat_mod[:, k], bins=50, density=True, color='#DC143C',histtype='step')
							axes[i,j*dim+k].hist(dat_std[:, k], bins=50, density=True, color='#4169E1',histtype='step')
				fig.savefig(path+'/finalpdf_Margin_'+str(epoch+1)+'.png',dpi=150)
				plt.close()
			else:
				pass
		else:
			pass

	def dat_trans(self,dat):
		# transfer [dim,T,N] type data to [N,dim*T] type
		dim_,NT_,NN_ = dat.shape
		re = np.zeros([NN_,dim_*NT_])
		for i in range(dim_):
			re[:,i::dim_] = dat[i].T
		return re

	## Ensemble related
	def Eva_Ensemble(self,modellist,DatVes,epoch):
		N_T = (DatVes.test_data).shape[1]
		data_ = DatVes.datachoose((np.vstack(DatVes.test_data)).T, DatVes.dim, np.zeros([DatVes.test_data.shape[-1],1],dtype=int), DatVes.N_his)
		Xsh = np.tile(data_[:,:DatVes.dim*DatVes.N_his],(len(modellist),1))
		pre = [Xsh]
		for i in range(N_T-DatVes.N_his):
			Xs = self.Mulmodel_Generate(modellist,Xsh)
			pre += [Xs]
			Xsh = np.concatenate([Xsh[:,DatVes.dim:],Xs], -1)
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

	def Mulmodel_Generate(self,modellist,Xs):
		Nmodel = len(modellist)
		modelid = np.random.randint(Nmodel, size=Xs.shape[0])
		Xre = np.zeros([Xs.shape[0],modellist[0].dim])
		for j in range(Nmodel):
			_id = np.where(modelid==j)[0]
			Xre[_id] = modellist[j].predict(Xs[_id]).detach().numpy()
		return Xre

class DataTran(NFSDE.DataTran):
	def __init__(self,config,Monitor=None):
		# Note: N_rec here denotes the time-length of data (diff with ResNetPDE), starting from 2
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dat_config = config.dat_config
		self.N_his  = self.net_config.N_his
		self.d_RNN  = self.net_config.N_rec-1
		self.n_ea_traj       = self.dat_config.n_ea_traj
		self.train_data_path = self.dat_config.TrainData_dir
		self.test_data_path  = self.dat_config.TestData_dir
		self.N_pred          = self.dat_config.N_pred
		self.Monitor         = Monitor

	def test_mdat1model(self,model,save_path,mode='w'):
		L_Nmax_Test = (self.test_data).shape[1]
		pred = self.test_tensordata(self.test_data,model,L_Nmax_Test)
		if mode=='w':
			self.pred = pred
			sio.savemat(save_path,{'pred':self.pred})
		elif mode=='a':
			self.pred = pred
			if os.path.exists(save_path):
				data_exist = (sio.loadmat(save_path))['pred']
				self.pred = np.concatenate([data_exist,self.pred],axis=-1)
			sio.savemat(save_path,{'pred':self.pred})
		elif mode=='d':
			return pred

	def train_data_trans(self,seed_):
		smaple_L_Nmax = self.L_Nmax-self.d_RNN-self.N_his
		# random setting
		np.random.seed(seed_)
		sample_init_L = np.random.randint(smaple_L_Nmax+1,size=(self.N_long_traj,self.n_ea_traj))
		temp_wu = np.random.permutation(self.n_train)
		if ('zeroinit' in self.dat_config.keys()) and self.dat_config.zeroinit:
			sample_init_L = np.zeros(sample_init_L.shape,dtype=int)
		# data merging
		data_ = (np.vstack(self.train_data)).T
		# set train inputs and outputs
		train_mat = np.zeros((self.n_train, self.dim*(self.d_RNN+self.N_his)))
		for i in range(self.n_ea_traj):
			train_mat[i*self.N_long_traj:(i+1)*self.N_long_traj] = self.datachoose(data_, self.dim, sample_init_L[:,[i]], self.d_RNN+self.N_his)
		self.train_mat  = train_mat[temp_wu,:]
		# monitor
		if self.Monitor.monitor_config.traindata_hist:
			# take the first variable
			# for i in range(self.dim):
			# 	self.Monitor.data2dhistogram(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],self.eqn_config.Delta,"Train_data"+str(i))
			pass
		if self.Monitor.monitor_config.traintransin_hist:
			# take the first variable
			# for i in range(self.dim):
			# 	self.Monitor.transprobinfo(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],"Train_data"+str(i))
			pass

	def test_singledata(self,test_data,model,N_T):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose(test_data, self.dim, 0, self.N_his)
		pre = np.zeros(N_T*self.dim)
		pre[:(self.N_his*self.dim)] = data_
		for i in range(N_T-self.N_his):
			next_time = model.predict(np.array([pre[self.dim*i:self.dim*(i+self.N_his)]]),seeds=None)
			pre[self.dim*(i+1):self.dim*(i+2)] = next_time
		pre = (pre.reshape([N_T,self.dim])).T
		return pre

	def test_tensordata(self,test_data,model,N_T,Nullstart=False):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), self.N_his)
		# Xs = torch.from_numpy(data_[:,:self.dim]).to(torch.float32)
		Xsh = data_[:,:(self.N_his*self.dim)]
		# if Nullstart:
		# 	Xs = torch.zeros(Xs.shape)
		pre = [Xsh] 
		for i in range(N_T-self.N_his):
			Xs = model.predict(Xsh,seeds=None)
			pre += [Xs.detach().numpy()]
			Xsh = np.concatenate((Xsh[:,self.dim:],Xs.detach().numpy()), -1)
		pre = np.concatenate(pre, -1)
		pre_ = np.zeros([self.dim, N_T, Xsh.shape[0]])
		for j in range(self.dim):
			pre_[j] = (pre[:,j::self.dim]).T
		return pre_