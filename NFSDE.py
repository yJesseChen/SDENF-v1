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
import myutils

import numpy as np


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

		self.optimizer = torch.optim.Adam(self.flows.parameters(), lr=self.l_rate, weight_decay=self.w_decay)
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
		elif flow=='DAMAF':
			return FLs.MAFCond_DANN
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
		return re

	def predict_discrete(self,x0):
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


class Monitor():
	def __init__(self,path,config,NFModel,Evaulation=None):
		self.eqn_config      = config.eqn_config
		self.net_config      = config.net_config
		self.dat_config      = config.dat_config
		self.monitor_config  = config.monitor_config
		self.rawpath         = path
		self.repdfpath       = self.rawpath+'repdfplot'
		self.dataplotpath    = self.rawpath+'dataplot'
		self.cond_mvpath     = self.rawpath+'condmeanvar'
		self.loss_path       = self.rawpath+'loss'
		self.Ens_save_path   = self.rawpath+'Ens_model/'
		self.Ens_cond_mvpath = self.rawpath+'Ens_cond_mv'
		self.Ens_repdfpath   = self.rawpath+'Ens_repdf'
		self.Ens_evapath     = self.rawpath+'Ens_Eva'
		self.Ens_endpdfpath  = self.rawpath+'Ens_Epdf'
		self.Best_save_path   = self.rawpath+'Best_model/'
		self.Best_repdfpath   = self.rawpath+'Best_repdf'
		self.Best_evapath     = self.rawpath+'Best_Eva'
		if Evaulation!=None:
			self.Evaulation = Evaulation
		if not os.path.exists(path):
			os.makedirs(path)
		if (self.monitor_config.repdf_display['if']) and (not os.path.exists(self.repdfpath)):
			os.makedirs(self.repdfpath)
		if (self.monitor_config.traindata_hist) and (not os.path.exists(self.dataplotpath)):
			os.makedirs(self.dataplotpath)
		if (self.monitor_config.traintransin_hist) and (not os.path.exists(self.dataplotpath)):
			os.makedirs(self.dataplotpath)
		if (self.monitor_config.cond_mv['if']) and (not os.path.exists(self.cond_mvpath)):
			os.makedirs(self.cond_mvpath)
		if (self.monitor_config.loss['if']) and (not os.path.exists(self.loss_path)):
			os.makedirs(self.loss_path)	
		if (self.monitor_config.Ens_monitor['if']) and (not os.path.exists(self.Ens_save_path)):
			os.makedirs(self.Ens_save_path)
		if (self.monitor_config.Ens_monitor['Ens_cond_mv']) and (not os.path.exists(self.Ens_cond_mvpath)):
			os.makedirs(self.Ens_cond_mvpath)	
		if (self.monitor_config.Ens_monitor['Ens_repdf']) and (not os.path.exists(self.Ens_repdfpath)):
			os.makedirs(self.Ens_repdfpath)	
		if (self.monitor_config.Ens_monitor['Ens_eva']) and (not os.path.exists(self.Ens_evapath)):
			os.makedirs(self.Ens_evapath)
		# if (self.monitor_config.Ens_monitor['Ens_endpdf']) and (not os.path.exists(self.Ens_endpdfpath)):
		# 	os.makedirs(self.Ens_endpdfpath)
		if (self.monitor_config.Best_monitor['if']) and (not os.path.exists(self.Best_save_path)):
			os.makedirs(self.Best_save_path)	
		if (self.monitor_config.Best_monitor['Best_repdf']) and (not os.path.exists(self.Best_repdfpath)):
			os.makedirs(self.Best_repdfpath)	
		if (self.monitor_config.Best_monitor['Best_eva']) and (not os.path.exists(self.Best_evapath)):
			os.makedirs(self.Best_evapath)
		# operate config
		# self.condpdf_plotting_points = np.array(self.monitor_config.pdf_monitor['points'])
		self.Delta    = self.eqn_config.Delta
		self.N_epochs = self.net_config.N_epochs
		# global operation
		self.config = config
		self.NFModel = NFModel

	## Excution
	def complete_condpdf(self,model,epoch,best=False,enforce=False):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.repdf_display['times']))==0) or (epoch==0) or enforce:
			# logging.info('--------------Plotting final pdf on Epoch %d'%(epoch+1))
			## check if model list
			path = self.repdfpath if type(model) != list else self.Ens_repdfpath
			if best:
				path = self.Best_repdfpath

			if self.eqn_config.dim==1 and self.eqn_config.eqn_name not in ['SSASchlogl']:
				## draw
				int_long = self.monitor_config.repdf_display['int_long']
				p_size = self.monitor_config.repdf_display['size']
				px,py = p_size
				l1,l2 = self.monitor_config.repdf_display['range']
				p_grid = (np.linspace(l1,l2,px*py)).reshape([px,py])
				fig, axes = plt.subplots(nrows=px, ncols=py, figsize=(py*3, px*2), constrained_layout=True, squeeze=False)
				for i in range(px):
					for j in range(py):
						axes[i,j].set_title("$X_s=$%.2f, ite %d"%(p_grid[i,j],epoch+1))
						self.condpdf_plotting_std(self.eqn_config.eqn_name,axes[i,j],int_long,p_grid[i,j],self.Delta)
						self.condpdf_plotting_data(self.eqn_config.eqn_name,axes[i,j],model,int_long,p_grid[i,j],self.Delta)
				fig.savefig(path+'/finalpdf'+str(epoch+1)+'.png',dpi=150)
				plt.close()
				## draw
				# logging.info('--------------End plotting final pdf on Epoch %d'%(epoch+1))
			elif self.eqn_config.eqn_name in ['SSASchlogl']:
				N = 10000
				data_dic = sio.loadmat(self.monitor_config.repdf_display['path'])
				px,py = (data_dic['size'].astype('int')).flatten()

				fig, axes = plt.subplots(nrows=px, ncols=py, figsize=(py*3, px*2), constrained_layout=True, squeeze=False)
				for i in range(px):
					for j in range(py):
						ini = data_dic[str(i*py+j)+'_i']
						dat_std = (data_dic[str(i*py+j)+'_d'])
						dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
						axes[i,j].set_title("$X_s=$%.1f, ite %d"%(ini,epoch+1))
						axes[i,j].hist(dat_mod.flatten(), bins=50, density=True, color='#DC143C',histtype='step')
						axes[i,j].hist(dat_std.flatten(), bins=50, density=True, color='#4169E1',histtype='step')
				fig.savefig(path+'/finalpdf_'+str(epoch+1)+'.png',dpi=150)
				plt.close()
			elif self.eqn_config.dim==2 or self.eqn_config.eqn_name in ['SSAautocatalytic']:
				if self.eqn_config.eqn_name in ['MdOU','SO']:
					level = [0,3,6,9,12,15,18,21,24]
					## draw
					int_long = self.monitor_config.repdf_display['int_long']
					p_size = self.monitor_config.repdf_display['size']
					px,py = p_size
					l1,l2 = self.monitor_config.repdf_display['range']
					p_gridx = np.linspace(l1[0],l1[1],px)
					p_gridy = np.linspace(l2[0],l2[1],py)
					p_gridx,p_gridy = np.meshgrid(p_gridx,p_gridy)
					p_grid = np.array((p_gridx.flatten(),p_gridy.flatten())).T
					fig, axes = plt.subplots(nrows=px, ncols=py*2, figsize=(py*3*2, px*2), constrained_layout=True, squeeze=False)
					for i in range(px):
						for j in range(py):
							axes[i,j*2].set_title("$X_s=$(%.2f,%.2f), ite %d"%(p_grid[(i*py+j)][0],p_grid[(i*py+j)][1],epoch+1))
							axes[i,j*2+1].set_title("$X_s=$(%.2f,%.2f), ite %d"%(p_grid[(i*py+j)][0],p_grid[(i*py+j)][1],epoch+1))
							# self.condpdf_plotting_std2D(self.eqn_config.eqn_name,axes[i,j*2],int_long,p_grid[(i*py+j)],self.Delta,level)
							# self.condpdf_plotting_data2D(self.eqn_config.eqn_name,axes[i,j*2+1],model,int_long,p_grid[(i*py+j)],self.Delta,level)
							self.condmargpdf_plotting_std2D(self.eqn_config.eqn_name,axes[i,j*2],axes[i,j*2+1],int_long,p_grid[(i*py+j)],self.Delta)
							self.condmargpdf_plotting_data2D(self.eqn_config.eqn_name,axes[i,j*2],axes[i,j*2+1],model,int_long,p_grid[(i*py+j)],self.Delta)
					fig.savefig(path+'/finalpdf'+str(epoch+1)+'.png',dpi=150)
					plt.close()
				elif self.eqn_config.eqn_name in ['SSALV','SSABrusselator','SSAautocatalytic']:
					N = 10000
					data_dic = sio.loadmat(self.monitor_config.repdf_display['path'])
					px,py = (data_dic['size'].astype('int')).flatten()
					fig, axes = plt.subplots(nrows=px, ncols=py*2, figsize=(py*3*2, px*2), constrained_layout=True, squeeze=False)
					for i in range(px):
						for j in range(py):
							ini = data_dic[str(i*py+j)+'_i'].flatten()
							dat_std = data_dic[str(i*py+j)+'_d']
							dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
							a_ = min(np.min(dat_std[:, 0]),np.min(dat_mod[:, 0]))
							b_ = max(np.max(dat_std[:, 0]),np.max(dat_mod[:, 0]))
							c_ = min(np.min(dat_std[:, 1]),np.min(dat_mod[:, 1]))
							d_ = max(np.max(dat_std[:, 1]),np.max(dat_mod[:, 1]))
							range_ = np.array(((a_,b_),(c_,d_)))
							axes[i,j*2].set_title("$X_s=$(%.1f,%.1f), ite %d"%(ini[0],ini[1],epoch+1))
							axes[i,j*2+1].set_title("$X_s=$(%.1f,%.1f), ite %d"%(ini[0],ini[1],epoch+1))
							axes[i,j*2].hist2d(dat_mod[:, 0], dat_mod[:, 1], bins=30, range=range_, density=True,cmap='Reds')
							axes[i,j*2+1].hist2d(dat_std[:, 0], dat_std[:, 1], bins=30, range=range_, density=True,cmap='Blues')
					fig.savefig(path+'/finalpdf_2D_'+str(epoch+1)+'.png',dpi=150)
					plt.close()

					fig, axes = plt.subplots(nrows=px, ncols=py*2, figsize=(py*3*2.5, px*2), constrained_layout=True, squeeze=False)
					for i in range(px):
						for j in range(py):
							ini = data_dic[str(i*py+j)+'_i'].flatten()
							dat_std = data_dic[str(i*py+j)+'_d']
							dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
							axes[i,j*2].set_title("$X_s=$(%.1f,%.1f), dim 1, ite %d"%(ini[0],ini[1],epoch+1))
							axes[i,j*2+1].set_title("$X_s=$(%.1f,%.1f), dim 2, ite %d"%(ini[0],ini[1],epoch+1))
							axes[i,j*2].hist(dat_mod[:, 0], bins=50, density=False, color='#DC143C',histtype='step')
							axes[i,j*2].hist(dat_std[:, 0], bins=50, density=False, color='#4169E1',histtype='step')
							axes[i,j*2+1].hist(dat_mod[:, 1], bins=50, density=False, color='#DC143C',histtype='step')
							axes[i,j*2+1].hist(dat_std[:, 1], bins=50, density=False, color='#4169E1',histtype='step')
					fig.savefig(path+'/finalpdf_Margin_'+str(epoch+1)+'.png',dpi=150)
					plt.close()
				elif self.eqn_config.eqn_name in ['SSAmRNAwDynk']:
					N = 10000
					dim = self.eqn_config.dim
					data_dic = sio.loadmat(self.monitor_config.repdf_display['path'])
					px,py = (data_dic['size'].astype('int')).flatten()
					if dim>=5:
						px = px*py
						py = 1

					fig, axes = plt.subplots(nrows=px, ncols=py*dim, figsize=(py*(dim+1)*2.5, px*2), constrained_layout=True, squeeze=False)
					for i in range(px):
						for j in range(py):
							ini = data_dic[str(i*py+j)+'_i'].flatten()
							para = data_dic['para_'+str(i*py+j)+'_i'].flatten()
							dat_std = data_dic[str(i*py+j)+'_d']
							dat_mod = (model.predict(np.tile(ini,[N,1]),np.tile(para,[N,1]))).detach().numpy()
							for k in range(dim):
								axes[i,j*dim+k].set_title("$X_s=$(%.1f,%.1f), dim %d, ite %d"%(ini[0],ini[1],k+1,epoch+1))
								axes[i,j*dim+k].hist(dat_mod[:, k], bins=50, density=True, color='#DC143C',histtype='step')
								axes[i,j*dim+k].hist(dat_std[:, k], bins=50, density=True, color='#4169E1',histtype='step')
					fig.savefig(path+'/finalpdf_Margin_'+str(epoch+1)+'.png',dpi=150)
					plt.close()
			elif self.eqn_config.dim>2:
				if self.eqn_config.eqn_name in ['SSAOregonator','SSACIRC73s','SSAVilar2002R','SHeatEqu','SHeatEqu_modal','SAdvDiff','SAdvDiff_modal']:
					N = 10000
					dim = self.eqn_config.dim
					data_dic = sio.loadmat(self.monitor_config.repdf_display['path'])
					px,py = (data_dic['size'].astype('int')).flatten()
					if dim>=5:
						px = px*py
						py = 1

					fig, axes = plt.subplots(nrows=px, ncols=py*dim, figsize=(py*(dim+1)*2.5, px*2), constrained_layout=True, squeeze=False)
					for i in range(px):
						for j in range(py):
							ini = data_dic[str(i*py+j)+'_i'].flatten()
							dat_std = data_dic[str(i*py+j)+'_d']
							dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
							for k in range(dim):
								axes[i,j*dim+k].set_title("$X_s=$(%.1f,%.1f), dim %d, ite %d"%(ini[0],ini[1],k+1,epoch+1))
								axes[i,j*dim+k].hist(dat_mod[:, k], bins=50, density=True, color='#DC143C',histtype='step')
								axes[i,j*dim+k].hist(dat_std[:, k], bins=50, density=True, color='#4169E1',histtype='step')
					fig.savefig(path+'/finalpdf_Margin_'+str(epoch+1)+'.png',dpi=150)
					plt.close()
				else:
					raise AttributeError('complete_condpdf: no this type of 2D example')
			else:
				pass
		else:
			pass

	def cond_meanvar(self,model,epoch,enforce=False):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.cond_mv['times']))==0) or (epoch==0) or enforce:
			## check if model list
			path = self.cond_mvpath if type(model) != list else self.Ens_cond_mvpath
			if self.eqn_config.dim==1:
				## compute
				Npoint = self.monitor_config.cond_mv['Npoint']
				l1,l2 = self.monitor_config.cond_mv['range']
				p_grid = np.linspace(l1,l2,Npoint+1)
				Mean_t, Std_t = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
				Mean_d, Std_d = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
				for i in range(p_grid.shape[0]):
					Mean_t[i],Std_t[i] = self.condmv_plotting_std_cont(self.eqn_config.eqn_name,p_grid[i],self.Delta)
					Mean_d[i],Std_d[i] = self.condmv_plotting_data(model,p_grid[i])
				if self.eqn_config.eqn_name=='Exp_OU':
					Mean_d = (np.log(Mean_d)-np.log(p_grid))/self.eqn_config.Delta
				else:
					Mean_d = (Mean_d-p_grid)/self.eqn_config.Delta
					Std_d = Std_d/np.sqrt(self.eqn_config.Delta)
				## draw
				fig, axes = plt.subplots(ncols=4, figsize=(24, 5), constrained_layout=True)
				axes[0].plot(p_grid,Mean_t,linestyle='-',color='black')
				axes[0].plot(p_grid,Mean_d,linestyle='dashed',color='#6495ED')
				axes[0].set_title("Comparasion of Mean $E(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0].set_xlabel('$X_s$')
				axes[1].plot(p_grid,np.zeros(p_grid.shape),linestyle='-',color='black')
				axes[1].plot(p_grid,Mean_d-Mean_t,linestyle='dashed',color='#6495ED')
				axes[1].set_title("Error of Mean $E(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1].set_xlabel('$X_s$')
				axes[2].plot(p_grid,Std_t,linestyle='-',color='black')
				axes[2].plot(p_grid,Std_d,linestyle='dashed',color='red')
				axes[2].set_title("Comparasion of Std $Std(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[2].set_xlabel('$X_s$')
				axes[3].plot(p_grid,np.zeros(p_grid.shape),linestyle='-',color='black')
				axes[3].plot(p_grid,Std_d-Std_t,linestyle='dashed',color='red')
				axes[3].set_title("Error of Std $Std(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[3].set_xlabel('$X_s$')

				if (epoch+1==self.N_epochs):
					Err_d = np.sqrt(np.sum((Mean_d-Mean_t)**2)*(l2-l1)/Npoint)/np.sqrt(np.sum((Mean_t)**2)*(l2-l1)/Npoint)
					Err_s = np.sqrt(np.sum((Std_d-Std_t)**2)*(l2-l1)/Npoint)/np.sqrt(np.sum((Std_t)**2)*(l2-l1)/Npoint)
					errpath  = './error.mat'
					if os.path.exists(errpath):
						data = sio.loadmat(errpath)
						data['Err_d'] = np.append(data['Err_d'],Err_d)
						data['Err_s'] = np.append(data['Err_s'],Err_s)
						sio.savemat(errpath,data)
					else:
						sio.savemat(errpath,{'Err_d':np.array((Err_d)),'Err_s':np.array((Err_s))})

				fig.savefig(path+'/cond_mvplot'+str(epoch+1)+'.png',dpi=150)
				plt.close()
			elif self.eqn_config.dim==2:
				# if self.eqn_config.eqn_name in ['MdOU']:
				# 	mzlim = [-3,3]
				# elif self.eqn_config.eqn_name in ['SO']:
				# 	mzlim = [-5.5,5.5]
				# else:
				# 	raise AttributeError('cond_meanvar: no this 2d distribution')
				## draw
				Npoint = self.monitor_config.cond_mv['Npoint']
				l1,l2 = self.monitor_config.cond_mv['range']
				p_gridx = np.linspace(l1[0],l1[1],Npoint+1)
				p_gridy = np.linspace(l2[0],l2[1],Npoint+1)
				p_gridx,p_gridy = np.meshgrid(p_gridx,p_gridy)
				p_grid = np.array((p_gridx.flatten(),p_gridy.flatten())).T
				PSh = p_grid.shape
				Mean_t, V_t, C_t = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
				Mean_d, V_d, C_d = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
				for i in range(p_grid.shape[0]):
					Mean_t[i],V_t[i],C_t[i] = self.condmv_plotting_std_cont2D(self.eqn_config.eqn_name,p_grid[i],self.Delta)
					Mean_d[i],V_d[i],C_d[i] = self.condmv_plotting_data2D(model,p_grid[i])
				## draw
				fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 15), constrained_layout=True, subplot_kw={"projection": "3d"})
				# means
				axes[0,0].plot_surface(p_gridx, p_gridy, Mean_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[0,0].set_title("Truth Mean $E_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,1].plot_surface(p_gridx, p_gridy, Mean_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[0,1].set_title("Estimated Mean $E_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,2].plot_surface(p_gridx, p_gridy, Mean_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[0,2].set_title("Truth Mean $E_2(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,3].plot_surface(p_gridx, p_gridy, Mean_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[0,3].set_title("Estimated Mean $E_2(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,0].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
				axes[0,1].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
				axes[0,2].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
				axes[0,3].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
				# variances
				axes[1,0].plot_surface(p_gridx, p_gridy, V_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[1,0].set_title("Truth variance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1,1].plot_surface(p_gridx, p_gridy, V_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[1,1].set_title("Estimated variance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1,2].plot_surface(p_gridx, p_gridy, V_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[1,2].set_title("Truth variance $Var_2(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1,3].plot_surface(p_gridx, p_gridy, V_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[1,3].set_title("Estimated variance $Var_2(\cdot|X_s)$, ite %d"%(epoch+1))
				# covariance
				axes[2,0].plot_surface(p_gridx, p_gridy, C_t.reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[2,0].set_title("Truth Covariance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[2,1].plot_surface(p_gridx, p_gridy, C_d.reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[2,1].set_title("Estimated Covariance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				fig.savefig(path+'/cond_mvplot'+str(epoch+1)+'.png',dpi=150)
				plt.close()
			else:
				pass
		else:
			pass

	def compare_stoppingtime(self,model,epoch,best=False,enforce=False):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.repdf_display['times']))==0) or enforce:
			if self.eqn_config.eqn_name=='SSALV':
				cretier = lambda x: (np.abs(x[:,0])<1.0e-8)+(np.abs(x[:,1])<1.0e-8)
				Nmax = 10000
			if best:
				path = self.Best_repdfpath

			path = self.repdfpath
			N = 10000
			data_dic = sio.loadmat(self.monitor_config.stoppingtime['path'])
			px,py = (data_dic['size'].astype('int')).flatten()

			fig, axes = plt.subplots(nrows=px, ncols=py, figsize=(py*6,px*4), constrained_layout=True, squeeze=False)
			for i in range(px):
				for j in range(py):
					ini = data_dic[str(i*py+j)+'_i'].flatten()
					dat_std = data_dic[str(i*py+j)+'_d']
					dat_mod = self.compute_stoppingtime(model,ini,N,cretier,Nmax)
					axes[i,j].set_title("$X_s=$(%.1f,%.1f), ite %d"%(ini[0],ini[1],epoch+1))
					axes[i,j].hist(np.array(dat_mod).flatten(), bins=50, density=False, color='#DC143C',histtype='step')
					axes[i,j].hist(np.array(dat_std).flatten(), bins=50, density=False, color='#4169E1',histtype='step')
			fig.savefig(path+'/stoppingtime'+str(epoch+1)+'.png',dpi=150)
			plt.close()

	def compute_stoppingtime(self,model,ini,N_data,cretier,NTmax):
		count = 1
		re = []
		dat = np.tile(ini,[N_data,1])
		while count<=NTmax:
			dat_mod = (model.predict(dat)).detach().numpy()
			id_stop = np.where(cretier(dat_mod))[0]
			id_rest = np.delete(np.arange(dat.shape[0]),id_stop)
			re += [self.eqn_config.Delta*count]*len(id_stop)
			dat = dat_mod[id_rest]
			count += 1
		if len(dat)>0:
			re += [int(NTmax*1.1)*self.eqn_config.Delta]*len(dat)
		return re

	def Eva_meanv(self,model,epoch,DatVes,predt_path):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.Evameanv['times']))==0) or (epoch==0):
			DatVes.test_mdat1model(model,predt_path)
			self.Evaulation.plot_meancompare(save=True,epoch=('E'+str(epoch+1)))
			if 'Eva_lag' in self.monitor_config.Evameanv.keys() and self.monitor_config.Evameanv['Eva_lag']>0:
				if not os.path.exists(self.rawpath+'Eva_Lag'):
					os.makedirs(self.rawpath+'Eva_Lag')
				try:
					pre_data = DatVes.test_mdat1model(model,'',mode='d',lag=self.monitor_config.Evameanv['Eva_lag'])
					self.Evaulation.plot_meancompare(save=True,pre_sav=[pre_data,self.rawpath+'Eva_Lag'],epoch=('E'+str(epoch+1)))
				except:
					pass
			if self.eqn_config.eqn_name in ['SSASchlogl']:
				self.Evaulation.plot_pdfcompare(save=True,epoch=('PDF_'+str(epoch+1)))
			if hasattr(self.monitor_config,"stoppingtime") and self.monitor_config.stoppingtime['if']:
				self.compare_stoppingtime(model,epoch)
			if 'sample' in self.monitor_config.Evameanv.keys() and self.monitor_config.Evameanv['sample']:
				if self.eqn_config.eqn_name in ['StochasticRes','SSALV','SSABrusselator','SSAOregonator','SSAautocatalytic','SSACIRC73s','SSAVilar2002R','SSAmRNAwDynk','SSASchlogl','CSFlockingModel']:
					self.Evaulation.plot_samples_block(save=True,epoch=('Block'+str(epoch+1)))
				if self.eqn_config.eqn_name in ['SSATransfer','SSALV','SSABrusselator','SSAOregonator','SSAautocatalytic','SSASchlogl','REx2_3DOssilator']:
					self.Evaulation.plot_samples_ens(save=True,epoch=('Ens'+str(epoch+1)))
				if self.eqn_config.eqn_name in ['SSATransfer','SSALV','SSABrusselator','SSAOregonator','SSAautocatalytic']:
					self.Evaulation.plot_sample_fft_block(save=True,epoch=('FFT'+str(epoch+1)))
			# if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if']:
			# 	self.Evaulation.plot_meancompare_Resplus(save=True,epoch=('E'+str(epoch+1)))
		else:
			pass

	def Eva_loss(self,model,epoch,Logprob_data,Logdet_data,LogprobTrue):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.loss['times']))==0):
			self.Evaulation.plot_train_hisNF(self.N_epochs,Logprob_data,Logdet_data,LogprobTrue,savepath=(self.loss_path+'/loss.png'))
		else:
			pass

	def Eva_lr(self,model,epoch,lr_data):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.loss['times']))==0):
			self.Evaulation.plot_lr_hisNF(self.N_epochs,lr_data,savepath=(self.loss_path+'/lr.png'))
		else:
			pass

	## Conditional mean and variance
	def condmv_plotting_data(self,model,x,N=100000):
		try:
			data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros(N)
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.repeat(x,modelsep[i+1]-modelsep[i])[:,None])).detach().numpy().flatten()
		# data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
		m,s = np.mean(data),np.std(data)
		return m,s

	def condmv_plotting_std_cont(self,name,x,Delta):
		if name=='Brownian Motion':
			m,s = x,np.sqrt(Delta)
			return m,s
		elif name=='Geometric Brownian Motion':
			m,s = self.eqn_config.mu*x,self.eqn_config.sigma*np.abs(x)
			return m,s
		elif name=='OU Process':
			m   = self.eqn_config.theta*(self.eqn_config.mu-x)
			var = self.eqn_config.sigma**2
			return m,np.sqrt(var)
		elif name=='Exp_diffusion':
			m   = -self.eqn_config.mu*x
			std = self.eqn_config.sigma*np.exp(-x**2)
			return m,std
		elif name=='Trig_drift':
			m   = np.sin(2*self.eqn_config.k*np.pi*x)
			std = abs(self.eqn_config.sigma*np.cos(2*self.eqn_config.k*np.pi*x))
			return m,std
		elif name=='Exp_OU':
			th,dt  = self.eqn_config.theta,Delta
			mu,sig = self.eqn_config.mu,self.eqn_config.sigma
			MU,SIG = (1-th*dt)*np.log(x)+th*mu*dt,sig*np.sqrt(dt)
			m = -th*np.log(x)+th*mu+sig**2/2
			var = (np.exp(SIG**2)-1)*np.exp(2*MU+SIG**2)
			return m,np.sqrt(var)
		elif name=='Double_well':
			m   = x-x**3
			std = self.eqn_config.sigma
			return m,std
		elif name=='Exp_dis':
			m = self.eqn_config.theta*x+self.eqn_config.sigma/np.sqrt(self.eqn_config.Delta)
			std = self.eqn_config.sigma
			return m,std
		elif name=='SSASchlogl':
			if abs(self.eqn_config.Delta-0.1)<1e-8:
				x_int = np.linspace(50,600,40).astype('int')
				m_int = np.array([ 52.781,  30.123,   8.501,  -8.653, -22.328, -32.778, -39.98,  -45.478, -45.653, 
						  -43.908, -37.981, -28.864, -22.884, -12.824,  -0.838,  13.087,  24.869,  40.901,
						  53.244,  74.764,  86.171,  96.292, 111.249, 118.588, 129.256, 141.608, 141.595, 
						  146.941, 142.821, 138.646, 132.425, 123.419, 107.178,  90.879,  73.821,  52.977, 
						  23.282, -11.865, -39.073, -84.41 ])
				s_int = np.array([18.6997755,  20.81017749, 22.44785736, 23.81267644, 26.00465423, 27.9065274, 30.21155342, 
							32.6175896,  34.70705057, 37.00211823, 39.32171619, 41.71077739, 44.52545625, 46.74045895, 
							49.50743152, 51.49067919, 54.20479945, 56.16505871, 58.87350547, 60.99364254, 63.73116095, 
							66.2839428,  68.22331713, 70.25066993, 72.60895707, 72.95817592, 76.6365748,  78.51384115, 
							79.59568579, 80.19355752, 82.26923445, 83.4868615,  83.04496151, 85.69703458, 85.93791361, 
							86.84290844, 89.16571565, 87.7298078,  90.40481772, 91.17570504])
				f_m = scipy.interpolate.interp1d(x_int, m_int)
				f_s = scipy.interpolate.interp1d(x_int, s_int)
				m   = f_m(x)
				std = f_s(x)
				return m,std
			else:
				return 0,0
		elif name=='LagvinSchlogl':
			m   = 3e-2*x*(x-1)/2-1e-4*x*(x-1)*(x-2)/6+200-3.5*x
			std = self.eqn_config.s
			return m,std
		else:
			print('The distribution %s is not supported'%(name))

	def condmv_plotting_std_cont2D(self,name,x,Delta):
		if name in ['MdOU','SO']:
			m = np.dot(np.array(self.eqn_config.mu),np.array((x)))
			s = np.array(self.eqn_config.sigma)
			cov = (s.T).dot(s)*Delta
			return m,np.diagonal(cov),cov[0,1]
		else:
			print('The distribution %s is not supported'%(name))

	def condmv_plotting_data(self,model,x,N=100000):
		try:
			data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros(N)
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.repeat(x,modelsep[i+1]-modelsep[i])[:,None])).detach().numpy().flatten()
		# data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
		m,s = np.mean(data),np.std(data)
		return m,s

	def condmv_plotting_data2D(self,model,x,N=5000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).detach().numpy()
		# data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		m,s = (np.mean(data,axis=0)-x)/self.eqn_config.Delta,np.cov(data.T)
		return m,np.diagonal(s),s[0,1]

	## Conditional pdf
	def condpdf_plotting_std(self,name,ax,intlong,x,Delta):
		if name=='Brownian Motion':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, x, np.sqrt(Delta)),color='#000080',label='Reference')
		elif name=='Geometric Brownian Motion':
			x_axis = np.linspace(x*np.exp(self.eqn_config.mu*Delta)-intlong/2*x/3,x*np.exp(self.eqn_config.mu*Delta)+intlong/2*x/3,200)
			cgeobw = (self.eqn_config.mu-(self.eqn_config.sigma**2)/2)
			GeoBpdf = np.zeros(x_axis.shape)
			_id = (x_axis>0)*(np.abs(x_axis)>1.0e-9)
			GeoBpdf[_id] = np.exp(-(np.log(x_axis[_id]/x)-cgeobw*Delta)**2/(2*self.eqn_config.sigma**2*Delta))/(np.sqrt(2*np.pi*Delta)*self.eqn_config.sigma*x_axis[_id])
			ax.plot(x_axis, GeoBpdf,color='#000080',label='Reference')
		elif name=='OU Process':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = self.eqn_config.mu+(x-self.eqn_config.mu)*np.exp(-self.eqn_config.theta*Delta)
			var = self.eqn_config.sigma**2/(2*self.eqn_config.theta)*(1-np.exp(-2*self.eqn_config.theta*Delta))
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, np.sqrt(var)),color='#000080',label='Reference')
		# elif name=='Exp_diffusion':
		# 	x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
		# 	mapsample = self.cond_sample_EM(name,x,Delta)
		# 	kde = scipy.stats.kde.gaussian_kde(mapsample)
		# 	ax.plot(x_axis, kde(x_axis), color='#000080',label='Reference')
		elif name=='Exp_diffusion':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = x-self.eqn_config.mu*x*Delta
			std = self.eqn_config.sigma*np.exp(-x**2)*np.sqrt(Delta)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, std),color='#000080',label='Reference')
		# elif name=='Trig_drift':
		# 	x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
		# 	mapsample = self.cond_sample_EM(name,x,Delta)
		# 	kde = scipy.stats.kde.gaussian_kde(mapsample)
		# 	ax.plot(x_axis, kde(x_axis), color='#000080',label='Reference')
		elif name=='Trig_drift':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = x+np.sin(2*self.eqn_config.k*np.pi*x)*Delta
			std = abs(self.eqn_config.sigma*np.cos(2*self.eqn_config.k*np.pi*x))*np.sqrt(Delta)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, std),color='#000080',label='Reference')
		elif name=='Exp_OU':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			x_axis = x_axis[x_axis>0]
			th,dt  = self.eqn_config.theta,self.eqn_config.Delta
			mu,sig = self.eqn_config.mu,self.eqn_config.sigma
			MU,SIG = (1-th*dt)*np.log(x)+th*mu*dt,sig*np.sqrt(dt)
			pdf = 1/(x_axis*SIG*np.sqrt(2*np.pi))*np.exp(-(np.log(x_axis)-MU)**2/(2*SIG**2))
			ax.plot(x_axis, pdf, color='#000080',label='Reference')
		# elif name=='Double_well':
		# 	x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
		# 	mapsample = self.cond_sample_EM(name,x,Delta)
		# 	kde = scipy.stats.kde.gaussian_kde(mapsample)
		# 	ax.plot(x_axis, kde(x_axis), color='#000080',label='Reference')
		elif name=='Double_well':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = x+(x-x**3)*Delta
			std = self.eqn_config.sigma*np.sqrt(Delta)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, std),color='#000080',label='Reference')
		elif name=='Exp_dis':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			a = x+self.eqn_config.theta*x*self.eqn_config.Delta
			b = self.eqn_config.sigma*np.sqrt(self.eqn_config.Delta)
			_id = x_axis>=a
			x_plot = x_axis[_id]
			pdf_plot = np.exp(-(x_axis[_id]-a)/b)/b
			ax.plot(x_plot, pdf_plot,color='#000080',label='Reference')
			ax.plot([a,a], [0,1/b],color='#000080')
		elif name=='Skew-Product SDE':
			NUM_data = 10000
			sample_x = x*np.ones(NUM_data)
			y = np.random.normal(0.0, np.sqrt(self.eqn_config.lambda_/self.eqn_config.alpha*2), NUM_data)
			sample_y = sample_x+Delta*(1-y**2)*sample_x
			kde = scipy.stats.kde.gaussian_kde(sample_y)
			dist_space = np.linspace(np.mean(sample_y)-4*np.std(sample_y),np.mean(sample_y)+4*np.std(sample_y),200)
			ax.plot(dist_space,kde(dist_space),linestyle='dashed',color='red')
		elif name=='LagvinSchlogl':
			mea = x+(3e-2*x*(x-1)/2-1e-4*x*(x-1)*(x-2)/6+200-3.5*x)*Delta
			std = self.eqn_config.s*np.sqrt(Delta)
			x_axis = np.linspace(mea-3*std,mea+3*std,200)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, std),color='#000080',label='Reference')
		else:
			pass
			# print('The distribution %s is not supported'%(name))

	def condpdf_plotting_std2D(self,name,ax,intlong,x,Delta,level):
		if name=='MdOU':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			distx,disty = np.meshgrid(distx,disty)
			rv = scipy.stats.multivariate_normal(Mean, Cov)
			f = rv.pdf(np.dstack((distx,disty)))
			cfset = ax.contourf(distx, disty, f, cmap='Blues')
			cset = ax.contour(distx, disty, f, colors='k')
			ax.clabel(cset, inline=1, fontsize=7)
		elif name=='SO':
			pass
		else:
			print('The distribution %s is not supported'%(name))

	def condpdf_combine_plotting_std2D(self,name,intlong,x,Delta,level):
		if name=='MdOU':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			data = np.random.multivariate_normal(Mean, Cov, size=500000)
			xlimit = [x[0]-intlong[0]/1.4,x[0]+intlong[0]/1.4]
			ylimit = [x[1]-intlong[1]/1.4,x[1]+intlong[1]/1.4]
			a = sns.jointplot(x=data[:,0], y=data[:,1], fill=True, kind="kde", color="#004C99", levels=level, xlim=xlimit,ylim=ylimit, height=6)
			font2 = {'size'   : 14,}
			patch = matplotlib.patches.Patch(color='#004C99', alpha=0.3, label='Reference')
			plt.legend(handles=[patch],prop=font2)
			return a
		elif name=='SO':
			pass
		else:
			print('The distribution %s is not supported'%(name))

	def condmargpdf_plotting_std2D(self,name,ax1,ax2,intlong,x,Delta):
		if name=='MdOU':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			ax1.plot(distx, scipy.stats.norm.pdf(distx, Mean[0], np.sqrt(Cov[0,0])),color='#000080',label='Reference')
			ax2.plot(disty, scipy.stats.norm.pdf(disty, Mean[1], np.sqrt(Cov[1,1])),color='#000080',label='Reference')
			# rv = scipy.stats.multivariate_normal(Mean, Cov)
			# f = rv.pdf(np.dstack((distx,disty)))
			# cfset = ax.contourf(distx, disty, f, cmap='Blues')
			# cset = ax.contour(distx, disty, f, colors='k')
			# ax.clabel(cset, inline=1, fontsize=7)
		elif name=='SO':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			distx = np.linspace(x[0]-intlong[0]/5,x[0]+intlong[0]/5,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			## ax1
			ax1.plot([Mean[0]],[0],color='#000080',label='Reference', marker=".", markersize=25)
			## alternate for ax1
			# ax1.set_ylim([-10,540])
			# ax1.annotate("",xy=(Mean[0], 520), xycoords='data',xytext=(Mean[0], 0), textcoords='data',arrowprops=dict(arrowstyle="-|>, head_width=0.1",mutation_scale=30,connectionstyle="arc3",color='#000080'),)
			# ax1.plot(distx,np.zeros(distx.shape),color='#000080')
			# ax1.plot([Mean[0]],[0],color='#000080',label='Reference', marker='o', markerfacecolor='white', markersize=8)
			# ax1.set_xlim([x[0]-intlong[0]/10,x[0]+intlong[0]/10])
			## ax2
			ax2.plot(disty, scipy.stats.norm.pdf(disty, Mean[1], np.sqrt(Cov[1,1])),color='#000080',label='Reference')
			ax2.set_xlim([x[1]-intlong[0]/8,x[1]+intlong[0]/8])
		else:
			print('The distribution %s is not supported'%(name))

	def condpdf_plotting_data(self,name,ax,model,intlong,x,Delta,N=10000):
		try:
			data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros(N)
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.repeat(x,modelsep[i+1]-modelsep[i])[:,None])).detach().numpy().flatten()
		# data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
		kde = scipy.stats.kde.gaussian_kde(data)
		if name=='Brownian Motion':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Geometric Brownian Motion':
			dist_space = np.linspace(x-intlong/2*x/3,x+intlong/2*x/3,200)
		elif name=='OU Process':
			m = self.eqn_config.mu+(x-self.eqn_config.mu)*np.exp(-self.eqn_config.theta*Delta)
			dist_space = np.linspace(m-intlong/2,m+intlong/2,200)
		elif name=='Exp_diffusion':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Trig_drift':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Exp_OU':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
			dist_space = dist_space[dist_space>0]
		elif name=='Double_well':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Exp_dis':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		else:
			pass
			# print('The distribution %s is not supported'%(name))
		# pdb.set_trace()
		# ax.plot(dist_space,kde(dist_space),linestyle='dashed',color='red')
		ax.hist(data, bins=50, alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')

	def condpdf_plotting_data2D(self,name,ax,model,intlong,x,Delta,level,N=10000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).detach().numpy()
		# data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		kde = scipy.stats.kde.gaussian_kde(data.T)
		if name in ['MdOU','SO']:
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			distx,disty = np.meshgrid(distx,disty)
		else:
			print('The distribution %s is not supported'%(name))
		# pdb.set_trace()
		f = np.reshape(kde(np.vstack([distx.ravel(), disty.ravel()])), distx.shape)
		cfset = ax.contourf(distx, disty, f, cmap='Reds')
		cset = ax.contour(distx, disty, f, colors='k')
		ax.clabel(cset, inline=1, fontsize=7)

	def condpdf_combine_plotting_data2D(self,name,model,intlong,x,Delta,level,N=10000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).detach().numpy()
		# data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		if name in ['MdOU','SO']:
			xlimit = [x[0]-intlong[0]/1.4,x[0]+intlong[0]/1.4]
			ylimit = [x[1]-intlong[1]/1.4,x[1]+intlong[1]/1.4]
		else:
			print('The distribution %s is not supported'%(name))
		# pdb.set_trace()
		a = sns.jointplot(x=data[:,0], y=data[:,1], fill=True, kind="kde", color="#990000", levels=level, xlim=xlimit,ylim=ylimit, height=6)
		font2 = {'size'   : 14,}
		patch = matplotlib.patches.Patch(color='#990000', alpha=0.3, label='Learned')
		plt.legend(handles=[patch],prop=font2)
		return a

	def condmargpdf_plotting_data2D(self,name,ax1,ax2,model,intlong,x,Delta,N=10000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).detach().numpy()
		# data = (model.predict(np.tile(x,[N,1]))).detach().numpy()
		if name=='MdOU':
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
		elif name=='SO':
			distx = np.linspace(x[0]-intlong[0]/5,x[0]+intlong[0]/5,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
		else:
			print('The distribution %s is not supported'%(name))
		kde = scipy.stats.kde.gaussian_kde(data[:,0])
		## ax1
		# ax1.plot(distx,kde(distx),color='#DC143C',linestyle='dashed',label='Learned')
		ax1.hist(data[:,0], bins=50, alpha=0.6, ec="k", density=True, histtype='stepfilled',label='Learned')
		## ax2
		ax2.hist(data[:,1], bins=50, alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')

	## Tool
	def cond_sample_EM(self,equname,x,Delta):
		N = 10000
		if equname=='Exp_diffusion':
			xt = x-self.eqn_config.mu*x*Delta+self.eqn_config.sigma*np.exp(-x**2)*np.random.normal(0.0, np.sqrt(Delta), N)
		elif equname=='Trig_drift':
			xt = x+np.sin(2*self.eqn_config.k*np.pi*x)*Delta+self.eqn_config.sigma*np.cos(2*self.eqn_config.k*np.pi*x)*np.random.normal(0.0, np.sqrt(Delta), N)
		elif equname=='Double_well':
			xt = x+(x-x**3)*Delta+self.eqn_config.sigma*np.random.normal(0.0, np.sqrt(Delta), N)
		else:
			print('The distribution %s is not supported'%(name))
		return xt

	def pdf_plotting_data(self,ax,data,intlong,line_,color_,label_='Ground Truth'):
		m = np.mean(data)
		kde = scipy.stats.kde.gaussian_kde(data)
		dist_space = np.linspace(m-intlong/2,m+intlong/2,200)
		ax.plot(dist_space,kde(dist_space),linestyle=line_,color=color_,label=label_)

	## Data plot
	def data2dhistogram(self,data,Delta,name):
		# data should in the form of [num of trajectory, trajectory]
		N_data, Trac_long = data.shape
		x = np.arange(Trac_long)*Delta
		# fine the data
		num_fine = 800
		x_fine = np.linspace(x.min(), x.max(), num_fine)
		y_fine = np.empty((N_data, num_fine), dtype=float)
		for i in range(N_data):
			y_fine[i, :] = np.interp(x_fine, x, data[i, :])
		data_ = y_fine.flatten()
		x_ = np.tile(x_fine, N_data)
		# draw
		fig, ax = plt.subplots(ncols=1, figsize=(12, 4), constrained_layout=True)
		h, xedges, yedges = np.histogram2d(x_, data_, bins=[400, 100])
		pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=plt.cm.plasma, vmax=np.max(h), rasterized=True)
		# pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=plt.cm.plasma, norm=matplotlib.colors.LogNorm(vmax=5.5e2), rasterized=True)
		fig.colorbar(pcm, ax=ax, label="# points", pad=0)
		ax.set_title("Hit Diagram of %s"%(name))
		ax.set_xlabel("Time")
		ax.set_ylabel("Value of Data")
		fig.savefig(self.dataplotpath+'/hist_'+name+'.png',dpi=250)
		plt.close()

	def transprobinfo(self,data,name):
		# data should in the form of [num of trajectory, trajectory]
		data_in = (data[:,:-1]).flatten()
		fig, ax = plt.subplots(ncols=1, figsize=(12, 4), constrained_layout=True)
		ax.hist(data_in, bins=50, alpha=0.6, ec="k", histtype='stepfilled')
		ax.set_title("Number of trasition input data $X_s$ from %s"%(name))
		ax.set_xlabel("Value of $X_s$")
		ax.set_ylabel("# of data")
		ax.grid(alpha=0.7)
		fig.savefig(self.dataplotpath+'/hist_input_'+name+'.png',dpi=250)
		plt.close()

	## Ensemble related
	def Eva_Ensemble(self,modellist,DatVes,epoch):
		N_T = (DatVes.test_data).shape[1]
		data_ = DatVes.datachoose((np.vstack(DatVes.test_data)).T, DatVes.dim, np.zeros([DatVes.test_data.shape[-1],1],dtype=int), 1)
		Xs = np.tile(data_[:,:DatVes.dim],(len(modellist),1))
		pre = [Xs]
		for i in range(N_T-1):
			Xs = self.Mulmodel_Generate(modellist,Xs)
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

	def Endpdf_Ensemble(self,modellist,DatVes,epoch):
		data_ = DatVes.datachoose((np.vstack(DatVes.test_data)).T, DatVes.dim, np.zeros([DatVes.test_data.shape[-1],1],dtype=int), 1)
		Xs = np.tile(data_[:,:DatVes.dim],(len(modellist),1))
		for i in range(modellist[0].d_RNN-1):
			Xs = self.Mulmodel_Generate(modellist,Xs)
		pre_ = Xs.T
		save_ = (self.Ens_endpdfpath+'/'+str(epoch+1)+'P'+'.pdf')
		fig,ax = self.Evaulation.plot_endpdfGeneralD(DatVes.test_data[:,-1,:],pre_,DatVes.dim,savepath=save_)
		plt.close()

	def Mulmodel_Generate(self,modellist,Xs):
		Nmodel = len(modellist)
		modelid = np.random.randint(Nmodel, size=Xs.shape[0])
		Xre = np.zeros(Xs.shape)
		for j in range(Nmodel):
			_id = np.where(modelid==j)[0]
			Xre[_id] = modellist[j].predict(Xs[_id]).detach().numpy()
		return Xre

	def Ens_monitor(self,epoch,ckptmanager,model,DatVes):
		if not (set(['Testepoches_comp','Testepoches_endp']) <= set(dir(self))):
			self.Testepoches_comp,self.Testepoches_endp = self.Last_epoch_schedule(model.Testepoches)
		if (epoch+1) in self.Testepoches_comp:
			ckptmanager.Ensemble_save(model)
			# evaluate
			if ((epoch+1) in self.Testepoches_endp) or (epoch+1==self.N_epochs):
				modellist = self.readMultiplemodel(ckptmanager,model)
				if self.monitor_config.Ens_monitor['Ens_cond_mv']:
					self.cond_meanvar(modellist,epoch,enforce=True)
				if self.monitor_config.Ens_monitor['Ens_repdf']:
					self.complete_condpdf(modellist,epoch,enforce=True)
				if self.monitor_config.Ens_monitor['Ens_eva']:
					self.Eva_Ensemble(modellist,DatVes,epoch)
				# if self.monitor_config.Ens_monitor['Ens_endpdf']:
				# 	self.Endpdf_Ensemble(modellist,DatVes,epoch)

	def Best_monitor(self,epoch,loss_v,model,DatVes):
		if not hasattr(model, 'min_loss'):
			model.min_loss = loss_v
		if (loss_v > model.min_loss) or (epoch==0):
			torch.save(model.state_dict(), self.Best_save_path+'model.pt')
			model.min_loss = loss_v
		# evaluate
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.Evameanv['times']))==0) or (epoch==0):
			model_ = self.readmodel(self.Best_save_path+'model.pt',model)
			try:
				model_.noirange = DatVes.noirange
			except:
				pass
			if 'Best_repdf' in self.monitor_config.Best_monitor.keys() and self.monitor_config.Best_monitor['Best_repdf']:
				self.complete_condpdf(model_,epoch,best=True,enforce=True)
				# self.compare_stoppingtime(model_,epoch,best=True,enforce=False)
			if 'Best_eva' in self.monitor_config.Best_monitor.keys() and self.monitor_config.Best_monitor['Best_eva']:
				pre_data = DatVes.test_mdat1model(model_,'',mode='d')
				self.Evaulation.plot_meancompare(save=True,pre_sav=[pre_data,self.Best_evapath],epoch=('E'+str(epoch+1)))
			if 'Best_eva_lag' in self.monitor_config.Best_monitor.keys() and self.monitor_config.Best_monitor['Best_eva_lag']>0:
				if not os.path.exists(self.Best_evapath+'_Lag'):
					os.makedirs(self.Best_evapath+'_Lag')
				try:
					pre_data = DatVes.test_mdat1model(model_,'',mode='d',lag=self.monitor_config.Best_monitor['Best_eva_lag'])
					self.Evaulation.plot_meancompare(save=True,pre_sav=[pre_data,self.Best_evapath+'_Lag'],epoch=('E'+str(epoch+1)))
				except:
					pass
			if 'sample' in self.monitor_config.Best_monitor.keys() and self.monitor_config.Best_monitor['sample']:
				if self.eqn_config.eqn_name in ['StochasticRes','SSALV','SSABrusselator','SSAOregonator','SSAautocatalytic','SSACIRC73s','SSAVilar2002R','SSAmRNAwDynk']:
					pre_data = DatVes.test_mdat1model(model_,'',mode='d') if 'pre_data' not in locals() else pre_data
					self.Evaulation.plot_sample_block(pre_data,self.Delta,savepath=self.Best_evapath+'/'+'Block'+str(epoch+1)+'.png')
				elif self.eqn_config.eqn_name in ['REx2_3DOssilator']:
					pre_data = DatVes.test_mdat1model(model_,'',mode='d') if 'pre_data' not in locals() else pre_data
					test_data = (sio.loadmat(DatVes.test_data_path))['data']
					self.Evaulation.plot_sample_ens(test_data,pre_data,self.Delta,savepath=self.Best_evapath+'/'+'Ens'+str(epoch+1)+'.png')

	def Last_epoch_schedule(self,Testepoches):
		# schedule ensemble monitor test
		step = int(self.N_epochs/self.monitor_config.Evameanv['times'])
		end_p = step*(np.arange(int(self.N_epochs/step))+1)
		long = Testepoches[-1]-Testepoches[0]
		if step<long:
			raise AttributeError('Last_epoch_schedule: too many times for Eva_meanv_Multiple_last, please change to lower')
		# end_p = end_p[(end_p>long)*(end_p<Testepoches[0])]
		end_p = end_p[(end_p>long)]
		com_p = np.zeros(Testepoches.shape[0]*end_p.shape[0],dtype=int)
		for i in range(end_p.shape[0]):
			com_p[i*Testepoches.shape[0]:(i+1)*Testepoches.shape[0]] = Testepoches-(Testepoches[-1]-end_p[i])
		return com_p,end_p

	def readMultiplemodel(self,ckptmanager,model):
		# This function is designed for test for multiple models
		modellist = []
		modeldict = ckptmanager.list_models()
		for i in range(len(modeldict)):
			ModelX = self.NFModel(self.config)
			ModelX.load_state_dict(torch.load(modeldict[i]))
			# ModelX.eval()
			modellist.append(ModelX)
		return modellist

	def readmodel(self,path,model):
		# This function is designed for test for single models
		ModelX = self.NFModel(self.config)
		ModelX.load_state_dict(torch.load(path))
		return ModelX


class DataTran():
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

		if self.eqn_config.eqn_name=='SSALV' and (hasattr(self.dat_config, 'ConstrainedPred') and self.dat_config.ConstrainedPred):
			self.test_mdat1model = self.test_mdat1model_SSALV_Cons
		else:
			self.test_mdat1model = self.test_mdat1model_regular

	def read_traindata(self):
		# train data is assumed to be stored under key 'data' of matfile
		# train data in this function is in the form of [dim,n_of_time_step,n_of_tracjectory]
		try:
			self.train_data = (sio.loadmat(self.train_data_path))['data']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file.')
		self.dim, self.L_Nmax, self.N_long_traj = (self.train_data).shape
		self.n_train = self.n_ea_traj * self.N_long_traj

	def read_testdata(self):
		try:
			self.test_data = (sio.loadmat(self.test_data_path))['data']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file.')
		self.dim = (self.test_data).shape[0]

	def test_mdat1model_regular(self,model,save_path,mode='w'):
		L_Nmax_Test = (self.test_data).shape[1]
		Nullstart = True if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if'] else False
		pred = self.test_tensordata(self.test_data,model,L_Nmax_Test,Nullstart)
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

	# def test_mdat1model_SSALV_Cons(self,model,save_path,mode='w'):
	# 	L_Nmax_Test = (self.test_data).shape[1]
	# 	Nullstart = True if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if'] else False

	# 	pred = np.zeros(self.test_data.shape)
	# 	Ndata = self.test_data.shape[-1]
	# 	count = 0
	# 	trail = 0
	# 	while count<=Ndata and trail<=10:
	# 		pred_   = self.test_tensordata(self.test_data,model,L_Nmax_Test,Nullstart)
	# 		pred_rv = np.unique(np.where(np.abs(pred_)<1.0e-8)[-1])
	# 		pred_id  = np.delete(np.arange(Ndata), pred_rv)
	# 		N_pred  = min(len(pred_id),Ndata-count)
	# 		pred[:,:,count:count+N_pred] = pred_[:,:,pred_id[:N_pred]]
	# 		count += N_pred
	# 		trail += 1
	# 	pred = pred[:,:,:max(count,10)]
		
	# 	if mode=='w':
	# 		self.pred = pred
	# 		sio.savemat(save_path,{'pred':self.pred})
	# 	elif mode=='a':
	# 		self.pred = pred
	# 		if os.path.exists(save_path):
	# 			data_exist = (sio.loadmat(save_path))['pred']
	# 			self.pred = np.concatenate([data_exist,self.pred],axis=-1)
	# 		sio.savemat(save_path,{'pred':self.pred})
	# 	elif mode=='d':
	# 		return pred

	def test_mdat1model_SSALV_Cons(self,model,save_path,mode='w'):
		L_Nmax_Test = (self.test_data).shape[1]
		Nullstart = True if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if'] else False

		pred = np.zeros(self.test_data.shape)
		Ndata = self.test_data.shape[-1]
		count = 0
		trail = 0
		const = lambda x: (np.abs(x[:,0])<1.0e-8)+(np.abs(x[:,1])<1.0e-8)
		while count<Ndata and trail<=10:
			pred_   = self.test_tensordata_withconstraint(self.test_data,model,L_Nmax_Test,const,Nullstart)
			N_pred  = min(pred_.shape[-1],Ndata-count)
			pred[:,:,count:count+N_pred] = pred_[:,:,:N_pred]
			count += N_pred
			trail += 1
		pred = pred[:,:,:max(count,10)]
		
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

	def test_tensordata_withconstraint(self,test_data,model,N_T,constraint,Nullstart=False):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
		# Xs = torch.from_numpy(data_[:,:self.dim]).to(torch.float32)
		Xs = data_[:,:self.dim]
		if Nullstart:
			Xs = torch.zeros(Xs.shape)
		pre_ = np.zeros([self.dim,N_T,test_data.shape[-1]])
		pre_[:,0,:] = Xs.T
		for i in np.arange(N_T-1)+1:
			with torch.no_grad():
				Xs = model.predict(Xs)
				id_stop = np.where(constraint(Xs))[0]
				id_rest = np.delete(np.arange(Xs.shape[0]),id_stop)
				if len(id_rest)>0:
					pre_ = pre_[:,:,id_rest]
					Xs   = Xs[id_rest]
					pre_[:,i,:] = (Xs.detach().numpy()).T
				else:
					pre_ = pre_[:,:,[]]
					break
		return pre_

	def test_mdat1model_SSALV_Cons_Twomodel(self,model,save_path,mode='w'):
		L_Nmax_Test = (self.test_data).shape[1]
		Nullstart = True if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if'] else False

		pred = np.zeros(self.test_data.shape)
		Ndata = self.test_data.shape[-1]
		count = 0
		trail = 0
		const = lambda x: (np.abs(x[:,0])<1.0e-8)+(np.abs(x[:,1])<1.0e-8)
		while count<Ndata and trail<=0:
			pred_   = self.test_tensordata_withconstraint_Twomodel(self.test_data,model,L_Nmax_Test,const,Nullstart)
			N_pred  = min(pred_.shape[-1],Ndata-count)
			pred[:,:,count:count+N_pred] = pred_[:,:,:N_pred]
			count += N_pred
			trail += 1
		pred = pred[:,:,:max(count,10)]
		
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

	def test_tensordata_withconstraint_Twomodel(self,test_data,model,N_T,constraint,Nullstart=False):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		# conssss = lambda x: (np.abs(x[:,0])>200)*(np.abs(x[:,0])<10000)*(np.abs(x[:,1])>200)*(np.abs(x[:,1])<10000)
		conssss = lambda x: ((np.abs(x[:,0])>-1)*(np.abs(x[:,0])<-1))+((np.abs(x[:,1])>-1)*(np.abs(x[:,1])<-1))
		# conssss = lambda x: (np.abs(x[:,1])>200)*(np.abs(x[:,1])<10000)
		
		data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
		# Xs = torch.from_numpy(data_[:,:self.dim]).to(torch.float32)
		Xs = data_[:,:self.dim]
		if Nullstart:
			Xs = torch.zeros(Xs.shape)
		pre_ = np.zeros([self.dim,N_T,test_data.shape[-1]])
		pre_[:,0,:] = Xs.T
		for i in np.arange(N_T-1)+1:
			print(i)
			with torch.no_grad():
				Xsnew = model.predict(Xs)
				print(len(Xsnew))
				id_twomodel = conssss(Xs)
				Xsnew[id_twomodel] = torch.from_numpy((self.Model2.predict(Xs[id_twomodel])).astype('float32'))
				# Xsnew[id_twomodel] = self.Model2.predict(Xs[id_twomodel])
				id_stop = np.where(constraint(Xsnew))[0]
				id_rest = np.delete(np.arange(Xsnew.shape[0]),id_stop)
				if len(id_rest)>0:
					pre_ = pre_[:,:,id_rest]
					Xsnew   = Xsnew[id_rest]
					pre_[:,i,:] = (Xsnew.detach().numpy()).T
				else:
					pre_ = pre_[:,:,[]]
					break
				# pre_[:,i,:] = (Xsnew.detach().numpy()).T
				Xs = Xsnew
		return pre_

	def train_data_trans_traj(self,seed_):
		smaple_L_Nmax = self.L_Nmax-self.d_RNN
		# random setting
		np.random.seed(seed_)
		
		sample_init_L = np.random.randint(smaple_L_Nmax+1,size=(self.N_long_traj,self.n_ea_traj))
		# sample_init_L = np.tile(np.arange(self.n_ea_traj),[self.N_long_traj,1])
		# print('Warning: It is using uniform points selection')
		
		temp_wu = np.random.permutation(self.n_train)
		if ('zeroinit' in self.dat_config.keys()) and self.dat_config.zeroinit:
			sample_init_L = np.zeros(sample_init_L.shape,dtype=int)
		# data merging
		data_ = (np.vstack(self.train_data)).T
		# set train inputs and outputs
		train_mat = np.zeros((self.n_train, self.dim*self.d_RNN))
		for i in range(self.n_ea_traj):
			train_mat[i*self.N_long_traj:(i+1)*self.N_long_traj] = self.datachoose(data_, self.dim, sample_init_L[:,[i]], self.d_RNN)
		self.train_mat  = train_mat[temp_wu,:]
		# monitor
		if self.Monitor.monitor_config.traindata_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.data2dhistogram(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],self.eqn_config.Delta,"Train_data"+str(i))
		if self.Monitor.monitor_config.traintransin_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.transprobinfo(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],"Train_data"+str(i))

	def train_data_trans_pair(self,seed_):
		self.N_base = self.dat_config.N_train_base
		# data merging
		data_ = (np.vstack(self.train_data)).T
		# set train inputs and outputs
		train_mat = np.zeros((self.n_ea_traj*self.N_base, self.dim*self.d_RNN))
		temp_wu = np.random.permutation(data_.shape[0])
		data_     = data_[temp_wu,:]
		for i in range(self.n_ea_traj):
			train_mat[i*self.N_base:(i+1)*self.N_base,:self.dim] = data_[i*self.N_base:(i+1)*self.N_base,:-1:2]
			train_mat[i*self.N_base:(i+1)*self.N_base,self.dim:] = data_[i*self.N_base:(i+1)*self.N_base,1::2]
		self.train_mat  = train_mat
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

	def test_singledata(self,test_data,model,N_T):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose(test_data, self.dim, 0, 1)
		pre = np.zeros(N_T*self.dim)
		pre[:self.dim] = data_
		for i in range(N_T-1):
			next_time = model.predict(np.array([pre[self.dim*i:self.dim*(i+1)]]))
			pre[self.dim*(i+1):self.dim*(i+2)] = next_time
		pre = (pre.reshape([N_T,self.dim])).T
		return pre

	def test_tensordata(self,test_data,model,N_T,Nullstart=False):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
		# Xs = torch.from_numpy(data_[:,:self.dim]).to(torch.float32)
		Xs = data_[:,:self.dim]
		if Nullstart:
			Xs = torch.zeros(Xs.shape)
		pre = [Xs] 
		for i in range(N_T-1):
			# print(i)
			with torch.no_grad():
				Xs = model.predict(Xs)
				pre += [Xs.detach().numpy()]
		pre = np.concatenate(pre, -1)
		pre_ = np.zeros([self.dim,N_T,test_data.shape[-1]])
		for j in range(self.dim):
			pre_[j] = (pre[:,j::self.dim]).T
		# pre_ = np.zeros([self.dim,N_T,test_data.shape[-1]])
		# for i in np.arange(N_T-1):
		# 	with torch.no_grad():
		# 		Xs = model.predict(Xs)
		# 		pre_[:,i,:] = (Xs.detach().numpy()).T
		return pre_

	# def test_tensordata(self,test_data,model,N_T,Nullstart=False):
	# 	# data is in the form of [dim*n_of_time_step]
	# 	# aranging as [dim1_tracj, dim2_tracj,...]
	# 	LAG = 50
	# 	data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), LAG)
	# 	# Xs = torch.from_numpy(data_[:,:self.dim]).to(torch.float32)
	# 	Xs_i = data_[:,:self.dim*LAG]
	# 	Xs   = Xs_i[:,-self.dim:]
	# 	if Nullstart:
	# 		Xs = torch.zeros(Xs.shape)
	# 	pre = [Xs_i] 
	# 	for i in range(N_T-LAG):
	# 		# print(i)
	# 		with torch.no_grad():
	# 			Xs = model.predict(Xs)
	# 			pre += [Xs.detach().numpy()]
	# 	pre = np.concatenate(pre, -1)
	# 	pre_ = np.zeros([self.dim,N_T,test_data.shape[-1]])
	# 	for j in range(self.dim):
	# 		pre_[j] = (pre[:,j::self.dim]).T
	# 	# pre_ = np.zeros([self.dim,N_T,test_data.shape[-1]])
	# 	# for i in np.arange(N_T-1):
	# 	# 	with torch.no_grad():
	# 	# 		Xs = model.predict(Xs)
	# 	# 		pre_[:,i,:] = (Xs.detach().numpy()).T
	# 	return pre_

	def datachoose(self,data,dim,start,nstep):
		# data (1D or 2D) is in the form of [N,dim*n_of_time_step]
		# aranging as N*[dim1_tracj, dim2_tracj,...]
		# this function will choose consecutive 'nstep' of data from 'start' ([N,1]) index
		Ndata = 1 if (data.ndim==1) else data.shape[0]
		traclen = int(data.shape[-1]/dim)
		if (traclen-nstep)<np.max(start):
			raise AttributeError('DataTran::datachoose: Cant take so long step')
		ind = np.tile(np.arange(dim)*traclen,(nstep*Ndata,1))
		temp = (np.tile(np.arange(nstep),Ndata))[:,None]
		ind = (ind+temp).reshape([Ndata,dim*nstep])+start
		if data.ndim==1:
			ind = ind[0]
		return np.take_along_axis(data, ind, axis=(data.ndim-1))