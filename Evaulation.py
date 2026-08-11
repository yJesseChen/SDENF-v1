from __future__ import division
import pdb
import os
import munch
import json
import logging

import torch
import torch.nn as nn

import numpy as np
import numpy.linalg
import matplotlib
from matplotlib import pyplot as plt
import scipy
import scipy.io as sio
from scipy import stats
from numpy.fft import fft, ifft

class Evaluate():
	def plot_ode_compare(self,testdata,predictdata,Delta,savepath=None):
		xt = np.arange(testdata.shape[0])*Delta
		xp = np.arange(predictdata.shape[0])*Delta
		T = max(xt[-1],xp[-1])
		fig1, ax1 = plt.subplots(figsize=[10,7])
		ax1.plot(xt, testdata, color='black')
		ax1.plot(xp, predictdata, 'o', color='#6495ED', markerfacecolor='none')
		plt.xlim([-0.1*T,1.1*T])
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_train_hisNF(self,Nepoc,Logprob_data,Logdet_data,LogprobTrue,axisreset=False,savepath=None):
		x = np.arange(len(Logprob_data))
		Logprob_data = np.array(Logprob_data)
		if axisreset:
			bais = np.min(Logprob_data)
			shift = - bais + abs(bais)*0.1
			Logprob_data = Logprob_data + shift
		fig1, ax1 = plt.subplots(figsize=[10,7])
		# slice
		stp = int(len(x)*0.3)
		gp = ax1.plot(x[stp:], Logprob_data[stp:], color='#4169E1', label='Log Prob')
		if LogprobTrue!='None':
			if axisreset:
				LogprobTrue +=shift
			ax1.plot(x, LogprobTrue*np.ones(x.shape), linestyle='dashed', color='black', label='Ground Truth Log Prob')
		if axisreset:
			ax1.set_yscale('log')
		ax1.set_xlim([-100,Nepoc+100])
		# fig1.tight_layout()
		# gdps = gp+dp
		# labs = [l.get_label() for l in gdps]
		ax1.legend()
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_lr_hisNF(self,Nepoc,lr_data,savepath=None):
		x = np.arange(len(lr_data))
		fig1, ax1 = plt.subplots(figsize=[10,7])
		# ax1.plot(x, G_loss, color='#4169E1', label='Generator')
		# ax1.plot(x, D_loss, color='#DC143C', label='Discriminator')
		gp = ax1.plot(x, lr_data, color='#4169E1', label='Learning Rate')
		ax1.set_yscale('log')
		ax1.set_xlim([-100,Nepoc+100])
		ax1.legend()
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_index(self,Nepoc,data,name,savepath=None,log=True):
		x = np.arange(Nepoc)
		fig1, ax1 = plt.subplots(figsize=[10,7])
		ax1.plot(x, data, color='#0000FF', label=name)
		if log:
			ax1.set_yscale('log')
		ax1.legend()
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_sample(self,testdata,predictdata,Delta,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		# plot
		fig1, ax1 = plt.subplots(1,2,figsize=[20,7])
		for i in range(min(testdata.shape[0],200)):
			ax1[0].plot(xt_test, testdata[i])
			ax1[1].plot(xt_pred, predictdata[i])
		ax1[0].set_title('Ground Truth')
		ax1[1].set_title('Prediction')
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_meanstd(self,testdata,predictdata,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		xmean_test = np.mean(testdata,axis=0)
		xstde_test = np.std(testdata,axis=0,ddof=1)
		xt_test,xmean_test,xstde_test = xt_test[slice:],xmean_test[slice:],xstde_test[slice:]
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		xmean_pred = np.mean(predictdata,axis=0)
		xstde_pred = np.std(predictdata,axis=0,ddof=1)
		xt_pred,xmean_pred,xstde_pred = xt_pred[slice:],xmean_pred[slice:],xstde_pred[slice:]
		# Resdata
		if Resdata is not None:
			xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
		# Bound
		test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
		pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
		# plot
		fig1, ax1 = plt.subplots(figsize=[10,7])
		ax1.plot(xt_test, xmean_test, color='#4169E1', label='Ground Truth')
		ax1.fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
		ax1.plot(xt_pred, xmean_pred, color='#DC143C', label='Prediction')
		ax1.fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2)
		ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# ax1.set_ylim([-1.5,2.5])
		ax1.legend()
		if savepath is not None:
			fig1.savefig(savepath,dpi=200)
		return fig1,ax1

	def plot_meanstd_GD(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		# N_plot = min(dim,10)
		# fig1, ax1 = plt.subplots(ncols=N_plot, figsize=(10*N_plot, 7), squeeze=False)
		n_col = 5
		n_row = dim//n_col+int(dim%n_col!=0)
		if dim<=5:
			n_col = dim
		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2), constrained_layout=True, squeeze=False)
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					# Test data
					testdata,predictdata = testdataMD[num].T,predictdataMD[num].T
					xt_test = np.arange(testdata.shape[-1])*Delta
					xmean_test = np.mean(testdata,axis=0)
					xstde_test = np.std(testdata,axis=0,ddof=1)
					xt_test,xmean_test,xstde_test = xt_test[slice:],xmean_test[slice:],xstde_test[slice:]
					# Predict data
					xt_pred = np.arange(predictdata.shape[-1])*Delta
					xmean_pred = np.mean(predictdata,axis=0)
					xstde_pred = np.std(predictdata,axis=0,ddof=1)
					xt_pred,xmean_pred,xstde_pred = xt_pred[slice:],xmean_pred[slice:],xstde_pred[slice:]
					# Resdata
					if Resdata is not None:
						xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
					# Bound
					test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
					pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
					# plot
					axes[i,j].plot(xt_test, xmean_test, color='#4169E1', label='Ground Truth')
					axes[i,j].fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
					axes[i,j].plot(xt_pred, xmean_pred, color='#DC143C', label='Prediction')
					axes[i,j].fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2)
					axes[i,j].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
					# axes.set_ylim([-1.5,2.5])
					axes[i,j].legend()
				else:
					break
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,axes

	def plot_meanstd_SPDE_modal(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		if self.eqn_config.eqn_name in ['SHeatEqu_modal','SAdvDiff_modal','SHeatEqu_wSource_modal']:
			Ix = np.linspace(0,2*np.pi,50)
			Nx = Ix.shape[0]
			# modal matrix
			Basis = np.zeros([Nx,dim])
			for k in range(dim):
				K = (k+1)//2
				if k==0:
					col = np.ones(Nx)
				elif k%2==1:
					col = np.cos(K*Ix)
				elif k%2==0:
					col = np.sin(K*Ix)
				Basis[:,k] = col
		else:
			Ix = np.linspace(0,2*np.pi,50)

		NT = predictdataMD.shape[1]
		T_index = np.linspace(0,NT-1,10).astype('int')

		n_col = 5
		n_row = 2
		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2), constrained_layout=True, squeeze=False)
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					# T
					T_i = T_index[num]
					T_d = Delta*T_i
					# Test data
					testdata,predictdata = testdataMD[:,T_i,:],predictdataMD[:,T_i,:]
					test_nodal = np.dot(Basis,testdata).T
					predict_nodal = np.dot(Basis,predictdata).T

					xmean_test = np.mean(test_nodal,axis=0)
					xstde_test = np.std(test_nodal,axis=0,ddof=1)
					Ix = Ix[slice:]
					xmean_test,xstde_test = xmean_test[slice:],xstde_test[slice:]
					# Predict data
					xmean_pred = np.mean(predict_nodal,axis=0)
					xstde_pred = np.std(predict_nodal,axis=0,ddof=1)
					xmean_pred,xstde_pred = xmean_pred[slice:],xstde_pred[slice:]
					# Bound
					test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
					pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
					# plot
					axes[i,j].plot(Ix, xmean_test, color='#4169E1', label='Ground Truth')
					axes[i,j].fill_between(Ix, test_l, test_u, color='#4169E1', alpha=0.2)
					axes[i,j].plot(Ix, xmean_pred, color='#DC143C', label='Prediction')
					axes[i,j].fill_between(Ix, pred_l, pred_u, color='#DC143C', alpha=0.2)
					axes[i,j].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
					# axes.set_ylim([-1.5,2.5])
					axes[i,j].legend()
					axes[i,j].set_title('T = %.4f'%(T_d))
				else:
					break
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,axes

	def plot_meanstd_SPDE_nodal(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		if self.eqn_config.eqn_name in ['SHeatEqu','SAdvDiff','SHeatEqu_wSource']:
			Ix = np.linspace(0,2*np.pi,dim+1)
			PDEbd = self.bd_periodic
		else:
			Ix = np.linspace(0,2*np.pi,dim+1)

		NT = predictdataMD.shape[1]
		T_index = np.linspace(0,NT-1,10).astype('int')
		Nx = Ix.shape[0]

		n_col = 5
		n_row = 2
		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2), constrained_layout=True, squeeze=False)
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					# T
					T_i = T_index[num]
					T_d = Delta*T_i
					# Test data
					testdata,predictdata = testdataMD[:,T_i,:],predictdataMD[:,T_i,:]
					test_nodal = (PDEbd(testdata)).T
					predict_nodal = (PDEbd(predictdata)).T

					xmean_test = np.mean(test_nodal,axis=0)
					xstde_test = np.std(test_nodal,axis=0,ddof=1)
					Ix = Ix[slice:]
					xmean_test,xstde_test = xmean_test[slice:],xstde_test[slice:]
					# Predict data
					xmean_pred = np.mean(predict_nodal,axis=0)
					xstde_pred = np.std(predict_nodal,axis=0,ddof=1)
					xmean_pred,xstde_pred = xmean_pred[slice:],xstde_pred[slice:]
					# Bound
					test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
					pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
					# plot
					axes[i,j].plot(Ix, xmean_test, color='#4169E1', label='Ground Truth')
					axes[i,j].fill_between(Ix, test_l, test_u, color='#4169E1', alpha=0.2)
					axes[i,j].plot(Ix, xmean_pred, color='#DC143C', label='Prediction')
					axes[i,j].fill_between(Ix, pred_l, pred_u, color='#DC143C', alpha=0.2)
					axes[i,j].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
					# axes.set_ylim([-1.5,2.5])
					axes[i,j].legend()
					axes[i,j].set_title('T = %.4f'%(T_d))
				else:
					break
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,axes

	def bd_periodic(self,data):
		# data in form of [dim,Ndata]
		return np.concatenate([data,data[[0]]])

	def bd_homdirichlet(self,data):
		# data in form of [dim,Ndata]
		return np.concatenate([np.zeros(data.shape[-1]),data,np.zeros(data.shape[-1])])

	# def plot_sample_block(self,predictdata,Delta,savepath=None):
	# 	# data should be in the form of Ndata*test
	# 	# Predict data
	# 	xt_pred = np.arange(predictdata.shape[-1])*Delta
	# 	N = 3
	# 	# plot
	# 	fig1, axes = plt.subplots(nrows=N, ncols=N, figsize=(N*5, N*4), constrained_layout=True, squeeze=False)
	# 	count = 0
	# 	for i in range(N):
	# 		for j in range(N):
	# 			axes[i,j].plot(xt_pred, predictdata[i])
	# 			count += 1
	# 	if savepath is not None:
	# 		fig1.savefig(savepath,dpi=200)
	# 	return fig1, axes

	def plot_sample_block(self,predictdata,Delta,savepath=None):
		# data should be in the form of Ndata*test
		dim = predictdata.shape[0]
		Nt = predictdata.shape[1]
		Ndata = predictdata.shape[2]
		# Predict data
		xt_pred = np.arange(Nt)*Delta
		N = 3
		# plot
		fig1, axes = plt.subplots(nrows=N, ncols=N, figsize=(N*6, N*4), constrained_layout=True, squeeze=False)
		cmap = plt.get_cmap('winter')
		colors = [cmap(i) for i in np.linspace(0.1, 1, dim)]
		count = 0
		for m in range(N):
			for n in range(N):
				for i in range(dim):
					color = colors[i]
					axes[m,n].plot(xt_pred, (predictdata[i].T)[count], color=color, label='$X_{'+str(i+1)+'}$')
				axes[m,n].legend()
				count += 1
		if savepath is not None:
			fig1.savefig(savepath,dpi=200)
		return fig1, axes

	def plot_fft_block(self,predictdata,Delta,savepath=None):
		# data should be in the form of Ndata*test
		dim = predictdata.shape[0]
		Nt = predictdata.shape[1]
		Ndata = predictdata.shape[2]
		# Predict data
		xt_pred = np.arange(Nt)*Delta
		N = 3
		# plot
		fig1, axes = plt.subplots(nrows=N, ncols=N, figsize=(N*6, N*4), constrained_layout=True, squeeze=False)
		cmap = plt.get_cmap('winter')
		colors = [cmap(i) for i in np.linspace(0.1, 1, dim)]
		count = 0
		for m in range(N):
			for n in range(N):
				self.fft_plot((predictdata[0].T)[count],Delta,axes[m,n])
				count += 1
		if savepath is not None:
			fig1.savefig(savepath,dpi=200)
		return fig1, axes

	def fft_plot(self,data,Delta,ax):
		X = np.fft.fft(data)
		N = len(X)
		n = np.arange(N)
		sr = 1/Delta
		T = N/sr
		freq = n/T
		n_oneside = N//2
		a = 1/(freq[:n_oneside][1:])
		b = (np.abs(X)[:n_oneside]/n_oneside)[1:]
		ext = a[np.argmax(b)]

		ax.plot(a, b, 'b')
		ax.text(ext, -.033, '%.4f'%(ext), color='red', transform=ax.get_xaxis_transform(),ha='center', va='top')
		ax.set_xlabel('period')

	def plot_sample_ens(self,testdata,predictdata,Delta,savepath=None):
		# data should be in the form of Ndata*test
		dim = testdata.shape[0]
		Nt = testdata.shape[1]
		Ndata = testdata.shape[2]
		# Test data
		xt_test = np.arange(Nt)*Delta
		# Predict data
		xt_pred = np.arange(Nt)*Delta
		# plot
		fig1, ax1 = plt.subplots(1,2,figsize=[20,7])
		cmap = plt.get_cmap('winter')
		colors = [cmap(i) for i in np.linspace(0.1, 1, dim)]
		for i in range(dim):
			color = colors[i]
			for j in range(min(Ndata,10)):
				if j!=0:
					ax1[0].plot(xt_test, (testdata[i].T)[j], color=color)
					ax1[1].plot(xt_pred, (predictdata[i].T)[j], color=color)
				else:
					ax1[0].plot(xt_test, (testdata[i].T)[j], color=color, label='$X_'+str(i+1)+'$')
					ax1[1].plot(xt_pred, (predictdata[i].T)[j], color=color, label='$X_'+str(i+1)+'$')
		ax1[0].legend()
		ax1[1].legend()
		ax1[0].set_title('Ground Truth')
		ax1[1].set_title('Prediction')
		if savepath is not None:
			fig1.savefig(savepath,dpi=200)
		return fig1,ax1

	def plot_meanstdGeneralD(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		N_plot = min(dim,10)
		fig1, ax1 = plt.subplots(ncols=N_plot, figsize=(10*N_plot, 7), squeeze=False)
		for i in range(N_plot):
			# Test data
			testdata,predictdata = testdataMD[i].T,predictdataMD[i].T
			xt_test = np.arange(testdata.shape[-1])*Delta
			xmean_test = np.mean(testdata,axis=0)
			xstde_test = np.std(testdata,axis=0,ddof=1)
			xt_test,xmean_test,xstde_test = xt_test[slice:],xmean_test[slice:],xstde_test[slice:]
			# Predict data
			xt_pred = np.arange(predictdata.shape[-1])*Delta
			xmean_pred = np.mean(predictdata,axis=0)
			xstde_pred = np.std(predictdata,axis=0,ddof=1)
			xt_pred,xmean_pred,xstde_pred = xt_pred[slice:],xmean_pred[slice:],xstde_pred[slice:]
			# Resdata
			if Resdata is not None:
				xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
			# Bound
			test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
			pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
			# plot
			ax1[0,i].plot(xt_test, xmean_test, color='#4169E1', label='Ground Truth')
			ax1[0,i].fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
			ax1[0,i].plot(xt_pred, xmean_pred, color='#DC143C', label='Prediction')
			ax1[0,i].fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2)
			ax1[0,i].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
			# ax1.set_ylim([-1.5,2.5])
			ax1[0,i].legend()
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_endpdfGeneralD(self,testdataMD,predictdataMD,dim,savepath=None):
		# data should be in the form of dim*Ndata
		x_axis = np.linspace(-5,5,200)
		n_col = 5
		n_row = dim//n_col+int(dim%n_col!=0)
		if dim<=5:
			n_col = dim
		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*5, n_row*4), constrained_layout=True, squeeze=False)
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					mt,st = np.mean(testdataMD[num]),np.std(testdataMD[num])
					mp,sp = np.mean(predictdataMD[num]),np.std(predictdataMD[num])
					x_axis = np.linspace(min(mt-3*st,mp-3*sp),max(mt+3*st,mp+3*sp),500)
					# kde = scipy.stats.kde.gaussian_kde(testdataMD[num])
					# axes[i,j].plot(x_axis, kde(x_axis), color='#4169E1',label='Ground Truth')
					# kde = scipy.stats.kde.gaussian_kde(predictdataMD[num])
					# axes[i,j].plot(x_axis, kde(x_axis), color='#DC143C',label='Prediction')
					axes[i,j].hist(testdataMD[num], bins=200, alpha=1.0,color='#4169E1', density=True, histtype='step',label='Ground Truth',linewidth=1.3)
					axes[i,j].hist(predictdataMD[num], bins=200, alpha=1.0,color='#DC143C', density=True, histtype='step',label='Learned', ls='--', linewidth=1.3)
					axes[i,j].legend()
				else:
					break
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,axes

	def readmodel(path,Model,config):
		# This function is designed for test for single models
		ModelX = Model(config)
		ModelX.load_state_dict(torch.load(path),strict=False)
		return ModelX

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

class SdeNFEva(Evaluate):
	def __init__(self,config,result_path,save_path):
		self.eqn_config  = config.eqn_config
		self.net_config  = config.net_config
		self.dat_config  = config.dat_config
		self.result_path = result_path
		self.save_path   = save_path
		self.dim = self.eqn_config.dim
		self.Delta   = self.eqn_config.Delta
		self.n_epochs = self.net_config.N_epochs
		self.test_data_path  = self.dat_config.TestData_dir
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

	def plot_samplecompare(self,save=False):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		for i in range(min(self.dim,10)):
			save_ = (self.save_path+'/S'+str(i+1)+'.pdf') if save else None
			# if i==0:
			# 	pdb.set_trace()
			fig,ax = self.plot_sample(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)

	def plot_losthist(self,save=False):
		try:
			with open(self.result_path+'/Test_history.json') as json_data_file:
				file = json.load(json_data_file)
				Logprob_data = file['Logprob']
				Logdet_data = file['LogDet']
				LogprobTrue = file['LogprobTrue']
		except:
			raise AttributeError('SdeNFEva::plot_losshist: Fail to find loss data')
		save_ = (self.save_path+'/loss_hist.pdf') if save else None
		fig,ax = self.plot_train_hisNF(self.n_epochs,Logprob_data,Logdet_data,LogprobTrue,savepath=save_)

	def plot_Wdistance(self,save=False):
		try:
			with open(self.result_path+'/Test_history.json') as json_data_file:
				file = json.load(json_data_file)
				W_dist_data = file['W_dist']
				save_ = (self.save_path+'/W_dist.pdf') if save else None
				fig,ax = self.plot_index(self.n_epochs,W_dist_data,'Wasserstein Distance',savepath=save_,log=False)
		except:
			pass

	def plot_meancompare(self,save=False,pre_sav=None,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		if pre_sav is None:
			try:
				pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
				save_path_ = self.save_path
			except:
				raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		else:
			pre_data,save_path_ = pre_sav

		if self.dim<=5:
			for i in range(min(self.dim,10)):
				save_ = (save_path_+'/'+epoch+'M'+str(i+1)+'.png') if save else None
				# pdb.set_trace()
				fig,ax = self.plot_meanstd(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)
				# fig,ax = self.plot_meanstdGeneralD(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)
				plt.close()
		else:
			save_ = (save_path_+'/'+epoch+'M'+'.pdf') if save else None
			fig,ax = self.plot_meanstd_GD(test_data,pre_data,self.dim,self.Delta,savepath=save_)

		if self.eqn_config.eqn_name in ['SHeatEqu_modal','SAdvDiff_modal','SHeatEqu_wSource_modal']:
			save_ = (save_path_+'/'+epoch+'_Nodal_M'+'.png') if save else None
			fig,ax = self.plot_meanstd_SPDE_modal(test_data,pre_data,self.dim,self.Delta,savepath=save_)

		if self.eqn_config.eqn_name in ['SHeatEqu','SAdvDiff','SHeatEqu_wSource']:
			save_ = (save_path_+'/'+epoch+'_Nodal_M'+'.png') if save else None
			fig,ax = self.plot_meanstd_SPDE_nodal(test_data,pre_data,self.dim,self.Delta,savepath=save_)

	# def plot_meancompare_Resplus(self,save=False,epoch=''):
	# 	test_data = (sio.loadmat(self.test_data_path))['data']
	# 	Res_data = (sio.loadmat(self.eqn_config.Resdata['path']))['pred']
	# 	try:
	# 		pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
	# 	except:
	# 		raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
	# 	for i in range(min(self.dim,10)):
	# 		save_ = (self.save_path+'/'+epoch+'M'+str(i+1)+'.pdf') if save else None
	# 		fig,ax = self.plot_meanstd(test_data[i].T,pre_data[i].T,self.Delta,Resdata=Res_data[i],savepath=save_)

	def plot_samples_block(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('plot_samples_block: Fail to find prediction data')
		save_ = (self.save_path+'/'+epoch+'.png') if save else None
		fig,ax = self.plot_sample_block(pre_data,self.Delta,savepath=save_)

	def plot_samples_ens(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('plot_samples_block: Fail to find prediction data')
		save_ = (self.save_path+'/'+epoch+'.png') if save else None
		# pdb.set_trace()
		fig,ax = self.plot_sample_ens(test_data,pre_data,self.Delta,savepath=save_)
		# fig,ax = self.plot_meanstdGeneralD(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)
		plt.close()

	def plot_sample_fft_block(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('plot_FFT_block: Fail to find prediction data')
		save_ = (self.save_path+'/'+epoch+'.png') if save else None
		# pdb.set_trace()
		fig,ax = self.plot_fft_block(pre_data,self.Delta,savepath=save_)
		# fig,ax = self.plot_meanstdGeneralD(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)
		plt.close()

	def plot_pdfcompare(self,save=False,pre_sav=None,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		if pre_sav is None:
			try:
				pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
				save_path_ = self.save_path
			except:
				raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		else:
			pre_data,save_path_ = pre_sav

		save_ = (save_path_+'/'+epoch+'.pdf') if save else None
		fig,ax = self.plot_endpdfGeneralD(test_data[:,-1,:],pre_data[:,-1,:],self.dim,savepath=save_)



