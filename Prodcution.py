from __future__ import division
import pdb
import os
import munch
import json
import logging
import time

import torch
import torch.nn as nn

import numpy as np
import numpy.linalg
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
import scipy.io as sio
from scipy import stats
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

import Chemical_Dynamics
import myutils

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

	def plot_train_hisNF(self,Nepoc,Logprob_data,Logdet_data,LogprobTrue,axisreset=True,savepath=None):
		x = np.arange(len(Logprob_data))
		Logprob_data = np.array(Logprob_data)
		if axisreset:
			bais = np.min(Logprob_data)
			shift = - bais + abs(bais)*0.1
			Logprob_data = Logprob_data + shift
		fig1, ax1 = plt.subplots(figsize=[10,7])
		# ax1.plot(x, G_loss, color='#4169E1', label='Generator')
		# ax1.plot(x, D_loss, color='#DC143C', label='Discriminator')
		gp = ax1.plot(x, Logprob_data, color='#4169E1', label='Log Prob')
		if LogprobTrue!='None':
			if axisreset:
				LogprobTrue +=shift
			ax1.plot(x, LogprobTrue*np.ones(x.shape), linestyle='dashed', color='black', label='Ground Truth Log Prob')
		# ax1.tick_params(axis='y', labelcolor='#4169E1')
		# ax2 = ax1.twinx()
		# dp = ax1.plot(x, Logdet_data, color='#DC143C', label='Log Det')
		# ax2.tick_params(axis='y', labelcolor='#DC143C')
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

	def plot_sample(self,name,testdata,predictdata,Delta,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		N_T = testdata.shape[1]
		dim = testdata.shape[0]
		ylabels = [None for i in range(dim)]

		if self.eqn_config.eqn_name=='Geometric Brownian Motion':
			dataxlim,trainlim,Num = [[0,1]],[1],20
		elif self.eqn_config.eqn_name=='OU Process':
			dataxlim,trainlim,Num = [[0,4]],[1],10
		elif self.eqn_config.eqn_name=='Exp_diffusion':
			dataxlim,trainlim,Num = [[0,10]],[1],5
		elif self.eqn_config.eqn_name=='Trig_drift':
			dataxlim,trainlim,Num = [[0,10]],[1],1
		elif self.eqn_config.eqn_name=='Exp_OU':
			dataxlim,trainlim,Num = [[0,5]],[1],10
		elif self.eqn_config.eqn_name=='Double_well':
			dataxlim,trainlim,Num = [[0,1]],[1],2
		elif self.eqn_config.eqn_name=='Exp_dis':
			dataxlim,trainlim,Num = [[0,5]],[1],10
		elif self.eqn_config.eqn_name=='MdOU':
			dataxlim,trainlim,Num = [[0,5],[0,5]],[1,1],10
		elif self.eqn_config.eqn_name=='SO':
			dataxlim,trainlim,Num = [[0,5],[0,5]],[1,1],10
		elif name=='DisturbOU':
			Num = 10
			ylabels = ['$x$']
		elif name=='Ex19BiStochsticOU':
			Num = 10
			ylabels = ['$x$']
		elif name=='Ex16Multiscale':
			Num = 10
			ylabels = ['$x$','$y$']
		elif name=="Ex17PredPrey":
			Num = 10
			ylabels = ['$x$','$y$']
		elif name=="StochasticRes":
			Num = 10
		elif name=="SSASchlogl":
			Num = 10
		else:
			Num = 5
		
		if name in ['StochasticRes','SSASchlogl']:
			plt.rcParams['text.usetex'] = True
			for i in range(Num):
				fig2,ax2 = self.plotmultipledata_Doublewell((predictdata[0].T)[[i]],[0,(N_T-1)*Delta],1,"OrRd",save=savepath+'/PredSample'+str(i+1)+'.pdf')
		else:
			for i in range(dim):
				plt.rcParams['text.usetex'] = True
				# "OrRd", "Blues"
				fig1,ax1 = self.plotmultipledata((testdata[i].T),[0,(N_T-1)*Delta],Num,"Blues",ylabels=ylabels[i],save=savepath+'/TruthSample'+str(i)+'.pdf')
				fig2,ax2 = self.plotmultipledata((predictdata[i].T)[20:],[0,(N_T-1)*Delta],Num,"OrRd",ylabels=ylabels[i],save=savepath+'/PredSample'+str(i)+'.pdf')
			if self.eqn_config.dim==2:
				if name=='Ex17PredPrey':
					Nump = 10
				else:
					Nump = 5
				plt.rcParams['text.usetex'] = True
				fig3,ax3 = self.plotmultipledata2Dphase(testdata[0].T,testdata[1].T,Nump,"Blues",save=savepath+'/TruthPhase'+'.pdf')
				fig4,ax4 = self.plotmultipledata2Dphase(predictdata[0].T,predictdata[1].T,Nump,"OrRd",save=savepath+'/PredPhase'+'.pdf')
			if self.eqn_config.eqn_name in ['REx2_3DOssilator','REx4_pendulum','REx7_YGO']:
				ylabels = [r'$\xi_%d$'%(i+1) for i in range(dim)]
				if self.eqn_config.eqn_name=='REx2_3DOssilator':
					delay_ = 50
				elif self.eqn_config.eqn_name=='REx4_pendulum':
					delay_ = 50
				elif self.eqn_config.eqn_name=='REx7_YGO':
					delay_ = 50
				else:
					delay_ = 1

				for i in range(dim):
					plt.rcParams['text.usetex'] = True
					# "OrRd", "Blues"
					fig1,ax1 = self.plotmultipledata((testdata[i].T),[0,(N_T-1)*Delta],Num,"Blues",ylabels=ylabels[i],save=savepath+'/TruthSample'+str(i)+'.pdf')
					fig2,ax2 = self.plotmultipledata_mem((predictdata[i].T)[20:],[0,(N_T-1)*Delta],Num,"OrRd",delay_,ylabels=ylabels[i],save=savepath+'/PredSample'+str(i)+'.pdf')

	def plot_sample_block(self,name,testdata,predictdata,Delta,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		N_T = testdata.shape[1]
		dim = testdata.shape[0]

		if name=='SSATransfer':
			Num = 9
		elif name=="SSALV":
			Num = 9
			configs = {'figsize':[4,4]}
			index = 18
		elif name=="SSABrusselator":
			Num = 9
			configs = {'figsize':[4,4]}
			index = 0
		elif name=='SSAOregonator':
			Num = 9
			configs = {'view':[20,45],'scale':[0.7,0.7,1.4],'12_figsize':[2.2,4],'13_figsize':[1.8,4],'23_figsize':[2.8,4]}
			index = 0
		elif name=='SSACIRC73s':
			Num = 9
			index_t = 1
			index_p = 3
			figman = [4,4]
		elif name=='SSAVilar2002R':
			Num = 9
			index_t = 5
			index_p = 4
			figman = [3,3]
		else:
			Num = 9
			configs = {}
			index = 0
		
		plt.rcParams['text.usetex'] = True
		if self.eqn_config.dim<=5:
			for i in range(Num):
				fig2,ax2 = self.plot_singledata_multidim(predictdata[:,:,index+i],[0,(N_T-1)*Delta],save=savepath+'/PredSample_block'+str(i+1)+'.pdf')
		else:
			self.plot_singledata_multidim(predictdata[:,:,index_p],[0,(N_T-1)*Delta],sepfig=figman,colorss='#DC143C',save=savepath+'/PredSample_block.pdf')
			self.plot_singledata_multidim(testdata[:,:,index_t],[0,(N_T-1)*Delta],sepfig=figman,colorss='#4169E1',save=savepath+'/TestSample_block.pdf')
		if self.eqn_config.dim==2:
			for i in range(Num):
				fig3,ax3 = self.plotsingledata2Dphase(predictdata[0,:,index+i],predictdata[1,:,index+i],"#004c99",configs,save=savepath+'/PredPhase'+str(i+1)+'.pdf')
		if self.eqn_config.dim==3:
			for i in range(Num):
				# fig3,ax3 = self.plotsingledata3Dphase(predictdata[0,:,index+i],predictdata[1,:,index+i],predictdata[2,:,index+i],"#004c99",configs,save=savepath+'/PredPhase'+str(i+1)+'.pdf')
				try:
					self.plotsingledata3Dpairphase(predictdata[0,:,index+i],predictdata[1,:,index+i],predictdata[2,:,index+i],"#004c99",configs,save=savepath+'/PredPhase'+str(i+1))
				except:
					pass
		if name in ['SSAmRNAwDynk']:
			if name=='SSAmRNAwDynk':
				labels = [r'M',r'P']
			for i in range(Num):
				self.plot_singledata_gilispie_version_biaxis(np.arange(N_T)*Delta,predictdata[:,:,index+i],colors=['#4169E1','#DC143C'],labels=labels,save=savepath+'/PredSample_block_biaxis_'+str(i+1)+'.pdf')
			for i in range(Num):
				self.plot_singledata_gilispie_version_biaxis(np.arange(N_T)*Delta,testdata[:,:,index+i],colors=['#ADCAC9','#6F4D46'],labels=labels,save=savepath+'/TestSample_block_biaxis_grid_'+str(i+1)+'.pdf')

	def plot_fft_block(self,name,predictdata,Delta,slice=0,savepath=None):
		# Disabled in the public package: original SSA/reference path plotting relied on local data files.
		pass

	def plot_glispie_ori(self,name,savepath=None):
		# Disabled in the public package: original SSA/Gillespie trajectory plots relied on local data files.
		pass

	def plot_multiscale_ori(self,name,savepath=None):
		# Disabled in the public package: original multiscale trajectory plots relied on local data files.
		pass

	def plot_sample_ens(self,name,testdata,predictdata,Delta,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		N_T = testdata.shape[1]
		dim = testdata.shape[0]

		if name=='SSATransfer':
			Num = 20
		elif name=="SSALV":
			Num = 30
		else:
			Num = 10
		
		plt.rcParams['text.usetex'] = True
		fig2,ax2 = self.plot_ens_multidim(predictdata,[0,(N_T-1)*Delta],Num,save=savepath+'/PredSample_ens'+'.pdf')

	def plot_fft(self,data,Delta,n_dom,color_,save=False):
		X = np.fft.fft(data)
		N = len(X)
		n = np.arange(N)
		sr = 1/Delta
		T = N/sr
		freq = n/T
		n_oneside = N//2
		# a = 1/(freq[:n_oneside][1:])
		a = freq[:n_oneside][1:]
		b = (np.abs(X)[:n_oneside]/n_oneside)[1:]

		topnlarge_inx = (-b).argsort()[:n_dom]
		bm = b[topnlarge_inx]
		ext = a[topnlarge_inx]

		# bm = b[np.argmax(b)]
		# ext = a[np.argmax(b)]

		fig,ax = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		ax.plot(a, b, color=color_, linewidth=1.3)
		# ax.text(ext, -.033, '%.4f'%(ext), color='red', transform=ax.get_xaxis_transform(),ha='center', va='top')
		for i in range(n_dom):
			ax.text(ext[i], bm[i]+bm[0]*0.07, r'\textbf{%.4f}'%(ext[i]), color='#DC143C',ha='center', va='top',size='large')
		ax.scatter(ext, bm, color='#DC143C', s=12.0)
		ax.set_ylim([-np.max(b)*0.05,np.max(b)*1.13])
		ax.set_xlabel('frequency (Hz)', font2)
		if save:
			fig.savefig(save, bbox_inches='tight')
		return fig,ax

	def plot_singledata_multidim(self,dataset,Tinterval,colorss=None,sepfig=None,save=False):
		Nx = dataset.shape[-1]
		dim = dataset.shape[0]
		x = np.linspace(Tinterval[0],Tinterval[1],Nx)
		colors = ['#4169E1','#DC143C','#2cc990']
		if sepfig is not None:
			px,py = sepfig
			fig1,ax1 = plt.subplots(nrows=px, ncols=py, figsize=[px*3,py*2], constrained_layout=True, squeeze=False)
			font2 = {'size'   : 14,}
			ccc = colorss if colorss is not None else 'black'
			for i in range(px):
				for j in range(py):
					ax1[i,j].plot(x, dataset[i*px+j], color=ccc, linewidth=1.3)
					ax1[i,j].set_xlabel('$T$', font2)
					ax1[i,j].set_ylabel('$X_{'+str(i*px+j+1)+'}$', font2)
		else:
			if dim>3:
				cmap = plt.get_cmap('RdYlBu')
				colors = [cmap(i) for i in np.linspace(0.3, 1, dim)]
			fig1,ax1 = plt.subplots(figsize=(6,4))
			font2 = {'size'   : 14,}
			for i in range(dim):
				ax1.plot(x, dataset[i], color=colors[i], linewidth=1.3, label='$X_{'+str(i+1)+'}$')
			ax1.set_xlabel('$T$', font2)
			ax1.legend(prop=font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plot_meanstd(self,name,testdata,predictdata,Delta,ylabels=None,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		if name=='Ex17PredPrey':
			fig_size = [12,4]
		elif name=='MultiscaleNonlinOclator':
			fig_size = [5.4,3.5]
		elif name=='SSAmRNAwDynk':
			fig_size = [5.4,3.5]
		else:
			fig_size = [6,4]
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
		fig1, ax1 = plt.subplots(figsize=fig_size)
		plt.rcParams['text.usetex'] = True
		ax1.plot(xt_test, xmean_test, linewidth=2.0, color='#000080', label='Ground Truth Mean')
		ax1.fill_between(xt_test, test_l, test_u, color='#000080', alpha=0.2, label='Ground Truth Std')
		ax1.plot(xt_pred, xmean_pred, linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
		ax1.fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2, label='Prediction Std')
		ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# ax1.set_ylim([-1.5,2.5])
		# ax1.legend(prop={'size': 13})
		# For MultiscaleNonlinOclator
		ax1.legend(prop={'size': 13},bbox_to_anchor=(-0.025, 1.00, 1., .102), loc=3, ncol=2)
		ax1.set_xlabel('$T$', {'size': 13})
		ax1.set_ylabel(ylabels, {'size': 13})
		if savepath is not None:
			if name=='StochasticRes':
				fig1.savefig(savepath, dpi=300)
			else:
				fig1.savefig(savepath, bbox_inches='tight')
		return fig1,ax1

	def plot_meanstd_sep(self,name,testdata,predictdata,Delta,ylabels=None,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		# if name=='Ex17PredPrey':
		# 	fig_size = [12,4]
		# elif name=='MultiscaleNonlinOclator':
		# 	fig_size = [5.4,3.5]
		# elif name=='SSAmRNAwDynk':
		# 	fig_size = [5.4,3.5]
		# else:
		# 	fig_size = [6,4]
		
		fig_size = [5,4]
		fig1, axes = plt.subplots(nrows=2, ncols=1, figsize=fig_size, constrained_layout=True, squeeze=False)
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
		# plot
		plt.rcParams['text.usetex'] = True
		axes[0][0].plot(xt_test, xmean_test, linewidth=2.0, color='#000080', label='Ground Truth')
		axes[0][0].plot(xt_pred, xmean_pred, linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction')
		axes[1][0].plot(xt_test, xstde_test, linewidth=2.0, color='#000080', label='Ground Truth')
		axes[1][0].plot(xt_pred, xstde_pred, linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction')
		
		# ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# ax1.set_ylim([-1.5,2.5])
		# ax1.legend(prop={'size': 13})
		# For MultiscaleNonlinOclator
		axes[0][0].legend(prop={'size': 13},bbox_to_anchor=(-0.025, 1.00, 1.04, .102), loc=3, ncol=2, mode="expand",handlelength=4.0)
		axes[1][0].set_xlabel('$T$', {'size': 13})
		axes[0][0].set_ylabel('Mean of '+ylabels, {'size': 13})
		axes[1][0].set_ylabel('STD of '+ylabels, {'size': 13})
		if savepath is not None:
			if name=='StochasticRes':
				fig1.savefig(savepath, dpi=300)
			else:
				fig1.savefig(savepath, bbox_inches='tight')
		return fig1,axes

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

	# def plot_acf(self,name,testdata,predictdata,Delta,savepath=None):
		# data should be in the form of Ndata*test
		# if name=='Ex17PredPrey':
			# fig_size = [6,4]
		# else:
			# fig_size = [6,4]
		# Test data
		# xt_test = np.arange(testdata.shape[-1]-1)*Delta
		# ac_test = self.compute_acf_ens(testdata)
		# Predict data
		# xt_pred = np.arange(predictdata.shape[-1]-1)*Delta
		# ac_pred = self.compute_acf_ens(predictdata)
		# plot
		# fig1, ax1 = plt.subplots(figsize=fig_size)
		# plt.rcParams['text.usetex'] = True
		# ax1.plot(xt_test, ac_test, linewidth=2.0, color='#000080', label='Ground Truth')
		# ax1.plot(xt_pred, ac_pred, linewidth=2.0, color='#DC143C', label='Prediction', linestyle='dashed')
		# ax1.set_xlabel(r'$\tau$', {'size': 13})
		# ax1.set_ylabel('ACF', {'size': 13})
		# ax1.legend(prop={'size': 13})
		# if savepath is not None:
			# fig1.savefig(savepath, bbox_inches='tight')
		# return fig1,ax1

	# def compute_acf(self,onedts,tau):
		# N = onedts.shape[0]
		# m = np.mean(onedts)
		# v = np.var(onedts,ddof=1)
		# acov = np.sum((onedts[:N-tau]-m)*(onedts[tau:]-m))/(N-1)
		# return acov/v

	# def compute_acf_ens(self,dataset):
		# for dataset in form of [Ndata,Ntime]
		# Ndata,Ntime = dataset.shape
		# re = np.zeros([Ndata,Ntime-1])
		# for i in range(Ndata):
			# print("ACF Number of trajectory: %d/%d \r"%(i+1,Ndata), sep=' ', end='', flush=True)
			# for j in range(Ntime-1):
				# re[i,j] = self.compute_acf(dataset[i],j)
		# return np.mean(re,axis=0)

	def plot_endpdfGeneralD(self,testdataMD,predictdataMD,dim,savepath=None):
		# data should be in the form of dim*Ndata
		# N_plot = min(dim,10)
		# fig1, ax1 = plt.subplots(ncols=N_plot, figsize=(10*N_plot, 7), squeeze=False)
		x_axis = np.linspace(-5,5,200)
		# for i in range(N_plot):
		# 	kde = scipy.stats.kde.gaussian_kde(testdataMD[i])
		# 	ax1[0,i].plot(x_axis, kde(x_axis), color='#4169E1',label='Ground Truth')
		# 	kde = scipy.stats.kde.gaussian_kde(predictdataMD[i])
		# 	ax1[0,i].plot(x_axis, kde(x_axis), color='#DC143C',label='Prediction')
		# 	ax1[0,i].legend()
		n_col = 5
		n_row = dim//n_col+int(dim%n_col!=0)
		if dim<=5:
			n_col = dim
		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2), constrained_layout=True, squeeze=False)
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					mt,st = np.mean(testdataMD[num]),np.std(testdataMD[num])
					mp,sp = np.mean(predictdataMD[num]),np.std(predictdataMD[num])
					x_axis = np.linspace(min(mt-3*st,mp-3*sp),max(mt+3*st,mp+3*sp),500)
					kde = scipy.stats.kde.gaussian_kde(testdataMD[num])
					axes[i,j].plot(x_axis, kde(x_axis), color='#4169E1',label='Ground Truth')
					kde = scipy.stats.kde.gaussian_kde(predictdataMD[num])
					axes[i,j].plot(x_axis, kde(x_axis), color='#DC143C',label='Prediction')
					axes[i,j].legend()
				else:
					break
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,axes

	def plot_mem_meanstd(self,name,testdata,predictdata,Delta,delay,ylabels=None,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		if name=='Ex17PredPrey':
			fig_size = [6,4]
		else:
			fig_size = [6,4]
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
		# Bound
		test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
		pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
		# plot
		fig1, ax1 = plt.subplots(figsize=fig_size)
		plt.rcParams['text.usetex'] = True
		# observed memory
		ax1.plot(xt_test[:delay+1], xmean_test[:delay+1], linewidth=2.0, color='grey', label='Observation Mean')
		ax1.fill_between(xt_test[:delay+1], test_l[:delay+1], test_u[:delay+1], color='grey', alpha=0.35, label='Observation Std')
		# prediction
		ax1.plot(xt_test[delay:], xmean_test[delay:], linewidth=2.0, color='#000080', label='Ground Truth Mean')
		ax1.fill_between(xt_test[delay:], test_l[delay:], test_u[delay:], color='#000080', alpha=0.2, label='Ground Truth Std')
		ax1.plot(xt_pred[delay:], xmean_pred[delay:], linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
		ax1.fill_between(xt_pred[delay:], pred_l[delay:], pred_u[delay:], color='#DC143C', alpha=0.2, label='Prediction Std')
		ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# seperate line
		ax1.plot(delay*Delta*np.ones(2), [min(np.min(test_l),np.min(pred_l))-10,max(np.max(test_u),np.max(pred_u))+10], linewidth=2.0, color='grey', alpha=1.0, linestyle=(0, (1, 1)))
		ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# ax1.set_ylim([-1.5,2.5])
		# ax1.legend(prop={'size': 13})
		# For MultiscaleNonlinOclator
		ax1.legend(prop={'size': 9.3},bbox_to_anchor=(-0.025, 1.00, 1., .102), loc=3, ncol=3)
		ax1.set_xlabel('$T$', {'size': 13})
		ax1.set_ylabel(ylabels, {'size': 13})
		if savepath is not None:
			if name=='StochasticRes':
				fig1.savefig(savepath, dpi=300)
			else:
				fig1.savefig(savepath, bbox_inches='tight')
		return fig1,ax1

	def plot_meanstd_SPDE_modal_fixtime(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		if self.eqn_config.eqn_name=='SHeatEqu_wSource_modal':
			n_col = 4
			n_row = 1
			T_index = [10,50,100,160]

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

		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2.3), constrained_layout=True, squeeze=False)
		plt.rcParams['text.usetex'] = True
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
					axes[i,j].plot(Ix, xmean_test, linewidth=2.0, color='#000080', label='Ground Truth Mean')
					axes[i,j].fill_between(Ix, test_l, test_u, color='#000080', alpha=0.2, label='Ground Truth Std')
					axes[i,j].plot(Ix, xmean_pred, linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
					axes[i,j].fill_between(Ix, pred_l, pred_u, color='#DC143C', alpha=0.2, label='Prediction Std')
					axes[i,j].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
					# axes.set_ylim([-1.5,2.5])
					axes[i,j].set_title('$T = %.2f$'%(T_d))
					axes[i,j].set_xlabel('$x$', {'size': 11})
				else:
					break
		# species for 1*4 figure
		axes[0,0].set_ylabel('$u(\cdot,T)$', {'size': 11})
		axes[0,0].legend(prop={'size': 11},bbox_to_anchor=(0.8, -0.55, 3.0, .102), loc=3, ncol=4, mode="expand", borderaxespad=0)
		if savepath is not None:
			fig1.savefig(savepath, bbox_inches='tight')
		return fig1,axes

	def plot_meanstd_SPDE_modal_fixpos(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		if self.eqn_config.eqn_name=='SHeatEqu_wSource_modal':
			n_col = 4
			n_row = 1
			x_index = [0,np.pi/2,np.pi,3*np.pi/2]
			x_latex = ['0','\pi/2','\pi','3\pi/2']

		if self.eqn_config.eqn_name in ['SHeatEqu_modal','SAdvDiff_modal','SHeatEqu_wSource_modal']:
			def basis_vec(x):
				Basis = np.zeros(dim)
				for k in range(dim):
					K = (k+1)//2
					if k==0:
						col = 1
					elif k%2==1:
						col = np.cos(K*x)
					elif k%2==0:
						col = np.sin(K*x)
					Basis[k] = col
				return Basis

		xt_test = np.arange(testdataMD.shape[1])*Delta
		xt_pred = np.arange(predictdataMD.shape[1])*Delta

		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2.3), constrained_layout=True, squeeze=False)
		plt.rcParams['text.usetex'] = True
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					# T
					x_i = x_index[num]
					Basis_v = basis_vec(x_i)
					# Test data
					test_nodal = np.tensordot(Basis_v,testdataMD,axes=1).T
					predict_nodal = np.tensordot(Basis_v,predictdataMD,axes=1).T

					xmean_test = np.mean(test_nodal,axis=0)
					xstde_test = np.std(test_nodal,axis=0,ddof=1)
					xt_test = xt_test[slice:]
					xmean_test,xstde_test = xmean_test[slice:],xstde_test[slice:]
					# Predict data
					xmean_pred = np.mean(predict_nodal,axis=0)
					xstde_pred = np.std(predict_nodal,axis=0,ddof=1)
					xt_pred = xt_pred[slice:]
					xmean_pred,xstde_pred = xmean_pred[slice:],xstde_pred[slice:]
					# Bound
					test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
					pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
					# plot
					axes[i,j].plot(xt_test, xmean_test, linewidth=2.0, color='#000080', label='Ground Truth Mean')
					axes[i,j].fill_between(xt_test, test_l, test_u, color='#000080', alpha=0.2, label='Ground Truth Std')
					axes[i,j].plot(xt_pred, xmean_pred, linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
					axes[i,j].fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2, label='Prediction Std')
					axes[i,j].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
					# axes.set_ylim([-1.5,2.5])
					axes[i,j].set_title('$x = %s $'%(x_latex[num]))
					axes[i,j].set_xlabel('$T$', {'size': 11})
				else:
					break
		# species for 1*4 figure
		axes[0,0].set_ylabel('$u(x,\cdot)$', {'size': 11})
		axes[0,0].legend(prop={'size': 11},bbox_to_anchor=(0.8, -0.55, 3.0, .102), loc=3, ncol=4, mode="expand", borderaxespad=0)
		if savepath is not None:
			fig1.savefig(savepath, bbox_inches='tight')
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

	def plot_meanqtl(self,name,testdata,predictdata,Delta,ylabels=None,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		if name=='Ex17PredPrey':
			fig_size = [12,4]
		elif name=='MultiscaleNonlinOclator':
			fig_size = [5.4,3.5]
		elif name=='SSAmRNAwDynk':
			fig_size = [5.4,3.5]
		else:
			fig_size = [6,4]
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		xmean_test = np.mean(testdata,axis=0)
		test_l     = np.quantile(testdata,0.25,axis=0)
		test_u     = np.quantile(testdata,0.75,axis=0)
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		xmean_pred = np.mean(predictdata,axis=0)
		pred_l     = np.quantile(predictdata,0.25,axis=0)
		pred_u     = np.quantile(predictdata,0.75,axis=0)
		# plot
		fig1, ax1 = plt.subplots(figsize=fig_size)
		plt.rcParams['text.usetex'] = True
		ax1.plot(xt_test, xmean_test, linewidth=2.0, color='#000080', label='Ground Truth Mean')
		ax1.fill_between(xt_test, test_l, test_u, color='#000080', alpha=0.2, label='Ground Truth IQR')
		ax1.plot(xt_pred, xmean_pred, linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
		ax1.fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2, label='Prediction IQR')
		ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# ax1.set_ylim([-1.5,2.5])
		# ax1.legend(prop={'size': 13})
		# For MultiscaleNonlinOclator
		ax1.legend(prop={'size': 13},bbox_to_anchor=(-0.025, 1.00, 1., .102), loc=3, ncol=2)
		ax1.set_xlabel('$T$', {'size': 13})
		ax1.set_ylabel(ylabels, {'size': 13})
		if savepath is not None:
			if name=='StochasticRes':
				fig1.savefig(savepath, dpi=300)
			else:
				fig1.savefig(savepath, bbox_inches='tight')
		return fig1,ax1

	def plot_mem_meanqtl(self,name,testdata,predictdata,Delta,delay,ylabels=None,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		if name=='Ex17PredPrey':
			fig_size = [6,4]
		else:
			fig_size = [6,4]
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		xmean_test = np.mean(testdata,axis=0)
		test_l     = np.quantile(testdata,0.25,axis=0)
		test_u     = np.quantile(testdata,0.75,axis=0)
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		xmean_pred = np.mean(predictdata,axis=0)
		pred_l     = np.quantile(predictdata,0.25,axis=0)
		pred_u     = np.quantile(predictdata,0.75,axis=0)
		# plot
		fig1, ax1 = plt.subplots(figsize=fig_size)
		plt.rcParams['text.usetex'] = True
		# observed memory
		ax1.plot(xt_test[:delay+1], xmean_test[:delay+1], linewidth=2.0, color='grey', label='Observation Mean')
		ax1.fill_between(xt_test[:delay+1], test_l[:delay+1], test_u[:delay+1], color='grey', alpha=0.35, label='Observation IQR')
		# prediction
		ax1.plot(xt_test[delay:], xmean_test[delay:], linewidth=2.0, color='#000080', label='Ground Truth Mean')
		ax1.fill_between(xt_test[delay:], test_l[delay:], test_u[delay:], color='#000080', alpha=0.2, label='Ground Truth IQR')
		ax1.plot(xt_pred[delay:], xmean_pred[delay:], linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
		ax1.fill_between(xt_pred[delay:], pred_l[delay:], pred_u[delay:], color='#DC143C', alpha=0.2, label='Prediction IQR')
		ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# seperate line
		ax1.plot(delay*Delta*np.ones(2), [min(np.min(test_l),np.min(pred_l))-10,max(np.max(test_u),np.max(pred_u))+10], linewidth=2.0, color='grey', alpha=1.0, linestyle=(0, (1, 1)))
		ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
		# ax1.set_ylim([-1.5,2.5])
		# ax1.legend(prop={'size': 13})
		# For MultiscaleNonlinOclator
		ax1.legend(prop={'size': 9.3},bbox_to_anchor=(-0.025, 1.00, 1., .102), loc=3, ncol=3)
		ax1.set_xlabel('$T$', {'size': 13})
		ax1.set_ylabel(ylabels, {'size': 13})
		if savepath is not None:
			if name=='StochasticRes':
				fig1.savefig(savepath, dpi=300)
			else:
				fig1.savefig(savepath, bbox_inches='tight')
		return fig1,ax1

	def plot_SPDE_modal_pdf(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		if self.eqn_config.eqn_name=='SHeatEqu_wSource_modal':
			n_col = 4
			n_row = 1
			x_index = [0.7,1.3,3.8,5.5]
			x_latex = ['0.7','1.3','3.8','5.5']
			T_index = [40,80,120,160]

		if self.eqn_config.eqn_name in ['SHeatEqu_modal','SAdvDiff_modal','SHeatEqu_wSource_modal']:
			def basis_vec(x):
				Basis = np.zeros(dim)
				for k in range(dim):
					K = (k+1)//2
					if k==0:
						col = 1
					elif k%2==1:
						col = np.cos(K*x)
					elif k%2==0:
						col = np.sin(K*x)
					Basis[k] = col
				return Basis

		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2.3), constrained_layout=True, squeeze=False)
		plt.rcParams['text.usetex'] = True
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					# T
					x_i = x_index[num]
					Basis_v = basis_vec(x_i)
					# Test data
					test_nodal = np.tensordot(Basis_v,testdataMD,axes=1).T
					predict_nodal = np.tensordot(Basis_v,predictdataMD,axes=1).T

					Test_data = test_nodal[:,T_index[num]]
					Predict_data = predict_nodal[:,T_index[num]]

					# plot
					axes[i,j].hist(Test_data, bins=30, alpha=1.0,color='#000080', density=True, histtype='step',label='Ground Truth',linewidth=1.3)
					axes[i,j].hist(Predict_data, bins=30, alpha=1.0,color='#DC143C', density=True, histtype='step',label='Learned', ls='--', linewidth=1.3)
					axes[i,j].set_title('$(x,t) = (%s,%.2f) $'%(x_latex[num],T_index[num]*Delta))
					axes[i,j].set_xlabel('$x$', {'size': 11})
					axes[i,j].legend(prop={'size': 12}, loc=4)
					axes[i,j].set_ylabel(r'pdf', {'size': 11})
				else:
					break
		# species for 1*4 figure
		# axes[0,0].set_ylabel(r'pdf', {'size': 11})
		if savepath is not None:
			fig1.savefig(savepath, bbox_inches='tight')
		return fig1,axes

	def plot_pdf(self,name,testdata,predictdata,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		if name=='Exp_diffusion':
			fig_size = [6,4]
			idx = [100,200,400,600,800,1000]
		elif name=='DisturbOU':
			fig_size = [5,3]
			idx = [200,400,800]
		elif name=='Ex16Multiscale':
			fig_size = [5,3]
			idx = [200,400,600,700,900]
		elif name=='Ex17PredPrey':
			fig_size = [5,3]
			idx = [500,2000,3000,3800,4000]
		elif name=='StochasticRes':
			fig_size = [5,3]
			idx = [40000,60000,100000,200000,300000,400000]
		elif name=='Ex19BiStochsticOU':
			fig_size = [5,3]
			idx = [200,600,800]
		elif name=='Double_well':
			fig_size = [6,4]
			idx = [-1,8000,5000,3000,1000,500,200,50,100]
		elif name=='Skew-Product SDE':
			fig_size = [6,4]
			idx = [10,200]
		elif name=="Multiscale_Stochastic_exp":
			fig_size = [6,4]
			idx = [10,200]
		elif name=="MultiScaleDuan3D":
			fig_size = [6,4]
			idx = [200,400,600,800]
		elif name=="MultiscaleNonlinOclator":
			fig_size = [6,4]
			idx = [400,800,1200,1600]
		elif name=="Ex38MultiscaleTriad":
			fig_size = [6,4]
			idx = [200,300,400]
		elif name=="SSAmRNAwDynk":
			fig_size = [6,4]
			idx = [10,100,220,420,500,600,780,900,1100,1130]
		elif name=="SSASchlogl":
			fig_size = [6,4]
			idx = [0,5,10,20,30,50,400,500]
		elif name=="SSALV":
			fig_size = [6,4]
			idx = [10,100,200,400,600,800,1000]
		elif name=="REx2_3DOssilator":
			fig_size = [6,4]
			idx = [100,150,200,250,300,350,400]
		elif name=="REx4_pendulum":
			fig_size = [6,4]
			idx = [100,150,200,250,300,350,400]
		elif name=="REx7_YGO":
			fig_size = [6,4]
			idx = [100,150,200,250,300,350,400]
		font2 = {'size'   : 15,}
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		xmean_test = np.mean(testdata,axis=0)
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		xmean_pred = np.mean(predictdata,axis=0)
		# plot
		for j in idx:
			fig1, ax1 = plt.subplots(figsize=fig_size)
			mt,st = np.mean(testdata[:,j-1]),np.std(testdata[:,j-1])
			mp,sp = np.mean(predictdata[:,j-1]),np.std(predictdata[:,j-1])
			x_axis = np.linspace(min(mt-3*st,mp-3*sp),max(mt+3*st,mp+3*sp),500)
			# kde vs step
			# kde = scipy.stats.kde.gaussian_kde(testdata[:,j-1], bw_method=0.04)
			# ax1.plot(x_axis, kde(x_axis), color='#000080',label='Reference')
			# kde = scipy.stats.kde.gaussian_kde(predictdata[:,j-1], bw_method=0.04)
			# ax1.plot(x_axis, kde(x_axis), color='#DC143C',linestyle='dashed',label='Learned')
			# ax1.legend(prop=font2)
			## ax1.hist(predictdata[:,j-1], bins=200, alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')
			
			# hist vs hist
			# pdb.set_trace()
			bins_ = self.bins_relative(testdata[:,j],predictdata[:,j],100)
			ax1.hist(testdata[:,j], bins=bins_, alpha=1.0,color='#000080', density=True, histtype='step',label='Ground Truth',linewidth=1.3)
			ax1.hist(predictdata[:,j], bins=bins_, alpha=1.0,color='#DC143C', density=True, histtype='step',label='Learned', ls='--', linewidth=1.3)
			ax1.legend(prop=font2)
			# ax1.set_xlabel('$x$',fontsize=20)
			# ax1.set_ylabel('pdf',fontsize=20)
			ax1.xaxis.set_tick_params(labelsize=20)
			ax1.yaxis.set_tick_params(labelsize=20)
			if savepath is not None:
				fig1.savefig(savepath+'_T'+str(int((j)))+'.pdf', bbox_inches='tight')
			plt.close()
		return fig1,ax1

	def plot_md_pdf(self,name,testdata,predictdata,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		if name=='SSACIRC73s':
			idx = [200,400,800]
			px,py = [4,4]
			title = ['$X_{%d}$'%(i+1) for i in range(16)]
		elif name=='SSAVilar2002R':
			idx = [200,400,800,1200,1600]
			px,py = [3,3]
			title = ['$X_{%d}$'%(i+1) for i in range(9)]

		font2 = {'size'   : 14,}
		dim   = testdata.shape[0]
		# # Test data
		# xt_test = np.arange(testdata.shape[1])*Delta
		# xmean_test = np.mean(testdata,axis=-1)
		# # Predict data
		# xt_pred = np.arange(predictdata.shape[1])*Delta
		# xmean_pred = np.mean(predictdata,axis=-1)
		# plot
		for j in idx:
			count = 1
			fig1, axes = plt.subplots(nrows=px, ncols=py, figsize=(py*3, px*2), constrained_layout=True, squeeze=False)
			for i in range(dim):
				# mt,st = np.mean(testdata[:,:,j-1]),np.std(testdata[:,:,j-1])
				# mp,sp = np.mean(predictdata[:,:,j-1]),np.std(predictdata[:,:,j-1])
				# x_axis = np.linspace(min(mt-3*st,mp-3*sp),max(mt+3*st,mp+3*sp),500)
				
				# hist vs hist
				bins_ = self.bins_relative(testdata[i,j-1,:],predictdata[i,j-1,:],50)
				axes[i//py,i%py].hist(testdata[i,j-1,:], bins=bins_, alpha=1.0,color='#000080', density=True, histtype='step',label='Ground Truth',linewidth=1.3)
				axes[i//py,i%py].hist(predictdata[i,j-1,:], bins=bins_, alpha=1.0,color='#DC143C', density=True, histtype='step',label='Learned', ls='--', linewidth=1.3)
				axes[i//py,i%py].legend(prop=font2)
				axes[i//py,i%py].set_title(title[i])
				# axes.set_xlabel('$x$')
				# axes.set_ylabel('pdf')
			if savepath is not None:
				fig1.savefig(savepath+'_T'+str(int(j*Delta))+'.pdf', bbox_inches='tight')
			plt.close()
		return fig1,axes

	def plotmultipledata(self,dataset,Tinterval,num,cmapname,ylabels=None,save=False):
		Nx = dataset.shape[-1]
		x = np.linspace(Tinterval[0],Tinterval[1],Nx)
		if cmapname=='DuoduoYishan' and num<=5:
			colors = (['#aa3e53','#d89c7c','#cf9198','#e7cfd4','#f6f0ee'])[::-1]
		elif cmapname=='XinxinXiangrong' and num<=5:
			colors = (['#1a3b30','#509579','#99bdbd','#c6e3d8','#e8f4f0'])[::-1]
		else:
			cmap = plt.get_cmap(cmapname)
			if num>1:
				colors = [cmap(i) for i in np.linspace(0.3, 1, num)]
			elif num==1:
				colors = [cmap(1.0)]
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		for i, color in enumerate(colors, start=0):
			ax1.plot(x, dataset[i], color=color)
		ax1.set_xlabel('$T$', font2)
		ax1.set_ylabel(ylabels, font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plotmultipledata_mem(self,dataset,Tinterval,num,cmapname,delay=0,ylabels=None,save=False):
		Nx = dataset.shape[-1]
		x = np.linspace(Tinterval[0],Tinterval[1],Nx)
		if cmapname=='DuoduoYishan' and num<=5:
			colors = (['#aa3e53','#d89c7c','#cf9198','#e7cfd4','#f6f0ee'])[::-1]
		elif cmapname=='XinxinXiangrong' and num<=5:
			colors = (['#1a3b30','#509579','#99bdbd','#c6e3d8','#e8f4f0'])[::-1]
		else:
			cmap = plt.get_cmap(cmapname)
			if num>1:
				colors = [cmap(i) for i in np.linspace(0.3, 1, num)]
			elif num==1:
				colors = [cmap(1.0)]
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		# observational data
		if delay>0:
			cmap = plt.get_cmap('Greys')
			if num>1:
				colors_g = [cmap(i) for i in np.linspace(0.3, 1, num)]
			elif num==1:
				colors_g = [cmap(1.0)]
			
			for i, color in enumerate(colors_g, start=0):
				ax1.plot(x[:delay+1], dataset[i,:delay+1], color=color)
		# prediction
		for i, color in enumerate(colors, start=0):
			ax1.plot(x[delay:], dataset[i,delay:], color=color)
		# seperate line
		yli = plt.gca().get_ylim()
		plt.autoscale(False)
		ax1.plot(delay*(x[1]-x[0])*np.ones(2), yli, linewidth=2.0, color='grey', alpha=1.0, linestyle=(0, (1, 1)))
		ax1.set_xlabel('$T$', font2)
		ax1.set_ylabel(ylabels, font2)


		##################################### legend ########################
		class HandlerColormap(matplotlib.legend_handler.HandlerBase):
			def __init__(self, cmap, num_stripes=8, **kw):
				HandlerBase.__init__(self, **kw)
				self.cmap = cmap
				self.num_stripes = num_stripes
			def create_artists(self, legend, orig_handle, 
							xdescent, ydescent, width, height, fontsize, trans):
				stripes = []
				for i in range(self.num_stripes):
					s = Rectangle([xdescent + i * width / self.num_stripes, ydescent], 
								width / self.num_stripes, height, 
								fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)), 
								transform=trans)
					stripes.append(s)
				return stripes
		cmaps = [plt.get_cmap('Greys'),plt.get_cmap(cmapname)]
		cmap_labels = ['Observation','Predictions']
		cmap_handles = [matplotlib.patches.Rectangle((0, 0), 1, 1) for _ in cmaps]
		handler_map = dict(zip(cmap_handles, [HandlerColormap(cm, num_stripes=8) for cm in cmaps]))
		ax1.legend(handles=cmap_handles, labels=cmap_labels, handler_map=handler_map, 
						prop={'size': 13},bbox_to_anchor=(-0.0, 1.04, 1., .101), loc=9, ncol=2)
		##################################### legend ########################

		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plotmultipledata_Doublewell(self,dataset,Tinterval,num,cmapname,save=False):
		Nx = dataset.shape[-1]
		x = np.linspace(Tinterval[0],Tinterval[1],Nx)
		cmap = plt.get_cmap(cmapname)
		if num>1:
			colors = [cmap(i) for i in np.linspace(0.3, 1, num)]
		elif num==1:
			colors = [cmap(0.8)]
		fig1,ax1 = plt.subplots(figsize=(12,4))
		ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		font2 = {'size'   : 14,}
		# ax1.plot(x,0.6*np.cos(self.eqn_config.omega*x), color='grey', alpha=0.8, linewidth=0.5)
		# ax1.plot(x,np.zeros(x.shape), color='grey', alpha=0.8, linewidth=0.5)
		for i, color in enumerate(colors, start=0):
			ax1.plot(x, dataset[i], color=color)
		ax1.set_xlabel('$T$', font2)
		ax1.set_ylabel('$S$', font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	# def plot_singledata_multidim(self,dataset,Tinterval,save=False):
	# 	Nx = dataset.shape[-1]
	# 	dim = dataset.shape[0]
	# 	x = np.linspace(Tinterval[0],Tinterval[1],Nx)
	# 	colors = ['#4169E1','#DC143C','#2cc990']
	# 	fig1,ax1 = plt.subplots(figsize=(6,4))
	# 	font2 = {'size'   : 14,}
	# 	for i in range(dim):
	# 		ax1.plot(x, dataset[i], color=colors[i], linewidth=1.3, label='$X_'+str(i+1)+'$')
	# 	ax1.set_xlabel('$T$', font2)
	# 	ax1.legend(prop=font2)
	# 	if save:
	# 		fig1.savefig(save, bbox_inches='tight')
	# 	return fig1,ax1

	def plot_singledata_multidim_tx_version(self,T,dataset,save=False):
		dim = dataset.shape[0]
		# colors = ['#0a0a0a','#9b9c9e','#6e6f70']
		# colors = ['#230C2B','#9EB7FF','#A95145']
		# colors = ['#230C2B','#ADCAC9','#697889']
		colors = ['#230C2B','#ADCAC9','#6F4D46']
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		for i in range(dim):
			ax1.plot(T, dataset[i], color=colors[i], linewidth=1.3, label='$X_'+str(i+1)+'$')
		ax1.set_xlabel('$T$', font2)
		ax1.legend(prop=font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')

		plt.close()
		return fig1,ax1

	def plot_singledata_multidim_multiscale_version(self,T,dataset,save=False):
		dim = dataset.shape[0]
		# colors = ['#0a0a0a','#9b9c9e','#6e6f70']
		# colors = ['#230C2B','#9EB7FF','#A95145']
		# colors = ['#230C2B','#ADCAC9','#697889']
		# colors = ['#230C2B','#ADCAC9','#6F4D46']
		colors = ['#957064','#c4c7b4','#ebe7e4']
		labels = ['$x$','$y$','$z$']
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		for i in range(dim):
			ax1.plot(T, dataset[i], color=colors[i], linewidth=1.3, label=labels[i], zorder=dim-i)
		ax1.set_xlabel('$T$', font2)
		ax1.legend(prop=font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plot_singledata_multidim_multiscale_version_biaxis(self,T,dataset,N_slow,legend_loc,save=False):
		dim = dataset.shape[0]
		# colors = ['#0a0a0a','#9b9c9e','#6e6f70']
		# colors = ['#230C2B','#9EB7FF','#A95145']
		# colors = ['#230C2B','#ADCAC9','#697889']
		# colors = ['#230C2B','#ADCAC9','#6F4D46']
		colors = ['#957064','#c4c7b4','#ebe7e4']
		labels = ['$x$','$y$','$z$']
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		
		# for i in range(dim):
		# 	ax1.plot(T, dataset[i], color=colors[i], linewidth=1.3, label=labels[i], zorder=dim-i)

		color = colors[0]
		ax1.set_xlabel('$T$', font2)
		ax1.set_ylabel('$x$',font2,  color=color)
		lns = []
		color_s = ['#957064','#c1aea1']
		# ls = ['solid','dotted','dotted']
		for i in range(N_slow):
			l1 = ax1.plot(T, dataset[i], color=color_s[i], label='$x_'+str(i+1)+'$ (slow)')
			lns += l1
		ax1.tick_params(axis='y', labelcolor=color)

		ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

		colorf = ['#c4c7b4','#ebe7e4']
		ax2.set_ylabel('$y$', font2, color=colorf[0])  # we already handled the x-label with ax1
		for j in range(dim-N_slow):
			l2 = ax2.plot(T, dataset[j+N_slow], color=colorf[j], label='$y_'+str(j+1)+'$ (fast)')
			lns += l2
		ax2.tick_params(axis='y', labelcolor=colorf[0])

		ax1.set_zorder(1)
		ax1.patch.set_visible(False)

		labs = [l.get_label() for l in lns]
		# ax2.legend(lns, labs, prop=font2, loc=legend_loc)
		# For MultiscaleNonlinOclator
		ax2.legend(lns, labs, prop=font2, bbox_to_anchor=(-0.03, 1.00, 1., .102), loc=3, ncol=3)

		fig1.tight_layout()  # otherwise the right y-label is slightly clipped
		
		# ax1.legend(prop=font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plot_singledata_gilispie_version_biaxis(self,T,dataset,colors,labels = ['$x$','$y$'],save=False):
		dim = dataset.shape[0]
		fig1,ax1 = plt.subplots(figsize=(7,4))
		font2 = {'size'   : 14,}

		ax1.set_xlabel('$T$', font2)
		ax1.set_ylabel(labels[0],font2,  color=colors[0])
		l1 = ax1.plot(T, dataset[0], color=colors[0])
		ax1.tick_params(axis='y', labelcolor=colors[0])

		ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

		ax2.set_ylabel(labels[1], font2, color=colors[1])  # we already handled the x-label with ax1
		l2 = ax2.plot(T, dataset[1], color=colors[1])
		ax2.tick_params(axis='y', labelcolor=colors[1])

		# ax1.set_zorder(1)
		# ax1.patch.set_visible(False)

		fig1.tight_layout()  # otherwise the right y-label is slightly clipped
		
		if save:
			fig1.savefig(save, bbox_inches='tight')

		plt.close()
		return fig1,ax1

	def plot_ens_multidim(self,dataset,Tinterval,Num,save=False):
		Nx = dataset.shape[1]
		dim = dataset.shape[0]
		x = np.linspace(Tinterval[0],Tinterval[1],Nx)
		colors = ['#4169E1','#DC143C','#2cc990']
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		for i in range(dim):
			for j in range(Num):
				if j==0:
					ax1.plot(x, dataset[i,:,j], color=colors[i], linewidth=1.0, label='$X_'+str(i+1)+'$')
				else:
					ax1.plot(x, dataset[i,:,j], color=colors[i], linewidth=1.0)
		ax1.set_xlabel('$T$', font2)
		ax1.legend(prop=font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plotmultipledata2Dphase(self,dataset1,dataset2,num,cmapname,save=False):
		if cmapname=='DuoduoYishan' and num<=5:
			colors = (['#aa3e53','#d89c7c','#cf9198','#e7cfd4','#f6f0ee'])[::-1]
		elif cmapname=='XinxinXiangrong' and num<=5:
			colors = (['#1a3b30','#509579','#99bdbd','#c6e3d8','#e8f4f0'])[::-1]
		else:
			cmap = plt.get_cmap(cmapname)
			if num>1:
				colors = [cmap(i) for i in np.linspace(0.3, 1, num)]
			elif num==1:
				colors = [cmap(1.0)]
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		for i, color in enumerate(colors, start=0):
			ax1.plot(dataset1[i], dataset2[i], color=color)
		ax1.set_xlabel('$x$', font2)
		ax1.set_ylabel('$y$', font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plotsingledata2Dphase(self,dataset1,dataset2,color_,configs,save=False):
		figsize_ = configs['figsize'] if ('figsize' in configs.keys()) else [6,4]

		fig1,ax1 = plt.subplots(figsize=figsize_)
		font2 = {'size'   : 14,}
		ax1.plot(dataset1, dataset2, color=color_, linewidth=1.3)
		ax1.set_xlabel('$X_1$', font2)
		ax1.set_ylabel('$X_2$', font2)
		plt.axis('equal')
		ax1.set_xlim([0,np.max([dataset1])*1.01])
		ax1.set_ylim([0,np.max([dataset2])*1.01])
		if save:
			fig1.savefig(save, bbox_inches='tight')

		plt.close()
		return fig1,ax1

	def plotsingledata3Dphase(self,dataset1,dataset2,dataset3,color_,configs,save=False):
		view = configs['view'] if ('view' in configs.keys()) else [20,45]
		scale = configs['scale'] if ('scale' in configs.keys()) else [0.7,0.7,1.4]

		fig1 = plt.figure(figsize=(10,8),facecolor='white')
		ax1  = fig1.add_subplot(111, projection='3d')
		ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
		ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
		ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
		ax1.view_init(elev=view[0], azim=view[1])
		ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([scale[0], scale[1], scale[2], 1]))
		font2 = {'size'   : 14,}
		ax1.plot(dataset1, dataset2, dataset3, color=color_)
		ax1.set_xlabel('$X_1$', font2)
		ax1.set_ylabel('$X_2$', font2)
		ax1.set_zlabel('$X_3$', font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def plotsingledata3Dpairphase(self,dataset1,dataset2,dataset3,color_,configs,save=False):
		configs['figsize'] = configs['12_figsize']
		font2 = {'size'   : 14,}
		fig1,ax1 = self.plotsingledata2Dphase(dataset1,dataset2,color_,configs,save=False)
		if save:
			fig1.savefig(save+'_12.pdf', bbox_inches='tight')
		
		configs['figsize'] = configs['13_figsize']
		fig1,ax1 = self.plotsingledata2Dphase(dataset1,dataset3,color_,configs,save=False)
		ax1.set_ylabel('$X_3$', font2)
		if save:
			fig1.savefig(save+'_13.pdf', bbox_inches='tight')
		
		configs['figsize'] = configs['23_figsize']
		fig1,ax1 = self.plotsingledata2Dphase(dataset2,dataset3,color_,configs,save=False)
		ax1.set_xlabel('$X_2$', font2)
		ax1.set_ylabel('$X_3$', font2)
		if save:
			fig1.savefig(save+'_23.pdf', bbox_inches='tight')
		return fig1,ax1

	def interpolate_wrt_t(self,react_time,react_numb,t_step):
		f = scipy.interpolate.interp1d(react_time,react_numb,kind='previous')
		return f(t_step)
	
	## ----------------------------------------------------Conditional pdf----------------------------------------------------
	def complete_condpdf(self,model,save=False):
		# logging.info('--------------Plotting final pdf on Epoch %d'%(epoch+1))
		## check if model list
		if self.eqn_config.dim==1 and self.eqn_config.eqn_name not in ['SSASchlogl']:
			if self.eqn_config.eqn_name=='Geometric Brownian Motion':
				# condpdf_plotting_points = [0.5,1.0,2.0]
				# int_long = [1.0,1.0,1.0]
				# xlimlen = [1.0,1.0,1.0]
				condpdf_plotting_points = [4.0,5.0,6.0]
				int_long = [3.0,3.0,3.0]
				xlimlen = [4.0,5.0,6.0]
			elif self.eqn_config.eqn_name=='OU Process':
				condpdf_plotting_points = [0.8,1.2,1.8]
				int_long = [0.3,0.3,0.3]
				xlimlen = [0.3,0.3,0.3]
			elif self.eqn_config.eqn_name=='Exp_diffusion':
				condpdf_plotting_points = [-0.3,0.0,0.3]
				int_long = [0.4,0.4,0.4]
				xlimlen = [0.4,0.4,0.4]
			elif self.eqn_config.eqn_name=='Trig_drift':
				condpdf_plotting_points = [0.4,0.5,0.6]
				int_long = [0.4,0.4,0.4]
				xlimlen = [0.4,0.4,0.4]
			elif self.eqn_config.eqn_name=='Exp_OU':
				condpdf_plotting_points = [0.4,0.7,1.0]
				int_long = [0.2,0.2,0.2]
				xlimlen  = [0.1,0.2,0.25]
			elif self.eqn_config.eqn_name=='Double_well':
				condpdf_plotting_points = [-1.5,0,1.5]
				# int_long = [1.0,1.0,1.0]
				# xlimlen  = [1.0,1.0,1.0]
				int_long = [0.5,0.5,0.5]
				xlimlen  = [0.5,0.5,0.5]
			elif self.eqn_config.eqn_name=='Exp_dis':
				condpdf_plotting_points = [0.34,0.52,0.72]
				int_long = [0.15,0.15,0.15]
				xlimlen  = [0.12,0.12,0.12]
			## draw
			font2 = {'size'   : 14,}
			int_long = self.monitor_config.repdf_display['int_long']
			p_size = self.monitor_config.repdf_display['size']
			px,py = p_size
			l1,l2 = self.monitor_config.repdf_display['range']
			p_grid = (np.linspace(l1,l2,px*py)).reshape([px,py])
			for i in range(px):
				for j in range(py):
					fig, axes = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
					plt.rcParams['text.usetex'] = True
					self.condpdf_plotting_std(self.eqn_config.eqn_name,axes,int_long,p_grid[i,j],self.Delta)
					self.condpdf_plotting_data(self.eqn_config.eqn_name,axes,model,int_long,p_grid[i,j],self.Delta)
					if self.eqn_config.eqn_name=='Exp_dis':
						axes.set_xlim([p_grid[i,j]-xlimlen[0]/4,p_grid[i,j]+xlimlen[0]*3/4])
					else:
						axes.set_xlim([p_grid[i,j]-xlimlen[0]/2,p_grid[i,j]+xlimlen[0]/2])
					axes.legend(prop=font2)
					fig.savefig(save+'condpdf_'+str(p_grid[i,j])+'.pdf',dpi=150)
			plt.close()
			## draw
			# logging.info('--------------End plotting final pdf on Epoch %d'%(epoch+1))
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
				fig.savefig(save+'.pdf')
				plt.close()
			elif self.eqn_config.eqn_name in ['SSALV','SSABrusselator','SSAautocatalytic','SSAmRNAwDynk','SSASchlogl']:
				if self.eqn_config.eqn_name=='SSAmRNAwDynk':
					labels = [r'M',r'P']
				else:
					labels = [r'$X_1$',r'$X_2$']
				N = 10000
				data_dic = sio.loadmat(self.monitor_config.repdf_display['path'])
				px,py = (data_dic['size'].astype('int')).flatten()
				# Choice 1
				fig, axes = plt.subplots(nrows=px, ncols=py*2, figsize=(py*3*2, px*2), constrained_layout=True, squeeze=False)
				for i in range(px):
					for j in range(py):
						ini = data_dic[str(i*py+j)+'_i'].flatten()
						dat_std = data_dic[str(i*py+j)+'_d']
						if self.eqn_config.eqn_name in ['SSAmRNAwDynk']:
							para = data_dic['para_'+str(i*py+j)+'_i'].flatten()
							dat_mod = (model.predict(np.tile(ini,[N,1]),np.tile(para,[N,1]))).detach().numpy()
						else:
							dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
						a_ = min(np.min(dat_std[:, 0]),np.min(dat_mod[:, 0]))
						b_ = max(np.max(dat_std[:, 0]),np.max(dat_mod[:, 0]))
						c_ = min(np.min(dat_std[:, 1]),np.min(dat_mod[:, 1]))
						d_ = max(np.max(dat_std[:, 1]),np.max(dat_mod[:, 1]))
						range_ = np.array(((a_,b_),(c_,d_)))
						fig1,ax1 = plt.subplots(figsize=(6,4))
						font2 = {'size'   : 14,}
						plt.rcParams['text.usetex'] = True
						ax1.hist2d(dat_mod[:, 0], dat_mod[:, 1], bins=30, range=range_, density=True,cmap='Reds',norm = matplotlib.colors.LogNorm())
						ax1.set_xlabel(labels[0], fontsize=30)
						ax1.set_ylabel(labels[1], fontsize=30)
						ax1.xaxis.set_tick_params(labelsize=20)
						ax1.yaxis.set_tick_params(labelsize=20)
						fig1.savefig(save+'pdf2d_model_'+'%d_%d'%(ini[0],ini[1])+'.pdf', bbox_inches='tight')
						plt.close()

						fig2,ax2 = plt.subplots(figsize=(6,4))
						ax2.hist2d(dat_std[:, 0], dat_std[:, 1], bins=30, range=range_, density=True,cmap='Blues',norm = matplotlib.colors.LogNorm())
						ax2.set_xlabel(labels[0], fontsize=30)
						ax2.set_ylabel(labels[1], fontsize=30)
						ax2.xaxis.set_tick_params(labelsize=20)
						ax2.yaxis.set_tick_params(labelsize=20)
						fig2.savefig(save+'pdf2d_std_'+'%d_%d'%(ini[0],ini[1])+'.pdf', bbox_inches='tight')
						plt.close()

				# Choice 2
				# dim = self.eqn_config.dim
				# px,py = (data_dic['size'].astype('int')).flatten()
				# fig, axes = plt.subplots(nrows=px, ncols=py*dim, figsize=(py*(dim+1)*2.5, px*2), constrained_layout=True, squeeze=False)
				# for i in range(px):
				# 	for j in range(py):
				# 		ini = data_dic[str(i*py+j)+'_i'].flatten()
				# 		dat_std = data_dic[str(i*py+j)+'_d']
				# 		dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
				# 		# mean = model.Model_drift.myevaluate(np.expand_dims(ini, axis=0))[0]
				# 		# for k in range(dim):
				# 		# 	axes[i,j*dim+k].set_title("$X_s=$(%.1f,%.1f), dim %d"%(ini[0],ini[1],k+1))
				# 		# 	# axes[i,j*dim+k].scatter(np.array((mean[k])), np.array((0)), color='green', s=100.0, clip_on=False)
				# 		# 	axes[i,j*dim+k].hist(dat_mod[:, k], bins=50, density=False, color='#DC143C',histtype='step',label='Learned', ls='--', linewidth=1.3)
				# 		# 	axes[i,j*dim+k].hist(dat_std[:, k], bins=50, density=False, color='#000080',histtype='step',label='Ground Truth', linewidth=1.3)

				# 		a_ = min(np.min(dat_std[:, 0]),np.min(dat_mod[:, 0]))
				# 		b_ = max(np.max(dat_std[:, 0]),np.max(dat_mod[:, 0]))
				# 		c_ = min(np.min(dat_std[:, 1]),np.min(dat_mod[:, 1]))
				# 		d_ = max(np.max(dat_std[:, 1]),np.max(dat_mod[:, 1]))
				# 		range_ = np.array(((a_,b_),(c_,d_)))
				# 		axes[i,j*2].set_title("$X_s=$(%.1f,%.1f), truth"%(ini[0],ini[1]))
				# 		axes[i,j*2].hist2d(dat_std[:, 0], dat_std[:, 1], bins=30, range=range_, density=True,cmap='Blues',norm = matplotlib.colors.LogNorm())
				# 		# axes[i,j*2+1].set_title("$X_s=$(%.1f,%.1f), learned"%(ini[0],ini[1]))
				# 		prob = self.probality_outdis_discrete(dat_std,dat_mod)
				# 		axes[i,j*2+1].set_title("$X_s=$(%.1f,%.1f), learned, %.2e"%(ini[0],ini[1],prob))
				# 		axes[i,j*2+1].hist2d(dat_mod[:, 0], dat_mod[:, 1], bins=30, range=range_, density=True,cmap='Reds',norm = matplotlib.colors.LogNorm())
				# 		# plt.close()
				
				# fig.savefig(save+'cond.pdf')
				# plt.close()
				# px,py = (data_dic['size'].astype('int')).flatten()

				# Choice 3
				for i in range(px):
					for j in range(py):
						ini = data_dic[str(i*py+j)+'_i'].flatten()
						dat_std = data_dic[str(i*py+j)+'_d']
						if self.eqn_config.eqn_name in ['SSAmRNAwDynk']:
							para = data_dic['para_'+str(i*py+j)+'_i'].flatten()
							dat_mod = (model.predict(np.tile(ini,[N,1]),np.tile(para,[N,1]))).detach().numpy()
						else:
							dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
						# axes[i,j*2].set_title("$X_s=$(%.1f,%.1f), dim 1, ite %d"%(ini[0],ini[1],epoch+1))
						# axes[i,j*2+1].set_title("$X_s=$(%.1f,%.1f), dim 2, ite %d"%(ini[0],ini[1],epoch+1))

						fig1,ax1 = plt.subplots(figsize=(6,4))
						font2 = {'size'   : 16,}
						plt.rcParams['text.usetex'] = True
						bins_ = self.bins_relative(dat_std[:, 0],dat_mod[:, 0],50)
						ax1.hist(dat_std[:, 0], bins=bins_, density=True, color='#000080',histtype='step',label='Ground Truth', linewidth=1.3)
						ax1.hist(dat_mod[:, 0], bins=bins_, density=True, color='#DC143C',histtype='step',label='Learned', ls='--', linewidth=1.3, alpha=0.8)
						ax1.xaxis.set_tick_params(labelsize=20)
						ax1.yaxis.set_tick_params(labelsize=20)
						ax1.legend(prop=font2,loc='upper right')
						fig1.savefig(save+'margin_dim1_'+'%d_%d'%(ini[0],ini[1])+'.pdf')
						plt.close()

						fig2,ax2 = plt.subplots(figsize=(6,4))
						bins_ = self.bins_relative(dat_std[:, 1],dat_mod[:, 1],50)
						ax2.hist(dat_std[:, 1], bins=bins_, density=True, color='#000080',histtype='step',label='Ground Truth',linewidth=1.3)
						ax2.hist(dat_mod[:, 1], bins=bins_, density=True, color='#DC143C',histtype='step',label='Learned', ls='--', linewidth=1.3, alpha=0.8)
						ax2.xaxis.set_tick_params(labelsize=20)
						ax2.yaxis.set_tick_params(labelsize=20)
						ax2.legend(prop=font2,loc='upper right')
						fig2.savefig(save+'margin_dim2_'+'%d_%d'%(ini[0],ini[1])+'.pdf')
						plt.close()
		elif self.eqn_config.dim>2 or self.eqn_config.eqn_name in ['SSASchlogl']:
			if self.eqn_config.eqn_name in ['SSAOregonator','SSATransfer','SSACIRC73s','SSAVilar2002R','SSASchlogl']:
				if self.eqn_config.eqn_name=='SSACIRC73s':
					index = [1,2]
					plx,ply = [4,4]
					title = ['$X_{%d}$'%(i+1) for i in range(16)]
				if self.eqn_config.eqn_name=='SSAVilar2002R':
					index = [0,1,2,3,4,5,6,7,8,9,10,11]
					plx,ply = [3,3]
					title = ['$X_{%d}$'%(i+1) for i in range(9)]
				elif self.eqn_config.eqn_name=='SSASchlogl':
					index = np.arange(16)
					# index = [15]
					plx,ply = [1,1]
					title = ['$X_{1}$']

				N = 10000
				dim = self.eqn_config.dim
				data_dic = sio.loadmat(self.monitor_config.repdf_display['path'])

				# px,py = (data_dic['size'].astype('int')).flatten()
				# fig, axes = plt.subplots(nrows=px, ncols=py*dim, figsize=(py*(dim+1)*2.5, px*2), constrained_layout=True, squeeze=False)
				# for i in range(px):
				# 	for j in range(py):
				# 		ini = data_dic[str(i*py+j)+'_i'].flatten()
				# 		dat_std = data_dic[str(i*py+j)+'_d']
				# 		dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
				# 		for k in range(dim):
				# 			axes[i,j*dim+k].set_title("$X_s=$(%.1f,%.1f), dim %d, ite %d"%(ini[0],ini[1],k+1,epoch+1))
				# 			bins_ = self.bins_relative(dat_mod[:, k],dat_std[:, k],50)
				# 			axes[i,j*dim+k].hist(dat_mod[:, k], bins=bins_, density=True, color='#DC143C',histtype='step',label='Learned', ls='--', linewidth=1.3)
				# 			axes[i,j*dim+k].hist(dat_std[:, k], bins=bins_, density=True, color='#000080',histtype='step',label='Ground Truth', linewidth=1.3)
				# fig.savefig(save+'.pdf')
				# plt.close()
				# px,py = (data_dic['size'].astype('int')).flatten()
				
				plt.rcParams['text.usetex'] = True
				for i in index:
					# fig, axes = plt.subplots(nrows=plx, ncols=ply, figsize=(ply*6, plx*4), constrained_layout=True, squeeze=False)
					fig, axes = plt.subplots(nrows=plx, ncols=ply, figsize=(ply*3, plx*2), constrained_layout=True, squeeze=False)
					plt.rcParams['text.usetex'] = True
					ini = data_dic[str(i)+'_i'].flatten()
					dat_std = data_dic[str(i)+'_d']
					dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
					for k in range(dim):
						axes[k//ply,k%ply].set_title(title[k],fontsize=17)
						# bins_ = self.bins_relative(dat_mod[:, k],dat_std[:, k],50)
						axes[k//ply,k%ply].hist(dat_std[:, k], bins=bins_, density=True, color='#000080',histtype='step',label='Ground Truth', linewidth=1.3)
						axes[k//ply,k%ply].hist(dat_mod[:, k], bins=bins_, density=True, color='#DC143C',histtype='step',label='Learned', ls='--', linewidth=1.3,alpha=0.8)
						# bins_ = self.bins_relative((dat_std[:, k]-ini)/self.eqn_config.Delta,(dat_mod[:, k]-ini)/self.eqn_config.Delta,83)
						# axes[k//ply,k%ply].hist((dat_std[:, k]-ini)/self.eqn_config.Delta, bins=bins_, density=True, color='#000080',histtype='step',label='Ground Truth', linewidth=1.3)
						# axes[k//ply,k%ply].hist((dat_mod[:, k]-ini)/self.eqn_config.Delta, bins=bins_, density=True, color='#DC143C',histtype='step',label='Learned', ls='--', linewidth=1.3,alpha=0.8)
						axes[k//ply,k%ply].xaxis.set_tick_params(labelsize=14)
						axes[k//ply,k%ply].yaxis.set_tick_params(labelsize=14)
						font2 = {'size'   : 13,}
						# axes[k//ply,k%ply].legend(prop=font2,loc='upper right')
					# fig.savefig(save+'condf_'+str(ini)+'.pdf')
					# axes[-1,2].legend(prop={'size': 16},bbox_to_anchor=(0.1, -0.7, 2.0, .102), loc='lower right', ncol=2, mode="expand", borderaxespad=0)
					fig.savefig(save+'condf_'+str(i)+'.pdf')
					plt.close()
			else:
				raise AttributeError('complete_condpdf: no this type of 2D example')
		else:
			pass

	def bins_relative(self,dat1,dat2,num):
		min_,max_ = min(np.min(dat1),np.min(dat2)),max(np.max(dat1),np.max(dat2))
		if max_-min_<=5:
			bins_ = np.linspace(min_-0.05,max_,20)
		else:
			bins_ = np.arange(min_-2,max_+2,np.ceil((max_-min_)/num+1.0e-8))-0.5
		return bins_

	def probality_outdis_discrete(self,truth,pred):
		tt = (truth.astype('int')).tolist()
		pp = (pred.astype('int')).tolist()
		N = len(pp)
		count=0
		for i in range(N):
			if pp[i] in tt:
				count+=1
		return 1-count/N

	def compare_stoppingtime(self,model,save=False):
		if self.eqn_config.eqn_name=='SSALV':
			cretier = lambda x: (np.abs(x[:,0])<1.0e-8)+(np.abs(x[:,1])<1.0e-8)
			Nmax = 10000
			N = 10000
		elif self.eqn_config.eqn_name=='SSASchlogl':
			cretier = lambda x: (x[:,0]>=563)
			Nmax = 1000
			N = 20

		data_dic = sio.loadmat(self.monitor_config.stoppingtime['path'])
		# data_dic['size'] = np.array((1,1))
		px,py = (data_dic['size'].astype('int')).flatten()

		# fig, axes = plt.subplots(nrows=px, ncols=py, figsize=(py*6,px*4), constrained_layout=True, squeeze=False)
		# for i in range(px):
		# 	for j in range(py):
		# 		ini = data_dic[str(i*py+j)+'_i'].flatten()
		# 		dat_std = data_dic[str(i*py+j)+'_d']
		# 		dat_mod = self.compute_stoppingtime(model,ini,N,cretier,Nmax)
		# 		axes[i,j].set_title("$X_s=$(%.1f,%.1f)"%(ini[0],ini[1]))
		# 		axes[i,j].hist(np.array(dat_mod).flatten(), bins=50, density=False, color='#DC143C',histtype='step')
		# 		axes[i,j].hist(np.array(dat_std).flatten(), bins=50, density=False, color='#4169E1',histtype='step')
		# fig.savefig(save+'stoppingtime.png',dpi=150)
		# plt.close()

		for i in range(px):
			for j in range(py):
				if i*py+j==0:
					pass
				else:
					continue

				ini = data_dic[str(i*py+j)+'_i'].flatten()
				dat_std = data_dic[str(i*py+j)+'_d']
				dat_mod = self.compute_stoppingtime(model,ini,N,cretier,Nmax)
				# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,4), constrained_layout=True, squeeze=True)
				# # axes.set_title("$X_s=$(%.1f,%.1f)"%(ini[0],ini[1]))
				# axes.hist(np.array(dat_std).flatten(), bins=50, density=True, color='#000080',histtype='step', linewidth=1.3,label='Ground Truth')
				# axes.hist(np.array(dat_mod).flatten(), bins=50, density=True, color='#DC143C',histtype='step',label='Prediction', ls='--', linewidth=1.3, alpha=0.8)
				# Debug-only local save disabled in the public package.
				# axes.set_xlabel('Stopping Time')
				# font2 = {'size'   : 14,}
				# axes.legend(prop=font2)
				# fig.savefig(save+'stoppingtime_'+str(i*px+j)+'.pdf')
				# plt.close()

				# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,4), constrained_layout=True, squeeze=True)
				# # axes.set_title("$X_s=$(%.1f,%.1f)"%(ini[0],ini[1]))
				# bins_ = np.linspace(0,900,50)
				# axes.hist(dm.flatten(), bins=bins_, density=True, color='#000080',histtype='step', linewidth=1.3,label='Ground Truth')
				# axes.hist(ds.flatten(), bins=bins_, density=True, color='#DC143C',histtype='step',label='Prediction', ls='--', linewidth=1.3, alpha=0.8)
				# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
				# axes.xaxis.set_tick_params(labelsize=13)
				# axes.yaxis.set_tick_params(labelsize=13)
				# axes.set_xlabel('Stopping Time')
				# font2 = {'size'   : 14,}
				# axes.legend(prop=font2)
				# plt.show()

				stoppingtimedatapath  = './Ex23_run_stoppingtime1.mat'
				if os.path.exists(stoppingtimedatapath):
					data = sio.loadmat(stoppingtimedatapath)
					data['1_d'] = np.append(data['1_d'],np.array(dat_mod))
					sio.savemat(stoppingtimedatapath,data)
				else:
					sio.savemat(stoppingtimedatapath,{'1_d':np.array(dat_mod)})

	def compute_stoppingtime(self,model,ini,N_data,cretier,NTmax):
		## second model if available
		# Model2 = Chemical_Dynamics.ChemicalDynamics(self.eqn_config)
		# Model2.predict = Model2.SSAsolverT(self.eqn_config.Delta)
		# conssss = lambda x: ((np.abs(x[:,0])>-1)*(np.abs(x[:,0])<5))+((np.abs(x[:,1])>-1)*(np.abs(x[:,1])<20))
		
		count = 1
		re = []
		dat = np.tile(ini,[N_data,1])
		while count<=NTmax:
			print(count*0.1)
			print(len(dat))
			dat_mod = (model.predict(dat)).detach().numpy()
			# id_twomodel = conssss(dat)
			# dat_mod[id_twomodel] = torch.from_numpy((Model2.predict(dat[id_twomodel])).astype('float32'))
			
			id_stop = np.where(cretier(dat_mod))[0]
			id_rest = np.delete(np.arange(dat.shape[0]),id_stop)
			re += [self.eqn_config.Delta*count]*len(id_stop)
			dat = dat_mod[id_rest]
			count += 1
		if len(dat)>0:
			re += [int(NTmax*1.1)*self.eqn_config.Delta]*len(dat)
		return re

	def trail_condpdf(self,model,ini,file,save=True):
		# data in file should be in form of [N_sample,dim]
		## check if model list
		N = 10000
		dim = self.eqn_config.dim
		data_dic = sio.loadmat(file)

		fig, axes = plt.subplots(nrows=1, ncols=dim, figsize=(dim*3, 2), constrained_layout=True, squeeze=False)
		dat_std = data_dic['data']
		dat_mod = (model.predict(np.tile(ini,[N,1]))).detach().numpy()
		for k in range(dim):
			axes[0,k].hist(dat_mod[:, k], bins=50, color='#DC143C',histtype='step',label='Learned', ls='--', linewidth=1.3)
			axes[0,k].hist(dat_std[:, k], bins=50, color='#000080',histtype='step',label='Ground Truth', linewidth=1.3, alpha=0.5)
		fig.savefig(save+'1.pdf')
		plt.close()

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
		else:
			print('The distribution %s is not supported'%(name))

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

	def condpdf_plotting_data(self,name,ax,model,intlong,x,Delta,N=100000):
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
			print('The distribution %s is not supported'%(name))
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

	# --------------------------------------------------------------------------------------------------------------------------------

	# ----------------------------------------------------- condmeanv ----------------------------------------------------------------
	def cond_meanvar(self,model,save=False):
		## check if model list
		font2 = {'size'   : 18,}
		if self.eqn_config.dim==1:
			## compute
			Npoint = self.monitor_config.cond_mv['Npoint']
			# Npoint = 200
			l1,l2 = self.monitor_config.cond_mv['range']
			# l1,l2 = 50,600
			p_grid = np.linspace(l1,l2,Npoint+1)
			Mean_t, Std_t = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
			Mean_d, Std_d = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
			for i in range(p_grid.shape[0]):
				Mean_t[i],Std_t[i] = self.condmv_plotting_std_cont(self.eqn_config.eqn_name,p_grid[i],self.Delta)
				Mean_d[i],Std_d[i] = self.condmv_plotting_data(model,p_grid[i],N=500000)
			if self.eqn_config.eqn_name=='Exp_OU':
				Mean_d = (np.log(Mean_d)-np.log(p_grid))/self.eqn_config.Delta
			else:
				Mean_d = (Mean_d-p_grid)/self.eqn_config.Delta
				Std_d = Std_d/np.sqrt(self.eqn_config.Delta)
			plt.rcParams['text.usetex'] = True
			## draw
			fig, axes = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
			plt.rcParams['text.usetex'] = True
			axes.plot(p_grid,Mean_t,linestyle='-', linewidth=2.5,color='#000080',label='Reference')
			axes.plot(p_grid,Mean_d,linestyle='dashed', linewidth=2.5,color='#DC143C',label='Learned')
			axes.legend(prop=font2)
			axes.xaxis.set_tick_params(labelsize=18)
			axes.yaxis.set_tick_params(labelsize=18)
			fig.savefig(save+'condmean'+'.pdf',dpi=150)

			# Condmean error plot and printed error metrics are disabled in the public package.
			# fig, axes = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
			# plt.rcParams['text.usetex'] = True
			# axes.plot(p_grid,np.zeros(p_grid.shape),linestyle='-', linewidth=2.5,color='#000080',label='Reference')
			# axes.plot(p_grid,Mean_t-Mean_d,linestyle='dashed', linewidth=2.5,color='#DC143C',label='Learned')
			# axes.legend(prop=font2)
			# axes.xaxis.set_tick_params(labelsize=18)
			# axes.yaxis.set_tick_params(labelsize=18)
			# fig.savefig(save+'condmeanerror'+'.pdf',dpi=150)
			# print('Mean Norm: %.4e'%(np.sqrt(p_grid[1]-p_grid[0])*np.sqrt(np.sum((Mean_t-Mean_d)**2))))

			fig, axes = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
			plt.rcParams['text.usetex'] = True
			axes.plot(p_grid,Std_t,linestyle='-', linewidth=2.5,color='#000080',label='Reference')
			axes.plot(p_grid,Std_d,linestyle='dashed', linewidth=2.5,color='#DC143C',label='Learned')
			axes.legend(prop=font2)
			axes.xaxis.set_tick_params(labelsize=18)
			axes.yaxis.set_tick_params(labelsize=18)
			if self.eqn_config.eqn_name=='OU Process':
				axes.set_ylim([0.27,0.33])
			elif self.eqn_config.eqn_name=='Double_well':
				axes.set_ylim([0.45,0.55])
			fig.savefig(save+'condstd'+'.pdf',dpi=150)
			plt.close()
			# print('STD Norm: %.4e'%(np.sqrt(p_grid[1]-p_grid[0])*np.sqrt(np.sum((Std_t-Std_d)**2))))
		elif self.eqn_config.dim==2 and self.eqn_config.eqn_name in ['MdOU','SO']:
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
			# axes[0,0].set_title("Truth Mean $E_1(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[0,1].plot_surface(p_gridx, p_gridy, Mean_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			# axes[0,1].set_title("Estimated Mean $E_1(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[0,2].plot_surface(p_gridx, p_gridy, Mean_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
			# axes[0,2].set_title("Truth Mean $E_2(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[0,3].plot_surface(p_gridx, p_gridy, Mean_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			# axes[0,3].set_title("Estimated Mean $E_2(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[0,0].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
			axes[0,1].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
			axes[0,2].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
			axes[0,3].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
			# variances
			axes[1,0].plot_surface(p_gridx, p_gridy, V_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
			# axes[1,0].set_title("Truth variance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[1,1].plot_surface(p_gridx, p_gridy, V_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			# axes[1,1].set_title("Estimated variance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[1,2].plot_surface(p_gridx, p_gridy, V_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
			# axes[1,2].set_title("Truth variance $Var_2(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[1,3].plot_surface(p_gridx, p_gridy, V_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			# axes[1,3].set_title("Estimated variance $Var_2(\cdot|X_s)$, ite %d"%(epoch+1))
			# covariance
			axes[2,0].plot_surface(p_gridx, p_gridy, C_t.reshape([Npoint+1,Npoint+1]), cmap='Blues')
			# axes[2,0].set_title("Truth Covariance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
			axes[2,1].plot_surface(p_gridx, p_gridy, C_d.reshape([Npoint+1,Npoint+1]), cmap='Reds')
			# axes[2,1].set_title("Estimated Covariance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
			fig.savefig(save+'.pdf',dpi=150)
			plt.close()
		else:
			pass

	def condmv_plotting_data(self,model,x,N=50000):
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
			m,s = self.eqn_config.mu*x,self.eqn_config.sigma*x
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
			elif abs(self.eqn_config.Delta-0.05)<1e-8:
				x_int = np.linspace(50,600,40).astype('int')
				m_int = np.array([ 56.232,  33.17 ,   9.796,  -6.666, -23.842, -32.414, -39.174,
							-44.124, -45.222, -43.286, -39.158, -30.398, -21.476, -13.406,
							-3.974,  10.278,  26.512,  37.434,  52.434,  69.96 ,  79.31 ,
							95.814, 107.92 , 121.512, 127.194, 134.292, 137.928, 141.808,
							140.784, 142.976, 132.154, 133.238, 121.428,  99.622,  86.26 ,
							54.256,  28.068,  -7.276, -43.536, -94.814])
				s_int = np.array([19.24874303, 21.28420906, 23.15499771, 24.83417448, 26.7513729 ,
							28.69175197, 30.88067174, 32.63436274, 35.20497887, 37.17970023,
							39.78203807, 41.10751852, 43.6171878 , 46.01777872, 47.56011319,
							50.8498981 , 52.2067399 , 55.30620927, 57.259792  , 59.88746047,
							61.31993311, 64.38979631, 66.30486921, 68.88545342, 70.71861365,
							73.011033  , 74.10301845, 76.08027704, 78.59171245, 81.16113093,
							82.74014755, 84.42115709, 86.97207623, 88.41343143, 90.37889475,
							90.95877265, 92.86104335, 93.84707237, 97.24492188, 97.5124057 ])
				f_m = scipy.interpolate.interp1d(x_int, m_int)
				f_s = scipy.interpolate.interp1d(x_int, s_int)
				m   = f_m(x)
				std = f_s(x)
				return m,std
			elif abs(self.eqn_config.Delta-0.02)<1e-8:
				x_int = np.linspace(50,600,40).astype('int')
				m_int = np.array([ 59.585,   30.935,    8.175,  -11.365,  -23.255,  -32.08,   -40.455,  -40.25,
							-47.2 ,   -42.63 ,  -34.59,   -35.68 ,  -21.75 ,   -6.4  ,    2.225,   10.89,
							27.315,   47.42  ,  52.88 ,   63.72  ,  76.76  ,  97.8   , 107.08  , 122.305,
							124.3 ,   134.34 ,  144.26,   144.605,  150.795,  149.745,  143.705,  125.795,
							117.31,   106.785,   84.79,    41.88 ,   37.155,   -3.01 ,  -47.   , -100.59 ])
				s_int = np.array([19.95939767,  21.65792039,  23.58067827,  25.38467521,  27.14385933,
							28.91535011,  31.3120242 ,  33.48878544,  35.23724166,  36.72987424,
							39.33459849,  41.70550026,  43.10474162,  45.13092953,  47.3329271 ,
							49.62950894,  52.50516942,  54.4237712 ,  56.47534074,  58.25646086,
							61.00949146,  63.79728207,  65.60607801,  67.40901082,  69.74854981,
							72.25105735,  75.17713115,  75.93071104,  78.7767882 ,  80.29616242,
							82.16740509,  86.52529896,  88.92427834,  88.71839874,  91.59188347,
							94.59228992,  95.87650974,  99.54726916,  99.90675653, 103.35663035])
				f_m = scipy.interpolate.interp1d(x_int, m_int)
				f_s = scipy.interpolate.interp1d(x_int, s_int)
				m   = f_m(x)
				std = f_s(x)
				return m,std
			elif abs(self.eqn_config.Delta-0.01)<1e-8:
				x_int = np.linspace(50,600,40).astype('int')
				m_int = np.array([ 58.61,  32.1,    8.36, -10.11, -23.85, -35.73, -44.04, -43.31, -43.35, -44.17,
							-38.22, -29.54, -21.42,  -8.96,   4.6,   10.12,  22.36,  39.15,  58.15,  61.57,
							84.01, 103.64, 100.32, 111.42, 133.95, 138.29, 131.73, 137.61, 137.79, 143.57,
							144.6,  128.85, 113.25, 106.47,  91.81,  49.41,  35.91,  -1.49, -27.94, -87.9 ])
				s_int = np.array([20.33122424,  22.01853537,  23.85248633,  25.2960052,   27.33206496,
							29.40244328,  31.28809333,  33.05741126,  34.99110994,  37.2094089,
							39.30511819,  41.15548425,  43.55584732,  45.41560507,  47.67586811,
							48.9615753,   51.37684599,  54.36563965,  55.98245953,  58.28791771,
							60.2202059,   63.1242228,   65.26024039,  67.44298211,  69.97116531,
							71.29339913,  72.68935322,  76.51957187,  78.87172598,  81.01312579,
							83.15689027,  85.44358826,  88.23352183,  90.58328428,  92.99967333,
							95.56242211,  96.87463403,  99.91039885,  101.7351147,  103.2359235])
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

	# def condmv_plotting_data(self,model,x,N=5000):
	# 	data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
	# 	try:
	# 		data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
	# 	except:
	# 		Nmodel = len(model)
	# 		modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
	# 		data = np.zeros(N)
	# 		for i in range(Nmodel):
	# 			data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.repeat(x,modelsep[i+1]-modelsep[i])[:,None])).detach().numpy().flatten()
	# 	# data = (model.predict(np.repeat(x,N)[:,None])).detach().numpy().flatten()
	# 	m,s = np.mean(data),np.std(data)
	# 	return m,s

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

	# --------------------------------------------------------------------------------------------------------------------------------
	# ---------------------------------------------Ensemble related-------------------------------------------------------------------
	def Eva_Ensemble(self,eqn_name,modellist,DatVes,save=False):
		N_T = (DatVes.test_data).shape[1]
		data_ = DatVes.datachoose((np.vstack(DatVes.test_data)).T, DatVes.dim, np.zeros([DatVes.test_data.shape[-1],1],dtype=int), 1)
		# Xs = np.tile(data_[:,:DatVes.dim],(len(modellist),1))
		# pre = [Xs]
		# for i in range(N_T-1):
		# 	Xs = self.Mulmodel_Generate(modellist,Xs)
		# 	pre += [Xs]
		# pre = np.concatenate(pre, -1)
		# pre_ = np.zeros([DatVes.dim,N_T,DatVes.test_data.shape[-1]*len(modellist)])
		# for j in range(self.eqn_config.dim):
		# 	pre_[j] = (pre[:,j::DatVes.dim]).T
		# pre_ = np.zeros([DatVes.dim,N_T,DatVes.test_data.shape[-1]*len(modellist)])
		Xs = data_[:,:DatVes.dim]
		pre_ = np.zeros([DatVes.dim,N_T,DatVes.test_data.shape[-1]])
		pre_[:,0,:] = Xs.T
		for i in np.arange(N_T-1)+1:
			with torch.no_grad():
				Xs = self.Mulmodel_Generate(modellist,Xs)
				pre_[:,i,:] = Xs.T
		for i in range(min(DatVes.dim,10)):
			if save:
				save_ = (save+'/'+'M'+str(i+1)+'.pdf')
				fig,ax = self.plot_meanstd(eqn_name,DatVes.test_data[i].T,pre_[i].T,self.eqn_config.Delta,savepath=save_)
		if save:
			save_ = (save+'/'+'P'+str(i+1))
			fig,ax = self.plot_pdf(eqn_name,DatVes.test_data[i].T,pre_[i].T,self.eqn_config.Delta,savepath=save_)
		plt.close()

	def Mulmodel_Generate(self,modellist,Xs):
		Nmodel = len(modellist)
		modelid = np.random.randint(Nmodel, size=Xs.shape[0])
		Xre = np.zeros(Xs.shape)
		for j in range(Nmodel):
			_id = np.where(modelid==j)[0]
			with torch.no_grad():
				Xre[_id] = modellist[j].predict(Xs[_id]).detach().numpy()
		return Xre
	# --------------------------------------------------------------------------------------------------------------------------------

	def readmodel(self,path,Model,config):
		# This function is designed for test for single models
		ModelX = Model(config)
		ModelX.load_state_dict(torch.load(path),strict=False)
		return ModelX

	def readMultiplemodel(self,ckptmanager,Model,config):
		# This function is designed for test for multiple models
		modellist = []
		modeldict = ckptmanager.list_models()
		for i in range(len(modeldict)):
			ModelX = Model(config)
			ModelX.load_state_dict(torch.load(modeldict[i]))
			# ModelX.eval()
			modellist.append(ModelX)
		return modellist

class SdeNFEva(Evaluate):
	def __init__(self,config,result_path,save_path,model_path=None,Mymodel=None):
		self.eqn_config      = config.eqn_config
		self.net_config      = config.net_config
		self.dat_config      = config.dat_config
		self.monitor_config  = config.monitor_config
		self.result_path = result_path
		self.save_path   = save_path
		self.dim = self.eqn_config.dim
		self.Delta   = self.eqn_config.Delta
		self.n_epochs = self.net_config.N_epochs
		self.test_data_path  = self.dat_config.TestData_dir
		self.model_path = model_path
		self.Mymodel    = Mymodel
		self.config = config
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

	def plot_samplecompare(self,save=False):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		if self.eqn_config.eqn_name in ['SSALV','SSABrusselator','SSAOregonator','SSAautocatalytic','SSAmRNAwDynk','SSACIRC73s','SSAVilar2002R']:
			self.plot_sample_block(self.eqn_config.eqn_name,test_data,pre_data,self.Delta,savepath=self.save_path)
		else:
			self.plot_sample(self.eqn_config.eqn_name,test_data,pre_data,self.Delta,savepath=self.save_path)
		if self.eqn_config.eqn_name in ['SSATransfer','SSALV']:
			self.plot_sample_ens(self.eqn_config.eqn_name,test_data,pre_data,self.Delta,savepath=self.save_path)
		
		# Original SSA/Gillespie path plots are disabled in the public package.
		# if self.eqn_config.eqn_name in ['SSALV','SSABrusselator','SSAOregonator','SSAautocatalytic']:
		# 	self.plot_glispie_ori(self.eqn_config.eqn_name,savepath=self.save_path)

		# Original multiscale path plots are disabled in the public package.
		# if self.eqn_config.eqn_name in ["Skew-Product SDE","Multiscale_Stochastic_exp","MultiScaleDuan3D","MultiscaleNonlinOclator","Ex38MultiscaleTriad"]:
		# 	self.plot_multiscale_ori(self.eqn_config.eqn_name,savepath=self.save_path)


	def plot_fftompare(self,save=False):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')

		if self.eqn_config.eqn_name in ['SSABrusselator','SSAOregonator','SSAautocatalytic','SSACIRC73s','SSAVilar2002R']:
			save_ = self.save_path+'/fft'
			if not os.path.exists(save_):
				os.makedirs(save_)
			self.plot_fft_block(self.eqn_config.eqn_name,pre_data,self.Delta,savepath=save_)

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

	def plot_meancompare(self,save=False,epoch=''):
		if self.eqn_config.eqn_name=='DisturbOU':
			ylabels = ['$x$']
		elif self.eqn_config.eqn_name=='SSALV':
			ylabels = ['$X_1$','$X_2$']
		elif self.eqn_config.eqn_name=='Ex19BiStochsticOU':
			ylabels = ['$x$']
		elif self.eqn_config.eqn_name=='Ex16Multiscale':
			ylabels = ['$x$','$y$']
		elif self.eqn_config.eqn_name=="Ex17PredPrey":
			ylabels = ['$x$','$y$']
		elif self.eqn_config.eqn_name=='SSAmRNAwDynk':
			ylabels = ['M','P']
		else:
			ylabels = [None for i in range(self.dim)]

		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		plt.rcParams['text.usetex'] = True
		for i in range(min(self.dim,10)):
			save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'M'+str(i+1)+'.pdf') if save else None
			if self.eqn_config.eqn_name=='StochasticRes':
				save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'M'+str(i+1)+'.png') if save else None
			# pdb.set_trace()
			fig,ax = self.plot_meanstd(self.eqn_config.eqn_name,test_data[i].T,pre_data[i].T,self.Delta,ylabels=ylabels[i],savepath=save_)
			# fig,ax = self.plot_meanstdGeneralD(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)
			if self.eqn_config.eqn_name in ['SSAmRNAwDynk','SSALV']:
				save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'sep_'+'M'+str(i+1)+'.pdf') if save else None
				fig,ax = self.plot_meanstd_sep(self.eqn_config.eqn_name,test_data[i].T,pre_data[i].T,self.Delta,ylabels=ylabels[i],savepath=save_)
			if self.eqn_config.eqn_name in ['REx2_3DOssilator','REx4_pendulum','REx7_YGO']:
				if self.eqn_config.eqn_name=='REx2_3DOssilator':
					delay_ = 50
				elif self.eqn_config.eqn_name=='REx4_pendulum':
					delay_ = 50
				elif self.eqn_config.eqn_name=='REx7_YGO':
					delay_ = 50
				else:
					delay_ = 1
				save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'mem_'+'M'+str(i+1)+'.pdf') if save else None
				fig,ax = self.plot_mem_meanstd(self.eqn_config.eqn_name,test_data[i].T,pre_data[i].T,self.Delta,delay_,ylabels=ylabels[i],savepath=save_)
			plt.close()
		if self.eqn_config.eqn_name in ['SHeatEqu_wSource_modal']:
			save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_fixtime'+'.pdf') if save else None
			fig,ax = self.plot_meanstd_SPDE_modal_fixtime(test_data,pre_data,self.dim,self.Delta,savepath=save_)
			save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_fixpos'+'.pdf') if save else None
			fig,ax = self.plot_meanstd_SPDE_modal_fixpos(test_data,pre_data,self.dim,self.Delta,savepath=save_)
			plt.close()

	# def plot_meanqtlcompare(self,save=False,epoch=''):
		# if self.eqn_config.eqn_name=='DisturbOU':
			# ylabels = ['$x$']
		# elif self.eqn_config.eqn_name=='SSALV':
			# ylabels = ['$X_1$','$X_2$']
		# elif self.eqn_config.eqn_name=='Ex19BiStochsticOU':
			# ylabels = ['$x$']
		# elif self.eqn_config.eqn_name=='Ex16Multiscale':
			# ylabels = ['$x$','$y$']
		# elif self.eqn_config.eqn_name=="Ex17PredPrey":
			# ylabels = ['$x$','$y$']
		# elif self.eqn_config.eqn_name=='SSAmRNAwDynk':
			# ylabels = ['M','P']
		# else:
			# ylabels = [None for i in range(self.dim)]

		# test_data = (sio.loadmat(self.test_data_path))['data']
		# try:
			# pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		# except:
			# raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		# plt.rcParams['text.usetex'] = True
		# for i in range(min(self.dim,10)):
			# save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'MQ'+str(i+1)+'.pdf') if save else None
			# pdb.set_trace()
			# fig,ax = self.plot_meanqtl(self.eqn_config.eqn_name,test_data[i].T,pre_data[i].T,self.Delta,ylabels=ylabels[i],savepath=save_)
			# fig,ax = self.plot_meanstdGeneralD(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)
			# if self.eqn_config.eqn_name in ['REx2_3DOssilator','REx4_pendulum','REx7_YGO']:
				# if self.eqn_config.eqn_name=='REx2_3DOssilator':
					# delay_ = 50
				# elif self.eqn_config.eqn_name=='REx4_pendulum':
					# delay_ = 50
				# elif self.eqn_config.eqn_name=='REx7_YGO':
					# delay_ = 50
				# else:
					# delay_ = 1
				# save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'mem_'+'MQ'+str(i+1)+'.pdf') if save else None
				# fig,ax = self.plot_mem_meanqtl(self.eqn_config.eqn_name,test_data[i].T,pre_data[i].T,self.Delta,delay_,ylabels=ylabels[i],savepath=save_)
			# plt.close()

	# def plotcompute_acf(self,save=False):
		# test_data = (sio.loadmat(self.test_data_path))['data']
		# try:
			# pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		# except:
			# raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		# plt.rcParams['text.usetex'] = True
		# for i in range(min(self.dim,10)):
			# save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+'ACF'+str(i+1)+'.pdf') if save else None
			# fig,ax = self.plot_acf(self.eqn_config.eqn_name,test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)

	def plot_pdfcompare(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		plt.rcParams['text.usetex'] = True
		if self.dim <=3:
			for i in range(min(self.dim,10)):
				save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'_pdf_X'+str(i+1)) if save else None
				fig,ax = self.plot_pdf(self.eqn_config.eqn_name,test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)
				plt.close()
		else:
			save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'_pdf') if save else None
			fig,ax = self.plot_md_pdf(self.eqn_config.eqn_name,test_data,pre_data,self.Delta,savepath=save_)
			plt.close()

		if self.eqn_config.eqn_name in ['SHeatEqu_wSource_modal']:
			save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_'+epoch+'_pdf.pdf') if save else None
			self.plot_SPDE_modal_pdf(test_data,pre_data,self.dim,self.Delta,savepath=save_)

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

	def plot_condpdfcompare(self,save=False):
		# NFModel = self.readmodel(self.model_path,self.Mymodel.NFSSDE,self.config)
		save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_condpdf/') if save else None
		if not os.path.exists(save_):
			os.makedirs(save_)
		self.complete_condpdf(self.NFModel,save=save_)

	def plot_stoppingtimecompare(self,save=False):
		# NFModel = self.readmodel(self.model_path,self.Mymodel.NFSSDE,self.config)
		save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_stoppingtime/') if save else None
		if not os.path.exists(save_):
			os.makedirs(save_)
		self.compare_stoppingtime(self.NFModel,save=save_)

	def Trail_condpdfcompare(self,ini,file,save=False):
		# NFModel = self.readmodel(self.model_path,self.Mymodel.NFSSDE,self.config)
		save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_trail_condpdf/') if save else None
		if not os.path.exists(save_):
			os.makedirs(save_)
		self.trail_condpdf(self.NFModel,ini,file,save=save_)

	def plot_condmeanvcompare(self,save=False,epoch=''):
		# self.NFModel = self.readmodel(self.model_path,self.Mymodel.NFSSDE,self.config)
		save_ = (self.save_path+'/'+self.eqn_config.eqn_name+'_condmeanv/') if save else None
		if not os.path.exists(save_):
			os.makedirs(save_)
		self.cond_meanvar(self.NFModel,save=save_)

	def plot_Enscompare(self,DatVes,ensmpath,save=False):
		SManager = myutils.SaveManager(path=ensmpath)
		NFModellist = self.readMultiplemodel(SManager,self.Mymodel.NFSSDE,self.config)
		save_ = (self.save_path+'/Ens/') if save else None
		if not os.path.exists(save_):
			os.makedirs(save_)
		self.cond_meanvar(NFModellist,save=save_)
		self.Eva_Ensemble(self.eqn_config.eqn_name,NFModellist,DatVes,save=save_)

	def prediction_time(self):
		N_test = 1
		NFModel = self.readmodel(self.model_path,self.Mymodel.NFSSDE,self.config)
		pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		Xs = np.tile((pre_data[:,0,[0]]).T,(N_test,1))
		N_T = pre_data.shape[1]
		st = time.time()
		for i in range(N_T-1):
			with torch.no_grad():
				Xs = NFModel.predict(Xs)
		print('%d Trajectory Prediction Time: %.8f'%(N_test,time.time()-st))

