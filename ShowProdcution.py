import json
import munch
import os
import sys
import shutil
import logging
import importlib as imp

import numpy as np
from absl import app
from absl import flags
from absl import logging as absl_logging
import pdb
import warnings

import NFSDE    as Mymodel1
import ResNFSDE as Mymodel2
import MixNFSDE as Mymodel3
import NFNonAutoSDE    as Mymodel4
import ResNFNonAutoSDE as Mymodel5
import MixNFNonAutoSDE as Mymodel6
import NFSDE_SSAconserve as Mymodel7
import ResNFSDE_SSAgenconserve as Mymodel8
import MixNFSDE_SSAgenconserve as Mymodel9
# import NFARSDE as Mymodel10
# import NFARSDE2 as Mymodel11
# import NFARSDE3 as Mymodel12
# import NFARSDE4 as Mymodel13

import Prodcution as Evaulation
imp.reload(Mymodel1)
imp.reload(Mymodel2)
imp.reload(Mymodel3)
imp.reload(Mymodel4)
imp.reload(Mymodel5)
imp.reload(Mymodel6)
imp.reload(Mymodel7)
imp.reload(Mymodel8)
imp.reload(Mymodel9)
# imp.reload(Mymodel10)
# imp.reload(Mymodel11)
# imp.reload(Mymodel12)
# imp.reload(Mymodel13)
imp.reload(Evaulation)

os.chdir(sys.path[0])

### Now support:
### 3. Ex3OU

flags.DEFINE_string('test_name',   'REx4_Mix_s2__1_prod',                'Name of test')
flags.DEFINE_string('test_case',    'prod',                        'Name of post test')
flags.DEFINE_string('model_name',   'NFSDE',         'Name of model')
FLAGS = flags.FLAGS

def Model_select(model_name):
	if model_name=='NFSDE':
		return Mymodel1
	elif model_name=='ResNFSDE':
		return Mymodel2
	elif model_name=='MixNFSDE':
		return Mymodel3
	elif model_name=='NFNonAutoSDE':
		return Mymodel4
	elif model_name=='ResNFNonAutoSDE':
		return Mymodel5
	elif model_name=='MixNFNonAutoSDE':
		return Mymodel6
	elif model_name=='NFSDE_SSAconserve':
		return Mymodel7
	elif model_name=='ResNFSDE_SSAgenconserve':
		return Mymodel8
	elif model_name=='MixNFSDE_SSAgenconserve':
		return Mymodel9
	# elif model_name=='NFARSDE':
	# 	return Mymodel10
	# elif model_name=='NFARSDE2':
	# 	return Mymodel11
	# elif model_name=='NFARSDE3':
	# 	return Mymodel12
	# elif model_name=='NFARSDE4':
	# 	return Mymodel13
	elif model_name==None:
		raise AttributeError("Model_select: please type in model name")
	else:
		raise AttributeError("Model_select: %s model is not supported"%(model_name))


def main(argv):
	del argv
	### Setup path
	result_path = './results'
	root_path = result_path+'/'+FLAGS.test_name+'/'+FLAGS.test_case if FLAGS.test_case else result_path+'/'+FLAGS.test_name
	save_path = root_path+'/'+'Eva'
	config_path = root_path+'/'+'Test_config.json'
	bestm_path  = result_path + '/' + FLAGS.test_name + '/Test_model/model.pt'
	ens_m_path  = result_path + '/' + FLAGS.test_name + '/' + 'Monitor/Ens_model'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if not os.path.exists(config_path):
		shutil.copy2(result_path+'/'+FLAGS.test_name+'/'+'Test_config.json', root_path)
	if not os.path.exists(root_path+'/'+'predict.mat'):
		shutil.copy2(result_path+'/'+FLAGS.test_name+'/'+'predict.mat', root_path)
	### Load configration
	with open(config_path) as json_data_file:
		config = json.load(json_data_file)
	config = munch.munchify(config)
	### Evaluation
	Mymodel = Model_select(FLAGS.model_name)
	Eva = Evaulation.SdeNFEva(config,root_path,save_path,bestm_path,Mymodel)
	NFModel = Eva.readmodel(bestm_path,Mymodel.NFSSDE,config)
	DatVes = Model_select(FLAGS.model_name).DataTran(config)
	# DatVes.read_testdata()
	Eva.NFModel = NFModel
	Eva.DatVes  = DatVes
	if FLAGS.model_name in ['MixNFSDE','MixNFNonAutoSDE','MixNFSDE_SSAgenconserve']:
		Eva.DatVes.train_hiddendata()
		Eva.NFModel.Model_drift = Eva.DatVes.Model_drift
	# if FLAGS.model_name in ['NFARSDE','NFARSDE2','NFARSDE3','NFARSDE4']:
	# 	DatVes.read_traindata()
	# 	DatVes.train_data_trans(1)
	# 	NFModel.noirange = DatVes.noirange
	# 	NFModel.Model_drift = DatVes.Model_drift
	try:
		showcf = config.show_config
	except:
		pass
	
	# Eva.plot_samplecompare(save=True)
	# Eva.plot_fftompare(save=True)
	Eva.plot_meancompare(save=True)
	# Eva.plot_meanqtlcompare(save=True)
	# Eva.plot_condpdfcompare(save=True)
	# Eva.plot_stoppingtimecompare(save=True)
	Eva.plot_condmeanvcompare(save=True)
	# Eva.plot_pdfcompare(save=True)
	# Eva.plot_Enscompare(DatVes,ens_m_path,save=True)
	# Eva.prediction_time()
	# Eva.plotcompute_acf(save=True)


if __name__ == '__main__':
	app.run(main)