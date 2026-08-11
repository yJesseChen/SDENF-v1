import json
import munch
import os
import torch
import sys
import logging
import importlib as imp
import pdb

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

from absl import app
from absl import flags
from absl import logging as absl_logging

import MixNFSDE  as Mymodel1
import MixNFSDE_SSAgenconserve as Mymodel2
import Evaulation
imp.reload(Mymodel1)
imp.reload(Evaulation)

os.chdir(sys.path[0])

### Now support:
### 1. Ex1GeoBrownian                    ./configs/Ex1GeoBrownian.json
### 1. Ex1GeoBrownianm2s1                ./configs/Ex1GeoBrownianm2s1.json
### 3. Ex3OU                             ./configs/Ex3OU.json
### 4. Ex4ExpDiff                        ./configs/Ex4Expdiff.json
### 5. Ex5Trig  						 ./configs/Ex5Trig.json
### 8. Ex8DW 						     ./configs/Ex8DW.json

flags.DEFINE_string('test_name',   'Ex23SSALV',                'Name of test')
flags.DEFINE_string('config_path', './configs/Ex23SSALV.json', 'Path of config file')
flags.DEFINE_string('model_name',  'MixNFSDE',                        'Name of model')
FLAGS = flags.FLAGS

def Model_select(model_name):
	if model_name=='MixNFSDE':
		return Mymodel1
	elif model_name==None:
		return Mymodel1
	elif model_name=='MixNFSDE_SSAgenconserve':
		return Mymodel2
	else:
		raise AttributeError("Model_select: %s model is not supported"%(model_name))

def main(argv):
	del argv
	### Setup path
	result_path = './results'
	root_path = result_path+'/'+FLAGS.test_name
	bestm_path = root_path + '/Best_model/'
	model_path = root_path + '/Test_model/'
	histy_path = root_path + '/Test_history.json'
	predt_path = root_path + '/predict.mat'
	setting_path = root_path + '/Test_config.json'
	Monitor_path = root_path + '/Monitor/'
	if not os.path.exists(root_path):
		os.makedirs(root_path)
		Retrain = 0
	else:
		# Retrain = 1
		# model_path_p = model_path
		# model_path = root_path + '/Test_model'+str(Retrain)+'/'
		# histy_path = root_path + '/Test_history'+str(Retrain)+'.json'
		# Monitor_path = root_path + '/Monitor'+str(Retrain)+'/'
		Retrain = 0
	### Load configration
	with open(FLAGS.config_path) as json_data_file:
		config = json.load(json_data_file)
	config = munch.munchify(config)
	json.dump(config, open(setting_path, 'w'), indent=2)
	### Setup logging information
	absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(asctime)s\t%(levelname)-10s %(message)s'))
	absl_logging.get_absl_handler().use_absl_log_file('Test', root_path)
	absl_logging.set_verbosity('info')
	logging.info('Begin to learn %s ' % config.eqn_config.eqn_name)
	### Model set & Monitor & Evaluation
	Mymodel = Model_select(FLAGS.model_name)
	# MyGansmodel,Mymodel = Model_select(FLAGS.model)
	Eva = Evaulation.SdeNFEva(config,root_path,Monitor_path+'Eva')
	NFMonitor = Mymodel.Monitor(Monitor_path,config,Mymodel.NFSSDE,Eva)
	### Data Manuplation
	logging.info('Start to manupulate data')
	DatVes = Mymodel.DataTran(config,NFMonitor)
	DatVes.read_traindata()
	DatVes.train_data_trans(1)
	DatVes.read_testdata()
	DatVes.train_hiddendata()
	### Model definition and trained
	logging.info('Start to train %s model'%(FLAGS.model_name))
	NFModel = Mymodel.NFSSDE(config)
	NFModel.Model_drift = DatVes.Model_drift
	if Retrain >= 1:
		NFModel.read_Model(model_path_p + 'model.pt')
	NFModel.train(DatVes.train_mat,model_path,histy_path,NFMonitor,DatVes,predt_path)
	### Test model
	##  This part has been transferred into GanModel.train function for general purpose
	# logging.info('Start to test model')
	# DatVes.test_mdat1model(GanModel,predt_path)
	# logging.info('End of test')


if __name__ == '__main__':
	app.run(main)