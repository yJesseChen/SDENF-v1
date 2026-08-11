import json
import munch
import os
import sys
import logging
import importlib as imp
import pdb

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

from absl import app
from absl import flags
from absl import logging as absl_logging

import MixNFNonAutoSDE  as Mymodule
import Evaulation
imp.reload(Mymodule)
imp.reload(Evaulation)

os.chdir(sys.path[0])

### Now support:
### 1. Ex12DisturbOU                    ./configs/Ex12DisturbOU.json

flags.DEFINE_string('test_name',   'Ex12DisturbOU',                'Name of test')
flags.DEFINE_string('config_path', './configs/Ex12DisturbOU.json', 'Path of config file')
flags.DEFINE_string('model',       None,                        'Name of model')
FLAGS = flags.FLAGS

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
	# MyGansmodel,Mymodule = Model_select(FLAGS.model)
	Eva = Evaulation.SdeNFEva(config,root_path,Monitor_path+'Eva')
	NFMonitor = Mymodule.Monitor(Monitor_path,config,Mymodule.NFSSDE,Eva)
	### Data Manuplation
	logging.info('Start to manupulate data')
	DatVes = Mymodule.DataTran(config,NFMonitor)
	DatVes.read_traindata()
	DatVes.train_data_trans(1)
	DatVes.read_testdata()
	DatVes.train_hiddendata()
	### Model definition and trained
	logging.info('Start to train %s model'%(FLAGS.model))
	NFModel = Mymodule.NFSSDE(config)
	NFModel.Model_drift = DatVes.Model_drift
	# if Retrain >= 1:
	# 	NFModel.read_Model(model_path_p + 'model.pt')
	NFModel.train(DatVes.train_mat,DatVes.para_mat,model_path,histy_path,NFMonitor,DatVes,predt_path)
	### Test model
	##  This part has been transferred into GanModel.train function for general purpose
	# logging.info('Start to test model')
	# DatVes.test_mdat1model(GanModel,predt_path)
	# logging.info('End of test')


if __name__ == '__main__':
	app.run(main)