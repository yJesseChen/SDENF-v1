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

import NFSDE    as Mymodel1
import ResNFSDE as Mymodel2
import MixNFSDE as Mymodel3
import NFNonAutoSDE    as Mymodel4
import ResNFNonAutoSDE as Mymodel5
import MixNFNonAutoSDE as Mymodel6
import NFSDE_SSAconserve as Mymodel7
# import NFARSDE as Mymodel8
# import NFARSDE2 as Mymodel9
# import NFARSDE3 as Mymodel10
# import NFARSDE4 as Mymodel11

import Evaulation
imp.reload(Mymodel1)
imp.reload(Mymodel2)
imp.reload(Mymodel3)
imp.reload(Mymodel4)
imp.reload(Mymodel5)
imp.reload(Mymodel6)
imp.reload(Mymodel7)
# imp.reload(Mymodel8)
# imp.reload(Mymodel9)
# imp.reload(Mymodel10)
# imp.reload(Mymodel11)
imp.reload(Evaulation)

os.chdir(sys.path[0])

flags.DEFINE_string('test_name',    'REx4_Mix_s2__1_prod',     'Name of test')
flags.DEFINE_string('model_name',   'NFSDE',         'Name of model')
flags.DEFINE_string('test_case',    'prod',                'Name of post test')
flags.DEFINE_string('test_file',    '',     'File dir of test data')
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
	# elif model_name=='NFARSDE':
	# 	return Mymodel8
	# elif model_name=='NFARSDE2':
	# 	return Mymodel9
	# elif model_name=='NFARSDE3':
	# 	return Mymodel10
	# elif model_name=='NFARSDE4':
	# 	return Mymodel11
	elif model_name==None:
		raise AttributeError("Model_select: please type in model name")
	else:
		raise AttributeError("Model_select: %s model is not supported"%(model_name))

def main(argv):
	del argv
	### Setup path
	result_path = './results'
	config_path = result_path + '/' + FLAGS.test_name + '/Test_config.json'
	bestm_path  = result_path + '/' + FLAGS.test_name + '/Test_model/model.pt'
	root_path    = result_path+'/'+FLAGS.test_name+'/'+FLAGS.test_case if FLAGS.test_case else result_path+'/'+FLAGS.test_name
	save_path    = root_path+'/'
	setting_path = root_path + '/Test_config.json'
	predt_path   = root_path + '/predict.mat'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	### Setup logging information
	absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(asctime)s\t%(levelname)-10s %(message)s'))
	absl_logging.get_absl_handler().use_absl_log_file('Test', root_path)
	absl_logging.set_verbosity('info')
	### Predict
	with open(config_path) as json_data_file:
		config = json.load(json_data_file)
	config = munch.munchify(config)
	if FLAGS.test_file: config.dat_config.TestData_dir = FLAGS.test_file
	json.dump(config, open(setting_path, 'w'), indent=2)
	Mymodel = Model_select(FLAGS.model_name)
	NFModel = Evaulation.Evaluate.readmodel(bestm_path,Mymodel.NFSSDE,config)
	### Average Predict
	DatVes = Mymodel.DataTran(config)
	if FLAGS.model_name in ['MixNFSDE','MixNFNonAutoSDE']:
		DatVes.train_hiddendata()
		NFModel.Model_drift = DatVes.Model_drift
	# if FLAGS.model_name in ['NFARSDE','NFARSDE2','NFARSDE3','NFARSDE4']:
	# 	DatVes.read_traindata()
	# 	DatVes.train_data_trans(1)
	# 	NFModel.noirange = DatVes.noirange
	# 	NFModel.Model_drift = DatVes.Model_drift
	DatVes.read_testdata()
	DatVes.test_mdat1model(NFModel,predt_path)


if __name__ == '__main__':
	app.run(main)