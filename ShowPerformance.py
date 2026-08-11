import json
import munch
import os
import sys
import logging
import importlib as imp

import numpy as np
from absl import app
from absl import flags
from absl import logging as absl_logging
import pdb
import warnings

import Evaulation
imp.reload(Evaulation)

os.chdir(sys.path[0])

### Now support:
### 3. Ex3OU

flags.DEFINE_string('test_name',   'ResNonAuto_Ex12_s1_3',                'Name of test')
flags.DEFINE_string('test_case',    'fre3',                        'Name of post test')
FLAGS = flags.FLAGS


def main(argv):
	del argv
	### Setup path
	result_path = './results'
	root_path = result_path+'/'+FLAGS.test_name+'/'+FLAGS.test_case if FLAGS.test_case else result_path+'/'+FLAGS.test_name
	save_path = root_path+'/'+'Eva'
	config_path = root_path+'/'+'Test_config.json'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	### Load configration
	with open(config_path) as json_data_file:
		config = json.load(json_data_file)
	config = munch.munchify(config)
	### Evaluation
	Eva = Evaulation.SdeNFEva(config,root_path,save_path)
	showcf = config.show_config
	if ('plot_samplecompare' in showcf.keys()) and (showcf.plot_samplecompare):
		Eva.plot_samplecompare(save=True)
	if ('plot_meancompare' in showcf.keys()) and (showcf.plot_meancompare):
		Eva.plot_meancompare(save=True)
	try:
		if ('plot_losthist' in showcf.keys()) and (showcf.plot_losthist):
			Eva.plot_losthist(save=True)
			Eva.plot_Wdistance(save=True)
	except:
		warnings.warn("plot_losthist:: No hist plot generated.")


if __name__ == '__main__':
	app.run(main)