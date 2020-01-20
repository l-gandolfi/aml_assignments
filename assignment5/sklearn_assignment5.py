from sklearn.neural_network import MLPClassifier

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.optimizer.acquisition import EI, PI, LCB
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.runhistory.runhistory import RunHistory
from smac.intensification.intensification import Intensifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

import logging
import typing
import numpy as np
import pandas as pd

def preprocess_labels(labels, encoder=None, categorical=True):
    """Encode labels with values among 0 and `n-classes-1`"""
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def preprocess_data(X, scaler=None):
    """Preprocess input data by standardise features 
    by removing the mean and scaling to unit variance"""
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def load_data(path):
	df = pd.read_csv(path)
	features_df = df[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]]
	labels_df = df[["Class"]]
	
	features = features_df.values.copy()
	labels = labels_df.values.copy()
	
	features, scaler = preprocess_data(features)
	labels, encoder = preprocess_labels(labels.ravel(), categorical=False)
	
	
	return features, labels


def nn_from_cfg(cfg):
	'''
	cfg: Configuration containing the parameters: learning rate and momentum
	model: Keras Sequential model containing layers and neurons
	'''
	
	PATH = '/home/luca/Desktop/Magistrale/Advanced Machine Learning/Assignment 5/php9pgo5r.csv'
	
	# Load the data
	X, Y = load_data(PATH)
	
	# Start the 10-fold cross validation manually
	kfold = StratifiedKFold(n_splits=10, shuffle=True)
	cvscores = []
	
	clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(4, 2), 
			    random_state=1, **cfg)

	
	print('Starting the k-fold ...')
	for train_index, test_index in kfold.split(X, Y):
		print('Fitting the model ...')
		clf.fit(X[train_index], Y[train_index])
		
		# evaluate the model
		scores = clf.score(X[test_index], Y[test_index])
		cvscores.append(scores)
	
	return 1 - np.mean(cvscores)  # Minimize!
	

# Logger
logging.basicConfig(level=logging.INFO)

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

lr = UniformFloatHyperparameter("learning_rate_init", 0.01, 0.1, default_value=0.1)
momentum = UniformFloatHyperparameter("momentum", 0.1, 0.9, default_value=0.1)
cs.add_hyperparameters([lr, momentum])

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 20,   # max. number of function evaluations;
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# It returns: Status, Cost, Runtime, Additional Infos
def_value = nn_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))


# Costruisco initial design di 5 points


initial_design = RandomConfigurations(tae_runner=nn_from_cfg, scenario=scenario,
								rng=np.random.RandomState(1), n_configs_x_params=5,
								traj_logger=TrajLogger,runhistory=RunHistory, 
								aggregate_func=typing.Callable, stats=Stats,
								intensifier=Intensifier)

# Optimize, using a SMAC-object
print("Optimizing ...")
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=nn_from_cfg, initial_design=RandomConfigurations, acquisition_function=LCB)

print('Calculating Incumbent ...')
# Obtain an incumbent configuration
incumbent = smac.optimize()

# Incumbet value
inc_value = nn_from_cfg(incumbent)
print("Optimized Value: %.2f" % (inc_value))

# Valuate
smac.validate(config_mode='inc',      # We can choose which configurations to evaluate
              #instance_mode='train+test',  # Defines what instances to validate
              n_jobs=1) 
