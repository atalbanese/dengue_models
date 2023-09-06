# Dengue Modeling Framework
A modeling workflow for predicting Dengue case rates in Brazil using GluonTS for time series manipulation and model evaluation, Metaflow for model orchestration, CometML for logging, and GeoPandas for mapping.

A single model can be trained and evaluated using 'python model_flow.py run'.

Multiple models and model parameters can be trained in parallel using 'python multi_model_tune.py run'. 

Submodels for a single model type can be run on clusters/subsets of the data using 'python model_cluster_flow.py run'

The following models are currently implemented: 
  deep_ar, deep_state, deep_factor, mqcnn, transformer_mx, transformer, feed_forward

## Using config.toml
  The workflow is controlled through a config file written as TOML (https://toml.io/en/).
  
  After cloning the repo, please run `git update-index --skip-worktree config.toml` to isolate local changes to the config file. 

  config.toml is organized as follows:
  ### base
  Defines configuration options common to all workflows: Data location, variables to ignore in data, and model type(s) to run. The model option can either be a single string for model_flow and model_cluster_flow, or a list of strings when running multi_model_tune.

  ### general_model
  #TODO: merge into base
  
  Defines model batch size, batches per epoch, prediction length, and how many samples to generate when evaluating models.

  ### reshape
  Parameters to set up data loading and reshaping, including which static and dynamic features to use, as well as the boundaries of training/testing/validation data.

  ### model specific parameters
  Initilization parameters for available model types. Model parameters start with [MODEL_NAME]. 
  
  At a minimum, a model specification must define whether to use a gluonts or pytorch based dataloader. (loader = 'gluonts' OR loader = 'torch')

  
  Multiple parameters can be tested by listing parameter options and running the multi_model_tune flow, for ex: add_age_feature = [true, false] instead of add_age_feature = true.
  
  Warning: all listed parameter combinations will be tested when running multi_model_tune, so testing many parameters at the same time can lead to 100s of models.
  In the future, a parameter optimization framework will be implemented to ease this process

  ### maps
  This is just the location of the shapefile for creating evaluation maps. Will be moved to base eventually

  ### [metrics]
  By using the GluonTS evaluator, we get many forecasting metrics. If you would like a map produced for a given metric, you can define it with a new [[metrics]] entry. You must provide the name of the metric. All other options should be valid GeoPandas plotting options

  
  ## Multiple configurations
  You may want to make multiple config files. By default, all model flows look for a local .config.toml file. Other files can be specified using:
  
  python model_flow.py run --config-file '/path/to/config/file'

## Using .env
The default logging framework, CometML, uses environment variables to store options. The local .env file defines those environment variables. In the future, we will be agnostic from Comet and this file may become optional.
After cloning the repo, please run `git update-index --skip-worktree .env` to isolate local changes to the .env file. 


## Adding additional models
  New models should be placed in the model_code folder. In the future, new models in the model_code folder will be automatically integrated into the workflow. For now, it should be possible to add a new model by modifying the utils.py loading functions and adding a model specific section to config.toml
  
  
