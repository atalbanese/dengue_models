#from comet_ml import init
from comet_ml.integration.metaflow import comet_flow, comet_skip
from metaflow import FlowSpec, step, IncludeFile, Flow, card, current
from dotenv import load_dotenv
from dill import dumps, loads

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from model_code import utils
import tomllib
import os

@comet_flow
class ModelFlow(FlowSpec):

    config_file = IncludeFile(
        'config_file',
        default= './config.toml',
        is_text = True,
        help='Configuration file for loading data and running model'
    )

    @comet_skip
    #@card
    @step
    def start(self):
        self.next(self.read_config)

    @comet_skip
    #@card
    @step
    def read_config(self):
        self.config = tomllib.loads(self.config_file)
        self.models = self.config['base']['model']
        
        self.next(self.load_data)

    @comet_skip
    #@card
    @step
    def load_data(self):
        self.dataset = utils.load_data(self.config['base']['file_loc'], self.config['base']['ignore_vars'])
        #print(self.dataset)
        self.next(self.make_config_grid, foreach='models')
    
    @comet_skip
    @step
    def make_config_grid(self):
        from sklearn.model_selection import ParameterGrid
        model_options = self.config[self.input]

        tuning_params = {k:v for k,v in model_options.items() if (isinstance(v, list)) and (len(v) > 1)}
        tuning_params['model'] = [self.input]
        print(tuning_params)
        self.grid_points = list(ParameterGrid(tuning_params))
        print(self.grid_points)
        self.next(self.train_model, foreach='grid_points')
        

    #@card
    @step
    def train_model(self):
        config = self.config
        model_type = self.input['model']
        self.input.pop('model')
        config['base']['model'] = model_type
        for k,v in self.input.items():
            self.config[model_type][k] = v
        model_data = utils.load_reshaper(self.dataset, config).get_data()
        #print('LOGGING TEST DATA')
        self.test_data = model_data['test']
        model = utils.load_model(config)
        self.model = dumps(model)

        predictor = model.train(
            training_data = model_data['train'],
            validation_data= model_data['valid']
        )
        self.predictor = dumps(predictor)
        self.next(self.evaluate_model)

    @comet_skip
    #@card
    @step
    def evaluate_model(self):
        predictor = loads(self.predictor)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset = self.test_data,
            predictor=predictor,
            num_samples=self.config['general_model']['num_test_samples']
        )
        #TODO: Figure out why evaluator errs when num_workers > 0, this is a big slowdown
        self.forecasts, self.test_data = list(forecast_it), list(ts_it)
        self.forecast_start_date = self.forecasts[0].start_date
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=0)
        self.agg_metrics, self.item_metrics = evaluator(self.test_data, self.forecasts)
        self.next(self.log_results)

    #@card
    @step
    def log_results(self):
        #TODO: Log actual predictions as simple csv or similar
        self.comet_experiment.log_parameters(self.config)
        self.comet_experiment.log_metrics(self.agg_metrics)
        self.comet_experiment.log_code('./model_code/pytorch_models.py')
        self.comet_experiment.log_code('./model_code/reshaper.py')
        self.comet_experiment.log_code('./model_code/utils.py')
        self.comet_experiment.log_table('prediction_metrics.csv', tabular_data=self.item_metrics, headers=True)
        utils.log_maps(self.config, self.item_metrics, self.comet_experiment, self.forecast_start_date)

        self.next(self.join)

    @comet_skip
    @step
    def join(self, inputs):
        self.next(self.join_2)
    
    @step
    def join_2(self, inputs):
        self.next(self.end)
    @comet_skip    
    @step
    def end(self):
        pass

if __name__ == '__main__':
    load_dotenv()
    ModelFlow()
