import duckdb
from metaflow import Metaflow
import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.mx.model.forecast import DistributionForecast as mxDF
from gluonts.torch.model.forecast import DistributionForecast as tDF
import polars as pl
from datetime import datetime
import re

class DBBuilder():
    def __init__(self, db_loc = ':memory:'):
        self.handler_dict = {
            'config': self.handle_config,
            'forecasts': self.handle_predictions,
            'item_metrics': self.handle_item_metrics,
            'agg_metrics': self.handle_agg_metrics,
        }
        self.all_forecasts = []
        self.all_item_metrics = []
        self.all_agg_metrics = []
        self.all_configs = []
        self.con = duckdb.connect(database=db_loc)

        self.init_tables()
        self.find_all_data()
        self.populate_db()
        
        
    def init_tables(self):
        self.init_model_runs_table()
        self.init_predictions_table()
        self.init_item_metrics_table()
        self.init_agg_metrics_table()

    def init_model_runs_table(self):
        self.con.sql("""CREATE TABLE IF NOT EXISTS model_runs
                     (
                        model_id VARCHAR,
                        task_id VARCHAR,
                        model_type VARCHAR,
                        train_limit DATETIME,
                        valid_limit DATETIME,
                        test_limit DATETIME,
                        batch_size USMALLINT,
                        prediction_length UTINYINT,
                        train_epochs UTINYINT,
                        frequency VARCHAR,
                        static_features VARCHAR[],
                        dynamic_features VARCHAR[],
                        model_args VARCHAR,
                     )
                     """)
        
    def init_predictions_table(self):
        self.con.sql("""CREATE TABLE IF NOT EXISTS predictions
                     (
                        model_id VARCHAR,
                        task_id VARCHAR,
                        muni_id VARCHAR,
                        start_date DATETIME,
                        predict_date DATETIME,
                        step UTINYINT,
                        time_unit VARCHAR,
                        mean DOUBLE
                     )
                    """)
        
    def init_agg_metrics_table(self):
        self.con.sql("""CREATE TABLE IF NOT EXISTS agg_metrics
                     (
                            Coverage_0_1_      DOUBLE,  
                            Coverage_0_5_      DOUBLE,  
                            Coverage_0_9_      DOUBLE,  
                            MAE_Coverage      DOUBLE,  
                                    MAPE      DOUBLE,  
                                    MASE      DOUBLE,  
                                    MSE      DOUBLE,  
                                    MSIS      DOUBLE,  
                                        ND      DOUBLE,  
                                    NRMSE      DOUBLE,  
                                    OWA      DOUBLE,  
                        QuantileLoss_0_1_      DOUBLE,  
                        QuantileLoss_0_5_      DOUBLE,  
                        QuantileLoss_0_9_      DOUBLE,  
                                    RMSE      DOUBLE,  
                                abs_error      DOUBLE,  
                        abs_target_mean      DOUBLE,  
                            abs_target_sum      DOUBLE,  
                        mean_wQuantileLoss      DOUBLE,  
                                model_id     VARCHAR,  
                                    sMAPE      DOUBLE,  
                            seasonal_error      DOUBLE,  
                                task_id     VARCHAR,  
                        wQuantileLoss_0_1_      DOUBLE,  
                        wQuantileLoss_0_5_      DOUBLE,  
                        wQuantileLoss_0_9_      DOUBLE,  
                mean_absolute_QuantileLoss      DOUBLE, 
                     ) 
                """)
        
    def init_item_metrics_table(self):
        self.con.sql("""CREATE TABLE IF NOT EXISTS item_metrics
                     (
                        item_id           VARCHAR,     
                        forecast_start    TIMESTAMP,   
                        MSE               DOUBLE,      
                        abs_error         DOUBLE,      
                        abs_target_sum    DOUBLE,      
                        abs_target_mean   DOUBLE,      
                        seasonal_error    DOUBLE,      
                        MASE              DOUBLE,      
                        MAPE              DOUBLE,      
                        sMAPE             DOUBLE,      
                        ND                DOUBLE,      
                        MSIS              DOUBLE,      
                        QuantileLoss_0_1_ DOUBLE,      
                        Coverage_0_1_     DOUBLE,      
                        QuantileLoss_0_5_ DOUBLE,      
                        Coverage_0_5_     DOUBLE,      
                        QuantileLoss_0_9_ DOUBLE,      
                        Coverage_0_9_     DOUBLE,      
                        model_id          VARCHAR,     
                        task_id           VARCHAR
                     )
            """)
        
    def populate_db(self):
        all_forecasts = pl.concat(self.all_forecasts)
        self.con.execute('INSERT INTO predictions SELECT * from all_forecasts')
        #pl is faster
        all_runs = pl.concat(self.all_configs)
        self.con.execute('INSERT INTO model_runs SELECT * from all_runs')
        #these are already stored as pd dataframes so we'll stick with pd :( (for now, ugh)
        all_item_metrics = pd.concat(self.all_item_metrics)
        self.con.execute('INSERT INTO item_metrics SELECT * from all_item_metrics')

        all_agg_metrics = pl.concat(self.all_agg_metrics)
        self.con.execute('INSERT INTO agg_metrics SELECT * from all_agg_metrics')

    def find_all_data(self):
        #Find all metaflow data artifacts from model runs
        flows = Metaflow().flows
        for flow in flows:
            for run in flow:
                for step in run.steps():
                    for task in step.tasks():
                        for entry in task.data._artifacts.items():
                            self.handle_data_artifact(entry)
    
    
    def handle_data_artifact(self, entry):
        key, artifact = entry
        run_id = str(artifact._object['run_number'])
        task_id = str(artifact._object['task_id'])
        unique_id = run_id + task_id
        if key in self.handler_dict:
            self.handler_dict[key](artifact, run_id, task_id)

    def handle_config(self, artifact, run_id, task_id):
        cfg = artifact.data
        df = pl.DataFrame(
            data = {
                'model_id': [run_id],
                'task_id': [task_id],
                'model_type': [cfg['base']['model']],
                'train_limit': [datetime.fromisoformat(cfg['reshape']['train_time_limit'])],
                'valid_limit': [datetime.fromisoformat(cfg['reshape']['valid_time_limit'])],
                'test_limit': [datetime.fromisoformat(cfg['reshape']['test_time_limit'])],
                'batch_size': [cfg['general_model']['train_batch_size']],
                'prediction_length': [cfg['general_model']['prediction_length']],
                'train_epochs':[ cfg['general_model']['epochs']],
                'frequency': [cfg['reshape']['frequency']],
                'static_features': [cfg['reshape']['static_features']],
                'dynamic_features': [cfg['reshape']['dynamic_features']],
                #TODO: Model types werent properly saved with config files for multi model flow
                #Probably similar problem for multi feature flow
                #Need to adjust all the multi_flows to make sure things are a-ok
                'model_args': [str(cfg[cfg['base']['model']])]
            }
        )

        self.all_configs.append(df)

    def handle_item_metrics(self, artifact, run_id, task_id):
        item_metrics = artifact.data
        #Holy shit pandas is bad, this would all be one parallel task in polars with no reassignments
        item_metrics.columns = item_metrics.columns.str.replace(r"\W+", "_",regex=True)
        item_metrics['item_id'] = item_metrics['item_id'].astype('string')
        item_metrics['model_id'] = run_id
        item_metrics['task_id'] = task_id
        item_metrics['forecast_start'] = item_metrics['forecast_start'].dt.to_timestamp()
        self.all_item_metrics.append(item_metrics)

    def handle_agg_metrics(self, artifact, run_id, task_id):
        agg_metrics = artifact.data
        agg_metrics['model_id'] = run_id
        agg_metrics['task_id'] = task_id
        self.all_agg_metrics.append(pl.DataFrame(dict(sorted(agg_metrics.items()))).rename({k:re.sub(r"\W+", '_',k) for k in agg_metrics.keys()}))

        
        

    def handle_actual_values(self):
        pass


    def handle_predictions(self, artifact, run_id, task_id):
        for forecast in artifact.data:
            self.handle_prediction(forecast, run_id, task_id)
        
            
    def handle_prediction(self, sample_forecast, run_id, task_id):
        try:
            dates = sample_forecast._index.start_time
            period = sample_forecast._index.freqstr
            muni_id = sample_forecast.item_id
            if isinstance(sample_forecast, QuantileForecast):
                means = sample_forecast.mean
            elif isinstance(sample_forecast, SampleForecast):
                means = sample_forecast.samples.mean(axis=0)
            elif isinstance(sample_forecast, mxDF) or isinstance(sample_forecast, tDF):
                means = sample_forecast.mean
            #stds = sample_forecast.samples.std(axis=0)
            start_date = dates[0].to_pydatetime()
            num_entries = len(dates)

            df = pl.DataFrame(
                {
                    'model_id': [str(run_id)]*num_entries,
                    'task_id': [str(task_id)]*num_entries,
                    'muni_id' : [muni_id]*num_entries,
                    'start_date': [start_date]*num_entries,
                    'predict_date': dates,
                    'step': list(range(num_entries)),
                    'time_unit' : [period]*num_entries,
                    'mean' : means,
                    #'std' : pd.Series(stds, dtype=np.float32),          
                })

            self.all_forecasts.append(df)

        except BaseException as e:
            #some quantile forecasts are missing key data. but why??
            print(e)
            print(sample_forecast)
            print(vars(sample_forecast))

    def save_table(self, table_name, out_loc):
        self.con.execute(f'SELECT * from {table_name}').pl().write_parquet(out_loc)




        




if __name__ == '__main__':
    import time
    start = time.time()
    test = DBBuilder()
    end = time.time()
    print(end - start)
    #test.save_table('predictions', 'predictions.parquet')
    #print(test.con.sql("SELECT * FROM predictions"))

    #import toml


    

