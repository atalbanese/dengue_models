import duckdb
from metaflow import Metaflow
import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast, QuantileForecast
import polars as pl

class DBBuilder():
    def __init__(self, db_loc = ':memory:'):
        self.handler_dict = {
            'config': self.handle_config,
            'forecasts': self.handle_predictions
        }
        self.con = duckdb.connect(database=db_loc)
        if db_loc == ':memory:':
            self.init_tables()
        self.find_all_data()
        
        self.all_data = {
            '':''
        }

        
    
        
    def init_tables(self):
        self.init_model_runs_table()
        self.init_predictions_table()
        pass

    def init_model_runs_table(self):
        self.con.sql("""CREATE TABLE model_runs
                     (
                        modelId UBIGINT PRIMARY KEY,
                        modelType VARCHAR,
                        trainLimit DATE,
                        validLimit DATE,
                        testLimit DATE,
                        context_length UTINYINT,
                        batchSize USMALLINT,
                        predictionLength UTINYINT,
                        trainEpochs UTINYINT,
                        frequency VARCHAR,
                        staticFeatures VARCHAR[],
                        dynamicFeatures VARCHAR[],
                        modelArgs VARCHAR,
                     )
                     """)
        
    def init_predictions_table(self):
        self.con.sql("""CREATE TABLE predictions
                     (
                        model_id BIGINT,
                        muni_id VARCHAR,
                        start_date DATETIME,
                        predict_date DATETIME,
                        step UTINYINT,
                        time_unit VARCHAR,
                        mean DOUBLE
                     )
                    """)
        

    def find_all_data(self):
        #Find all metaflow data artifacts from model runs
        flows = Metaflow().flows
        print(flows)
        for flow in flows:
            for run in flow:
                if run.data != None:
                    for entry in run.data._artifacts.items():
                        self.handle_data_artifact(entry)
    
    def handle_data_artifact(self, entry):
        key, artifact = entry
        run_id = artifact._object['run_number']
        if key in self.handler_dict:
            self.handler_dict[key](artifact, run_id)

    def handle_config(self, artifact, run_id):
        cfg = artifact.data

    def handle_item_metrics(self, artifact, run_id):
        pass
    def handle_agg_metrics(self, artifact, run_id):
        pass
    def handle_actual_values(self):
        pass

    def handle_predictions(self, artifact, run_id):
        for forecast in artifact.data:
            self.handle_prediction(forecast, run_id)
            
    def handle_prediction(self, sample_forecast, run_id):
        dates = sample_forecast._index.start_time
        period = sample_forecast._index.freqstr
        muni_id = sample_forecast.item_id
        if isinstance(sample_forecast, QuantileForecast):
            means = sample_forecast.mean
        elif isinstance(sample_forecast, SampleForecast):
            means = sample_forecast.samples.mean(axis=0)
        #stds = sample_forecast.samples.std(axis=0)
        start_date = dates[0]
        num_entries = len(dates)

        df = pd.DataFrame.from_dict(
            {
                'model_id': pd.Series([int(run_id)]*num_entries, dtype=np.int64),
                'muni_id' : pd.Series([muni_id]*num_entries, dtype='string'),
                'start_date': pd.Series([start_date]*num_entries),
                'predict_date': dates,
                'step': pd.Series(list(range(num_entries)), dtype=np.int8),
                'time_unit' : pd.Series([period]*num_entries, dtype='string'),
                'mean' : pd.Series(means, dtype=np.float32),
                #'std' : pd.Series(stds, dtype=np.float32),          
             })
        #self.con.append('predictions', df)
        self.con.execute('INSERT INTO predictions SELECT * from df')

    def save_table(self, table_name, out_loc):
        self.con.execute(f'SELECT * from {table_name}').pl().write_parquet(out_loc)




        




if __name__ == '__main__':
    test = DBBuilder()
    test.save_table('predictions', 'predictions.parquet')
    print(test.con.sql("SELECT * FROM predictions"))

    #import toml


    

