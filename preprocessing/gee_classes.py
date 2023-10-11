import ee
import time
import gdown
from collections import deque
import tomllib
import click
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import polars as pl
import glob


class GEEDownloader():
    def __init__(self, config):
        self.config = config
        self.jobs = deque()
        self.requested_jobs = list()
        self.completed_jobs = list()
        self.failed_jobs = list()

    def add_job(self, job):
        self.jobs.append(job)
        self.requested_jobs.append(job)

    def run_exports(self):
        running = len(self.jobs) > 0
        print('EXPORTING TO DRIVE...')
        print('Check https://code.earthengine.google.com/tasks for task status')
        while running:
            sleep_time = 5
            job = self.jobs.pop()
            state = job['job'].status()['state']
            if state == 'UNSUBMITTED':
                job['job'].start()
                sleep_time = 0
                self.jobs.appendleft(job)
            elif state == 'READY':
                self.jobs.appendleft(job)
            elif state == 'COMPLETED':
                self.completed_jobs.append(job)
                #print(f'COMPLETED: {job.status()}')
            elif state == 'RUNNING':
                self.jobs.appendleft(job)
                #print(f'RUNNING: {job.status()}')
            elif state == 'FAILED':
                self.handle_failure(job)
                #failed_jobs.append(job)
                #print(f'FAILED: {job.status()}')
            else:
                print(f'NOT SURE: {job.status()}')
            running = len(self.jobs) > 0
            time.sleep(sleep_time)
        return len(self.failed_jobs) == 0  
        
    def handle_failure(self, job):
        max_retries = self.config['base']['retries']
        cur_retries = job['retries']
        if cur_retries < max_retries:
            print(f'{job} FAILED. RETRYING...')
            job_1, job_2 = self.split_job(job)
            cur_retries += 1
            job['owner'].export_over_time(job_1, retries=cur_retries)
            job['owner'].export_over_time(job_2, retries=cur_retries)
        else:
            print(f'TOTAL FAILURE: {job}')
            self.failed_jobs.append(job)

    def download_folder(self):
        gdown.download_folder(url=self.config['base']['drive_link'], output=self.config['base']['save_dir'], quiet=True)


class GEERequestor():
    def __init__(self, config, downloader: GEEDownloader = None):
        self.config = config
        self.assets = dict()
        self.downloader = GEEDownloader(config) if downloader is None else downloader
        self.dataset_key = 'datasets'

        self.populate_assets()
        # Need these functions to be compiled after assets have been populated and also have them be static methods, hence the constructors
        self.agg_to_munis = self.get_muni_aggregator()
        self.clip_to_munis = lambda img: img.clip(self.assets['bbox'])
    
    def populate_assets(self):
        self.assets['population'] = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq('country', 'BRA'))
        self.assets['bbox'] = ee.Geometry.Polygon(
            [[[-76.58124317412843, 6.639971446351328],
            [-76.58124317412843, -34.761991448401204],
            [-29.82343067412843, -34.761991448401204],
            [-29.82343067412843, 6.639971446351328]]])
        self.assets['munis_simple_1'], self.assets['munis_simple_2'] = (ee.FeatureCollection(self.config['base']['munis_1']), 
                                                                        ee.FeatureCollection(self.config['base']['munis_2']))
        self.assets['reducer'] = {'median': ee.Reducer.median(),
                            'mean': ee.Reducer.mean()}[self.config['base']['reducer']]
        self.assets['crs'] = self.config['base']['crs']
        self.assets['scale'] = self.config['base']['scale']

    def create_exports(self):
        for dataset in self.config[self.dataset_key]:
            self.generate_requests(dataset)
           
    def generate_requests(self, dataset):
        start_date, end_date = datetime.fromisoformat(self.config['base']['start_date']), datetime.fromisoformat(self.config['base']['end_date'])
        
        while start_date < end_date:
            self.export_over_time({
                'start_date': start_date.isoformat(),
                'num_units': self.config['base']['num_units'] if (
                    start_date + relativedelta(**{self.config['base']['agg_unit']:self.config['base']['num_units']}) <= end_date
                    ) else (
                    self.calc_delta(end_date, start_date, self.config['base']['agg_unit'])
                    ), 
                'collection': dataset['collection'],
                'parameter': dataset['parameter'],
                'time_unit': self.config['base']['agg_unit'],
                'drive_folder': self.config['base']['drive_folder']
            })

            start_date = start_date + relativedelta(**{self.config['base']['agg_unit']:self.config['base']['num_units']})
            
    
    def export_over_time(self, args_dict, retries=0):
        #relativedelta uses plural units, GEE uses singular
        time_unit = args_dict["time_unit"][:-1]
        col = ee.ImageCollection(args_dict["collection"]).select(args_dict["parameter"])

        base_date = ee.Date(args_dict["start_date"])

        def time_mapper(n):
            return self.agg_to_munis(
                col.filterDate(
                    ee.DateRange(
                        base_date.advance(n, time_unit), 
                        base_date.advance(ee.Number(n).add(1), time_unit)
                    )
                ).map(self.clip_to_munis)
                .median()
                .addBands(
                    self.assets['population'].filter(ee.Filter.eq('year', base_date.advance(n, time_unit).get('year'))).first().unitScale(0, 21171)
                )
            ).map(lambda f: f.set({
                'start_date': base_date.advance(n, time_unit).format('YYYY-MM-dd'),
                'end_date': base_date.advance(ee.Number(n).add(1), time_unit).format('YYYY-MM-dd')
                })
            )

        month_ranges = ee.FeatureCollection(
            #Sequence is inclusive
            ee.List.sequence(0, args_dict["num_units"]-1)
            .map(time_mapper)
        ).flatten()

        self.downloader.add_job({
        'job': ee.batch.Export.table.toDrive(
                collection=month_ranges,
                fileFormat='CSV',
                selectors = ['CD_MUN', 'median', 'start_date', 'end_date'],
                folder=args_dict['drive_folder'],
                description=f'{args_dict["parameter"]}_{args_dict["start_date"]}'
            ),
        'retries': retries,
        'args': args_dict,
        'owner':self
        })
    
    def get_muni_aggregator(self):
        @staticmethod
        def agg_to_munis(img):
            img_stats_1 =  img.reduceRegions(**{
                'collection': self.assets['munis_simple_1'],
                'reducer':  self.assets['reducer'].splitWeights(),
                'scale': self.assets['scale'],  # meters
                'crs': self.assets['crs'],
            })

            img_stats_2 =  img.reduceRegions(**{
                'collection': self.assets['munis_simple_2'],
                'reducer': self.assets['reducer'].splitWeights(),
                'scale': self.assets['scale'],
                'crs': self.assets['crs'],
            })

            return img_stats_1.merge(img_stats_2)
        return agg_to_munis
    

    @staticmethod
    def calc_delta(end_date, start_date, unit):
        mult = {
            'months': [12, 1, 0],
            'weeks': [52, 4, 1]
                }[unit]
        delta = relativedelta(end_date, start_date)
        return delta.years*mult[0] + delta.months*mult[1] + delta.weeks*mult[2]
    
class GEESDMRequestor(GEERequestor):
    def __init__(self, config, downloader:GEEDownloader=None):
        super.__init__(config, downloader=downloader)

        self.add_additional_assets()

    def add_additional_assets(self):
        self.assets['albo_recs'] = ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/albo_gbif_date_filtered")
        self.assets['aeg_recs'] = ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/aegypti_gbif_date_filtered")
        self.assets['absence_recs'] = ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/dated_absence")

    def fix_recs(self, in_points, presence_num):

        def map_features(feature):
            cur_date = ee.Date.fromYMD(
                {
                  'year': feature.get('year'), 
                  'month': feature.get('month'), 
                  'day': feature.get('day')
                })
            back_date = cur_date.advance(-self.config['temporal_sdm']['lookback_window'],'day' )

            return feature.set(
                  {
                'date': cur_date,
                'back_date': back_date,
                'presence': presence_num 
                }
            )

        return in_points.map(map_features)


    def get_training_data(self):
        pass

    def train_classifier(self):
        pass

    def export_over_time(self, args_dict, retries=0):
        return super().export_over_time(args_dict, retries)

    
if __name__ == '__main__':
    ee.Initialize()
    with open('preprocessing/gee_config.toml', 'rb') as f:
        CFG = tomllib.load(f)
    test = GEERequestor(CFG)
    test.create_exports()
    test.downloader.run_exports()
    test.downloader.download_folder()
    print('here')