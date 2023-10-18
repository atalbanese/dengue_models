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
    def __init__(self, config, merge_only=False):
        self.config = config
        self.jobs = deque()
        self.requested_jobs = list()
        self.completed_jobs = list()
        self.failed_jobs = list()
        self.merge_only = merge_only


    @property
    def requested_datasets(self):
        out = set()
        if not self.merge_only:
            for job in self.requested_jobs:
                out.add((job['collection'], job['parameter']))
        else:
            for job in self.config['datasets']:
                out.add((job['collection'], job['parameter']))
            if 'temporal_sdm' in self.config:
                out.add(('temporal_sdm',  f'temporal_sdm_{self.config["temporal_sdm"]["species"]}'))
        return out
         

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
                print(f'NOT SURE: {state}')
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

    def check_dataset(self, parameter):
        for job in self.failed_jobs:
            if job['args']['parameter'] == parameter:
                return False
        else:
            if len(glob.glob(os.path.join(self.config['base']['save_dir'], f'{parameter}_*.csv'))) == 0:
                return False
        return True

    def download_folder(self):
        gdown.download_folder(url=self.config['base']['drive_link'], output=self.config['base']['save_dir'], quiet=True)

    def merge_all(self):
        for dataset in self.requested_datasets:
            if self.check_dataset(dataset[1]):
                self.merge_downloads(dataset[1])
                self.write_metadata(dataset)
            else:
                print(f'{dataset} failed to export completely. You may need to adjust the num_units and retries parameters')
        return True
    
    def merge_downloads(self, parameter):
        (
            pl.read_csv(os.path.join(self.config["base"]["save_dir"], f'{parameter}_*.csv'), try_parse_dates=True)
            .rename({'median': parameter,
                    'CD_MUN': 'muni_id'})
            .sort(['muni_id', 'start_date'])
            .fill_null(-999)
            #Filter out munis that are just lakes
            .filter(pl.col('muni_id')!= 430000)
            .with_columns((pl.col('start_date').str.to_date('%Y-%m-%d'),pl.col('end_date').str.to_date('%Y-%m-%d')))
            .write_parquet(os.path.join(self.config["base"]["save_dir"], f'{parameter}_{self.config["base"]["start_date"]}_{self.config["base"]["end_date"]}.parquet'))
        )
        pass
    
    def write_metadata(self, dataset):
        with open('./preprocessing/metadata.txt', 'a') as f:
            f.write(
                f'Collection: {dataset[0]}\n\
        Parameter: {dataset[1]}\n\
        Start Date: {self.config["base"]["start_date"]}\n\
        End Date: {self.config["base"]["end_date"]}\n\
        Time Units: {self.config["base"]["agg_unit"]}\n\
        Date Downloaded: {datetime.utcnow().isoformat()}\n\
        Population Weighted: True\n\
        Time aggregation: Median\n\
        Space aggregation: {self.config["base"]["reducer"]}\n'
        )


    @staticmethod
    def split_job(job):
        job = job['args']
        start_date = job['start_date']
        num_units = job['num_units']

        #Ensuring we get deep copies of the original job to modify
        job_1 = {k:v for k, v in job.items()}
        job_2 = {k:v for k, v in job.items()}

        #split job in half
        job_1['num_units'] = job['num_units']//2

        #get whatever time remains after job_1
        job_2['num_units'] = num_units - job_1['num_units']

        #increment start date
        job_2['start_date'] = (datetime.fromisoformat(start_date) + relativedelta(**{job['time_unit']:job_1['num_units']})).isoformat()

        return job_1, job_2


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
        self.assets['munis_simple'] = ee.FeatureCollection(self.config['base']['munis']).randomColumn()
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
        'collection': args_dict['collection'],
        'parameter': args_dict['parameter'],
        'retries': retries,
        'args': args_dict,
        'owner':self
        })
    
    # def get_muni_aggregator(self):
    #     @staticmethod
    #     def agg_to_munis(img):
    #         img_stats_1 =  img.reduceRegions(**{
    #             'collection': self.assets['munis_simple_1'],
    #             'reducer':  self.assets['reducer'].splitWeights(),
    #             'scale': self.assets['scale'],  # meters
    #             'crs': self.assets['crs'],
    #         })

    #         img_stats_2 =  img.reduceRegions(**{
    #             'collection': self.assets['munis_simple_2'],
    #             'reducer': self.assets['reducer'].splitWeights(),
    #             'scale': self.assets['scale'],
    #             'crs': self.assets['crs'],
    #         })

    #         return img_stats_1.merge(img_stats_2)
    #     return agg_to_munis

    # def get_muni_aggregator(self):
    #     @staticmethod
    #     def agg_to_munis(img):

    #         return self.assets['munis_simple'].map(lambda f: ee.Feature(f.geometry(), img.reduceRegion(
    #             reducer = self.assets['reducer'].splitWeights(),
    #             geometry = f.geometry(),
    #             scale = self.assets['scale'],
    #             crs = self.assets['crs'],
    #             bestEffort = True
    #         ).combine(f.toDictionary())))

    #     return agg_to_munis
    def get_muni_aggregator(self):
        num_splits = 4
        @staticmethod
        def agg_to_munis(img):
            cols = []
            for i in range(num_splits):
                lower_bound = i/num_splits
                upper_bound = (i+1)/num_splits

                cols.append(img.reduceRegions(**{
                'collection': self.assets['munis_simple'].filter(ee.Filter([ee.Filter.gte('random', lower_bound), ee.Filter.lt('random', upper_bound)])),
                'reducer':  self.assets['reducer'].splitWeights(),
                'scale': self.assets['scale'],  # meters
                'crs': self.assets['crs'],
                }))


            return ee.FeatureCollection(cols).flatten()

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
        super().__init__(config, downloader=downloader)

        self.add_additional_assets()
        self.classifier = self.train_classifier()

    def add_additional_assets(self):
        #TODO: just make this recs depending on which one was requested
        #Could also just combine into a 'dengue species SDM'
        presence_recs = {
            'albopictus': self.fix_recs(ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/albo_gbif_date_filtered"), 1),
            'aegypti': self.fix_recs(ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/aegypti_gbif_date_filtered"), 1),
            'both': (
                self.fix_recs(ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/albo_gbif_date_filtered"), 1).merge(
                     self.fix_recs(ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/aegypti_gbif_date_filtered"), 1)
                )
            )
        }[self.config['temporal_sdm']['species']]
        #We have 8000 premade date pseudo-absence points, we only need a small percent of them
        absence_percent = self.config['temporal_sdm']['num_absence_points']/8000
        absence_recs = self.fix_recs(ee.FeatureCollection("projects/ee-dengue-proof-of-concept/assets/dated_absence"), 0).randomColumn().filter(f'random <= {absence_percent}')

        self.assets['all_points'] = presence_recs.merge(absence_recs)

    def assemble_data(self, start_date, end_date):
        #TODO: Add in population flag

        return (
            ee.ImageCollection([self.sample_collection(start_date, end_date, dset) for dset in self.config['sdm_datasets']]
            ).toBands(
            ).clip(self.assets['bbox'])
        )

    def sample_collection(self, start_date, end_date, dataset):
        #Samples a histogram with set breaks then does some array magic to get each bin as its own image
        return (ee.ImageCollection(dataset['collection']
                ).filter(
                    ee.Filter.date(start_date,end_date)
                ).select(dataset['parameter']
                ).map(self.clip_to_munis
                ).reduce(
                    ee.Reducer.fixedHistogram(
                        min=dataset['min'],
                        max=dataset['max'],
                        steps=dataset['steps']
                    )
                ).arraySlice(1,1,2
                ).arrayProject([0]
                ).arrayFlatten(
                    [   
                        ee.List.sequence(1, dataset['steps']).map(
                            lambda i : ee.String(ee.Number(i).toInt().format()).cat(dataset['parameter'])
                        )
                    ]
                )
            )
        
    def fix_recs(self, in_points, presence_num):

        def map_features(feature):
            cur_date = ee.Date.fromYMD(
                  year = feature.get('year'), 
                  month = feature.get('month'), 
                  day =  feature.get('day')
                )
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
        return (
            self.assets['all_points']
            .filter('year <= 2022')
            .map(lambda f: (
                self.assemble_data(f.get('back_date'), f.get('date'))
                .sampleRegions(collection = ee.FeatureCollection([f]), scale=3000)
                .first()
            ), True)
        )

    def create_exports(self):
        self.generate_requests({
            'collection': 'temporal_sdm',
            'parameter': f'temporal_sdm_{self.config["temporal_sdm"]["species"]}'
        })
    def train_classifier(self):
        return (
            ee.Classifier.amnhMaxent(
                randomTestPoints = 25,
                seed=42
            ).train(
                features = self.get_training_data(),
                classProperty = 'presence',
                inputProperties = self.assemble_data(ee.Date('2020-01-01'), ee.Date('2020-03-01')).bandNames()
            )
        )

    def export_over_time(self, args_dict, retries=0):
        time_unit = args_dict["time_unit"][:-1]

        def time_mapper(n):
            base_date = ee.Date(args_dict['start_date']).advance(n, time_unit)
            return self.agg_to_munis(
                self.assemble_data(
                    base_date.advance(-self.config['temporal_sdm']['lookback_window'], 'day'),
                    base_date
                ).classify(self.classifier
                ).select('probability'
                ).addBands(
                    self.assets['population'].filter(ee.Filter.eq('year', base_date.get('year'))).first().unitScale(0, 21171)
                )
            ).map(lambda f : f.set(
                {
                'end_date' : base_date.format('YYYY-MM-dd'),
                'start_date' : base_date.advance(-self.config['temporal_sdm']['lookback_window'], 'day').format('YYYY-MM-dd')
                }
            ))
        
        predict_data = ee.FeatureCollection(
            ee.List.sequence(0, args_dict['num_units']-1)
            .map(time_mapper)
        ).flatten()

        self.downloader.add_job({
        'job': ee.batch.Export.table.toDrive(
                collection=predict_data,
                fileFormat='CSV',
                selectors = ['CD_MUN', 'median', 'start_date', 'end_date'],
                folder=args_dict['drive_folder'],
                description=f'{args_dict["parameter"]}_{args_dict["start_date"]}'
            ),
        'retries': retries,
        'collection': args_dict['collection'],
        'parameter': args_dict['parameter'],
        'args': args_dict,
        'owner':self
        })
    
if __name__ == '__main__':
    #ee.Authenticate(auth_mode='notebook')
    ee.Initialize()
    with open('preprocessing/gee_config.toml', 'rb') as f:
        CFG = tomllib.load(f)
    test = GEESDMRequestor(CFG)
    test.create_exports()
    test.downloader.run_exports()
    test.downloader.download_folder()
    print('here')