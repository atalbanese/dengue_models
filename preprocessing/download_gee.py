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

@click.command
@click.option('--config-file', type=str, default='preprocessing/gee_config.toml')
@click.option('--clean', type=bool, default=True)
@click.option('--auth', type=bool, default=False)
def main(config_file, clean, auth): 
    if auth:
        ee.Authenticate(auth_mode='notebook')
    ee.Initialize()
    with open(config_file, 'rb') as f:
        config = tomllib.load(f)

    populate_assets(config)
    download_all(config)
    merge_all(config)
    parquets = join_all(config)
    if clean:
        cleanup(config, parquets)

def join_all(config):
    all_parquets = glob.glob(os.path.join(config['base']['save_dir'], '*.parquet'))
    check_base_exists = glob.glob(os.path.join(config['base']['save_dir'], f'*{config["base"]["agg_unit"]}.parquet'))
    if len(check_base_exists)>0:
        all_parquets = list(set(all_parquets)-set(check_base_exists))
        to_read = check_base_exists[0]
        start_index=0
    else:
        to_read = all_parquets[0]
        start_index=1
    base_df = pl.read_parquet(to_read)
    if len(all_parquets)>1:
        for f in all_parquets[start_index:]:
            base_df = base_df.join(pl.read_parquet(f), on=['muni_id', 'start_date', 'end_date'])
    base_df.write_parquet(
        os.path.join(
            config['base']['save_dir'],
            f'all_parameters_{config["base"]["start_date"]}_{config["base"]["end_date"]}_{config["base"]["agg_unit"]}.parquet'
            )
        )
    return all_parquets


def cleanup(config, all_parquets):
    all_csvs = glob.glob(os.path.join(config['base']['save_dir'], '*.csv'))
    for csv in all_csvs:
        os.remove(csv)
    for parquet in all_parquets:
        os.remove(parquet)

def populate_assets(config):
    #assets = dict()
    assets['population'] = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq('country', 'BRA'))
    assets['bbox'] = ee.Geometry.Polygon(
        [[[-76.58124317412843, 6.639971446351328],
          [-76.58124317412843, -34.761991448401204],
          [-29.82343067412843, -34.761991448401204],
          [-29.82343067412843, 6.639971446351328]]])
    assets['munis_simple_1'], assets['munis_simple_2'] = load_munis(config['base']['munis_1'], config['base']['munis_2'])
    assets['reducer'] = {'median': ee.Reducer.median(),
                         'mean': ee.Reducer.mean()}[config['base']['reducer']]
    assets['crs'] = config['base']['crs']
    assets['scale'] = config['base']['scale']

def generate_requests(config, dataset):
    start_date, end_date = datetime.fromisoformat(config['base']['start_date']), datetime.fromisoformat(config['base']['end_date'])
    requests = list()
    while start_date < end_date:
        requests.append({
            'start_date': start_date.isoformat(),
            'num_units': config['base']['num_units'], #if config['base']['agg_unit']+start_date < end_date else end_date-start_date
            'collection': dataset['collection'],
            'parameter': dataset['parameter'],
            'time_unit': config['base']['agg_unit'],
            'drive_folder': config['base']['drive_folder']
        })
        start_date = start_date + relativedelta(**{config['base']['agg_unit']:config['base']['num_units']})
        
    return requests
    
#Downloading locally has lower computational limits than exporting to google drive so we are gonna run a lot of parallel chunks here
#If I cant get this to work consistently will switch to drive export then download from there
def download_all(config):
    for dataset in config['datasets']:
        #Need to split up time otherwise we reach GEE limits
        requests = generate_requests(config, dataset)
        fulfill_requests(requests)
    monitor_exports(config)
    gdown.download_folder(url=config['base']['drive_link'], output=config['base']['save_dir'], quiet=True)


def merge_all(config):
    for dataset in config['datasets']:
        if check_dataset(dataset):
            merge_downloads(config, dataset)
            write_metadata(config, dataset)
        else:
            print(f'{dataset} failed to export completely. You may need to adjust the num_units and retries parameters')
    return True

def write_metadata(config, dataset):
    with open('./preprocessing/metadata.txt', 'a') as f:
        f.write(
            f'Collection: {dataset["collection"]}\n\
            Parameter: {dataset["parameter"]}\n\
            Start Date: {config["base"]["start_date"]}\n\
            End Date: {config["base"]["end_date"]}\n\
            Time Units: {config["base"]["agg_unit"]}\n\
            Date Downloaded: {datetime.utcnow().isoformat()}\n'
        )

def check_dataset(dataset):
    for job in failed_jobs:
        if job['args']['parameter'] == dataset['parameter']:
            return False
    else:
        return True

def merge_downloads(config, dataset):
    (
        pl.read_csv(os.path.join(config["base"]["save_dir"], f'{dataset["parameter"]}_*.csv'), try_parse_dates=True)
        .rename({'median': dataset["parameter"],
                 'CD_MUN': 'muni_id'})
        .sort(['muni_id', 'start_date'])
        .fill_null(-999)
        #Filter out munis that are just lakes
        .filter(pl.col('muni_id')!= 430000)
        .write_parquet(os.path.join(config["base"]["save_dir"], f'{dataset["parameter"]}_{config["base"]["start_date"]}_{config["base"]["end_date"]}.parquet'))
    )


def fulfill_requests(requests):
    for req in requests:
        export_over_time(req)


#Helper fns
#We have municipios split into two chunks since there are over 5000 munis which is over the GEE feature collection limit
#There is a bug in uploading shapefiles so we are stuck using already uploaded assets
def load_munis(munis_1, munis_2):
    return ee.FeatureCollection(munis_1), ee.FeatureCollection(munis_2)

def clip_to_munis(img):
    return(img.clip(assets['bbox']))

#Population weighted aggregation
def agg_to_munis(img):
    img_stats_1 =  img.reduceRegions(**{
        'collection': assets['munis_simple_1'],
        'reducer':  assets['reducer'].splitWeights(),
        'scale': assets['scale'],  # meters
        'crs': assets['crs'],
    })

    img_stats_2 =  img.reduceRegions(**{
        'collection': assets['munis_simple_2'],
        'reducer': assets['reducer'].splitWeights(),
        'scale': assets['scale'],
        'crs': assets['crs'],
    })

    return img_stats_1.merge(img_stats_2)


#def export_over_time(start_date, num_units, collection, parameter, time_unit, save_dir):
def export_over_time(args_dict, retries=0):
    #relativedelta uses plural units, GEE uses singular
    time_unit = args_dict["time_unit"][:-1]
    col = ee.ImageCollection(args_dict["collection"]).select(args_dict["parameter"])

    base_date = ee.Date(args_dict["start_date"])

    def time_mapper(n):
        return agg_to_munis(
            col.filterDate(
                ee.DateRange(
                    base_date.advance(n, time_unit), 
                    base_date.advance(ee.Number(n).add(1), time_unit)
                )
            ).map(clip_to_munis)
            .median()
            .addBands(
                assets['population'].filter(ee.Filter.eq('year', base_date.advance(n, time_unit).get('year'))).first().unitScale(0, 21171)
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

    jobs.append({
        'job': ee.batch.Export.table.toDrive(
                collection=month_ranges,
                fileFormat='CSV',
                selectors = ['CD_MUN', 'median', 'start_date', 'end_date'],
                folder=args_dict['drive_folder'],
                description=f'{args_dict["parameter"]}_{args_dict["start_date"]}'
            ),
        'retries': retries,
        'args': args_dict
        }
    )
    return True

def monitor_exports(config):
    running = len(jobs) > 0
    print('EXPORTING TO DRIVE...')
    print('Check https://code.earthengine.google.com/tasks for task status')
    while running:
        sleep_time = 5
        job = jobs.pop()
        state = job['job'].status()['state']
        if state == 'UNSUBMITTED':
            job['job'].start()
            sleep_time = 0
            jobs.appendleft(job)
        elif state == 'READY':
            jobs.appendleft(job)
        elif state == 'COMPLETED':
            completed_jobs.append(job)
            #print(f'COMPLETED: {job.status()}')
        elif state == 'RUNNING':
            jobs.appendleft(job)
            #print(f'RUNNING: {job.status()}')
        elif state == 'FAILED':
            handle_failure(job, config)
            #failed_jobs.append(job)
            #print(f'FAILED: {job.status()}')
        else:
            print(f'NOT SURE: {job.status()}')
        running = len(jobs) > 0
        time.sleep(sleep_time)
    return len(failed_jobs) == 0  

def handle_failure(job, config):
    max_retries = config['base']['retries']
    cur_retries = job['retries']
    if cur_retries < max_retries:
        print(f'{job} FAILED. RETRYING...')
        job_1, job_2 = split_job(job)
        cur_retries += 1
        export_over_time(job_1, retries=cur_retries)
        export_over_time(job_2, retries=cur_retries)
    else:
        print(f'TOTAL FAILURE: {job}')
        failed_jobs.append(job)

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

if __name__ == '__main__':
    #global nonsense to accomodate GEE
    assets = dict()
    jobs = deque()
    completed_jobs = list()
    failed_jobs= list()
    main()
