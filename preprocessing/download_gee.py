import ee
import time
import gdown
from collections import deque
import tomllib
import click
#import geopandas as gpd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from pebble import ProcessPool
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
    # print("COMPLETED:\n")
    # print(completed_jobs)

    # print("FAILED:\n")
    # print(failed_jobs)
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
            'num_units': config['base']['agg_chunks'], #if config['base']['agg_unit']+start_date < end_date else end_date-start_date
            'collection': dataset['collection'],
            'parameter': dataset['parameter'],
            'time_unit': config['base']['agg_unit'],
            'drive_folder': config['base']['drive_folder']
        })
        start_date = start_date + relativedelta(**{config['base']['agg_unit']:config['base']['agg_chunks']})
        
    return requests
    
#Downloading locally has lower computational limits than exporting to google drive so we are gonna run a lot of parallel chunks here
#If I cant get this to work consistently will switch to drive export then download from there
def download_all(config):
    for dataset in config['datasets']:
        #Need to split up time otherwise we reach GEE limits
        requests = generate_requests(config, dataset)
        fulfill_requests(requests, config)
    success = monitor_exports()
    if success:
        gdown.download_folder(url=config['base']['drive_link'],output=config['base']['save_dir'])
    else:
        print('failed to export all jobs to drive, not downloading locally')
        print(failed_jobs)
        raise BaseException
        #merge_downloads(config, dataset)
    return True

def merge_all(config):
    for dataset in config['datasets']:
        merge_downloads(config, dataset)
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


def fulfill_requests(requests, config):

    for req in requests:
        export_over_time(req)
    # with ProcessPool(max_workers=config['base']['max_workers']) as pool:
    #     future = pool.map(export_over_time, requests, timeout=600)
    #     iterator = future.result()

    #     while True:
    #         try:
    #             result = next(iterator)
    #         except StopIteration:
    #             break
    #         except TimeoutError as error:
    #             print(error)
    #         except Exception as error:
    #             print(error)
    # return True

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
def export_over_time(args_dict):
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
        #Sequence is inclusive []
        ee.List.sequence(0, args_dict["num_units"]-1)
        .map(time_mapper)
    ).flatten()

    #print(month_ranges)
    # geemap.common.ee_export_vector(month_ranges, 
    #                                os.path.join(args_dict["save_dir"], f'{args_dict["parameter"]}_{args_dict["start_date"]}.csv'),
    #                                selectors = ['CD_MUN', 'median', 'start_date', 'end_date'])
    jobs.append(
    ee.batch.Export.table.toDrive(
        collection=month_ranges,
        fileFormat='CSV',
        selectors = ['CD_MUN', 'median', 'start_date', 'end_date'],
        folder=args_dict['drive_folder'],
        description=f'{args_dict["parameter"]}_{args_dict["start_date"]}'
    ))
    return True

def monitor_exports():
    running = len(jobs) > 0
    print('EXPORTING TO DRIVE...')
    while running:
        sleep_time = 5
        job = jobs.pop()
        state = job.status()['state']
        if state == 'UNSUBMITTED':
            job.start()
            sleep_time = 0
            jobs.appendleft(job)
        elif state == 'READY':
            jobs.appendleft(job)
        elif state == 'COMPLETED':
            completed_jobs.append(job)
            print(f'COMPLETED: {job.status()}')
        elif state == 'RUNNING':
            jobs.appendleft(job)
            print(f'RUNNING: {job.status()}')
        elif state == 'FAILED':
            failed_jobs.append(job)
            print(f'FAILED: {job.status()}')
        else:
            print(f'NOT SURE: {job.status()}')
        running = len(jobs) > 0
        time.sleep(sleep_time)
    return len(failed_jobs) == 0  

if __name__ == '__main__':
    #global nonsense to accomodate GEE
    assets = dict()
    jobs = deque()
    completed_jobs = list()
    failed_jobs= list()
    main()
