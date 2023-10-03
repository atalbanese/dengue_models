import ee
#Notebook mode needed to run remotely
#ee.Authenticate(auth_mode='notebook')
ee.Initialize()
import geemap
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
def main(config_file, clean): 
    with open(config_file, 'rb') as f:
        config = tomllib.load(f)

    #populate_assets(config)
    #download_all(config)
    join_all(config)
    #if clean:
    #    cleanup(config)

def join_all(config):
    all_parquets = glob.glob(os.path.join(config['base']['save_dir'], '*.parquet'))
    check_base_exists = glob.glob(os.path.join(config['base']['save_dir'], 'all_parameters_*'))
    if len(check_base_exists)>0:
        all_parquets = list(set(all_parquets)-set(check_base_exists))
    base_df = pl.read_parquet(all_parquets[0])
    if len(all_parquets)>1:
        for f in all_parquets[1:]:
            base_df = base_df.join(pl.read_parquet(f), on=['muni_id', 'start_date', 'end_date'])
    base_df.write_parquet(os.path.join(config['base']['save_dir'],f'all_parameters_{config["start_date"]}_{config["end_date"]}.parquet'))
    return True


def cleanup(config):
    all_csvs = glob.glob(os.path.join(config['base']['save_dir'], '*.csv'))
    for csv in all_csvs:
        os.remove(csv)

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
            'save_dir': config['base']['save_dir']
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
    with ProcessPool(max_workers=config['base']['max_workers']) as pool:
        future = pool.map(export_over_time, requests, timeout=600)
        iterator = future.result()

        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                print(error)
            except Exception as error:
                print(error)
    return True

#Helper fns
#We have municipios split into two chunks since there are over 5000 munis which is over the GEE feature collection limit
#There is a bug in uploading shapefiles so we are stuck using already uploaded assets
def load_munis(munis_1, munis_2):
    #return geemap.geojson_to_ee(munis_1), geemap.geojson_to_ee(munis_2)
    #return geemap.shp_to_ee(munis_1), geemap.shp_to_ee(munis_2)
    #munis_1 = gpd.read_file(munis_1)[0:250]
    #munis_2 = gpd.read_file(munis_2)[0:10]

    #return geemap.geopandas_to_ee(munis_1), geemap.geopandas_to_ee(munis_2)
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
    print(args_dict)
    col = ee.ImageCollection(args_dict["collection"]).select(args_dict["parameter"])
    print(col.propertyNames())
    base_date = ee.Date(args_dict["start_date"])

    def month_mapper(n):
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
        .map(month_mapper)
    ).flatten()

    #print(month_ranges)
    geemap.common.ee_export_vector(month_ranges, 
                                   os.path.join(args_dict["save_dir"], f'{args_dict["parameter"]}_{args_dict["start_date"]}.csv'),
                                   selectors = ['CD_MUN', 'median', 'start_date', 'end_date'])
    return True

if __name__ == '__main__':
    #global nonsense to accomodate GEE
    assets = dict()
    main()
