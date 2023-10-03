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

@click.command
@click.option('--config-file', type=str, default='preprocessing/gee_config.toml')
def main(config_file): 
    with open(config_file, 'rb') as f:
        config = tomllib.load(f)

    populate_assets(config)
    download_all(config)

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
            'start_date': start_date,
            'num_units': config['base']['agg_chunks'], #if config['base']['agg_unit']+start_date < end_date else end_date-start_date
            'collection': dataset['collection'],
            'parameter': dataset['parameter'],
            'time_unit': config['base']['agg_unit']
        })
        start_date = start_date + relativedelta(**{config['base']['agg_unit']:config['base']['agg_chunks']})
        
    return requests
    
#Downloading locally has lower computational limits than exporting to google drive so we are gonna run a lot of parallel chunks here
#If I cant get this to work consistently will switch to drive export then download from there
def download_all(config):
    for dataset in config['datasets']:
        #Need to split up time otherwise we reach GEE limits
        requests = generate_requests(config, dataset)
        pass
        #export_over_time(config['base']['start_month_1'], config['base']['num_months_1'], dataset['collection'], dataset['parameter'])
        #export_over_time(config['base']['start_month_2'], config['base']['num_months_2'], dataset['collection'], dataset['parameter'])

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


def export_over_time(start_date, num_units, collection, parameter, time_unit):
    #relativedelta uses plural units, GEE uses singular
    time_unit = time_unit[:-1]
    col = ee.ImageCollection(collection).select(parameter)
    base_date = ee.Date(start_date)

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
        ee.List.sequence(0, num_units)
        .map(month_mapper)
    ).flatten()

    #print(month_ranges)
    geemap.common.ee_export_vector(month_ranges, 
                                   'test.csv',
                                   selectors = ['CD_MUN', 'median', 'start_date', 'end_date'])

if __name__ == '__main__':
    #global nonsense to accomodate GEE
    assets = dict()
    main()
