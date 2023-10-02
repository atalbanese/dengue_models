import ee
#Notebook mode needed to run remotely
#ee.Authenticate(auth_mode='notebook')
ee.Initialize()
import geemap
import tomllib
import click


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
    
    
def download_all(config):
    for dataset in config['datasets']:
        #Need to split up time otherwise we reach GEE limits
        export_over_time(config['base']['start_month_1'], config['base']['num_months_1'], dataset['collection'], dataset['parameter'])
        export_over_time(config['base']['start_month_2'], config['base']['num_months_2'], dataset['collection'], dataset['parameter'])

#Helper fns
#We have municipios split into two chunks since there are over 5000 munis which is over the GEE feature collection limit
def load_munis(munis_1, munis_2):
    return geemap.geojson_to_ee(munis_1), geemap.geojson_to_ee(munis_2)

def clip_to_munis(img):
    return(img.clip(assets['bbox']))

#Population weighted aggregation
def agg_to_munis(img):
    img_stats_1 =  img.reduceRegions({
        'collection': assets['munis_simple_1'],
        'reducer':  assets['reducer'].splitWeights(),
        'scale': assets['scale'],  # meters
        'crs': assets['crs'],
    })

    img_stats_2 =  img.reduceRegions({
        'collection': assets['munis_simple_2'],
        'reducer': assets['reducer'].splitWeights(),
        'scale': assets['scale'],
        'crs': assets['crs'],
    })

    return img_stats_1.merge(img_stats_2)


def export_over_time(start_mo, num_months, collection, parameter):
    col = ee.ImageCollection(collection).select(parameter)
    base_date = ee.Date(start_mo)

    def month_mapper(n):
        return agg_to_munis(
            col.filterDate(
                ee.DateRange(
                    base_date.advance(n, 'month'), 
                    base_date.advance(ee.Number(n).add(1), 'month')
                )
            ).map(clip_to_munis)
            .median()
            .addBands(
                assets['population'].filter(ee.Filter.eq('year', base_date.advance(n, 'month').get('year'))).first().unitScale(0, 21171)
            )
        ).map(lambda f: f.set({
            'start_date': base_date.advance(n, 'month').format('YYYY-MM-dd'),
            'end_date': base_date.advance(ee.Number(n).add(1), 'month').format('YYYY-MM-dd')
            })
        )

    month_ranges = ee.FeatureCollection(
        ee.List.sequence(0, num_months)
        .map(month_mapper)
    ).flatten()

    geemap.common.ee_export_vector(month_ranges, 
                                   'test.csv',
                                   selectors = ['CD_MUN', 'median', 'start_date', 'end_date'])

if __name__ == '__main__':
    #global nonsense to accomodate GEE
    assets = dict()
    main()
