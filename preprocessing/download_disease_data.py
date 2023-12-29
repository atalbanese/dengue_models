from pysus.online_data import SINAN
import polars as pl
import geopandas as gpd
from datetime import datetime
import click
from tqdm import tqdm
import glob
import os


@click.group(chain=True)
def cli():
    pass

@cli.command('download')
@click.option('--save-dir', type=str, default='./data/cases/', help='location to save yearly disease data')
@click.option('--disease', default='DENG', help='four character code specifying target disease, codes can be looked up through pysus or SINAN')
@click.option('--start-year', default=2000, help='start year for disease data')
@click.option('--end-year', default=2021, help='end year for disease data')
def download(save_dir, disease, start_year, end_year):
    requested_years = set(range(start_year, end_year+1))
    completed_years = []
    print(f'Downloading {disease} data for {requested_years}')
    for year in tqdm(requested_years):
        try:
            year = str(year)
            if save_dir == '':
                SINAN.download(disease, year)
            else:
                SINAN.download(disease, year, data_path=save_dir)
            completed_years.append(year)
        except Exception as e:
            
            print(f'Download failed on {year}')
            print(f'Error: {e}')
            print(f'Years already completed: {completed_years}')
            print(f'Files may need cleaning up')
    consolidate()

def consolidate(loc = './data/cases/*BR*', save_dir = './data/cases/processed'):
    for f in glob.glob(loc):
        save_file = os.path.join(save_dir, f.split('/')[-1])
        if not os.path.exists(save_file):
            pl.read_parquet(os.path.join(f, '*.parquet')).write_parquet(save_file)

@cli.command('agg-data')
@click.option('--pop-data', type=str, default='./data/brazil/muni_pop.xlsx', help='location of population data spreadsheet downloaded from IBGE')
@click.option('--muni-data', type=str, default='./data/brazil/munis/munis_simple.shp', help='location of municipio shapefile downloaded from IBGE')
@click.option('--disease-data', type=str, default='./data/cases/processed/*.parquet', help='save dir specified using download command')
@click.option('--week', is_flag=True, type=bool, default=False, help='aggregate case counts to weekly totals')
@click.option('--month', is_flag=True, type=bool, default=False, help='aggregate case counts to monthly totals')
@click.option('--day', is_flag=True, type=bool, default=False, help='aggregate case counts to daily totals')
@click.option('--all', is_flag=True, type=bool, default=False, help='combine all datasets without temporal aggregation')
@click.option('--save-dir', type=str, default='./data/cases/agged/', help='save dir for aggregated files')
@click.option('--start-year', default=2001, help='year to start processing files')
@click.option('--end-year', default=2021, help='year to end processing files')
def agg_data(
    pop_data,
    muni_data,
    disease_data,
    week,
    month,
    day,
    all,
    save_dir,
    start_year,
    end_year
):
    #globals so they can be used in apply function :(
    global muni_pop
    global munis
    print('Loading pop data')
    muni_pop = load_muni_pop(loc=pop_data, start_year=start_year, end_year=end_year)
    print('Loading shape data')
    munis = load_muni_shape(loc=muni_data)
    print('Loading disease data')
    all_years = (
        load_parquets(loc=disease_data, start_year=start_year)
        .join(munis, left_on='ID_MUNICIP', right_on='CD_MUN', how='left')
        .drop_nulls()
        .with_columns(pl.col('ID_MUNICIP').str.slice(offset=0,length=6).cast(pl.Int64))
        .sort('DT_NOTIFIC')
        .rename({
                'DT_NOTIFIC': 'start_date',
                'ID_MUNICIP': 'muni_id'
        }).with_columns(pl.col('start_date').alias('month').dt.month())
        )
    
    if all:
        all_years.write_parquet(os.path.join(save_dir, 'all_dengue_cases.parquet'))

    if month:
        print('Aggregating months...')
        agg_over_time(all_years, '1mo').write_parquet(os.path.join(save_dir, 'dengue_per_month.parquet'))
    if week:
        print('Aggregating weeks...')
        agg_over_time(all_years, '1w').write_parquet(os.path.join(save_dir, 'dengue_per_week.parquet'))
    if day: 
        print('Aggregating days...')
        agg_over_time(all_years, '1d').write_parquet(os.path.join(save_dir, 'dengue_per_day.parquet'))
    

def load_muni_pop(loc = './data/brazil/muni_pop.xlsx', start_year = 2001, end_year=2021):
    return (
                pl.read_excel(
                    loc, 
                    read_csv_options={
                        'null_values':['...'], 'dtypes':{'ID_MUNICIP':pl.Utf8}
                        }
                )
                .with_columns(
                    [
                        pl.lit(None, dtype=pl.Int64).alias('2007'),
                        pl.lit(None, dtype=pl.Int64).alias('2010'),
                        pl.col('ID_MUNICIP').str.slice(0, 6)
                    ]
                )
                .select(
                    ['ID_MUNICIP'] + [str(x) for x in range(start_year, end_year)]
                )
                .melt(
                    id_vars=['ID_MUNICIP'],
                    value_vars=[str(x) for x in range(start_year, end_year)],
                    variable_name='year',
                    value_name='pop')
                .groupby('ID_MUNICIP')
                .apply(
                    lambda group_df: group_df.with_columns(
                        pl.col('pop')
                        .interpolate())
                        .with_columns(pl.col('pop')
                        .backward_fill()
                        )
                )
            )

def load_muni_shape(loc = './data/brazil/munis/munis_simple.shp'):
    munis = gpd.read_file(loc)
    munis = munis.to_crs('EPSG:5880')
    #munis['wkt'] = munis.geometry.to_wkt()

    #munis.CD_MUN = (munis.CD_MUN - (munis.CD_MUN%10))//10
    munis['x_centroid'] = munis.centroid.geometry.x
    munis['y_centroid'] = munis.centroid.geometry.y
    munis = munis.drop(columns=['geometry'])

    munis = pl.from_pandas(munis)
    return munis.with_columns(pl.col('CD_MUN').str.slice(0, 6))

def load_parquets(start_year = 2001, loc = './data/cases/processed/*.parquet'):
    return pl.read_parquet(
        loc,
        columns=[
            'DT_NOTIFIC', 
            'SEM_NOT', 
            'NU_ANO', 
            'SG_UF_NOT', 
            'ID_MUNICIP', 
            'ID_REGIONA', 
            'ID_UNIDADE', 
            'DT_SIN_PRI', 
            'SEM_PRI', 
            #'RESUL_SORO',
            #'SOROTIPO'
            ]
        ).with_columns([
            pl.col('DT_NOTIFIC').str.strptime(pl.Date, format="%Y%m%d", strict=False),
            pl.col('ID_MUNICIP').str.slice(0, 6),
            pl.col('NU_ANO').alias('year')
            ]
        ).filter(
            pl.col('DT_NOTIFIC') >= datetime(start_year,1,1)
        ).join(
            muni_pop, on=['ID_MUNICIP', 'year'], how='left'
        ).drop_nulls()

def add_blanks(group_df:pl.DataFrame):

    missing_ids = set(muni_pop.unique('muni_id').select('muni_id').to_series()) - set(group_df.unique('muni_id').select('muni_id').to_series())
    cur_date = group_df.select('start_date').to_series()[0]
    cur_year = group_df.select('year').to_series()[0]

    to_append = {
        'muni_id': list(missing_ids),
        'start_date': [cur_date] * len(missing_ids),
        'year': [cur_year] * len(missing_ids),
        'cases_per_100k': [0.0] * len(missing_ids),
        'count': [0] * len(missing_ids)
        }
    
    return (
        group_df.extend(
            pl.DataFrame(to_append,
                    schema_overrides = {
                        'count': pl.UInt32,
                    }
                )
                .join(muni_pop, on=['muni_id', 'year'], how='left')
                .join(munis, left_on='muni_id', right_on='CD_MUN', how='left')
                .select(group_df.columns)
            )
        )

def agg_over_time(df: pl.DataFrame, period):
    return (df
            .sort('start_date')
            .groupby_dynamic('start_date' , every=period, by='muni_id').agg([
                pl.count(),
                pl.first('x_centroid'),
                pl.first('y_centroid'),
                pl.first('NM_MUN'),
                pl.first('pop'),
                pl.first('year')
            ])
            .with_columns(
                (pl.col('count')/(pl.col('pop')/100000)).alias('cases_per_100k'))
            .groupby('start_date')
            .apply(
                add_blanks
            )
    )

if __name__ == '__main__':
    cli()