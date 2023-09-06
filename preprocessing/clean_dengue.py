import polars as pl
import geopandas as gpd
from datetime import datetime
import glob

MUNI_POP = (
                    pl.read_excel('muni_pop.xlsx', read_csv_options={'null_values':['...'], 'dtypes':{'ID_MUNICIP':pl.Utf8}})
                    .with_columns(
                                    [
                                        pl.lit(None, dtype=pl.Int64).alias('2007'),
                                        pl.lit(None, dtype=pl.Int64).alias('2010'),
                                        pl.col('ID_MUNICIP').str.slice(0, 6)
                                    ]
                                )
                    .select(
                        ['ID_MUNICIP'] + [str(x) for x in range(2001, 2022)]
                    )
                    .melt(
                        id_vars=['ID_MUNICIP'],
                        value_vars=[str(x) for x in range(2001, 2022)],
                        variable_name='year',
                        value_name='pop')
                    .groupby('ID_MUNICIP')
                    .apply(
                        lambda group_df: group_df.with_columns(pl.col('pop').interpolate()).with_columns(pl.col('pop').backward_fill())
                    )
                )

print('loading municipio spatial information')
munis = gpd.read_file('/home/tony/municipio_data/BR_Municipios_2021.shp')
munis = munis.to_crs('EPSG:5880')
#munis['wkt'] = munis.geometry.to_wkt()

#munis.CD_MUN = (munis.CD_MUN - (munis.CD_MUN%10))//10
munis['x_centroid'] = munis.centroid.geometry.x
munis['y_centroid'] = munis.centroid.geometry.y
munis = munis.drop(columns=['geometry'])

munis = pl.from_pandas(munis)
munis = munis.with_columns(pl.col('CD_MUN').str.slice(0, 6))

def main():

    #Loads population data
    #2007, 2010 are missing
    #We don't have 2000 at all
    #Some munis have missing data at beginning of time sequence (ie from 2001 to 2006)
    #For 2007, 2010 we interpolate
    #For missing data at beginning of time sequence we just fill with the next closest value
    print('loading population data')
    muni_pop = MUNI_POP

    print('loading case data')

    #test = pl.read_parquet('/home/tony/dengue_data/processed_files/DENGBR21.parquet')

    # all_files = glob.glob('/home/tony/dengue_data/processed_files/*.parquet')

    # for p in all_files:
    #     cur_columns = pl.read_parquet(p).columns
    #     print(p)
    #     print(f'Has results: {"RESUL_SORO" in cur_columns}')
    #     print(f'Has serotype: {"SOROTIPO" in cur_columns}' )

    all_years = pl.read_parquet(
        '/home/tony/dengue_data/processed_files/with_sorotipo/*.parquet',
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
            #'CS_SEXO', 
            #'CS_RACA',
            'RESUL_SORO',
            'SOROTIPO'
            ]
        ).with_columns([
            pl.col('DT_NOTIFIC').str.strptime(pl.Date, format="%Y%m%d", strict=False),
            pl.col('ID_MUNICIP').str.slice(0, 6),
            pl.col('NU_ANO').alias('year')
            ]
        ).filter(
            pl.col('DT_NOTIFIC') >= datetime(2001,1,1)
        ).join(
            muni_pop, on=['ID_MUNICIP', 'year'], how='left'
        ).drop_nulls()



    print('joining based on municipio_id')


    all_years_spatial = all_years.join(munis, left_on='ID_MUNICIP', right_on='CD_MUN', how='left').drop_nulls()
    #all_years_spatial.write_parquet('/home/tony/dengue/all_cases_with_municipios.parquet')
    print('aggregating cases over time')
    by_month = (all_years_spatial
            .sort('DT_NOTIFIC')
            .groupby_dynamic('DT_NOTIFIC' , every='1mo', by='ID_MUNICIP').agg([
                pl.count(),
                pl.first('x_centroid'),
                pl.first('y_centroid'),
                pl.first('NM_MUN'),
                pl.first('pop'),
                pl.first('year')
            ])
            .with_columns(
                (pl.col('count')/(pl.col('pop')/100000)).alias('cases_per_100k'))
            .groupby('DT_NOTIFIC')
            .apply(
                add_blanks
            )
    )
    by_week = (all_years_spatial
            .sort('DT_NOTIFIC')
            .groupby_dynamic('DT_NOTIFIC' , every='1w', start_by='monday', by='ID_MUNICIP').agg([
                pl.count(),
                pl.first('x_centroid'),
                pl.first('y_centroid'),
                pl.first('NM_MUN'),
                pl.first('pop')
            ])
            ).with_columns(
                (pl.col('count')/(pl.col('pop')/100000)).alias('cases_per_100k')
            )
    
    by_day = (all_years_spatial
            .sort('DT_NOTIFIC')
            .groupby_dynamic('DT_NOTIFIC' , every='1d', by='ID_MUNICIP').agg([
                pl.count(),
                pl.first('x_centroid'),
                pl.first('y_centroid'),
                pl.first('NM_MUN'),
                pl.first('pop')
            ])
            ).with_columns(
                (pl.col('count')/(pl.col('pop')/100000)).alias('cases_per_100k')
            )
    
    print('saving case data')
    #by_month.write_parquet('/home/tony/dengue/dengue_per_month_no_blanks.parquet')
    # by_week.write_parquet('/home/tony/dengue/dengue_per_week.parquet')
    # by_day.write_parquet('/home/tony/dengue/dengue_per_day.parquet')

def add_blanks(group_df:pl.DataFrame):

    missing_ids = set(MUNI_POP.unique('ID_MUNICIP').select('ID_MUNICIP').to_series()) - set(group_df.unique('ID_MUNICIP').select('ID_MUNICIP').to_series())
    cur_date = group_df.select('DT_NOTIFIC').to_series()[0]
    cur_year = group_df.select('year').to_series()[0]

    to_append = {
                'ID_MUNICIP': list(missing_ids),
                 'DT_NOTIFIC': [cur_date] * len(missing_ids),
                 'year': [cur_year] * len(missing_ids),
                 'cases_per_100k': [0.0] * len(missing_ids),
                 'count': [0] * len(missing_ids)
                 }
    
    return (group_df.extend(
                        pl.DataFrame(to_append,
                              schema_overrides = {
                                  'count': pl.UInt32,
                              }
                              )
                            .join(MUNI_POP, on=['ID_MUNICIP', 'year'], how='left')
                            .join(munis, left_on='ID_MUNICIP', right_on='CD_MUN', how='left')
                            .select(group_df.columns)
                 )
                )

if __name__ == '__main__':
    main()