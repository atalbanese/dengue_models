import polars as pl
import click


@click.command
@click.option('--rs-data', type=str, default='./data/gee_exports/all_parameters_2001-01-01_2021-01-01_months.parquet')
@click.option('--dengue-data', type=str, default='./data/cases/agged/dengue_per_month.parquet')
@click.option('--save-loc', type=str, default='./data/all_dengue_data.parquet')
@click.option('--validate', is_flag=True, type=bool, default=True)
def main(rs_data, dengue_data, save_loc, validate):
    rs = pl.read_parquet(rs_data)
    dengue = pl.read_parquet(dengue_data)
    #TODO: All these can be moved to earlier in preprocessing so we dont have to check
    if 'ID_MUNICIP' in dengue.columns:
        dengue = dengue.rename({'ID_MUNICIP': 'muni_id'})
    if 'DT_NOTIFIC' in dengue.columns:
        dengue = dengue.rename({'DT_NOTIFIC': 'start_date'})
    if 'NM_MUN' in dengue.columns:
        dengue = dengue.rename({'NM_MUN': 'muni_name'})

    if rs.schema['muni_id'] != dengue.schema['muni_id']:
        rs = rs.with_columns(pl.col('muni_id').cast(dengue.schema['muni_id']))

    if rs.schema['start_date'] != dengue.schema['start_date']:
        rs = rs.with_columns(pl.col('start_date').str.to_date('%Y-%m-%d'))


    rs_rows = rs.shape[0]
    dengue_rows = dengue.shape[0]

    if validate:
        if rs_rows != dengue_rows:
            click.confirm(f'Datasets do not have the same number of rows\n\
Remote Sensing rows: {rs_rows}\n\
Disease Data Rows{dengue_rows}\n\
Continue?', abort=True)
    
    new_df = rs.join(dengue, on=['muni_id', 'start_date'])

    if validate:
        if (new_df.shape[0] != rs_rows) or (new_df.shape[0] != dengue_rows):
            click.confirm(f'Merged dataset does not have the same number of rows\n\
Remote Sensing rows: {rs_rows}\n\
Disease Data Rows: {dengue_rows}\n\
Merged Rows: {new_df.shape[0]}\n\
Continue?', abort=True)

    new_df.write_parquet(save_loc)

if __name__ == '__main__':
    main()