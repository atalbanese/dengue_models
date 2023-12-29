import ee
import re
import tomllib
from gee_classes import GEEDownloader, GEERequestor, GEESDMRequestor
import click
import os
import polars as pl
import glob



@click.command
@click.option('--config-file', type=str, default='preprocessing/gee_config.toml', help='Location of config file')
@click.option('--clean', is_flag=True, type=bool, default=False, help='Clean up individual .csv files after merging')
@click.option('--auth', is_flag=True, type=bool, default=False, help='Run GEE authorization. Do this if you are getting auth errors')
@click.option('--merge-only', is_flag=True, type=bool, default=False, help='Do not download anything, just attempt to merge existing files. Use this if you have manually downloaded files or in the event of a crash during downloading')
@click.option('--sdm', is_flag=True, type=bool, default=False, help='Train and download SDM output as specified in the config file')
@click.option('--dynamic', is_flag=True, type=bool, default=False, help='Train and download dynamic variables as specified in the config file')
def main(config_file, clean, auth, merge_only, sdm, dynamic): 
    #TODO: test combined reducer to get mean, median, minmax var, std, all at same time
    if auth:
        ee.Authenticate(auth_mode='notebook')
    ee.Initialize()
    with open(config_file, 'rb') as f:
        config = tomllib.load(f)
    downloader = GEEDownloader(config, merge_only=merge_only)
    if not merge_only:
        
        dynamic_req = GEERequestor(config, downloader=downloader)
        sdm_req = GEESDMRequestor(config, downloader=downloader)
        if sdm:
            sdm_req = GEESDMRequestor(config, downloader=downloader)
            sdm_req.create_exports()
        if dynamic:
            dynamic_req = GEERequestor(config, downloader=downloader)
            dynamic_req.create_exports()
        downloader.run_exports()
        downloader.download_folder()

    downloader.merge_all()
    parquets = join_all(config)
    if clean:
        cleanup(config, parquets)

def join_all(config):
    #thank you chatgpt for this hilarious regex
    pattern = r'(?<=\/)(\w+)(?=_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.parquet)'
    all_parquets = glob.glob(os.path.join(config['base']['save_dir'], '*.parquet'))
    check_base_exists = glob.glob(os.path.join(config['base']['save_dir'], f'all_parameters_{config["base"]["start_date"]}_{config["base"]["end_date"]}_{config["base"]["agg_unit"]}.parquet'))
    if len(check_base_exists)>0:
        all_parquets = list(set(all_parquets)-set(check_base_exists))
        to_read = check_base_exists[0]
        start_index=0
    else:
        to_read = all_parquets[0]
        start_index=1
    base_df = pl.read_parquet(to_read)
    if len(all_parquets)>0:
        for f in all_parquets[start_index:]:
            suffix = re.search(pattern, f)[0]
            base_df = base_df.join(pl.read_parquet(f), on=['muni_id', 'end_date'], suffix=f'_{suffix}')
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






if __name__ == '__main__':
    main()
    