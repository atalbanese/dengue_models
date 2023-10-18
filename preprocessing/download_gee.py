import ee
import re
import time
import gdown
from collections import deque
import tomllib
from gee_classes import GEEDownloader, GEERequestor, GEESDMRequestor
import click
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import polars as pl
import glob


@click.command
@click.option('--config-file', type=str, default='preprocessing/gee_config.toml')
@click.option('--clean', is_flag=True, type=bool, default=False)
@click.option('--auth', is_flag=True, type=bool, default=False)
@click.option('--merge-only', is_flag=True, type=bool, default=False)
@click.option('--sdm', is_flag=True, type=bool, default=False)
@click.option('--dynamic', is_flag=True, type=bool, default=False)
def main(config_file, clean, auth, merge_only, sdm, dynamic): 
    if auth:
        ee.Authenticate(auth_mode='notebook')
    ee.Initialize()
    with open(config_file, 'rb') as f:
        config = tomllib.load(f)
    # downloader = GEEDownloader(config, merge_only=merge_only)
    # if not merge_only:
        
    #     dynamic_req = GEERequestor(config, downloader=downloader)
    #     sdm_req = GEESDMRequestor(config, downloader=downloader)
    #     if sdm:
    #         sdm_req = GEESDMRequestor(config, downloader=downloader)
    #         sdm_req.create_exports()
    #     if dynamic:
    #         dynamic_req = GEERequestor(config, downloader=downloader)
    #         dynamic_req.create_exports()
    #     downloader.run_exports()
    #     downloader.download_folder()

    # downloader.merge_all()
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
    if len(all_parquets)>1:
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
    