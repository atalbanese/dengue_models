from pysus.online_data import SINAN, FTP_Inspect, parquets_to_dataframe
import click
from tqdm import tqdm

@click.command()
@click.option('--save-dir', type=str, default='')
@click.option('--disease', default='Dengue')
@click.option('--start-year', default=2000)
@click.option('--end-year', default=2021)
def main(save_dir, disease, start_year, end_year):
    all_years = {int(x) for x in SINAN.get_available_years(disease)}
    requested_years = set(range(start_year, end_year+1))
    missing_years = requested_years - all_years
    if len(missing_years) != 0:
        print(f'Requested years outside of available range for {disease}')
        print(f'Available years: {all_years}')
        print(f'Unvailable years: {missing_years}')
        print(f'New request would be: {requested_years - missing_years}')
        if not click.confirm('Continue with these years?'):
            click.echo('Quitting')
            quit()
        requested_years = requested_years - missing_years
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


if __name__ == '__main__':
    main()