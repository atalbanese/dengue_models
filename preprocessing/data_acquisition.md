# Data Acquisition Scripts
    Data acquisition is largely done through two scripts, one which downloads google earth engine data and one which downloads disease data from SINAN. Both are run through the command line.

## download_gee.py

    Allows user to aggregate environmental variables from Google Earth Engine over space and time and export those values to google drive.
    This script is (mostly) controlled through a TOML configuration file, gee_config.toml. Google earth engine request parameters are all specified in the configuration files. 

    You will need to authorize GEE through the commandline and also have access to the municipios shapefile uploaded to Tony Albanese's (tony.t.albanese@gmail.com) google earth engine account. You can get that access by visiting https://code.earthengine.google.com/?asset=projects/ee-dengue-proof-of-concept/assets/munis_simple . If that doesnt work, you can upload the shapefile provided with this document to your own account.

    To download the environmental variables specified in your config file
    python download_gee.py --dynamic

    To train and download the results of the species distribution model specified in your config file
    python download_gee.py --sdm

    Other options are described by running python download_gee.py --help

    Categorical variables can be downloaded using the script at https://code.earthengine.google.com/7fc40ac39549466a0bbbb71f2df4cdf9?asset=projects%2Fee-dengue-proof-of-concept%2Fassets%2Fmunis_simple

    Additional assets may be required, they can be found here: 

    https://code.earthengine.google.com/?asset=projects/ee-dengue-proof-of-concept/assets/aegypti_gbif_date_filtered

    https://code.earthengine.google.com/?asset=projects/ee-dengue-proof-of-concept/assets/albo_gbif_date_filtered

    https://code.earthengine.google.com/?asset=projects/ee-dengue-proof-of-concept/assets/munis_simple_1

    https://code.earthengine.google.com/?asset=projects/ee-dengue-proof-of-concept/assets/munis_simple_2

## download_disease_data.py

    Downloads disease data from SINAN. Works in two parts, download and agg-data. More info can be found using python download_disease_data.py --help

    python download_disease_data.py download
        After download you can specify save directory, which disease you are targeting, and the yearly range

    python download_disease_data.py agg-data
        Downloaded data is saved into individual years. Agg-data allows you to combine those years, and aggregate cases to daily, monthly, or weekly case counts. You will need access to population data and a municipio shapefile to perform this aggregation. Those should be included with this package, they have been acquired from IBGE. 

