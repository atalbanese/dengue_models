# Data Acquisition Scripts
    Data acquisition is largely done through two scripts, one which downloads google earth engine data and one which downloads disease data from SINAN. Both are run through the command line.

## download_gee.py

    Allows user to aggregate environmental variables from Google Earth Engine over space and time and export those values to google drive.
    This script is (mostly) controlled through a TOML configuration file, gee_config.toml. Google earth engine request parameters are all specified in the configuration files. 

    You will need to authorize GEE through the commandline and also have access to the municipios shapefile uploaded to Tony Albanese's google earth engine account. You can get that access by visiting THIS LINK

    To download the environmental variables specified in your config file
    python download_gee.py --dynamic

    To train and download the results of the species distribution model specified in your config file
    python download_gee.py --sdm

    Other options are described by running python download_gee.py --help

    Categorical variables can be downloaded using the script at THIS LINK

## download_disease_data.py

    Downloads disease data from SINAN. Works in two parts, download and agg-data. More info can be found using python download_disease_data.py --help

    python download_disease_data.py download
        After download you can specify save directory, which disease you are targeting, and the yearly range

    python download_disease_data.py agg-data
        Downloaded data is saved into individual years. Agg-data allows you to combine those years, and aggregate cases to daily, monthly, or weekly case counts. You will need access to population data and a municipio shapefile to perform this aggregation. Those should be included with this package, they have been acquired from IBGE. 

