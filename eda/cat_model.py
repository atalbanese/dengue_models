import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from sklearn.decomposition import PCA
import dill as pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, SplineTransformer, KernelCenterer, PolynomialFeatures, PowerTransformer, QuantileTransformer, Normalizer
from itertools import chain
import math

def create_data_dict(start_date, 
                    end_date, 
                    all_data: pl.DataFrame, 
                    target_var = 'cases_per_100k',
                    case_lookback=24,
                    case_lag=1,
                    env_lag=0, 
                    env_lookback = 12,
                    env_list = [
                    'total_precipitation_sum', 
                    'temperature_2m', 
                    ],
                    additional_features = ['month', 'pop']
                    ):
    return (
        all_data
        .select(['muni_id', 'start_date', target_var] + env_list + additional_features)
        .filter((pl.col('start_date')>=datetime.fromisoformat(start_date)-relativedelta(years=math.ceil(case_lookback/12))) & (pl.col('start_date')<datetime.fromisoformat(end_date)))
        .with_columns(
            list(
                chain.from_iterable(
                    [
                        [pl.col(target_var).shift(i).over('muni_id').alias(f'{i}_shifted_cases')] for i in range(case_lag,case_lookback+case_lag)
                    ]
                    +
                    
                    [
                        [pl.col(e_var).shift(i).over('muni_id').alias(f'{i}_shifted_{e_var}') for i in range(env_lag,env_lookback+env_lag)] 
                            for e_var in env_list
                    ]
                    )
                )
        )
        .drop_nulls()
        .select(pl.exclude(env_list))
        .rename({target_var: 'target'})
        .partition_by('muni_id', as_dict=True)
    )

def get_features_for_muni(df, cat_fn=None, check_zeros=False):
    if check_zeros:
        if df.select(pl.col('target')).to_series().sum() == 0:
            return None
    selected = df if cat_fn is None else cat_fn(df)
    target_key = 'target' if cat_fn is None else 'cat_target'
    target = selected.select(pl.col(target_key)).to_series().to_numpy().astype(float) 

    features = selected.select(pl.exclude(['muni_id', 'start_date', 'target', 'count', 'cat_target'])).to_numpy()
    dates = selected.select(pl.col('start_date')).to_series()
    muni_id = selected.select(pl.col('muni_id').first()).item()

    return {'X': features,
            'y': target,
            'dates': dates,
            'muni_id': muni_id}

def handle_zero_case(muni_id):
    return pl.DataFrame({
        'predictions': [-999.0],
        'ground_truth': [-999.0],
        'date': [datetime.fromisoformat('1900-01-01')],
        'muni_id': [muni_id],
        'cat_style': ['NA'],
        'error': ['Only zeros in training data']
    })

def write_results(df: pl.DataFrame, save_dir, save_prefix, muni_id):
    #muni_id = df.select('muni_id').head(n=1).item()
    df.write_csv(os.path.join(save_dir, f'{muni_id}_{save_prefix}.csv'))

def make_simple_binary(df:pl.DataFrame):
    return df.with_columns(pl.col('target').cut([300], labels=['0','1']).alias('cat_target'))

def make_simple_ternary(df:pl.DataFrame):
    return df.with_columns(pl.col('target').cut([100,300], labels=['0','1', '2']).alias('cat_target'))

def train_and_test_clas(train, test, cat_style=''):

    #n_components = 15
    n_components = train['X'].shape[1]
    
    reg = HistGradientBoostingClassifier(
    random_state=42,
    categorical_features = [n_components-1],
    #l2_regularization=.05,
    #categorical_features=[15],
    max_iter=1000, 
    #learning_rate=0.5,
    #max_leaf_nodes=None, 
    #min_samples_leaf=10,
    #max_bins=255,
    early_stopping=False,
    class_weight='balanced',
    #validation_fraction=0.3
    )

    ct = ColumnTransformer([
        #('min_max', MinMaxScaler(), list(range(0, n_components-1))),
        ('min_max', RobustScaler(), list(range(1, n_components))),
    ],
    remainder='passthrough')

    train_x = ct.fit_transform(train['X'])
    test_x = ct.transform(test['X'])

    #sample_weight = compute_sample_weight('balanced', train['y'])
    reg.fit(train_x, train['y'], 
            #sample_weight=sample_weight
            )
    z =reg.predict(test_x)

    return pl.DataFrame({
        'predictions': z,
        'ground_truth': test['y'],
        'date': test['dates'],
        'muni_id': [test['muni_id']]*len(z),
        'cat_style': [cat_style]*len(z),
        'error': ['NONE'] * len(z)
    })

def train_models():
    all_results = []

    train_dict = create_data_dict(TRAIN_START, TRAIN_END, ALL_DATA, env_list=[])
    test_dict = create_data_dict(TEST_START, TEST_END, ALL_DATA, env_list=[])

    for k, v in tqdm(train_dict.items()):
        train_data = get_features_for_muni(v, CAT_FN, check_zeros=True)
        if train_data is None:
            results = handle_zero_case(k)
            all_results.append(results)
            write_results(results, SAVE_DIR, SAVE_PREFIX, k)
            continue

        test_data = get_features_for_muni(test_dict[k], CAT_FN, check_zeros=False)
    #Train Classifier
    #Test classifier
    #Log results to dataframe
        try:
            results = train_and_test_clas(train_data, test_data, cat_style=CAT_STYLE)
        except BaseException as e:
            print(e)
            results = pl.DataFrame({
        'predictions': [-999.0],
        'ground_truth': [-999.0],
        'date': [datetime.fromisoformat('1900-01-01')],
        'muni_id': [k],
        'cat_style': ['NA'],
        'error': [str(e)]
        })
        all_results.append(results)
        #Save individual dataframe
        write_results(results, SAVE_DIR, SAVE_PREFIX, k)
    return all_results


if __name__ == '__main__':
    monthly_cases = pl.read_parquet('./data/cases/agged/dengue_per_month.parquet')
    monthly_params = pl.read_parquet('./data/gee_exports/all_parameters_2001-01-01_2021-01-01_months.parquet')

    monthly_params = monthly_params.with_columns((pl.col('start_date').str.to_date('%Y-%m-%d'),pl.col('end_date').str.to_date('%Y-%m-%d')))
    monthly_cases = monthly_cases.with_columns(pl.col('ID_MUNICIP').str.slice(offset=0,length=6).cast(pl.Int64)).sort('DT_NOTIFIC').rename({
    'DT_NOTIFIC': 'start_date',
    'ID_MUNICIP': 'muni_id'
    })

    ALL_DATA = monthly_cases.join(monthly_params, how='left', on=['muni_id', 'start_date']).with_columns(pl.col('end_date').alias('month').dt.month())
    TRAIN_START = '2005-01-01'
    TRAIN_END = '2018-01-01'

    TEST_START = '2018-01-01'
    TEST_END = '2020-01-01'

    EL = 12
    LC = 24
    SAVE_DIR = '/home/tony/dengue/dengue_models/results/ternary'
    SAVE_PREFIX = 'simple_ternary'
    CAT_STYLE = 'simple_ternary'
    CAT_FN = make_simple_ternary
    all_results_trained = train_models()
    all_results_trained_df = pl.concat(all_results_trained)
    all_results_trained_df.write_csv(os.path.join(SAVE_DIR, f'{SAVE_PREFIX}_all_results.csv'))