from itertools import chain
from datetime import datetime
import polars as pl
import geopandas as gpd
from typing import Union
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from functools import partial
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import colorsys
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


## Reshaping 
def reshape_data(
    start_date: str,
    end_date: str,
    all_data: pl.DataFrame,
    target_var: str = "cases_per_100k",
    case_lookback: int = 24,
    case_lag: int = 1,
    env_lag: int = 0,
    env_lookback: int = 12,
    env_list: list[str] = [
        'temperature_2m'
    ],
    additional_features: list[str] = [
        "month",
    ],
    specific_cases: list[int] = None,
    specific_env: list[int] = None,
) -> pl.DataFrame:
    """Function which reshapes input data into target and feature columns. Time shifts historical cases and exogenous variables into a row of features 
      for each target day between start_date and end_date. Can use different lag for cases and historical cases, i.e. we want to model having a 4 month lag
      on case data but up to date/predicted environmental variables. Also handles categorical features. 


    Args:
        start_date (str): Start date of reshaped target data. Data from prior to start date will be included if case_lookback > 0. Inclusive
        end_date (str): End date of reshaped target data. Exclusive
        all_data (pl.DataFrame): Dataframe containing case data and any exogenous variables
        target_var (str, optional): Column in input dataframe we are targeting. Change to 'count' to target raw counts instead of case rate. Defaults to "cases_per_100k".
        case_lookback (int, optional): How many months of case history we want to include. Overridden if specific_cases is set. Defaults to 24.
        case_lag (int, optional): Lag between included known case history and prediction target month. Increase to increase distance of prediction window. Ignored if specific_cases is set. Defaults to 1.
        env_lag (int, optional): Lag between environmental variables and prediction target month. Ignored if specific_env is set. Defaults to 0.
        env_lookback (int, optional): How many months of environmental history we want to include. Overridden if specific_env is set. Defaults to 12.
        env_list (list[str], optional): Names of environmental variables to include. Defaults to [ 'temperature_2m' ].
        additional_features (list[str], optional): Static variables to include. Defaults to [ "month", ].
        specific_cases (list[int], optional): List of integers indicating specific case history months to include. For example [1,12,24] to get months n-1,n-12,n-24. Overrides case_lookback and ignores case_lag. Defaults to None.
        specific_env (list[int], optional): List of integers indicating specific environmental history months to include. For example [1,12,24] to get months n-1,n-12,n-24. Overrides env_lookback and ignores env_lag. Defaults to None.

    Returns:
        pl.DataFrame: Dataframe with 'target' column, with one entry per prediction month. Additional columns consist of lagged case history, environmental history, and static features. 

    """
    

    return (
        all_data.select(
            ["muni_id", "start_date", target_var] + env_list + additional_features
        )
        .with_columns(
            list(
                chain.from_iterable(
                    [
                        [
                            pl.col(target_var)
                            .shift(i)
                            .over("muni_id")
                            .alias(f"{i}_shifted_cases")
                        ]
                        for i in (
                            range(case_lag, case_lookback + case_lag)
                            if specific_cases is None
                            else specific_cases
                        )
                    ]
                    + [
                        [
                            pl.col(e_var)
                            .shift(i)
                            .over("muni_id")
                            .alias(f"{i}_shifted_{e_var}")
                            for i in (
                                range(env_lag, env_lookback + env_lag)
                                if specific_env is None
                                else specific_env
                            )
                        ]
                        for e_var in env_list
                    ]
                )
            )
        )
        .drop_nulls()
        .filter(
            (pl.col("start_date") >= datetime.fromisoformat(start_date))
            & (pl.col("start_date") < datetime.fromisoformat(end_date))
        )
        .select(pl.exclude(env_list))
        .rename({target_var: "target"})
    )

def get_features(df: pl.DataFrame, 
                 cat_fn: callable = None, 
                 cat_vars: list[str] = None) -> dict:
    """Transforms reshaped dataframe into a dictionary with entries compatible for use with the sklearn api. Can also transform target variable and cast categorical variables to the right datatype. 

    Args:
        df (pl.DataFrame): Dataframe with target column and requested features as produced by reshape_data
        cat_fn (callable, optional): Function to transform target column. Generally used if creating a classification model, can also be used to modify regression target (ie make target rate of change instead of case rate). Defaults to None.
        cat_vars (list[str], optional): List of categorical variables used, they will be cast to categorical type if not already. Defaults to None.

    Returns:
        dict: Dictionary with feature and target data frames that are compatible with the sklearn api, as well as date and municipio information for recovery later. If provided categorical function outputs information needed to transform testing data without leakage, this is included as well.
                'X': training features, pandas dataframe
                'y : target features, numpy series
                'dates': date for each row of training/target, numpy series
                'muni_id': municipio id for each row of training/target, numpy series
                'expectations': optional additional output from cat_fn, used to transform test data with information from training data to prevent data leakage and ensure consistency between transformations.
    """
    df = df.sort('start_date')
    selected = df if cat_fn is None else cat_fn(df)
    expectations = None
    #Some cat_fns return a tuple including expectations for category breaks based on historical data. If so, we want to pass these from training to testing data.
    if isinstance(selected, tuple):
        selected, expectations = selected
    target_key = 'target' if cat_fn is None else 'cat_target'
    target = selected.select(pl.col(target_key)).to_series().to_numpy().astype(float) 
    
    features = selected.select(pl.exclude(['muni_id', 'start_date', 'target', 'count', 'cat_target'])).to_pandas()
    if cat_vars is not None:
        features[cat_vars] = features[cat_vars].astype('category')
    dates = selected.select(pl.col('start_date')).to_series()
    muni_id = selected.select(pl.col('muni_id')).to_series()

    return {'X': features,
            'y': target,
            'dates': dates,
            'muni_id': muni_id,
            'expectations': expectations}

## Categorical Target Transformation 

def make_relative_ternary(df:pl.DataFrame, expectations: pl.DataFrame = None):
    """Relative ternary categorical fn

    Args:
        df (pl.DataFrame): _description_
        expectations (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple(pl.Dataframe, pl.Dataframe): Dataframe with categorical targets column, dataframe with historical expectations used to calculate targets column
    """
    if expectations is None:
        # want to save training expectations so they can be applied to test data
        expectations = df.with_columns(pl.col('target').mean().over(['muni_id', 'month']).alias('expected')).group_by(['muni_id', 'month']).agg(pl.col('expected').first()).sort('month')

    return (
            (
                df
                .join(expectations, on=['muni_id', 'month'])
                .with_columns(
                    pl.when(pl.col('target')<=pl.col('expected')*.5)
                    .then(pl.lit('0'))
                    .when((pl.col('target')>pl.col('expected')*.5)&(pl.col('target')<=pl.col('expected')))
                    .then(pl.lit('1'))
                    .otherwise(pl.lit('2'))
                    .alias('cat_target')
                    )
                .select(pl.exclude('expected'))
            ),
            expectations
    )

def make_relative_binary(df:pl.DataFrame, expectations: pl.DataFrame = None):
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        expectations (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple(pl.Dataframe, pl.Dataframe): Dataframe with categorical targets column, dataframe with historical expectations used to calculate targets column
    """
    if expectations is None:
        #want to save training expectations so they can be applied to test data
        expectations = df.with_columns(pl.col('target').mean().over(['muni_id','month']).alias('expected')).group_by(['muni_id', 'month']).agg(pl.col('expected').first()).sort('month')

    return (
            (
                df
                .join(expectations, on=['muni_id', 'month'])
                .with_columns(
                    pl.when(pl.col('target')>pl.col('expected'))
                    .then(pl.lit('1'))
                    .otherwise(pl.lit('0'))
                    .alias('cat_target')
                    )
                .select(pl.exclude('expected'))
            ),
            expectations
    )

def make_simple_binary(df:pl.DataFrame, expectations=None):
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        expectations (_type_, optional): Unused, included for compatibility with relative functions. Defaults to None.

    Returns:
        pl.Dataframe: Dataframe with categorical targets column
    """
    return df.with_columns(pl.col('target').cut([300], labels=['0','1']).alias('cat_target'))

def make_simple_ternary(df:pl.DataFrame, expectations=None):
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        expectations (_type_, optional):  Unused, included for compatibility with relative functions. Defaults to None.

    Returns:
        pl.Dataframe: Dataframe with categorical targets column
    """
    return df.with_columns(pl.col('target').cut([100,300], labels=['0','1', '2']).alias('cat_target'))

def get_cat_dict():
    return {
    'relative_ternary': make_relative_ternary,
    'relative_binary': make_relative_binary,
    'simple_binary': make_simple_binary,
    'simple_ternary': make_simple_ternary
}

## Model Training

def run_model(
        data : pl.DataFrame,
        train_start : str,
        train_end : str,
        test_end : str,
        clf_model: Union[RidgeClassifier, HistGradientBoostingClassifier],
        env_list: list[str] = [],
        additional_features: list[str] = [],
        cat_vars: list[str] = ['month'],
        cat_fn: callable = make_simple_binary,
        cat_style: str = 'simple_binary',
        case_lookback: int = 24,
        case_lag: int = 1,
        specific_cases: list[int] = None,
        env_lookback: int = 12,
        env_lag: int = 0,
        specific_env: list[int] = None
) -> dict:
    """Trains and tests a model, first reshaping provided data based on specified arguments. Returns model predictions on the test set as well as information on input parameters.

    Args:
        data (pl.DataFrame): Dataframe containing dengue data
        train_start (str): Start date for training data
        train_end (str): End date for training data
        test_end (str): End Date for testing data
        clf_model (Union[RidgeClassifier, HistGradientBoostingClassifier]): A classification model compatible with the sklearn api
        env_list (list[str], optional): List of environmental variables to include, if any. Defaults to [].
        additional_features (list[str], optional): List of static features to include, if any. Defaults to [].
        cat_vars (list[str], optional): List of categorical variables to include. Defaults to ['month'].
        cat_fn (callable, optional): Function to transform continuous target (case rate) to categorical target. Defaults to make_simple_binary.
        cat_style (str, optional): Descriptive name of cat_fn. Defaults to 'simple_binary'.
        case_lookback (int, optional): How many months of prior case data to include. Overridden by setting specific_cases. Defaults to 24.
        case_lag (int, optional): Gap between prediction month and prior cases. Increase to increase prediction window. Defaults to 1.
        specific_cases (list[int], optional): List of prior case months to include, overrides case_lookback. For ex, [1,12,24] includes cases from 1 month ago, 12 months ago, and 24 months ago but not the months in-between. Defaults to None.
        env_lookback (int, optional): Prior months of environmental features to include. Defaults to 12.
        env_lag (int, optional): Lag between prediction month and known environmental variables. Defaults to 0.
        specific_env (list[int], optional): Similar to specific_cases, use if you only want to include environmental data from certain months. Defaults to None.

    Returns:
        dict: Dict containing dataframe of results on the test set and the trained model
    """

    additional_features = additional_features + cat_vars

    train_shaped = reshape_data(
        train_start, 
        train_end, 
        data, 
        case_lookback=case_lookback, 
        case_lag=case_lag, 
        specific_cases=specific_cases,
        env_lookback=env_lookback,
        env_lag=env_lag,
        specific_env=specific_env,
        env_list=env_list,
        additional_features=additional_features
    )

    test_shaped = reshape_data(
        train_end, 
        test_end, 
        data, 
        case_lookback=case_lookback, 
        case_lag=case_lag, 
        specific_cases=specific_cases,
        env_lookback=env_lookback,
        env_lag=env_lag,
        specific_env=specific_env,
        env_list=env_list,
        additional_features=additional_features
    )

    train_data = get_features(train_shaped, cat_fn, cat_vars)
    expectations = train_data['expectations']
    new_cat_fn = partial(cat_fn, expectations = expectations)
    test_data = get_features(test_shaped, new_cat_fn, cat_vars)

    z, p_hat, model = train_and_test_model(train_data, test_data, clf_model)

    return {
            'results': pl.DataFrame({
                'predictions': z,
                'ground_truth': test_data['y'],
                'date': test_data['dates'],
                'muni_id': test_data['muni_id'],
                'cat_style': [cat_style]*len(z),
                'probabilities': p_hat,
                'env_vars': [env_list]*len(z),
                'additional_vars': [additional_features]*len(z),
                'case_lag': [case_lag]*len(z),
                'env_lag': [env_lag]*len(z),
                'specific_cases': [specific_cases]*len(z) if specific_cases is not None else [[specific_cases]]*len(z),
                'specific_env': [specific_env]*len(z) if specific_env is not None else [[specific_env]]*len(z)
            }),
            'model': model
    }

def train_and_test_model(
        train_data, 
        test_data, 
        clf_model, 
    ) -> dict:

    ct = make_column_transformer(
        (OrdinalEncoder(), make_column_selector(dtype_include='category')),
        (RobustScaler(), make_column_selector(dtype_exclude='category')),
    
    remainder='passthrough',
    verbose_feature_names_out=False)

    model = make_pipeline(ct, clf_model).set_output(transform='pandas')

    model.fit(train_data['X'], train_data['y'])

    z = model.predict(test_data['X'])
    if hasattr(model, 'predict_proba'):
        p_hat = model.predict_proba(test_data['X'])
    else:
        p_hat = [[None]] *len(z)

    return z, p_hat, model


## Results Analysis
def classify_results(predictions: pl.DataFrame):
    import matplotlib.pyplot as plt
    f1_report = classification_report(predictions.select('ground_truth').to_series(),predictions.select('predictions').to_series())
    print(f1_report)
    cm = ConfusionMatrixDisplay.from_predictions(predictions.select('ground_truth').to_series(),predictions.select('predictions').to_series())
    #cm.plot()
   # plt.show()

    return f1_report, cm

## Plotting Results
### Helpers
def rgb_to_hsl(rgb):
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, l, s

# Function to convert HSL to RGB
def hsl_to_rgb(hls):
    h, l, s = hls
    r, g, b = colorsys.hls_to_rgb(h+.02, min(1.0, l + 0.02), min(1.0, s - 0.25))
    return r, g, b

def plot_filled_ternary_gradient(axis, cmap='viridis'):
    values = np.array([1.0, 0.5, 0.0])  #End values for triangle points
    # Define the vertices of the ternary plot
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    axins = inset_axes(axis, width='25%', height='25%', loc='lower left', borderpad=3)
    axins.set_aspect('equal')

    # Create a ternary grid
    triang = tri.Triangulation(corners[:, 0], corners[:, 1])

    # Plot the filled ternary diagram with a continuous color gradient
    axins.tripcolor(triang,
                         values, 
                         cmap=cmap, shading='gouraud')
    
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)

    axins.text(-.15,-.16, 'Low', fontsize=16, c='black')

    axins.text(.75,-.16, 'Medium', fontsize=16, c='black')

    axins.text(.35, .95, 'High', fontsize=16, c='black')

def plot_ternary(
        predictions: pl.DataFrame, 
        munis: gpd.GeoDataFrame, 
        month: str, 
        title:str,
        accuracy:bool=False):
    import matplotlib.pyplot as plt
    from palettable.lightbartlein.diverging import BlueGrey_2 as imp_palette

    plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
    plt.rcParams['axes.grid'] = False
    plt.rc('legend',fontsize=10)
    custom_cmap = LinearSegmentedColormap.from_list('custom', 
                                                    [hsl_to_rgb(rgb_to_hsl((1, 0, 0))), 
                                                     hsl_to_rgb(rgb_to_hsl((0, 1, 0))), 
                                                     hsl_to_rgb(rgb_to_hsl((0, 0, 1)))])

    current = (predictions
           .with_columns(
               pl.col('probabilities').list.reverse()
           )
           .with_columns(
               
               pl.col('probabilities').list.to_struct().struct.rename_fields(['high_p','medium_p', 'low_p' ]).alias('x'),
               pl.when(pl.col('predictions')==pl.col('ground_truth')).then(pl.lit('Correct Prediction')).otherwise(pl.lit('Incorrect Prediction')).alias('correct'),
               pl.when(pl.col('ground_truth')==0).then(1.0).otherwise(0).alias('low'),
               pl.when(pl.col('ground_truth')==1).then(1.0).otherwise(0).alias('medium'),
               pl.when(pl.col('ground_truth')==2).then(1.0).otherwise(0).alias('high'),
               )
            .with_columns(
                pl.concat_list(['high', 'medium', 'low']).alias('truth_color')
            )
           .unnest('x')
           .filter(pl.col('date')== datetime.fromisoformat(month))
           )

    results = munis.merge(current.to_pandas(), left_on='CD_MUN', right_on='muni_id', how='right')

    results['pastel_colors'] = results['probabilities'].apply(lambda rgb: hsl_to_rgb(rgb_to_hsl(rgb)))

    results['truth_color'] = results['truth_color'].apply(lambda rgb: hsl_to_rgb(rgb_to_hsl(rgb)))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16.33,7.5), constrained_layout=True)

    results.plot(color=results['pastel_colors'], ax=ax[0], edgecolor='white', linewidth=0.05)
    ax[0].set_title(f'Predicted Probabilities - {month}', fontsize=24, )

    if not accuracy:
        results.plot(color=results['truth_color'], ax=ax[1], edgecolor='white', linewidth=0.05)
        ax[1].set_title(f'Ground Truth - {month}', fontsize=24)

    else:
        results.plot(column='correct', 
                  legend=True, 
                  legend_kwds={"fontsize": 16, 'loc': 'lower right', 'bbox_to_anchor': (.4, .03)},
                  categorical=True, 
                  cmap=imp_palette.mpl_colormap, ax=ax[1], edgecolor='white', linewidth=0.05)
        ax[1].set_title(f'Prediction Accuracy - {month}', fontsize=24)

    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    fig.suptitle(title, fontsize=30, fontweight='demibold')

    plot_filled_ternary_gradient(ax[0], cmap=custom_cmap)

    return fig, ax


def plot_binary(
        predictions: pl.DataFrame, 
        munis: gpd.GeoDataFrame, 
        month: str, 
        title:str,
        accuracy:bool=False):
    import matplotlib.pyplot as plt
    from palettable.lightbartlein.diverging import BlueGrey_2 as imp_palette

    plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
    plt.rcParams['axes.grid'] = False
    plt.rc('legend',fontsize=10)

    current = (predictions
           .with_columns(
               pl.col('probabilities').list.to_struct().struct.rename_fields(['low_p', 'high_p']).alias('x'),
               pl.when(pl.col('predictions')==pl.col('ground_truth')).then(pl.lit('Correct Prediction')).otherwise(pl.lit('Incorrect Prediction')).alias('correct'),
               pl.when(
                   pl.col('ground_truth')==1.0
               ).then(pl.lit('High Case Rate'))
               .otherwise(pl.lit('Low Case Rate')).alias('truth_label')
               )
           .unnest('x')
           .filter(pl.col('date')== datetime.fromisoformat(month))
           )
    
    results = munis.merge(current.to_pandas(), left_on='CD_MUN', right_on='muni_id', how='right')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16.33,8.25), constrained_layout=True)

    
    results.plot(ax= ax[0], column=results['high_p'],  cmap='coolwarm',edgecolor='white', linewidth=0.05, legend=True, 
                  legend_kwds={'orientation': 'horizontal', 'shrink': 0.3, 'anchor': (.9,-25),'label': 'High Case Rate Probability'})
    ax[0].set_title(f'Predicted Probabilities - {month}', fontsize=20, )

    if not accuracy:
        results.plot(ax= ax[1],column=results['truth_label'],  cmap='coolwarm_r', edgecolor='white', linewidth=0.05, legend=True, categorical=True,  legend_kwds={"fontsize": 16, 'loc': 'lower right', 'bbox_to_anchor': (.95, .12)})
        ax[1].set_title(f'Ground Truth - {month}', fontsize=20)
    else:
        results.plot(ax= ax[1],column=results['correct'],  cmap=imp_palette.mpl_colormap, edgecolor='white', linewidth=0.05, legend=True, categorical=True,  legend_kwds={"fontsize": 16, 'loc': 'lower right', 'bbox_to_anchor': (1.0, .12)})
        ax[1].set_title(f'Prediction Accuracy - {month}', fontsize=20)
    
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    fig.suptitle(title, fontsize=30, fontweight='demibold')

    return fig, ax

