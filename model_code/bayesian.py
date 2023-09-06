import pymc as pm
import arviz as az
import polars as pl
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pymc.sampling.jax as pmjax
import pytensor.tensor as pt

def load_data(drops=None):
    if drops is None:
        return pl.read_parquet('/home/tony/dengue/all_dengue_data.parquet').drop_nulls().with_columns(rank = pl.col('muni_id').rank('dense'), muni_id = pl.col('muni_id').cast(pl.Int32))
    else:
        return pl.read_parquet('/home/tony/dengue/all_dengue_data.parquet').select(pl.exclude(drops)).drop_nulls().with_columns(rank = pl.col('muni_id').rank('dense'), muni_id = pl.col('muni_id').cast(pl.Int32))
    
def one_hot_months(d_set: pl.DataFrame):
    return (d_set
            .with_columns(
                pl.col('case_start_month').dt.month()
                .alias('month')
                )
            .to_dummies('month')
            .select(pl.exclude('month'))
            )

def lag_cases(d_set: pl.DataFrame):
    return (d_set
            .sort('case_start_month')
            .with_columns(
                pl.col('cases_per_100k')
                .shift_and_fill(
                None,
                periods=LAG_GAP)
                .over('muni_id')
                .alias('case_lag')
            ).drop_nulls()
    )

def select_cluster(d_set: pl.DataFrame, cluster:int):
    return d_set.filter(pl.col('cluster') == cluster)

def split_data(d_set:pl.DataFrame, train_prop, valid_prop, test_prop):
    rng = np.random.default_rng(seed=42)

    unique_munis = d_set.unique('muni_id', maintain_order=True).select('muni_id').to_series().to_list()

    num_munis = len(unique_munis)
    train_num = int(train_prop*num_munis)
    valid_num = int(valid_prop*num_munis)

    train_munis = rng.choice(unique_munis, size=train_num, replace=False)
    unique_munis = np.setdiff1d(unique_munis, train_munis)

    valid_munis = rng.choice(unique_munis, size=valid_num, replace=False)

    test_munis = np.setdiff1d(unique_munis, valid_munis)

    test = d_set.filter(pl.col('muni_id').is_in(test_munis))
    train = d_set.filter(pl.col('muni_id').is_in(train_munis))
    valid = d_set.filter(pl.col('muni_id').is_in(valid_munis))

    return train, valid, test

def make_model(inp_data:pl.DataFrame):
    #coords = {'muni_id': }
    inp_data = inp_data.to_pandas()
    log_cases = np.log(inp_data['cases_per_100k']+0.1)
    log_lag = np.log(inp_data['case_lag']+0.1)

    muni, bz_munis = inp_data.muni_id.factorize()
    coords = {'muni_id': bz_munis}
    coords["param"] = ["alpha", "beta"]
    coords["param_bis"] = ["alpha", "beta"]
    with pm.Model(coords=coords) as model:
        #Vars
        #Lagged Cases
        lag_ind = pm.MutableData('case_lag', log_lag)
        #Muni
        muni_idx = pm.MutableData('muni', muni)

        # prior stddev in intercepts & slopes (variation across counties):
        sd_dist = pm.Exponential.dist(0.5, shape=(2,))

        # get back standard deviations and rho:
        chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist)

        # priors for average intercept and slope:
        mu_alpha_beta = pm.Normal("mu_alpha_beta", mu=0.0, sigma=5.0, shape=2)

        # population of varying effects:
        z = pm.Normal("z", 0.0, 1.0, dims=("param", "muni_id"))
        alpha_beta_county = pm.Deterministic(
            "alpha_beta_county", pt.dot(chol, z).T, dims=("muni_id", "param")
        )

        theta = (
        mu_alpha_beta[0]
        + alpha_beta_county[muni_idx, 0]
        + (mu_alpha_beta[1] + alpha_beta_county[muni_idx, 1]) * lag_ind
        )

        sigma = pm.Exponential("sigma", 1.0)

        y = pm.Normal("y", theta, sigma=sigma, observed=log_cases)

        covariation_intercept_slope_trace = pm.sample(
            1000,
            tune=1000,
            target_accept=0.95,
            idata_kwargs={"dims": {"chol_stds": ["param"], "chol_corr": ["param", "param_bis"]}},
        )

        print('here')

    pass

def run_model():
    pass

if __name__ == '__main__':
    RANDOM_SEED = 42
    LAG_GAP = 1
    ONE_HOT_MONTHS = True
    INCLUDE_LAG = True
    PREDICTORS = []
    DROPS = ['ntl']
    data = load_data(DROPS)

    if ONE_HOT_MONTHS:
        data = one_hot_months(data)
        PREDICTORS = PREDICTORS + [f'month_{x}' for x in range(1,13)]
    
    if INCLUDE_LAG:
        data = lag_cases(data)
        PREDICTORS = PREDICTORS + ['case_lag']

    data = select_cluster(data, 6)

    train_d, valid_d, test_d = split_data(data, 0.6, 0.2, 0.2)

    make_model(train_d)

