import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from scipy.cluster.hierarchy import  linkage
from scipy.cluster.hierarchy import to_tree
import dill as pickle

monthly_cases = pl.read_parquet('/home/tony/dengue/dengue_models/data/cases/agged/dengue_per_month.parquet')
monthly_params = pl.read_parquet('/home/tony/dengue/dengue_models/data/gee_exports/all_parameters_2001-01-01_2021-01-01_months.parquet')

monthly_params = monthly_params.with_columns((pl.col('start_date').str.to_date('%Y-%m-%d'),pl.col('end_date').str.to_date('%Y-%m-%d')))

monthly_cases = monthly_cases.with_columns(pl.col('ID_MUNICIP').str.slice(offset=0,length=6).cast(pl.Int64)).sort('DT_NOTIFIC').rename({
    'DT_NOTIFIC': 'start_date',
    'ID_MUNICIP': 'muni_id'
})

all_data = monthly_cases.join(monthly_params, how='left', on=['muni_id', 'start_date']).with_columns(pl.col('end_date').alias('month').dt.month())


cluster_features = ['cases_per_100k', 
                     #'temperature_2m', 
                     #'total_precipitation_min', 
                    # 'EVI'
                    ]
case_data = (
                all_data
                #weekly_cases
               .select(pl.col(['end_date', 'muni_id',  'month'] + cluster_features))
               .filter(
                    (pl.col('end_date') >= datetime(2010,1,1)) &
                    (pl.col('end_date') < datetime(2019,1,1)) &
                    (pl.col('month').is_in([11, 12, 1, 2, 3, 4, 5]))
                    )
               .sort('end_date')
               .group_by('muni_id', maintain_order=True)
               .agg([pl.col(f) for f in cluster_features])
               .sort('muni_id')
               )
case_matrix = np.vstack(
               case_data
               .select(pl.concat_list(cluster_features))
               .to_series())
muni_matrix = np.vstack(
               case_data.select(pl.col('muni_id'))
               .to_series()
               )

X = case_matrix
Z = linkage(X, 'ward')

rootnode, nodelist = to_tree(Z, rd=True)

def child_mapper(child_list, muni_matrix):
    num_ids = case_matrix.shape[0]
    new_list = []
    for child in child_list:
        if child < num_ids:
            new_list.append(muni_matrix[child][0])
    
    return new_list

cluster_dict = {node.id : {'child_models': node.pre_order(lambda x: x.id),
                           'include_munis' : child_mapper(node.pre_order(lambda x: x.id), muni_matrix),
                           'model': HistGradientBoostingRegressor(loss='poisson', max_iter=5000),
                           'predictions': None} for node in nodelist}
for node in nodelist:
    left = node.get_left()
    right = node.get_right()
    if left is not None:
        cluster_dict[left.id]['parent'] = node.id
    if right is not None:
        cluster_dict[right.id]['parent'] = node.id

sorted_clusters = {k:v for k, v in sorted(cluster_dict.items(), key=lambda x: len(x[1]['include_munis']), reverse=True)}

train_end=datetime.fromisoformat('2018-01-01')
start_date = datetime.fromisoformat('2005-01-01')

calced_clusters = {}

for k, v in tqdm(sorted_clusters.items()):
    
    #step 1: agg data at level
    munis_to_agg = v['include_munis']
    this_data = (all_data
                 .filter(pl.col('muni_id').is_in(munis_to_agg))
                 .group_by('start_date')
                 .agg((pl.col('count').sum(),pl.col('pop').sum()))
                 .with_columns((pl.col('count')/pl.col('pop')*pl.lit(100000)).alias('cases_per_100k'))
    )
    train_df = this_data.filter(pl.col('start_date')<=train_end)
    test_df = this_data.filter(pl.col('start_date')>train_end-relativedelta(years=2))

    train_features = []
    test_features = []

    train_targets= []
    test_targets = []

    for i in train_df.filter(pl.col('start_date')>=start_date).select('start_date').to_series():
        target = train_df.filter(pl.col('start_date') == i).select('cases_per_100k').item()
        local_cases = train_df.filter((pl.col('start_date')<i) & (pl.col('start_date')>=i-relativedelta(years=2))).select('cases_per_100k').unstack(step=1).to_numpy()

        train_features.append(local_cases)
        train_targets.append(target)
    
    for i in test_df.filter(pl.col('start_date')>train_end).select('start_date').to_series():
        target = test_df.filter(pl.col('start_date') == i).select('cases_per_100k').item()
        local_cases = test_df.filter((pl.col('start_date')<i) & (pl.col('start_date')>=i-relativedelta(years=2))).select('cases_per_100k').unstack(step=1).to_numpy()

        test_features.append(local_cases)
        test_targets.append(target)

    train_x = np.vstack(train_features)
    train_y = np.array(train_targets)

    test_x = np.vstack(test_features)
    test_y = np.array(test_targets)

    orig_train_x = train_x.copy()
    orig_test_x = test_x.copy()
    
    #step 2: add parent predictions if has parent
    if 'parent' in v:
        parent = v['parent']

        train_x = np.hstack((train_x, calced_clusters[parent]['train_predictions']))
        test_x = np.hstack((test_x, calced_clusters[parent]['test_predictions']))
    
    #step 3: train 

    loss_fn = 'poisson' if train_y.sum() != 0 else 'squared_error'

    reg = HistGradientBoostingRegressor(
    random_state=42, 
    max_iter=1000, 
    loss=loss_fn,
    max_leaf_nodes=None, 
    min_samples_leaf=10,
    l2_regularization=2.0, 
    max_bins=255,
    early_stopping=False)
    reg.fit(train_x, train_y)

    train_preds = reg.predict(train_x)
    test_preds = reg.predict(test_x)
    calced_clusters[k] = {'train_predictions' : np.expand_dims(train_preds, axis=1), 
                        'test_predictions':  np.expand_dims(test_preds,axis=1), 
                        #'model': reg,
                        'score': reg.score(test_x, test_y),
                        'rmse': np.sqrt(((test_preds - test_y) ** 2).mean())
                        }
    if len(v['child_models']) == 1:
        reg = HistGradientBoostingRegressor(
        random_state=42, 
        max_iter=1000, 
        loss=loss_fn,
        max_leaf_nodes=None, 
        min_samples_leaf=10,
        l2_regularization=2.0, 
        max_bins=255,
        early_stopping=False)
          
        reg.fit(orig_train_x, train_y)

        train_preds = reg.predict(orig_train_x)
        test_preds = reg.predict(orig_test_x)

        calced_clusters[k]['no_parent_score'] = reg.score(orig_test_x, test_y)
        calced_clusters[k]['no_parent_rmse'] = np.sqrt(((test_preds - test_y) ** 2).mean())
        calced_clusters[k]['parent_score_delta'] = calced_clusters[k]['no_parent_score'] - calced_clusters[k]['score']
        calced_clusters[k]['parent_rmse_delta'] = calced_clusters[k]['no_parent_rmse'] - calced_clusters[k]['rmse']
    
    with open('calced_clusters.pkl', 'wb') as f:
        pickle.dump(calced_clusters, f)

        