import polars as pl
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    RemoveFields,
    AddAgeFeature,
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    VstackFeatures,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    AddTimeFeatures
)
from typing import Union
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.dataset.loader import TrainDataLoader

class GluonReshaper():
    def __init__(self, dataset: Union[pl.DataFrame, pd.DataFrame], config):
        self.config = config
        if isinstance(dataset, pl.DataFrame):
            #Cast to pandas
            dataset = dataset.to_pandas()
        #Ensure any float64 columns are cast to float32
        float64_cols = list(dataset.select_dtypes(include='float64'))
        dataset[float64_cols] = dataset[float64_cols].astype('float32')
        dataset[config['reshape']['item_id']+'_cat'] = dataset[config['reshape']['item_id']].astype('category')
        self.dataset = dataset


    def _recast_data(self, time_limit: str):
        config = self.config['reshape']
        return PandasDataset.from_long_dataframe(
                    self.dataset[self.dataset[config['time_stamp']] <= time_limit], 
                    item_id=config['item_id'], 
                    timestamp=config['time_stamp'], 
                    static_feature_columns= config['static_features'] + [config['item_id'] + '_cat']
                    if 'static_features' in config 
                    else None, 
                    target=config['target'], 
                    freq=config['frequency'], 
                    feat_dynamic_real=config['dynamic_features']
                    if 'dynamic_features' in config 
                    else None
                )
    
    def _get_train(self):
        return self._recast_data(self.config['reshape']['train_time_limit'])

    def _get_test(self):
        return self._recast_data(self.config['reshape']['test_time_limit'])

    def _get_valid(self):
        return self._recast_data(self.config['reshape']['valid_time_limit'])

    def get_data(self):
        return {
            'train': self._get_train(),
            'test': self._get_test(),
            'valid': self._get_valid()
        }
    
class GluonBatchLoader(GluonReshaper):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)

    def _get_train_loader(self):
        return TrainDataLoader(
        # We cache the dataset, to make training faster
            Cached(self._get_train()),
            batch_size=self.config['general_model']['train_batch_size'],
            stack_fn=batchify,
            transform=get_transforms(self.config) + get_splitter(self.config, 'train'),
            num_batches_per_epoch=self.config['general_model']['batches_per_epoch'],
            )

    def _get_valid_loader(self):
        return TrainDataLoader(
        # We cache the dataset, to make training faster
            Cached(self._get_valid()),
            #TODO: Can do smaller validation batches to speed up training
            batch_size=self.config['general_model']['valid_batch_size'],
            stack_fn=batchify,
            transform=get_transforms(self.config) + get_splitter(self.config, 'valid'),
            num_batches_per_epoch=self.config['general_model']['batches_per_epoch'],
            )


    def get_data(self):
        return {
            'train': self._get_train_loader(),
            'test': self._get_test(),
            'valid': self._get_valid_loader()
        }

def get_transforms(config):
    remove_field_names = []

    if len(config['reshape']['static_features']) == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if len(config['reshape']['dynamic_features']) == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    return Chain(
        [
            RemoveFields(field_names=remove_field_names),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            #This is likely not necessary with 'time_embed' and 'age'
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(config['reshape']['frequency']),
                pred_length=config['general_model']['prediction_length'],
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config['general_model']['prediction_length'],
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if len(config['reshape']['dynamic_features']) > 0
                    else []
                )
            ),
        ]
    )

def get_splitter(config, mode):
    return InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler= get_sampler(config, mode),
        past_length=config['general_model']['context_length'],
        future_length=config['general_model']['prediction_length'],
        time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
    )

def get_sampler(config, mode):
    return {
        'train':ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=config['general_model']['prediction_length'],
        ),
        'test':TestSplitSampler(),
        'valid':ValidationSplitSampler(min_future=config['general_model']['prediction_length'])
    }[mode]