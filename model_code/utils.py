import polars as pl
import pandas as pd
import geopandas as gpd
import io
from PIL import Image
from .reshaper import GluonReshaper, GluonBatchLoader
#from metaflow.cards import Markdown, Artifact, Image, Table
from gluonts.mx import DeepAREstimator, Trainer, DeepStateEstimator, DeepFactorEstimator, MQCNNEstimator,MQRNNEstimator, TransformerEstimator
import matplotlib.pyplot as plt
from .pytorch_models import TorchEstimator, LightningFeedForwardNetwork, LightningTransformer


def load_data(file_loc, drops=None):
    if drops is None:
        return pl.read_parquet(file_loc).drop_nulls()
    else:
        return pl.read_parquet(file_loc).select(pl.exclude(drops)).drop_nulls()
    
def load_reshaper(dataset, config):
    model_type = config['base']['model']
    return {
        'gluonts': GluonReshaper,
        'torch': GluonBatchLoader
    }[config[model_type]['loader']](
        dataset,
        config
    )

def load_model(config: dict):
    model_type = config['base']['model']
    return {
        'gluonts': load_gluonts,
        'torch': load_torch_model,
    }[config[model_type]['loader']](
        config
    )


def load_gluonts(config: dict):

    model_type = config['base']['model']
    config[model_type].pop('loader')
    return {
        'deep_ar': DeepAREstimator,
        'deep_state': DeepStateEstimator,
        'deep_factor': DeepFactorEstimator,
        'mqcnn': MQCNNEstimator,
        'transformer_mx': TransformerEstimator

    }[model_type](
        trainer = Trainer(
            epochs=config['general_model']['epochs'],
            num_batches_per_epoch=config['general_model']['epochs']
            ),
        freq=config['reshape']['frequency'],
        prediction_length=config['general_model']['prediction_length'],
        batch_size = config['general_model']['train_batch_size'],
        **config[model_type]
    )


def load_torch_model(config: dict):
    model_type = config['base']['model']
    lightning_module = {
        'feed_forward': LightningFeedForwardNetwork,
        'transformer': LightningTransformer,
    }[model_type](
        **(config['general_model'] | config[model_type])
    )

    return TorchEstimator(lightning_module, config)



def log_maps(config: dict, results: pd.DataFrame, exp, start_date):
    metric_config = config['metrics']

    munis = gpd.read_file(config['maps']['muni_file'])
    munis['CD_MUN'] = munis['CD_MUN'].str.slice(start=0, stop=6)

    joined = munis.merge(results, left_on='CD_MUN', right_on='item_id')
    for metric in metric_config:
        metric_name = metric.pop('name')
        fig, ax = plt.subplots(1, 1, figsize=(10,8))
        joined.plot(ax = ax, column=metric_name, legend=True, **metric,
                    )
        fig.suptitle(f'{config["base"]["model"]} - {metric_name} - {start_date} - Prediction Months: {config["general_model"]["prediction_length"]}')
        ax.set_axis_off()

        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        exp.log_image(img, name=metric_name)

    #return Table(plots)

