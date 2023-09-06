import torch
import torch.nn as nn
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from typing import List, Callable
import pytorch_lightning as pl
from .reshaper import get_splitter, get_transforms

def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dimensions: List[int],
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
        scaling: Callable = mean_abs_scaling,
        **kwargs
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.distr_output = distr_output
        self.batch_norm = batch_norm
        self.scaling = scaling

        dimensions = [context_length] + hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(
            self.__make_lin(dimensions[-1], prediction_length * hidden_dimensions[-1])
        )

        self.nn = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[-1])

    @staticmethod
    def __make_lin(dim_in, dim_out):
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin

    def forward(self, context):

    
        scale = self.scaling(context)

        scaled_context = context / scale
        nn_out = self.nn(scaled_context)
        nn_out_reshaped = nn_out.reshape(
            -1, self.prediction_length, self.hidden_dimensions[-1]
        )
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, torch.zeros_like(scale), scale

    def get_predictor(self, input_transform, batch_size=32, device='cuda'):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
            device=device,
        )
    

class TransformerNetwork(nn.Module):
    def __init__(self,
        prediction_length: int,
        context_length: int,
        hidden_dimensions: List[int],
        input_dims: int = 6,
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
        scaling: Callable = mean_abs_scaling,
        augment_p: float = 0.5,
        attn_heads: int = 4,
        num_layers: int = 4,
        **kwargs
    ) -> None:
        super().__init__()
        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.distr_output = distr_output
        self.batch_norm = batch_norm
        self.scaling = scaling

        self.nn = torch.nn.Transformer(d_model=hidden_dimensions[0],
                                       nhead=attn_heads,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers,
                                       dim_feedforward=hidden_dimensions[1],
                                       batch_first=True)
        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[0])

        self.embed_past = torch.nn.Linear(input_dims, hidden_dimensions[0])

        self.embed_future = torch.nn.Linear(input_dims-1, hidden_dimensions[0])
        
    def forward(self, past_time, future_time, past_target):
        # past_time = batch['past_time_feat']
        # future_time = batch['future_time_feat']

        # past_target = batch['past_target']

        scale = self.scaling(past_target)
        scaled_target = past_target/scale

        src = self.embed_past(torch.concat([scaled_target.unsqueeze(2), past_time],dim=2))
        tgt = self.embed_future(future_time)

        nn_out = self.nn(src=src, tgt=tgt)

        distr_args = self.args_proj(nn_out)
        return distr_args, torch.zeros_like(scale), scale
    
    def get_predictor(self, input_transform, batch_size=32, device='cuda'):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=["past_time_feat", "future_time_feat", "past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
            device=device,
        )


    
class LightningFeedForwardNetwork(FeedForwardNetwork, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = kwargs['lr']

    def training_step(self, batch, batch_idx):
        context = batch["past_target"]
        target = batch["future_target"]

        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length

        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)

        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        context = batch["past_target"]
        target = batch["future_target"]

        # print(context.shape)
        # print(target.shape)
        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length

        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)

        return loss.mean()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class LightningTransformer(TransformerNetwork, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = kwargs['lr']

    def training_step(self, batch, batch_idx):
        print('TRAINING')
        past_time = batch['past_time_feat']
        future_time = batch['future_time_feat']

        past_target = batch['past_target']
        target = batch["future_target"]

        distr_args, loc, scale = self(past_time, future_time, past_target)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)

        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        past_time = batch['past_time_feat']
        future_time = batch['future_time_feat']

        past_target = batch['past_target']

        target = batch["future_target"]

        distr_args, loc, scale = self(past_time, future_time, past_target)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)

        return loss.mean()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



class TorchEstimator():
    def __init__(self, lightning_net, config):
        model_type = config['base']['model']
        self.device = config[model_type]['device']
        self.lightning_net = lightning_net
        self.config = config
        #TODO: Move device to config
        self.trainer = pl.Trainer(max_epochs=config['general_model']['epochs'], accelerator=self.device)
        self.test_splitter = get_transforms(self.config) + get_splitter(self.config, 'test')

    def train(self, training_data, validation_data) -> PyTorchPredictor:
        self.trainer.fit(self.lightning_net, 
                         train_dataloaders=training_data, 
                         val_dataloaders=validation_data)
        
        return self.lightning_net.get_predictor(self.test_splitter, device=self.device)

