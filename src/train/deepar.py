import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import DeepAR
pd.set_option('display.max_columns', None)


def make_model_deepar(config, train_dataset, log_dir):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    logger = TensorBoardLogger(log_dir)  # logging results to a tensorboard

    pl.seed_everything(config['seed'])

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        gpus=config['n_gpus'],
        weights_summary="top",
        gradient_clip_val=config['gradient_clip_val'],
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[early_stop_callback],
        logger=logger,
    )

    model = DeepAR.from_dataset(
        train_dataset,
        cell_type='LSTM',
        hidden_size=config['hidden_size'],  # most important hyperparameter default 10
        rnn_layers=config['rnn_layers'],  # important hyperparameter default 2
        dropout=config['dropout'],  # between 0.1 and 0.3 are good values
        log_interval=config['log_interval'],
    )
    print(f"Number of parameters in network: {model.size() / 1e3:.1f}k")

    return model, trainer


def hyper_parameter_tuning():
    # HYPERPARAMETER TUNING
    return None


### extra utility ###
def deepar_plot_prediction(x, out, idx=0, ax=None, show_future_observed=True):
    out = {'prediction': np.array(out)}

    # all true values for y of the first sample in batch
    encoder_target = x["encoder_target"]
    decoder_target = x["decoder_target"]

    y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
    max_encoder_length = x["encoder_lengths"].max()
    y = torch.cat(
        (
            y_all[: x["encoder_lengths"][idx]],
            y_all[max_encoder_length: (max_encoder_length + x["decoder_lengths"][idx])],
        ),
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    n_pred = y.shape[0]
    x_obs = np.arange(-(y.shape[0] - n_pred), 0)
    x_pred = np.arange(n_pred)
    prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
    obs_color = next(prop_cycle)["color"]
    pred_color = next(prop_cycle)["color"]

    # plot observed history
    if len(x_obs) > 0:
        if len(x_obs) > 1:
            plotter = ax.plot
        else:
            plotter = ax.scatter
        plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
    if len(x_pred) > 1:
        plotter = ax.plot
    else:
        plotter = ax.scatter

    # plot observed prediction
    if show_future_observed:
        plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

    # plot prediction
    plotter(x_pred[n_pred-len(out['prediction'][0]):], out['prediction'][0], label="predicted", c=pred_color)