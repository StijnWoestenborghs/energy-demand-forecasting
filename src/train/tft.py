import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
pd.set_option('display.max_columns', None)


def make_model_tft(config, train_dataset, log_dir):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    # lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger(log_dir)  # logging results to a tensorboard

    pl.seed_everything(config['seed'])

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        gpus=config['n_gpus'],
        weights_summary="top",
        gradient_clip_val=config['gradient_clip_val'],
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[early_stop_callback],  # lr_logger
        logger=logger,
    )

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,  # not meaningful for finding the learning rate but otherwise very important
        learning_rate=config['learning_rate'],
        hidden_size=config['hidden_size'],  # most important hyperparameter apart from learning rate
        attention_head_size=1, # number of attention heads. Set to up to 4 for large datasets
        dropout=config['dropout'],  # between 0.1 and 0.3 are good values
        hidden_continuous_size=config['hidden_continuous_size'],  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=config['log_interval'],
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {model.size() / 1e3:.1f}k")

    return model, trainer


def optimize_lr(config, training, train_dataloader, eval_dataloader):
    # make sure to set 'log_interval': 0 when execute=True, otherwise learning_rate optimization is logged
    config['log_interval'] = 0

    model_lr, trainer_lr = make_model_tft(config, train_dataset=training, log_dir=None)

    res = trainer_lr.tuner.lr_find(
        model_lr,
        train_dataloader=train_dataloader,
        val_dataloaders=eval_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    plt.show()


def hyper_parameter_tuning():
    # # HYPERPARAMETER TUNING
    # from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    # import pickle
    #
    # # create study
    # study = optimize_hyperparameters(
    #     train_dataloader,
    #     val_dataloader,
    #     model_path="optuna_test",
    #     n_trials=200,
    #     max_epochs=50,
    #     gradient_clip_val_range=(0.01, 1.0),
    #     hidden_size_range=(8, 128),
    #     hidden_continuous_size_range=(8, 128),
    #     attention_head_size_range=(1, 4),
    #     learning_rate_range=(0.001, 0.1),
    #     dropout_range=(0.1, 0.3),
    #     trainer_kwargs=dict(limit_train_batches=30),
    #     reduce_on_plateau_patience=4,
    #     use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    # )
    #
    # # save study results - also we can resume tuning at a later point in time
    # with open("test_study.pkl", "wb") as fout:
    #     pickle.dump(study, fout)
    #
    # # show best hyperparameters
    # print(study.best_trial.params)
    return None
