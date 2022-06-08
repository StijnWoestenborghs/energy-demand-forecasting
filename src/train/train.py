import os, sys
import json
import torch
import matplotlib.pyplot as plt

from tft import make_model_tft, optimize_lr
from deepar import make_model_deepar, deepar_plot_prediction
from pytorch_forecasting import Baseline, TemporalFusionTransformer
from pytorch_forecasting import Baseline, DeepAR



def check_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(os.listdir(save_dir)) != 0:
        proceed = input('Experiment directory not empty, continue? [y/n]: ')
        if proceed != 'y':
            print('Aborted')
            sys.exit()


def load_data(config):
    try:
        name = config["experiment_name"]
        training = torch.load(f"logs/{config['experiment_name']}/data/training")
        validation = torch.load(f"logs/{config['experiment_name']}/data/validation")
        train_dataloader = torch.load(f"logs/{config['experiment_name']}/data/train-loader.pth")
        eval_dataloader = torch.load(f"logs/{config['experiment_name']}/data/eval-loader.pth")
        return training, validation, train_dataloader, eval_dataloader
    except:
        raise ValueError("Experiment not in data logs")


def calculate_metrics(model, eval_dataloader, plot=True):
    # Mean Absolute Error
    actual = torch.cat([y[0] for z, y in iter(eval_dataloader)])
    predictions, x = model.predict(eval_dataloader, return_x=True)
    mae = (actual - predictions).abs().mean()

    print(f"MAE: {mae}")

    if plot:
        # plot actual vs predictions by variables
        predictions_vs_actual = model.calculate_prediction_actual_by_variable(x, predictions)
        model.plot_prediction_actual_by_variable(predictions_vs_actual)
        plt.show()


if __name__ == "__main__":
     # LOAD experiment config
    with open("./src/config.json", 'r') as f:
        config = json.load(f)

    # Define save directory
    save_dir = f"logs/{config['experiment_name']}/models"
    check_save_dir(save_dir)

    # Load ts & data loaders
    training, validation, train_dataloader, eval_dataloader = load_data(config)

    # (Optional) Optimize learning rate TFT
    optim_lr = False
    if config["type"] == "tft" and optim_lr == True:
        optimize_lr(config, training, train_dataloader, eval_dataloader)

    # Create model
    if config["type"] == "tft":
        model, trainer = make_model_tft(config, training, save_dir)
    elif config["type"] == "deepar":
        model, trainer = make_model_deepar(config, training, save_dir)

    # Train model
    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=eval_dataloader,
    )

    # load the best model according to the validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    if config["type"] == "tft":
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    elif config["type"] == "deepar":
        best_model = DeepAR.load_from_checkpoint(best_model_path)

    # EVALUATE PERFORMANCE on evaluation data
    calculate_metrics(model=best_model, eval_dataloader=eval_dataloader, plot=True)

    # plot eval prediction
    if config["type"] == "tft":
        raw_predictions, x = best_model.predict(eval_dataloader, mode="raw", return_x=True)
        for i in range(min(10, config["n_train_hh"])):
            best_model.plot_prediction(x, raw_predictions, idx=i, add_loss_to_title=True, plot_attention=False)
            plt.show()
    elif config["type"] == "deepar":
        predictions, x = best_model.predict(eval_dataloader, return_x=True)
        for i in range(min(10, config["n_train_hh"])):
            deepar_plot_prediction(x, predictions, idx=i)
            plt.show()
