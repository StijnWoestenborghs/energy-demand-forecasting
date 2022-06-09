import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

from pytorch_forecasting import TemporalFusionTransformer, DeepAR


def check_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(os.listdir(save_dir)) != 0:
        proceed = input('Experiment directory not empty, continue? [y/n]: ')
        if proceed != 'y':
            print('Aborted')
            sys.exit()


def update(i):
    idx_available_low = data.index[data['time_idx'] == i][0]
    idx_available_high = data.index[data['time_idx'] == encoder_length + i][0]
    idx_target_low = data.index[data['time_idx'] == encoder_length + i][0]
    idx_target_high = data.index[data['time_idx'] == encoder_length + i + prediction_length][0]

    available_data = data[idx_available_low:idx_available_high]
    target_data = data[idx_target_low:idx_target_high]
    # target_data.loc[target_data.index, 'cumulative_energy_volume'] = np.zeros(prediction_length)

    test_prediction, test_x = best_model.predict(pd.concat([available_data, target_data], ignore_index=True), mode="raw", return_x=True)

    ax.clear()
    best_model.plot_prediction(test_x, test_prediction, idx=h, show_future_observed=True, plot_attention=False, ax=ax)


def animate(households, frames, render=False, save_dir='animations'):
    global h
    for h in households:
        global ax
        fig, ax = plt.subplots()

        anim = animation.FuncAnimation(fig, func=update, frames=frames)

        writer = animation.writers['ffmpeg']()
        anim.save('{}/pred-{}-hh-{}.mp4'.format(save_dir, config['prediction_length'], h), writer=writer, dpi=100)
        if render:
            plt.show()


if __name__ == '__main__':

    # load model
    experiment = "logs/baseline"
    best_model_path = f"{experiment}/models/default/version_0/checkpoints/epoch=2-step=89.ckpt"
   
    # Define save directory
    save_dir = f"{experiment}/animations"
    check_save_dir(save_dir)
    with open(f"{experiment}/config.json", 'r') as f:
        config = json.load(f)

    if config["type"] == "tft":
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    elif config["type"] == "deepar":
        best_model = DeepAR.load_from_checkpoint(best_model_path)

    # Load test data
    data = pd.read_csv(f"{experiment}/data/test.csv")

    num = {'15min': 15, '30min': 30, '1hour': 60}
    prediction_length = int(config['prediction_length'] * 60 / num[config['resolution']])
    encoder_length = int(config['context_length'] * 60 / num[config['resolution']])

    animate(households=[0], frames=2*144, save_dir='animations', render=False)

