import os, sys
import pandas as pd
import datetime
import numpy as np
import json
import torch

from pytorch_forecasting import TimeSeriesDataSet
from utils.plots import plot_raw_data, plot_freq_target
pd.set_option('display.max_columns', None)


def load_data(file, hh_start, hh_end, start=None, stop=None):
    data = pd.read_csv(file, delimiter=';', decimal=',')
    data.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)

    time = []
    for time_str in data['datetime']:
        time += [datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')]

    data.drop('datetime', axis=1, inplace=True)
    data.index = time
    data.index.name = 'datetime'

    if start is not None and stop is not None:
        start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        stop = datetime.datetime.strptime(stop, '%Y-%m-%d %H:%M:%S')
        data = data[start:stop]

    return data[data.columns[hh_start:hh_end]]


def generate_time_series(data, resolution):
    num = {'15min': 15, '30min': 30, '1hour': 60}

    for i, column in enumerate(data.columns):

        data_hh = pd.DataFrame(np.transpose(np.array([data[column], [column]*len(data)])),
                               index=data.index, columns=['energy', 'group_ids'])

        # add cumulative energy, for the desired resolution
        data_hh = calculate_cumulative_energy(data_hh, resolution)

        # One time_idx step should represents resolution step in time
        data_hh = datetime_to_unix_timestamp(data_hh)
        data_hh['time_idx'] = (((data_hh['time_unix'] - data_hh['time_unix'].min()) / 60) / num[resolution]).astype(
            'int32')
        data_hh.drop('time_unix', axis=1, inplace=True)

        if i == 0:
            ts = data_hh
        else:
            ts = ts.append(data_hh)

    # Change data index: must be unique
    ts = ts.sort_values(by=['time_idx'])
    return ts


def calculate_cumulative_energy(dataframe, resolution):
    res_dict = {'15min': [0, 15, 30, 45],
                 '30min': [0, 30],
                 '1hour': [0]}

    idx, cumulative_energy, energy = [], [], []
    energy_c, energy_a = 0, 0
    time_of_day, day_of_week, day_of_year = [], [], []
    for time, volume in zip(dataframe.index, dataframe['energy']):
        energy_c += float(volume)
        energy_a += float(volume)
        if (resolution != '4hour' and time.minute in res_dict[resolution]) or (resolution == '4hour' and time.hour in res_dict[resolution] and time.minute == 0):
            idx += [time]
            cumulative_energy += [energy_c]
            energy += [energy_a]
            time_of_day += [60*time.hour + time.minute]
            day_of_week += [time.timetuple().tm_wday]
            day_of_year += [time.timetuple().tm_yday]
            energy_a = 0
        if time.hour == 0 and time.minute == 0:
            energy_c, energy_a = 0, 0

    d = {'cumulative_energy': cumulative_energy,
         'energy': energy,
         'group_ids': dataframe['group_ids'],
         'time_of_day': time_of_day,
         'day_of_week': day_of_week,
         'day_of_year': day_of_year}
    ts = pd.DataFrame(data=d, index=idx)
    ts.index.name = 'date_time'
    return ts


def datetime_to_unix_timestamp(dataframe):
    unix_timestamps = []
    for date_time in dataframe.index:
        unix_timestamps.append(date_time.replace(tzinfo=datetime.timezone.utc).timestamp())
    dataframe['time_unix'] = unix_timestamps
    return dataframe


def cleanup(data):
    for column in data.columns:
        # drop household if mean consumption is below one in the first week
        start = datetime.datetime.strptime('2012-01-01 00:15:00', '%Y-%m-%d %H:%M:%S')
        stop = datetime.datetime.strptime('2012-01-07 00:15:00', '%Y-%m-%d %H:%M:%S')
        if np.mean(data[column][start:stop]) < 1:
            data.drop([column], axis=1, inplace=True)
    return data


def create_dataset(data, config):
    # dataframe index has to be unique
    data.index = np.arange(len(data))

    # prediction and context length in terms of time_idx
    num = {'15min': 15, '30min': 30, '1hour': 60}
    max_prediction_length = int(config['prediction_length'] * 60 / num[config['resolution']])
    max_encoder_length = int(config['context_length'] * 60 / num[config['resolution']])
    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=config['target'],
        group_ids=['group_ids'],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx", "time_of_day"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[config['target']],
        # add_relative_time_idx=True,
        # add_target_scales=True,
        # add_encoder_length=True,
        allow_missings=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    return training, validation


if __name__ == '__main__':
    # LOAD experiment config
    with open("./src/config.json", 'r') as f:
        config = json.load(f)

    # Define save directory
    save_dir = f"logs/{config['experiment_name']}/data"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(os.listdir(save_dir)) != 0:
        proceed = input('Experiment directory not empty, continue? [y/n]: ')
        if proceed != 'y':
            print('Aborted')
            sys.exit()

    # LOAD DATA
    file = './src/data/LD2011_2014.txt'
    data = load_data(file, hh_start=0, hh_end=config["n_train_hh"] + config["n_test_hh"], start='2012-01-01 00:15:00', stop="2014-01-01 00:00:00")
    data = cleanup(data)

    data_train_eval = data.iloc[:, :config["n_train_hh"]]
    data_test = data.iloc[:, config["n_train_hh"]:]

    ts = generate_time_series(data_train_eval, resolution=config['resolution'])

    # Visualize data
    visualize = False
    if visualize:
        plot_households = 10
        plot_raw_data(data_train_eval, plot_households=plot_households, start='2012-01-01 00:15:00', stop='2012-01-07 00:15:00')
        
        ts_15 = generate_time_series(data_train_eval, resolution='15min')
        ts_30 = generate_time_series(data_train_eval, resolution='30min')
        ts_1 = generate_time_series(data_train_eval, resolution='1hour')
        plot_freq_target(data_train_eval, ts_15, ts_30, ts_1, plot_households=plot_households, start='2012-01-01 00:15:00', stop='2012-01-07 00:15:00')


    # CREATE DATASET AND DATALOADER
    training, validation = create_dataset(ts, config)

    train_dataloader = training.to_dataloader(train=True, batch_size=config['batch_size'], num_workers=0)  # os.cpu_count()
    eval_dataloader = validation.to_dataloader(train=False, batch_size=config['batch_size'], num_workers=0)  # os.cpu_count()
    
    # save dat & loader
    training.save(f'{save_dir}/training')
    validation.save(f'{save_dir}/validation')
    torch.save(train_dataloader, f'{save_dir}/train-loader.pth')
    torch.save(eval_dataloader, f'{save_dir}/eval-loader.pth')