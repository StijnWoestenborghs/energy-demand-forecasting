import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import seaborn as sns
import numpy as np
import json

from pytorch_forecasting import TemporalFusionTransformer, DeepAR
from utils.metrics import calculate_metric

sys.path.insert(0, './src/prep')
from preprocess import load_data, generate_time_series, cleanup


def change_index(data, config):
    data_hh = datetime_to_unix_timestamp(data)
    num = {'15min': 15, '30min': 30, '1hour': 60}
    data_hh['time_idx'] = (((data_hh['time_unix'] - data_hh['time_unix'].min()) / 60) / num[config['resolution']]).astype(
        'int32')
    data_hh.drop('time_unix', axis=1, inplace=True)
    data_hh.index = data_hh['time_idx']
    return data_hh


def datetime_to_unix_timestamp(dataframe):
    unix_timestamps = []
    for date_time in dataframe.index:
        unix_timestamps.append(date_time.replace(tzinfo=datetime.timezone.utc).timestamp())
    dataframe['time_unix'] = unix_timestamps
    return dataframe


def create_groupids(data, name):
    data['group_ids'] = np.array(len(data)*['{}'.format(name)])
    return data


def load_models(experiments):
    models, configs = [], []
    for experiment in experiments:
        try:
            with open(f"logs/{experiment}/config.json", 'r') as f:
                config = json.load(f)

            save_train = f'logs/{experiment}/models/default'
            versions = [int(el[len('version_'):]) for el in os.listdir(save_train) if not el.startswith('.')]
            save_dir_model = f'{save_train}/version_{np.max(versions)}/checkpoints'
            model_path = f'{save_dir_model}/{os.listdir(save_dir_model)[0]}'

            if config["type"] == "tft":
                model = TemporalFusionTransformer.load_from_checkpoint(model_path)
            elif config["type"] == "deepar":
                model = DeepAR.load_from_checkpoint(model_path)

            models += [model]
            configs += [config]
        except:
            raise ValueError(f"Experiment/model not in logs: {experiment}")
    
    return models, configs


if __name__ == '__main__':

    # TEST SPECIFICATIONS
    n_hh = 1                                        # number of households to test
    EXPERIMENTS = ["baseline", "baseline_deepar"]    # experiments to test
    test_length = 7*24                               # hours
    test_horizon = 4                                 # hours

    # LOAD SPECIFIC TEST DATA PERIOD
    file = './src/data/LD2011_2014.txt'
    data_test = load_data(file, hh_start=-n_hh-1, hh_end=-1, start='2014-01-01 00:15:00', stop="2015-01-01 00:00:00")
    data_test = cleanup(data_test)

    # load models
    models, configs = load_models(EXPERIMENTS)

    MAE_household = np.zeros(shape=(n_hh, len(models)))

    for h in range(n_hh):
        print('household: {}/{}'.format(h + 1, n_hh*len(models)))
        for i, experiment in enumerate(EXPERIMENTS):
            model = models[i]
            config = configs[i]

            # select household
            data_hh = generate_time_series(data_test.iloc[:, h:h+1], resolution=config['resolution'])
            data_hh.index = data_hh['time_idx']

            MAE = calculate_metric(data=data_hh,
                                    model=model,
                                    test_horizon=test_horizon,
                                    test_length=test_length,
                                    config=config,
                                    type='mae',
                                    model_target=('energy', 'energy'))

            MAE_household[h][i] = MAE

            print('Metric for household {} for experiment {}: {}'.format(h, experiment, MAE))

    df = pd.DataFrame(MAE_household, index=[f'hh_{i}' for i in range(n_hh)], columns=EXPERIMENTS)
    df.style.background_gradient(cmap='viridis').set_properties(**{'font-size': '20px'})

    plt.figure(figsize=(10, 7))
    sns.heatmap(df, cmap='Blues', annot=True, annot_kws={'size': 10}, fmt='g')
    plt.title('Mean of MAE total horizon for {} hours testing'.format(test_horizon))
    plt.show()
