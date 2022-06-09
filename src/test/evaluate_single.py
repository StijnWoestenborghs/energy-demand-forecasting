import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import seaborn as sns
import numpy as np

from pytorch_forecasting import TemporalFusionTransformer, DeepAR
from src.prep.preprocess import load_data, generate_time_series, cleanup
from src.test.utils.metrics import calculate_metric


def change_index(data, config):
    data_hh = datetime_to_unix_timestamp(data)
    num = {'15min': 15, '30min': 30, '1hour': 60}
    data_hh['time_idx'] = (((data_hh['time_unix'] - data_hh['time_unix'].min()) / 60) / num[config['frequency']]).astype(
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


if __name__ == '__main__':

    # LOAD DATA
    file = '../dataset/LD2011_2014.txt'
    data = load_data(file, hh_start=0, hh_end=370, start='2014-01-01 00:15:00', stop="2015-01-01 00:00:00")
    data = cleanup(data)
    data_test = data.iloc[:, 300:321]
    # data_single = pd.DataFrame(np.array(data_test.iloc[:, 10:11]), columns=['single'], index=data_test.index)

    # load model
    model_abs_00 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_0/checkpoints/epoch=17-step=539.ckpt'
    model_abs_01 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_1/checkpoints/epoch=4-step=149.ckpt'
    model_abs_02 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_2/checkpoints/epoch=3-step=119.ckpt'
    model_abs_03 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_3/checkpoints/epoch=25-step=779.ckpt'
    model_abs_04 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_4/checkpoints/epoch=6-step=209.ckpt'
    model_abs_05 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_5/checkpoints/epoch=30-step=929.ckpt'
    model_abs_06 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_6/checkpoints/epoch=3-step=119.ckpt'
    model_abs_07 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_7/checkpoints/epoch=14-step=449.ckpt'
    model_abs_08 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_8/checkpoints/epoch=3-step=119.ckpt'
    model_abs_09 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_9/checkpoints/epoch=20-step=629.ckpt'
    model_abs_10 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_10/checkpoints/epoch=4-step=149.ckpt'
    model_abs_11 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_11/checkpoints/epoch=15-step=479.ckpt'
    model_abs_12 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_12/checkpoints/epoch=6-step=209.ckpt'
    model_abs_13 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_13/checkpoints/epoch=14-step=449.ckpt'
    model_abs_14 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_14/checkpoints/epoch=9-step=299.ckpt'
    model_abs_15 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_15/checkpoints/epoch=3-step=119.ckpt'
    model_abs_16 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_16/checkpoints/epoch=32-step=989.ckpt'
    model_abs_17 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_17/checkpoints/epoch=31-step=959.ckpt'
    model_abs_18 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_18/checkpoints/epoch=9-step=299.ckpt'
    model_abs_19 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/single-models/version_19/checkpoints/epoch=0-step=29.ckpt'

    model_cumsum_00 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_20/checkpoints/epoch=9-step=299.ckpt'
    model_cumsum_01 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_21/checkpoints/epoch=39-step=1199.ckpt'
    model_cumsum_02 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_22/checkpoints/epoch=13-step=419.ckpt'
    model_cumsum_03 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_23/checkpoints/epoch=34-step=1049.ckpt'
    model_cumsum_04 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_24/checkpoints/epoch=12-step=389.ckpt'
    model_cumsum_05 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_25/checkpoints/epoch=12-step=389.ckpt'
    model_cumsum_06 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_26/checkpoints/epoch=39-step=1199.ckpt'
    model_cumsum_07 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_27/checkpoints/epoch=12-step=389.ckpt'
    model_cumsum_08 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_28/checkpoints/epoch=36-step=1109.ckpt'
    model_cumsum_09 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_29/checkpoints/epoch=32-step=989.ckpt'
    model_cumsum_10 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_30/checkpoints/epoch=34-step=1049.ckpt'
    model_cumsum_11 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_31/checkpoints/epoch=38-step=1169.ckpt'
    model_cumsum_12 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_32/checkpoints/epoch=9-step=299.ckpt'
    model_cumsum_13 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_33/checkpoints/epoch=13-step=419.ckpt'
    model_cumsum_14 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_34/checkpoints/epoch=53-step=1619.ckpt'
    model_cumsum_15 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_35/checkpoints/epoch=12-step=389.ckpt'
    model_cumsum_16 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_36/checkpoints/epoch=30-step=929.ckpt'
    model_cumsum_17 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_37/checkpoints/epoch=69-step=2099.ckpt'
    model_cumsum_18 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_38/checkpoints/epoch=20-step=629.ckpt'
    model_cumsum_19 = '/Users/stijn.woestenborghs/git/energy-demand-forecasting-ml/logs/tft_logs/default/version_39/checkpoints/epoch=70-step=2129.ckpt'

    model_path_abs = [model_abs_00, model_abs_01, model_abs_02, model_abs_03, model_abs_04, model_abs_05, model_abs_06, model_abs_07, model_abs_08, model_abs_09,
                      model_abs_10, model_abs_11, model_abs_12, model_abs_13, model_abs_14, model_abs_15, model_abs_16, model_abs_17, model_abs_18, model_abs_19]

    model_path_cumsum = [model_cumsum_00, model_cumsum_01, model_cumsum_02, model_cumsum_03, model_cumsum_04, model_cumsum_05, model_cumsum_06, model_cumsum_07, model_cumsum_08, model_cumsum_09,
                         model_cumsum_10, model_cumsum_11, model_cumsum_12, model_cumsum_13, model_cumsum_14, model_cumsum_15, model_cumsum_16, model_cumsum_17, model_cumsum_18, model_cumsum_19]

    models_abs = []
    for model_path in model_path_abs:
        models_abs += [TemporalFusionTransformer.load_from_checkpoint(model_path)]

    models_cumsum = []
    for model_path in model_path_cumsum:
        models_cumsum += [TemporalFusionTransformer.load_from_checkpoint(model_path)]

    models = [list(models_abs), list(models_cumsum)]
    print(np.shape(models))

    MAE_household = [[], []]

    for h in range(20):
        for i, target in enumerate(['energy', 'cumulative_energy']):

            model = models[i][h]

            config = {
                'frequency': '15min',
                'target': target,
                'prediction_length': 24,  # [hours]
                'context_length': 7 * 24,  # [hours]
            }

            # select household
            data_hh = generate_time_series(data_test.iloc[:, h:h+1], frequency=config['frequency'])
            data_hh.index = data_hh['time_idx']

            test_length = 7*24   # hours
            test_horizon = 4     # hours

            if h != 12:
                MAE = calculate_metric(data=data_hh,
                                       model=model,
                                       test_horizon=test_horizon,
                                       test_length=test_length,
                                       config=config,
                                       type='mae',
                                       model_target=(target, 'energy'))

                MAE_household[i] += [MAE]

            print('Metric for household {} for {} hours testing: {}'.format(h, test_length, MAE))


    df = pd.DataFrame(np.transpose(MAE_household), index=['hh_{}'.format(i) for i in range(19)], columns=['absolute', 'cumulative'])
    print('mean abs: {}'.format(np.mean(df['absolute'])))
    print('mean cumsum: {}'.format(np.mean(df['cumulative'])))
    df.style.background_gradient(cmap='viridis').set_properties(**{'font-size': '20px'})

    plt.figure(figsize=(10, 7))
    sns.heatmap(df, cmap='Blues', annot=True, annot_kws={'size': 10}, fmt='g')
    plt.title('Metric rolling MAE for {} hours testing'.format(test_length))
    plt.show()
