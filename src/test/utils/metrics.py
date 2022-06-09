import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.rolling_plot import plot_rolling_metric


def calculate_metric(data, model, config, start_idx, test_length, test_horizon, update, evaluation_target, type='mae', visualize=False):
    '''
    data: dataframe of one specific household
    model, config: model & specs
    test_length: number of evaluated time_idx's
    test_horizon: evaluated horizon in time idx's
    update: time idx step per update.
    evaluation target: metric based on absolute or cumulative signal
    type: Metric type one of mae, mape, mase, wsnmae
    '''

    # define model characteristics
    num = {'15min': 15, '30min': 30, '1hour': 60}
    prediction_length = int(config['prediction_length'] * 60 / num[config['resolution']])
    encoder_length = int(config['context_length'] * 60 / num[config['resolution']])

    if start_idx-encoder_length < 0:
        raise ValueError('start_idx must at least include encoder length. Increase start_idx by at least: {}'.format(abs(start_idx-encoder_length)))
    if type not in ['mae', 'mape', 'mase', 'wsnmae']:
        raise ValueError("type should be on of [mae, mape, mase, wsnmae]")

    # model target & evaluation target
    mt = config["target"]       # output signal of the model
    et = evaluation_target

    m_ar, time_store, pred_store = [], [], []

    for i in range(start_idx, start_idx+test_length, update):

        # update prediction
        time, actual, prediction = predict(i, data, model, encoder_length, prediction_length, mt)

        # only evaluate test horizon
        t, a, p = time[:test_horizon], actual[:test_horizon], prediction[:test_horizon]
        
        # preprocess to evaluation target
        a, p = prep_eval_target(a, p, mt, et)

        # calculate chosen metric
        m = choose_metric(a, p, type)

        m_ar += [m]
        time_store += [t]
        pred_store += [p]

    if visualize:
        # always plot in function of evaluation target (also pred_store in fucntion of et)
        plot_rolling_metric(data[et], config, start_idx, test_length, test_horizon, time_store, pred_store, title=f'Rolling MAE: {np.mean(m_ar)} L')
    
    return np.mean(m_ar)


def predict(i, data, model, encoder_length, prediction_length, mt):
    idx_available_low = data.index[data['time_idx'] == i - encoder_length][0]
    idx_available_high = data.index[data['time_idx'] == i][0]
    idx_target_low = data.index[data['time_idx'] == i][0]
    idx_target_high = data.index[data['time_idx'] == i + prediction_length][0]

    available_data = data[idx_available_low:idx_available_high]
    target_data = data[idx_target_low:idx_target_high]

    test_prediction, test_x = model.predict(pd.concat([available_data, target_data], ignore_index=True),
                                        return_x=True)

    time = np.array(data[idx_target_low:idx_target_high].index)
    actual = np.array(target_data[mt])                  # compare predictions with model target (mt)
    prediction = np.array(test_prediction[0])

    return time, actual, prediction


def choose_metric(a, p, type):
    if type == 'mae':
        m = mae(a, p)
    elif type == 'mape':
        m = mape(a, p)
    elif type == 'mase':
        m = mape(a, p)
    elif type == 'wsnmae':
        m = wsnmae(a, p)
    return m


def mae(actual, prediction):
    return np.mean(np.absolute(np.array(prediction)-np.array(actual)))


def mape(actual, prediction):
    error = np.array(prediction)-np.array(actual)
    try:
        return np.mean(100*np.abs((error/np.array(actual))))
    except ZeroDivisionError:
        return None


def mase(actual, prediction):
    raise NotImplementedError


def wsnmae(actual, prediction):
    error_under = np.clip(np.array(prediction)-np.array(actual), a_min=None, a_max=0)
    error_over = np.clip(np.array(prediction)-np.array(actual), a_min=0, a_max=None)
    return np.mean(np.absolute(error_over) + np.absolute(error_under)**2)


def cumsum_to_abs(signal):
    # TODO idx time mn clip negative
    out = np.zeros(len(signal))
    s = np.append([np.array(signal)[0]], np.array(signal))
    for i, (a, b) in enumerate(zip(s, s[1:])):
        out[i] = b-a
    out = np.clip(out, a_min=0, a_max=None)
    return out


def abs_to_cumsum(signal):
    # TODO
    raise NotImplementedError


def prep_eval_target(a, p, mt, et):
    if mt == 'energy' and et == 'cumulative_energy':
        # convert model output to cumulative
        a = abs_to_cumsum(a)
        p = abs_to_cumsum(p)
    if mt == 'cumulative_energy' and et == 'energy':
        # convert model output to absolute
        a = cumsum_to_abs(a)
        p = cumsum_to_abs(p)
    return a, p
