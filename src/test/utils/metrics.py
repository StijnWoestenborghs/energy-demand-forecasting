import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_metric(data, model, test_horizon, test_length, config, type='mae', model_target=()):
    '''
    data, config: same input as long format from generate_time_series, only index changed to time_idx
    test_horizon: first # hours of prediction to be evaluated
    test_length: hours of testing
    type: Metric type one of mae, mape, mase, wsnmae
    '''

    # define model characteristics
    num = {'15min': 15, '30min': 30, '1hour': 60}
    prediction_length = int(config['prediction_length'] * 60 / num[config['resolution']])
    encoder_length = int(config['context_length'] * 60 / num[config['resolution']])
    time_frames = int((test_length*60) / num[config['resolution']])

    # model target & evaluation target
    (mt, et) = model_target

    m_ar = np.zeros(time_frames)
    for i in range(time_frames):
        if type in ['mae', 'mape', 'mase', 'wsnmae']:
            idx_available_low = data.index[data['time_idx'] == i][0]
            idx_available_high = data.index[data['time_idx'] == encoder_length + i][0]
            idx_target_low = data.index[data['time_idx'] == encoder_length + i][0]
            idx_target_high = data.index[data['time_idx'] == encoder_length + i + prediction_length][0]
            available_data = data[idx_available_low:idx_available_high]
            target_data = data[idx_target_low:idx_target_high]

            test_prediction, test_x = model.predict(pd.concat([available_data, target_data], ignore_index=True),
                                                    return_x=True)

            actual = np.array(target_data[config['target']])
            prediction = np.array(test_prediction[0])

            # only evaluate test horizon
            a = actual[0:int((test_horizon*60)/(num[config['resolution']]))]
            p = prediction[0:int((test_horizon*60)/(num[config['resolution']]))]
            idx = target_data.index[0:int((test_horizon*60)/(num[config['resolution']]))]

            # preprocess to evaluation target
            a, p = prep_eval_target(a, p, idx, mt, et)

            # calculate chosen metric
            m = choose_metric(a, p, type)
        else:
            raise NotImplementedError
        m_ar[i] = m
    return np.mean(m_ar)


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


def prep_eval_target(a, p, idx, mt, et):
    # if mt == 'energy' and et == 'cumulative_energy':
    #     # split on mn
    #     mn = np.where(idx % 96 == 0)
    #     if len(mn[0]) != 0:
    #         a1 = pd.Series(a[0:int(mn[0])]).cumsum()
    #         a2 = pd.Series(a[int(mn[0]):]).cumsum()
    #         p1 = pd.Series(p[0:int(mn[0])]).cumsum()
    #         p2 = pd.Series(p[int(mn[0]):]).cumsum()
    #         a = np.append(np.array(a1), np.array(a2))
    #         p = np.append(np.array(p1), np.array(p2))
    #     else:
    #         a = np.array(pd.Series(a, index=idx).cumsum())
    #         p = np.array(pd.Series(p, index=idx).cumsum())
    if mt == 'cumulative_energy' and et == 'energy':
        # convert mt to absolute
        a = cumsum_to_abs(a)
        p = cumsum_to_abs(p)
    return a, p
