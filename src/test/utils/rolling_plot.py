import os
import matplotlib.pyplot as plt
import numpy as np


def plot_rolling_metric(data_et, config, start_idx, test_length, test_horizon, time_store, pred_store, title):
    test_data = data_et[start_idx:start_idx+test_length+test_horizon]
    time = np.array(test_data.index)
    actual = np.array(test_data)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(time, actual, label='actual')
    ax.plot(time_store[0], pred_store[0], c='darkorange', label='predictions')
    for i in range(1, len(time_store)):
        ax.plot(time_store[i], pred_store[i], c='darkorange')
    plt.legend()
    plt.xlabel('Time index')
    plt.ylabel('Evaluation target')
    plt.title(title)

    plt.show()
