
import matplotlib.pyplot as plt
import datetime
import numpy as np


def plot_raw_data(data, plot_households, start=None, stop=None):
    if start is not None and stop is not None:
        start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        stop = datetime.datetime.strptime(stop, '%Y-%m-%d %H:%M:%S')
        data = data[start:stop]

    plt.figure()
    for household in range(plot_households):
        ax = plt.subplot2grid((plot_households, 1), (household, 0))
        ax.plot(data.index, data[data.columns[household]])
    plt.show()


def plot_compare_time_series(ts_15, ts_30, ts_1, target, start, stop, axes):
    ts_15[target].loc[start:stop].plot(ax=axes[0])
    ts_30[target].loc[start:stop].plot(ax=axes[1])
    ts_1[target].loc[start:stop].plot(ax=axes[2])
    axes[0].set_title('Time Series: 15min')
    axes[1].set_title('Time Series: 30min')
    axes[2].set_title('Time Series: 1hour')


def plot_freq_target(data, ts_15, ts_30, ts_1, plot_households, start, stop):

    for group in data.columns[0:plot_households]:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 5))
        plot_compare_time_series(ts_15.loc[ts_15['group_ids'] == group],
                                 ts_30.loc[ts_30['group_ids'] == group],
                                 ts_1.loc[ts_1['group_ids'] == group],
                                 'energy', start=start, stop=stop,
                                 axes=(ax1, ax2, ax3))
        plot_compare_time_series(ts_15.loc[ts_15['group_ids'] == group],
                                 ts_30.loc[ts_30['group_ids'] == group],
                                 ts_1.loc[ts_1['group_ids'] == group],
                                 'cumulative_energy', start=start, stop=stop,
                                 axes=(ax4, ax5, ax6))
        plt.show()