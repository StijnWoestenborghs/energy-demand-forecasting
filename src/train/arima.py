from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from src.preprocess import load_data, cleanup
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# LOAD DATA
file = 'dataset/LD2011_2014.txt'
data = load_data(file, hh_start=0, hh_end=370, start='2012-01-01 00:15:00', stop="2014-01-01 00:00:00")
data = cleanup(data)
data_arima_train = data.iloc[:, 300:321]

data_2 = load_data(file, hh_start=0, hh_end=370, start='2014-01-01 00:15:00', stop="2015-01-01 00:00:00")
data_2 = cleanup(data_2)
data_arima_test = data_2.iloc[:, 300:321]

MAE = []

for h in range(20):
    print('Progress: {}/{}'.format(h,20))

    train = np.array(data_arima_train.iloc[:, h:h + 1])[-672:]
    test = np.array(data_arima_test.iloc[:, h:h + 1])[0:672+96]
    history = [x for x in train]

    actual = []
    predictions = []
    m_ar = []

    for t in tqdm(range(len(test)-96)):
        model = ARIMA(history, order=(1, 1, 1))
        model_fit = model.fit()
        output = model_fit.forecast(steps=96)
        predictions.append(output)
        obs = test[0:96]
        actual.append(obs)

        # plt.plot(np.arange(96), output, label='prediction')
        # plt.plot(np.arange(96), obs, label='actual')
        # plt.legend()
        # plt.show()

        m = mean_absolute_error(obs[:16], output[:16])
        m_ar += [m]

        history.append(test[0])
        history = history[1:]
        test = test[1:]

    MAE += [np.mean(m_ar)]

print(MAE)

# 12th hh left out
MAE = [16.909043401223546, 26.615066702683674, 22.616618890733672, 10.820601840382576, 13.187733893240393, 16.028684722397855, 113.22507773618024, 17.732914647761856, 33.551418876186084, 79.80439307259365, 17.219452556079904, 56.185861664998995, 163.38250610826736, 13.018716639262705, 49.65989848767008, 84.38825833885558, 27.55027572268615, 12.65790524128087, 19.275515978437575]
print(np.mean(MAE))
#
# df = pd.DataFrame(np.transpose(MAE), index=['hh_{}'.format(i) for i in range(19)],
#                   columns=['ARIMA'])
# df.style.background_gradient(cmap='viridis').set_properties(**{'font-size': '20px'})
#
# plt.figure(figsize=(10, 7))
# sns.heatmap(df, cmap='Blues', annot=True, annot_kws={'size': 10}, fmt='g')
# plt.title('Metric rolling MAE for {} hours testing'.format(168))
# plt.show()
