import numpy as np
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


corrente = np.array([5, 10, 14, 2, 1.5, 6])
tempo = np.array([1,2,4,6,7,10])


data = np.column_stack((corrente,tempo))

iforest = IsolationForest(n_estimators = 100, contamination = 0.03, max_samples ='auto')
prediction = iforest.fit_predict(data)
print(prediction[:20])
print("Number of outliers detected: {}".format(prediction[prediction < 0].sum()))
print("Number of normal samples detected: {}".format(prediction[prediction > 0].sum()))

normal_data = data[np.where(prediction > 0)]
outliers = data[np.where(prediction < 0)]
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(normal_data[:, 0], normal_data[:, 1])
ax.scatter(outliers[:, 0], outliers[:, 1])
plt.show()