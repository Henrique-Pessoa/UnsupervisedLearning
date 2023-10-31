import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


Leucocitos = np.array([2000,4000,5000,6500])

Plaquetas = np.array([100000,20000,80000,145000])

Linfocitos = np.array([2.3,4.5,6,5])


data = np.column_stack((Leucocitos,Plaquetas,Linfocitos))

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
