import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


Leucocitos = np.array([2000,4000,5000,6500])

Plaquetas = np.array([100000,20000,80000,145000])

Linfocitos = np.array([2.3,4.5,6,5])


data = np.column_stack((Leucocitos,Plaquetas,Linfocitos))

scaler = StandardScaler()
scaledData = scaler.fit_transform(data)

model = DBSCAN(eps=30, min_samples=2)
model.fit(scaledData)
labels = model.labels_
n_clusters = len(set(labels)) -(1 if -1 in labels else 0)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(Leucocitos,Plaquetas,Linfocitos)

print(f"numero de cluster: {n_clusters}")

plt.show()
