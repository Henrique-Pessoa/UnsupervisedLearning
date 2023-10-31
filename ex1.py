import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


Area  =  np.array([120,145,80,160,200,90,110,130,180,160])

Valor = np.array([300,450,550,600,350,420,550,780,360,575])

Dist_praia =  np.array([15,15,8,25,12,15,22,8,5,14])


data = np.column_stack((Area,Valor,Dist_praia))

scaler = StandardScaler()
scaledData = scaler.fit_transform(data)

model = DBSCAN(eps=30, min_samples=2)
model.fit(scaledData)
labels = model.labels_
n_clusters = len(set(labels)) -(1 if -1 in labels else 0)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(Area,Dist_praia,Valor)

print(f"numero de cluster: {n_clusters}")

plt.show()
