#%%
# modul yang digunakan
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from scipy import stats
# %%
euro = pd.read_csv("euroclub.csv")
euro.head(6)
# %%
# mengubah nama tim sebagai index/row names
euro = euro.set_index("team")
euro.index.names = [None]
euro.head(6)
# %%
# standardisasi nilai variabel
scaler = MinMaxScaler()
euro[['goal','shots','yc','rc','poss','pass','aerials','rating']] = scaler.fit_transform(euro[['goal','shots','yc','rc','poss','pass','aerials','rating']])
euro.head(6)
#%%
# membuat elbow plot sebagai penentuan jumlah k
wss = []
for i in range(1,11): #1-11 adalah range axis elbow plot
    model = KMeans(n_clusters=i, init='k-means++', random_state=18)
    model.fit(euro)
    wss.append(model.inertia_)
plt.plot(range(1,11), wss)
plt.title("Penentuan Jumlah k dengan Metode Elbom")
plt.xlabel("Jumlah Klaster")
plt.ylabel("WSS")
plt.show()
# %%
# membentuk model kmeans
model = KMeans(n_clusters=3, init='k-means++', random_state=18)
predict = model.fit_predict(euro)
# %%
# membuat plot
plt.scatter(euro.iloc[predict == 0, 0], euro.iloc[predict == 0, 1], s=100, c='orange', label ='Cluster 1')
plt.scatter(euro.iloc[predict == 1, 0], euro.iloc[predict == 1, 1], s=100, c='green', label ='Cluster 2')
plt.scatter(euro.iloc[predict == 2, 0], euro.iloc[predict == 2, 1], s=100, c='brown', label ='Cluster 2')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='blue', label ='Centroids')
plt.title("Klaster Klub-Klub Eropa")
plt.legend()
plt.show()
# %%
model.inertia_ # lower, better
# END