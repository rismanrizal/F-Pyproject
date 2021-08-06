#CONTOH DENGAN REGRESI LOGISTIK
#%%
from operator import index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy as rp
import statsmodels.api as sm
# %%
#import data from csv
waterpot = pd.read_csv("water_potability.csv")
waterpot.head(10)
#%%
print("#Jumlah Total Sampel:" + str(len(waterpot)))
#%%
#mengetahui proporsi DV
sns.countplot(x = "potability", data = waterpot)
rp.summary_cat(waterpot['potability'])
#%%
#mengetahui karakteristik dataset dan NaN pengamatan
waterpot.info()
waterpot.isnull().sum()
#memvisualisasikan NaN pengamatan
sns.heatmap(waterpot.isnull(), yticklabels=False)
#%%
#menghilangkat NaN data dalam row dengan nama variabel yg sama
waterpot.dropna(inplace=True)
print('Updated Dataframe:')
print(waterpot)
#%%
#cek karakteristik datase
waterpot.info()
print(waterpot.dtypes)
#%%
#kolom 'solid' adalah object {string}
#perlu diubah ke dalam float
waterpot.solid = waterpot.solid.str.replace(',', '').astype(float)
#%%
#membentuk array variabel dependen
Y = waterpot['potability'].values
print("Tipe array adalah: ", type(Y))
print("Tipe elemen array adalah: ", Y.dtype)
#%%
#mengubah Y dari float ke integer
Y = Y.astype('int')
print("Tipe elemen array adalah: ", Y.dtype)
#%%
#membentuk dataframe independen variabel
X = waterpot.drop(labels=['potability'], axis=1)
print(X.head(4))
#%%
#membentuk kelas training dan testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=135)
#%%
#membentuk model regresi logistik
from sklearn.linear_model import LogisticRegression
mymodel = LogisticRegression()
#%%
#evaluasi model fit
mymodel.fit(X_train, y_train)
# %%
#prediksi model
prediction = mymodel.predict(X_test)
#%%
from sklearn import metrics
print("Akurasi Model = ", metrics.accuracy_score(y_test, prediction))
#%%
#coefisien dari model
modelcoef = pd.Series(mymodel.coef_[0], index=X.columns.values)
print(modelcoef)
#%%
#overall akurasi dari model
print(metrics.classification_report(y_test, prediction, 
zero_division=0))
#%%
#confusion matriks
print(metrics.confusion_matrix(y_test, prediction, labels=
[0, 1]))
#%%
#membuat kurva ROC
metrics.plot_roc_curve(mymodel, X_test, y_test)
#----END----