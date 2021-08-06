#%%
from io import IncrementalNewlineDecoder
from matplotlib import style
from numpy.core.fromnumeric import mean, std
import pandas as pd
import numpy as np
from pandas.core.series import Series
from sklearn import tree
from sklearn import metrics
# %%
#IMPOR DATASET
drug = pd.read_csv("drug200.csv")
drug.head(5)
# %%
#CEK MISSING VALUE
drug.isnull().sum()
# %%
#MEMISAHKAN INISIAL HURUF TERAKHIR PADA KOLOM OBAT
#membuatnya menjadi kolom baru
drug['Drugs'] = drug['Drug'].str[-1:]
drug.head(4)
#%%
#menghapus kolom 'Drug'
del drug['Drug']
#%%
#MELIHAT KARAKTERISTIK DATA
drug.info()
drug.head(4)
#%%
#mengubah type beberapa kolom pada dataframe
#object menjadi numeric
def Sex_to_numeric(x):
    if x=='F': return 2
    if x=='M': return 1
drug['Sex_d'] = drug['Sex'].apply(Sex_to_numeric)
#%%
#mengubah type beberapa kolom pada dataframe
#object menjadi numeric
def BP_to_numeric (y):
    if y=='HIGH': return 3
    if y=='NORMAL': return 2
    if y=='LOW': return 1
drug['BP_d'] = drug['BP'].apply(BP_to_numeric)
# %%
#mengubah type beberapa kolom pada dataframe
#object menjadi numeric
def Chol_to_numeric(z):
    if z=='HIGH': return 2
    if z=='NORMAL': return 1
drug['Chol_d'] = drug['Cholesterol'].apply(Chol_to_numeric)
#%%
#mengubah type beberapa kolom pada dataframe
#object menjadi numeric
def drug_to_numeric(a):
    if a=='A': return 1
    if a=='B': return 2
    if a=='C': return 3
    if a=='X': return 4
    if a=='Y': return 5
drug['drugscode'] = drug['Drugs'].apply(drug_to_numeric)
#atau untuk mengubah categorical variables with ordinal encoding, gunakan:
#import category_encoders as ce
#encoder = ce.OrdinalEncoder(cols=['column names'])
#X_train = encoder.fit_transform(X_train)
# %%
#mengecek terakhir kondisi data
drug.info()
drug.head(5)
#%%
#MEMBUAT DATAFRAME X DAN Y DENGAN MENGELUARKAN BEBERAPA VARIABEL
X = drug.drop(['Sex','BP','Drugs','Cholesterol','drugscode'],axis=1)
Y = drug['drugscode']
#%%
#MEMBAGI DATASET MENJADI TRAIN DAN TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=10)
#%%
#MEMBUAT MODEL RF CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=10)
#%%
#MENGEVALUASI MODEL
RF.fit(X_train, y_train)
#%%
#1 prediksi keakuratan model
prediksi = RF.predict(X_test)
RF.score(X_test, y_test)
print('Akurasi model: {0:0.4f}'. format(metrics.accuracy_score(y_test, prediksi)))
#%%
#2a confusion matriks
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, prediksi)
CM
#%%
#2b visualisasi confusion matriks
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(6,4))
sn.heatmap(CM, annot=True)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
#%%
#3 accuration report
from sklearn.metrics import classification_report
print(classification_report(y_test, prediksi))
#%%
#4a variabel importance
#menyimpan nama2 kolom X untuk digunakan pada visualisasi variabel importance
variabel_list = list(X.columns)
var_importance = pd.Series(RF.feature_importances_, index=X_train.columns).sort_values(ascending=False)
#atau bisa juga: var_importance = list(RF.feature_importances_)
#%%
#4b visualisasi variabel importance
import matplotlib.pyplot as plt
#%matplotlib inline 
plt.style.use('Solarize_Light2')
x_values = list(range(len(var_importance)))
plt.bar(x_values, var_importance, orientation = 'vertical')
plt.xticks(x_values, variabel_list, rotation = 'vertical')
plt.ylabel('Importance'); plt.xlabel('Variabel'); plt.title('Variabel Importance')
#%%
#5 visualisasi salah satu tree
import pydot
import graphviz
from sklearn import tree
treeplot = tree.plot_tree (RF.estimators_[0],feature_names=X.columns, filled=True)
# %%
#SIMULASI MEMBUAT SATU PENGAMATAN UNTUK PREDIKSI
uji = {'Age': [33],
        'Na_to_K': [10.1806],
        'Sex_d': [2],
        'BP_d':[1],
        'Chol_d':[2]
        }
ujicoba = pd.DataFrame(uji)
onesampel = RF.predict(ujicoba)
onesampel
# %%
#VALIDASI MODEL RF DENGAN CROSS VALIDATION
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=18)
n_scores = cross_val_score(RF, X, Y, scoring='accuracy', cv=CV, n_jobs=1, error_score='raise')
# %%
#evaluasi model setelah dilakukan validasi
from numpy import mean
from numpy import std
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#END