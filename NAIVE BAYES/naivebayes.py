# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
# %%
data = pd.read_csv("haberman.data")
data.head(6)
# membuat dataframe variabel prediktor dan target
X = data.drop(['status'], axis=1)
y = data['status']
# %%
# membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=18)
# %%
# standardize variabel prediktor
scaller = StandardScaler()
X_train = scaller.fit_transform(X_train) # selalu pertama
X_test = scaller.transform(X_test)
# dapat juga menggunakan RobustScaler
# %%
# membuat model NB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# %%
# membuat prediksi
y_pred = classifier.predict(X_test)
# %%
# mengukur akurasi
akurasi = accuracy_score(y_test,y_pred)
confmatrix = confusion_matrix(y_test, y_pred)
print(akurasi)
print(confmatrix) 
# %%
# membuat confusion matriks dan heatmap
confmatrix_hm = pd.DataFrame(data=confmatrix)
sns.heatmap(confmatrix_hm, annot=True, fmt='d', cmap='YlGnBu')
# %%
# classification report
print(classification_report(y_test, y_pred))
# %%
# tuning model dengan menggunakan k-CV
scores = cross_val_score(classifier, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# %%
# menghitung rata-rata dari CV
print('Average cross-validation score: {:.4f}'.format(scores.mean()))
# ada improvement akurasi, maka model ini digunakan dibandingkan model original
# END #