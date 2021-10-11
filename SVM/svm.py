# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
# %%
data = pd.read_csv("abalone.data")
data.head(5)
# %%
data['sex'] = data['sex'].astype('category')
# %%
data.info()
# %%
sns.pairplot(data,hue='sex',palette='Dark2')
# %%
X = data.drop(['sex'], axis=1)
y = data['sex']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=88) 
# %%
model = svm.SVC()
# %%
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
# %%
print("Akurasi:", metrics.accuracy_score(y_test, y_pred))
# %%
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %%
# tuning model
#applying Gridsearchcv to find the best model
parameters = [{'C': [0.1, 1,10,100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}]
grid = GridSearchCV(estimator= model,
                          param_grid = parameters, refit = True, scoring = 'accuracy',cv = 10)
grid.fit(X_train, y_train)
# %%
accuracy = grid.best_score_
print(accuracy)
# %%
