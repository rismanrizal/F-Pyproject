# NOTES: MODEL INI BUKAN MERUPAKAN MODEL TERBAIK
# NAMPAKNYA PERLU DILAKUKAN 'TUNING' MODEL 
# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
# %%
# DATA
data = pd.read_csv("class_cmc.data", sep=",")
data.head(6)
data.isnull().sum()
# %%
# MEMBAGI DATA MENJADI 'X' dan 'y'; DEPENDENT DAN INDEPENDENT VARIABEL
indepvar = ['age', 'education', 'child', 'religion', 'working', 'living']
X = data[indepvar]
y = data.c_method
# %%
# MEMBAGI DATA TRAIN DAN TEST 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)
# %%
# MEMBENTUK MODEL DECISION TREE
# INI BAGIAN KRUSIAL DALAM MENENTUKAN MODEL TERBAIK
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
mymodel = clf.fit(X_train, y_train) # membuat model dari train dataset
# %%
# MEMBUAT PREDIKSI ATAS TEST DATASET
y_pred = mymodel.predict(X_test)
# %%
# MODEL AKURASI
print("Akurasi:", metrics.accuracy_score(y_test, y_pred))
# %%
# VISUALISASI TREE MODEL
dot_data = StringIO()
export_graphviz(mymodel, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=indepvar, class_names=['1','2','3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('kontrasepsi.png')
Image(graph.create_png())
# %%
# CONFUSION MATRIX MODULE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# %%
confmatrix = confusion_matrix(y_test, y_pred)
confmatrix
# %%
# VISUALISASI CONFUSION MATRIX
plt.figure(figsize=(6,4))
sns.heatmap(confmatrix, annot=True, cmap='rainbow')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
# %%
# CLASSIFICATION REPORT
classreport = classification_report(y_test, y_pred, output_dict=True)
classreport
# %%
# MEMBUAT CLASSIFICATION REPORT DALAM BENTUK TABEL
tabelcr = pd.DataFrame(classreport).transpose()
tabelcr
# %%
# END