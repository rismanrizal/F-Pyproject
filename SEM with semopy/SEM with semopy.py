#!/usr/bin/env python
# coding: utf-8

# # LIBRARY

# In[1]:


import semopy as sem
import pandas as pd
import numpy as np
import pingouin
import scipy as stats
import graphviz


# # IMPORT DATA

# In[2]:


mydatadf = pd.read_csv("edu.csv")
mydatadf.head(6)


# # UJI NORMALITAS (UNIVARIATE DAN MULTIVARIATE)

# In[3]:


# UNIVARIATE
# mengubah dataframe ke array
mydataarray = mydatadf.to_numpy()


# In[4]:


# mendefinisikan dan menghitung Mahalanobis distance
def mahalanobis(x=None, data=None, cov=None):
    x_mu = x - np.mean(mydatadf)
    if not cov:
        cov = np.cov(mydatadf.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


# In[5]:


# menambahkan kolom 'mahalanobis' pada dataframe
mydatadf ['mahalanobis'] = mahalanobis(x=mydatadf, 
data=mydatadf[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','y1','y2','y3']])


# In[6]:


# menghitung pvalue dan menambahkan kolom 'pvalue' pada dataframe
from scipy.stats import chi2
mydatadf ['pvalue'] = 1 - chi2.cdf(mydatadf['mahalanobis'], 14)
#angka terakhir adalah nilai df (k-1, jumlah variabel dikurangi 1)


# In[7]:


mydatadf.head(6).apply(lambda s: s.apply('{0:.3f}'.format))
# agar menampilkan 3 angka belakang decimal
# nilai pvalue kurang dari 0,01 menandakan tidak univariate normal


# In[8]:


# MULTIVARIATE
# mengembalikan df ke semula (tanpa mahalanobis dan pvalue)
mydata = mydatadf.drop(['mahalanobis', 'pvalue'], axis=1)


# In[9]:


from pingouin import multivariate_normality
multivariate_normality(mydata, alpha=0.05)


# # MENGUJI MULTIKOLINEARITAS

# In[12]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# membentuk df khusus independen variabel
IV = mydata[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']]

# membentuk df VIF 
vif_data = pd.DataFrame()
vif_data["Variabel"] = IV.columns

# calculating VIF for each variabel
vif_data["VIF"] = [variance_inflation_factor(IV.values, i)
                          for i in range(len(IV.columns))]
  
print(vif_data)


# # MEMBENTUK MODEL SEM

# In[13]:


# mendefinisikan model
mymodel = """
advising =~ x1 + x2 + x3 + x4
tutoring =~ x5 + x6 + x7 + x8
value =~ x9 + x10 + x11 + x12
satisfaction =~ y1 + y2 + y3
value ~ tutoring + advising
satisfaction ~ value + tutoring + advising
"""


# In[14]:


# membentuk model
modelsem = sem.Model(mymodel)


# In[15]:


# karena non-normality, coba digunakan UWLS sebagai metode estimasi
modelsem.fit(mydata, obj="DWLS", solver="SLSQP")


# In[16]:


# memprediksi factor loadings
factors = modelsem.predict_factors(mydata)
print(factors.head())


# In[17]:


# mengoptimasi model
opt = sem.Optimizer(modelsem)
obj = opt.optimize()


# In[18]:


from semopy.inspector import inspect
inspect(opt)


# In[19]:


# Model fit
stats = sem.calc_stats(modelsem)
print(stats.T)


# In[21]:


# membentuk grafik SEM
gg = sem.semplot(modelsem, filename = "semedugg.png")
gg


# In[23]:


# Create Report
from semopy import ModelMeans
from semopy import report
modelsem2 = ModelMeans(mymodel)
modelsem2.fit(mydata)
report(modelsem2, "Education Report SEM")


# # END
