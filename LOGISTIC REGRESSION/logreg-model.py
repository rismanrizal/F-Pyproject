#%%
#running regresi logistik hanya sampai pembentukan model saja
from operator import index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy as rp
import statsmodels.api as sm
# %%
#import data
mydata = pd.read_csv("datafinal.csv")
mydata.head (4)
# %%
#jumlah pengamatan
print("#jumlah emiten:" + str(len(mydata)))
# %%
#melihat proporsi dari DV
sns.countplot(x="ca", data=mydata)
rp.summary_cat(mydata['ca'])
# %%
#cek karakteristik data
mydata.info()
# %%
#cek apakah ada data NaN
mydata.isnull()
mydata.isnull().sum()
# %%
#memvisualisasikan NaN data dalam bentuk plot
sns.heatmap(mydata.isnull(), yticklabels=False)
#%%
#membangun model regresi logistik (import module)
import statsmodels.formula.api as smf
#%%
#membangun model regresi logistik
modelrl = smf.logit("ca ~ price + marketcap + tlod + ir + gc + reg", data=mydata).fit()
#%%
#model summary
modelrl.summary()
#----END----