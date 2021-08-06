#%%
import pandas as pd
import numpy as np
# %%
humor = pd.read_csv('humor.csv')
humor.head(4)
# %%
#MEMBUAT PENGELOMPOKKAN KOLOM BERDASARKAN VARIABEL
AFF = pd.DataFrame (humor, columns = ['af1','af2','af3','af4','af5','af6','af7','af8'])
SEL = pd.DataFrame (humor, columns = ['se1','se2','se3','se4','se5','se6','se7','se8'])
AGG = pd.DataFrame (humor, columns = ['ag1','ag2','ag3','ag4','ag5','ag6','ag7','ag8'])
DEF = pd.DataFrame (humor, columns = ['sd1','sd2','sd3','sd4','sd5','sd6','sd7','sd8'])
# %%
import scipy.stats as stats
from factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
# %%
#MENGUJI PARAMETER STATISTIK VALIDITAS
#1 Tes Bartlett, misal untuk variabel AFF
bartlett_AFF, pvalue_AFF = calculate_bartlett_sphericity(DEF)
print(bartlett_AFF, pvalue_AFF)
# %%
#2 Tes KMO, misal untuk variabel SEL
KMO_SEL, KMO_modelSEL = calculate_kmo(SEL)
print(KMO_SEL, KMO_modelSEL)
# %%
#MENGUJI RELIABILITAS
#1 Alpha Cronbach, misal untuk variabel DEF
import pingouin as pg
pg.cronbach_alpha(data=DEF)
# END