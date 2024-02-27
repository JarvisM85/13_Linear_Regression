# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:00:26 2024

@author: sahil
"""

import pandas as pd
import numpy as np
import seaborn as sns
cars = pd.read_csv("C:/DS2/8_Linear_Regression/Cars.csv")


cars.describe()


import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,82,1))
sns.distplot(cars.HP)
plt.boxplot(cars.HP)

sns.distplot(cars.MPG)
plt.boxplot(cars.MPG)

sns.distplot(cars.VOL)
plt.boxplot(cars.VOL)

sns.distplot(cars.SP)
plt.boxplot(cars.SP)

sns.distplot(cars.WT)
plt.boxplot(cars.WT)


import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])

plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])



### QQ PLOT
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist="norm",plot=pylab)
plt.show()


import seaborn as sns
sns.pairplot(cars.iloc[:,:])
# Linearity : #direction : #strength  :
cars.corr()


import statsmodels.formula.api as smf
ml1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()


#####
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
cars_new = cars.drop(cars.index[[76]])

ml_new = smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new


rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)
vif_hp

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt = 1/(1-rsq_wt)


rsq_vol = smf.ols('VOL~HP+WT+SP',data=cars).fit().rsquared
vif_vol = 1/(1-rsq_vol)

rsq_sp = smf.ols('SP~HP+WT+VOL',data=cars).fit().rsquared
vif_sp = 1/(1-rsq_sp)


d1 ={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame = pd.DataFrame(d1)
vif_frame

final_ml = smf.ols('MPG~VOL+SP+HP',data=cars).fit()
final_ml.summary()


pred = final_ml.predict(cars)

## QQ PLOT
res = final_ml.resid
sm.qqplot(res)
plt.show()

stats.probplot(res,dist="norm",plot=pylab)
plt.show()

sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt

from sklearn.model_selection import train_test_split
cars_train,cars_test = train_test_split(cars,test_size=0.2)

model_train = smf.ols('MPG')







