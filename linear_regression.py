# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:11:43 2024

@author: sahil
"""

import pandas as pd
import numpy as np

wcat = pd.read_csv("C:/DS2/8_Linear_Regression/wc-at.csv")

#exploratory data analysis
#1.measure the central tendancy
#2.measures of dispersion
#3.Third moment business desicion
#4.Fourth moment bussiness decision
wcat.info()
wcat.describe()
#Graphical Representation 
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)
#data is right skewed
#scatter plot
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')
#direction: positive, linearity:moderate, Strength:poor
#now let us calculate correlation coefficient
np.corrcoef(wcat.Waist,wcat.AT)
#let us check the direction using covar factor
cov_output = np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output

#now let us apply to linear regression model
import statsmodels.formula.api as smf
#all machine learning algorithms are implemented using sklearn;
#but for this statsmodel
#package is being used because it gives you
#backed calculations of bita-0 and bita-1
model=smf.ols('AT~Waist',data=wcat).fit()
model.summary()
#OLS helps to find best fit model, which causes least square error
#first you check R squared value = 0.670, if R square= 0.8 means that model is 
#fit, if  R square=0.8 to 0.6 moderate fit.
#next you check P>|t|=0 it means less than alpha, alpha is 0.05, hence the model is accepted


#Regression Line
pred1=model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist,pred1,'r')
plt.show()

#error calculations
res1 = wcat.AT-pred1
np.mean(res1)

#it must be zero and here it 10^-14=~0
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1

#32.76.76 lesser the value better the model
#how to improve this model, transformation of 
plt.scatter(x=np.log(wcat['Waist']),y=wcat['AT'],color='brown')
np.correcoef(np.log(wcat.Waist),wcat.AT)
#r value is 0.82 < 0.85 hence moderate linearity
model2 = smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()
#again check the R square value of 0.67 whch is less than 0.8
#p value is 0 less than 0.05
pred2=model2.predict(pd.DataFrame(wcat['Waist']))
#check wcat and pred2 from variable explorer
#scatter diagram
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend(['predicated line','observed data'])
#error calculations
res2 = wcat.AT-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

#there no considerable changes
#now let us change y value instead of x
plt.scatter(x=wcat['Waist'],y=np.log(wcat['AT']),color='orange')
np.corrcoef(wcat.Waist,np.log(wcat.AT))
#r value is 0.84 < 0.85 hence moderate linearity
model3 = smf.ols('np.log(AT)~Waist',data = wcat).fit()
model3.summary()

#again check the R sqare value = 0.707 which is less than 0.8
#p value is 0.02 less than 0.05
pred3 = model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at=np.exp(pred3)

#check wcat and pred3_at from variable explorer
#scatter diagram

plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,'r')
plt.legend(['predicated line','observed data'])
plt.show()

#error calculations
res3 = wcat.AT-pred3_at
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3

#RMSE is 38.53 
#polynomial transformation
#x=Waist,x^2=Waist*Waist, y=log(at)
model4 = smf.ols('np.log(AT)~Waist+I(Waist*Waist)')