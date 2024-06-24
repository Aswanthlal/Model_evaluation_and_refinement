import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

df = pd.read_csv(filepath)
df.head()

#load module for linearregression
from sklearn.linear_model import LinearRegression
#creating linear regression object
#simple linear regression
lm = LinearRegression()
lm
#predictor and response variable
X = df[['highway-mpg']]
Y = df['price']
#fit the line
lm.fit(X,Y)

Yhat=lm.predict(X)
Yhat[0:5]
#value of intercept and slope
lm.intercept_
lm.coef_
#Yhat=a+bx
#price=38423.1-821.7*highway-mpg

#creating another object
lm1=LinearRegression()
lm1
lm1.fit(df[['engine-size']], df[['price']])
lm1
lm1.coef_
lm1.intercept_
Yhat=-7963.34 + 166.86*X
Price=-7963.34 + 166.86*df['engine-size']

#multiple linear regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#fit the model
lm.fit(Z, df['price'])
#intercept and coefficients
lm.intercept_
lm.coef_
#Yhat=a+ b1x1 + b2x2 + b3x3 +b4x4
#price=-15678.742628061467+52.65851272*horsepower+4.69878948*curb-weight+81.95906216*engine-size+33.58258185*highway-mpg

#creating and training another model
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])
lm2.coef_

#model Evaluation using Visualization
import seaborn as sns
#an excellent way to visualize the fit of simple regression model is by using regression plots.
#which shows a combination of scatterplot and linear regression line
##highway-mpg as predictor
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.show()
#price is negatively correlated to highway-mpg since regression slope is negative

#regression plot of peal-rpm
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.show()
#comparing both regression plots 
#the points for highway-mpg are mnuch closer to the generated line.
df[["peak-rpm","highway-mpg","price"]].corr

#Residual plot(residual is the difference bw observed and predicted value.)
#in regression plot residal is the distance from the data point to the fitted regression line
#residual plot shows residuals on y-axis and independent variable on x-axis
#if the points are randomly spread around the x axis then the linear model is appropriate for the data
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()
#residuals are not randomly spread around the x axis
#maybe a non linear model is appropriate for this data 


#multiple linear regression
#cant visualize it effectively with regression or residual plot
#one way to lokk at it is by the distribution plot and compare with actual value
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()
#fitted values are reasonably close to the actual value since the two distributions overlap


#polynomial regression and pipelines
#we get non linear relations bysetting higher order terms pf the predictor variables

#function to plot the data
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

#variables    
x = df['highway-mpg']
y = df['price']

#fitting the model using polyfit and polynomial of 3rd order
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

#plotting 
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)
#this polynomial model performs better than the linear model

#creating a 11 order polynomial
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')

#performig polynomial transform on multiple features
from sklearn.preprocessing import PolynomialFeatures

#creating a polynomial feature
pr=PolynomialFeatures(degree=2)
pr
Z_pr=pr.fit_transform(Z)
Z.shape
#in original data there are 201 samples and 4 features
Z_pr.shape
#after transformation 201 samples and 15 features


#pipeline
#data pipelines simplify the steps of preprocessing data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#create the pipeline bycreating a list of tuples including the name of model estimator and corresponding constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

#input list as an argument to the pipeline constructor
pipe=Pipeline(Input)
pipe

#convert z to float to avoid conversation warnings
Z = Z.astype(float)

#normalizing and fitting
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]

#another pipeline features z and target y
Input=[('scale',StandardScaler()),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]

#Measure for in sample evaluation

#model1 simple LR 
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2(measure to indicate how close the data is to the fitted regression line)
print('The R-square is: ', lm.score(X, Y))
#~49.659% of variation of price is explained by this model
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

#MSE
from sklearn.metrics import mean_squared_error
#mse is the measure of difference bw actual value and estimated value
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#model 2 multiple linear regression
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
#~80.896% of the variation is explainded by this model

Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

#model 3 polynomial fit
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
mean_squared_error(df['price'], p(x))

#Prediction and decision making

#create a new input
new_input=np.arange(1, 100, 1).reshape(-1, 1)
#fit the model
lm.fit(X, Y)
lm
#prediction
yhat=lm.predict(new_input)
yhat[0:5]
#ploting data
plt.plot(new_input, yhat)
plt.show()


# Better R squared value is better fit for the data.
# Small MSE is a better fit for the data.

# SLR vs MLR 
# R-Squared in combining with MSE show that MLR seems like a better model fit incase compare to SLR. 

# SLR vs Polynomial fit 
# polynomial fit resulted in a lower MSE and Higher R-squared, we can conclude that this was a better fit model.

# MLR vs polynomial fit 
# MSE for MLR smaller than MSE for polynomial fit
# R-squared for MLR is much larger than for the Polynomial fit
    

# COnclusion
# MLR model is the best model to predict price from our dataset. This result makes sense since we have 27 variables
# in total and we know that more than one of those variables are potential predictors of the final car price 