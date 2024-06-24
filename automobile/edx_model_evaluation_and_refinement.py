import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

#importing dataset
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(filepath)

#use only numeric data
df = df._get_numeric_data()

#libraries for plotting
from ipywidgets import interact, interactive, fixed, interact_manual

#functions for plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    ax1 = sns.kdeplot(RedFunction, color='r', label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color='b', label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of cars')
    plt.show()
    plt.close()

def pollyplot(xtrain,xtest,y_train,y_test,lr,poly_transform):
    width=12
    height=10
    plt.figure(figsize=(width,height))


    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
    xmax=max([xtrain.values.max(),xtest.values.max()])
    xmin=min([xtrain.values.min(),xtest.values.min()])

    x=np.arange(xmin,xmax,0.1)

    plt.plot(xtrain,y_train,'ro',label='Training Data')
    plt.plot(xtest,y_test,'go',label='Test Data')
    plt.plot(x,lr.predict(poly_transform.fit_transform(x.reshape(-1,1))),label='predicted function')
    plt.ylim(-10000,60000)
    plt.ylabel('price')
    plt.legend()


#training and testing

y_data=df['price']
x_data=df.drop('price',axis=1)
from sklearn.model_selection import train_test_split

#splitting data into training and testing(10% for testing)
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.10,random_state=1)

print('number of test samples :',x_test.shape[0])
print('number of training samples:',x_train.shape[0])

#using 40%
x_train1,x_test1,y_train1,y_test1=train_test_split(x_data,y_data,test_size=0.4,random_state=0)
print('number of test samples:',x_test1.shape[0])
print('number of train samples:',x_train1.shape[0])

#importing
from sklearn.linear_model import LinearRegression

#creating a LR object, fitting the model and calculating R^2 
lre=LinearRegression()
lre.fit(x_train[['horsepower']],y_train)
lre.score(x_test[['horsepower']],y_test)
#r^2 is much smaller using the test data compared to training daata
lre.score(x_train[['horsepower']],y_train)

#R^2 on test data using 40% for testing
x_train1,x_test1,y_train1,y_test1=train_test_split(x_data,y_data,test_size=0.4,random_state=0)
lre.fit(x_train1[['horsepower']],y_train1)
lre.score(x_test1[['horsepower']],y_test1)


#Cross validation score
from sklearn.model_selection import cross_val_score
#input horsepower and trget data y_data, CV determine the number of folds
Rcross=cross_val_score(lre,x_data[['horsepower']],y_data,cv=4)
Rcross

#avg and std deviation
print('The mean of the folds are',Rcross.mean(), 'and the standard deviation is',Rcross.std())
#we can use negative squared error as a score
-1*cross_val_score(lre,x_data[['horsepower']],y_data,cv=4,scoring='neg_mean_squared_error')

#R^2 using  two fold
Rc=cross_val_score(lre,x_data[['horsepower']],y_data,cv=2)
Rc.mean()

#can also use cross_val_predict to predict the output
from sklearn.model_selection import cross_val_predict
yhat=cross_val_predict(lre,x_data[['horsepower']],y_data,cv=4)
yhat[0:5]


#over fitting under fitting and model selection
#Creating a multiple linear model and training the model
lr=LinearRegression()
lr.fit(x_train[['horsepower','curb-weight','engine-size','highway-mpg']],y_train)
#prediction using training data
yhat_train=lr.predict(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
yhat_train[0:5]

#prediction using test data
yhat_test=lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]

#model evaluation
import matplotlib.pyplot as plt
import seaborn as sns


#plot of predicted values using training data compared to the actual values of the training data
Title='Distribution plot of predicted value using training data vs training data distribution'
DistributionPlot(y_train,yhat_train,'Actual values(Train)','Predicted values (Train)',Title)
#the model seems to be doing well in learning from the training dataset
#when the model grnerates new values from testing datathe distribution of predicted value is much different from actual target values 


#Plot of predicted value using the test data compared to the actual values of the test data
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
#it is evident that the distribution of the test data in Figure 1 is much better at fitting the data. 
#This difference in Figure 2 is apparent in the range of 5000 to 15,000. This is where the shape of the distribution is extremely different.

#polynomial regression 
from sklearn.preprocessing import PolynomialFeatures

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.45,random_state=0)

pr=PolynomialFeatures(degree=5)
x_train_pr=pr.fit_transform(x_train[['horsepower']])
x_test_pr=pr.fit_transform(x_test[['horsepower']])
pr

#creating a regression model and training it
poly=LinearRegression()
poly.fit(x_train_pr,y_train)

yhat=poly.predict(x_test_pr)
yhat[0:5]

print('predicted values:',yhat[0:4])
print('true values:', y_test[0:4].values)

#previously defined function to display training data, testing data, and the predicted function
pollyplot(x_train['horsepower'],x_test['horsepower'],y_train,y_test,poly,pr)

#R^2 score
poly.score(x_train_pr,y_train)
poly.score(x_test_pr,y_test)
#the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87.  The lower the R^2, the worse the model. A negative R^2 is a sign of overfitting.

#R^2 changes on the test data for different order polynomials and then plot the results
Rsqu_test=[]
order=[1,2,3,4]
for n in order:
    pr=PolynomialFeatures(degree=n)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr,y_train)
    Rsqu_test.append(lr.score(x_test_pr,y_test))
plt.plot(order,Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 using test data')
plt.text(3,0.75,'maximum R^2')
plt.show()
#We see the R^2 gradually increases until an order three polynomial is used. 
#Then, the R^2 dramatically decreases at an order four polynomial

#The following function will be used in the next section
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)


#experiment with different polynomial orders and different amounts of data.
interact(f,order=(0,6,1),test_data=(0.05,0.95,0.05))
pr1=PolynomialFeatures(degree=2)

x_train_pr1=pr1.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
x_test_pr1=pr1.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])
x_train_pr1.shape

poly1=LinearRegression().fit(x_train_pr1,y_train)
yhat_test1=poly1.predict(x_test_pr1)
Title='Distribution plot of predicted value using test data vs distribution of test data'
DistributionPlot(y_test,yhat_test1,'Actual values (Test)','Predicted values(Test)',Title)
#The predicted value is higher than actual value for cars where the price $10,000 range, conversely the predicted price is lower than the price cost in the $30,000 to $40,000 range. As such the model is not as accurate in these ranges.

#Ridgr regression
#perform a degree two polynomial transformation 
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
#import ridge
from sklearn.linear_model import Ridge
#setting the regularization parameter (alpha) to 0.1
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train_pr,y_train)
yhat=RidgeModel.predict(x_test_pr)
print('predicted:',yhat[0:4])
print('test set:',y_test[0:4].values)

#We select the value of alpha that minimizes the test error. To do so, we can use a for loop.
#We have also created a progress bar to see how many iterations we have completed so far.
from tqdm import tqdm
Rsqu_test=[]
Rsqu_train=[]
dummy1=[]
Alpha=10*np.array(range(0,1000))
pbar=tqdm(Alpha)

for alpha in pbar:
    RidgeModel=Ridge(alpha=alpha)
    RidgeModel.fit(x_train_pr,y_train)
    test_score,train_score=RidgeModel.score(x_test_pr,y_test),RidgeModel.score(x_train_pr,y_train)

    pbar.set_postfix({'test score':test_score,'train score':train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

#plot output of R^2 for diff alphas
width=12
height=10
plt.figure(figsize=(width,height))
plt.plot(Alpha,Rsqu_test,label='Validation data')
plt.plot(Alpha,Rsqu_train,'r', label='training data')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()
#The blue line represents the R^2 of the validation data, and the red line represents the R^2 of the training data.
#The x-axis represents the different values of Alpha.
#training and testing data are same here
#The blue line represents the R^2 on the validation data. 
#As the value for alpha increases, the R^2 increases and converges at a point

RidgeModel=Ridge(alpha=10)
RidgeModel.fit(x_train_pr,y_train)
RidgeModel.score(x_test_pr,y_test)

#Grid seearch
from sklearn.model_selection import GridSearchCV
#create a dictionary of parameter values:
parameters1=[{'alpha':[0.001,0.1,1,10,100,1000,10000,100000,10000000]}]
#ridge regression object
RR=Ridge()
RR

#grid search object
Grid1=GridSearchCV(RR,parameters1,cv=4)
#fit the model
Grid1.fit(x_data[['horsepower','curb-weight','engine-size','highway-mpg']],y_data)
#The object finds the best parameter values on the validation data

#obtaining the estimator with the best parameters
BestRR=Grid1.best_estimator_
BestRR
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test)

# grid search for the alpha parameter and the normalization parameter
parameters2=[{'alpha':[0.001,0.1,1,10,100,1000,10000,100000,10000000]}]
Grid2=GridSearchCV(Ridge(),parameters2,cv=4)
Grid2.fit(x_data[['horsepower','curb-weight','engine-size','highway-mpg']],y_data)
best_alpha=Grid2.best_params_['alpha']
best_ridge_model=Ridge(alpha=best_alpha)
best_ridge_model.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
