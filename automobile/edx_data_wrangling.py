import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pyplot 


file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

#Reading dataset from the URL and adding the related headers(Create a python list containing name of headers.)
headers=["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

#load data from web aadress. Set the parameter 'names' equel to python list 'headres'. 
df=pd.read_csv(file_path, header=None, names=headers)

#To see what the data looks like use head() method
df.head()

#Assign column headers to dataframe.
df.columns=headers
df.head(10)

#replace missing values with python's default missing value marker(For reasons of computational speed and convenience)
df.replace('?',np.nan, inplace=True)
df.head(10)

#To identify the missing values(output will be boolean values)
missing_data=df.isnull()
missing_data.head(10)

#To count the missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

#Dealing with missing values.
#Eaither drop the data(Drop the whole row or colunm) or replace the data(Replace by mean,frequency or based on other functions)
##Calculating mean values and replacing nan with mean values.
avg_norm_loss=df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan,avg_norm_loss,inplace=True)
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
df['bore'].replace(np.nan,avg_bore,inplace=True)
avg_stroke=df["stroke"].astype('float').mean(axis=0)
print("Average of stroke:",avg_stroke)
df["stroke"].replace(np.nan,avg_stroke,inplace=True)
avg_horsepower=df["horsepower"].astype('float').mean(axis=0)
print("Average horsepower", avg_horsepower)
df['horsepower'].replace(np.nan,avg_horsepower,inplace=True)
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm", avg_peakrpm)
df['peak-rpm'].replace(np.nan,avg_peakrpm,inplace=True)
df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()

#Replacing with most frequent value
df['num-of-doors'].replace(np.nan,"four",inplace=True)

#Droping colunm
df.dropna(subset=['price'],axis=0,inplace=True)

#Reseting index
df.reset_index(drop=True,inplace=True)
df.head()

#to check the data types for each colunm. 
df.dtypes

#Convertong datatypes to proper formats
df[["bore","stroke"]]=df[["bore","stroke"]].astype("float")
df[["normalized-losses"]]=df[["normalized-losses"]].astype("int")
df[["price"]]=df[["price"]].astype("float")
df[["peak-rpm"]]=df[["peak-rpm"]].astype("float")
df.dtypes
df.head(5)

#Data standardization (transsforming data to a common formatnto make meaningful comparison)
df['city-L/100km']=235/df["city-mpg"]
df.head()
df["highway-L/100km"]=235/df['highway-mpg']
df.rename(columns={'"highway-mpg"':'highway-L/100km'},inplace=True)
df.head(10)

#Data normalization (transforming values in to a similar range)
#Scaling the variables so the variable avg is 0, so the variance is 1, or the variable values ranges from 0 to 1
## replace (original value) by (original value)/(maximum value)
df['length']=df['length']/df['length'].max()
df['width']=df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()
df[["length",'width','height']].head()

#Binning(The ptocess of transforming cts numerical variables into discrete categotical bins for grouped analysis)
##converting data to correct format
df['horsepower']=df['horsepower'].astype(int,copy=True)

#plotting histogram to see the distribution of horsepower
plt.pyplot.hist(df['horsepower'])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# TO find 3 bins of equel size bandwidth using linespace 
bins=np.linspace(min(df["horsepower"]),max(df["horsepower"]),4)
group_names=['Low','Medium','High']

##apply function cut to determine what each value belongs to
df['horsepower-binned']=pd.cut(df['horsepower'],bins,labels=group_names,include_lowest=True)
df[['horsepower','horsepower-binned']].head(20)

#number of vehicle in each bin
df['horsepower-binned'].value_counts()

#plot the distribution of each bin
plt.bar(group_names,df['horsepower-binned'].value_counts())
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

#bin visualization
plt.hist(df["horsepower"],bins=3)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

#indicator or dummy variables(used to label categories)
#we use indicator variables so we can use categorical variables for analysis
df.columns
dummy_variable_1=pd.get_dummies(df['fuel-type'])
dummy_variable_1.head()

#change colunm names for clarity
dummy_variable_1.rename(columns={'gas':'fuel-type-gas','diesel':'fuel-type-diesel'},inplace=True)
dummy_variable_1.head()

# merge data frame "df" and "dummy_variable_1"
df=pd.concat([df,dummy_variable_1],axis=1)

# drop original column "fuel-type" from "df"
df.drop('fuel-type',axis=1,inplace=True)
df.head()
dummy_variable_2=pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'},inplace=True)
dummy_variable_2.head()
df=pd.concat([df,dummy_variable_2],axis=1)
df.drop("aspiration",axis=1,inplace=True)
df.to_csv("clean_df.csv")
