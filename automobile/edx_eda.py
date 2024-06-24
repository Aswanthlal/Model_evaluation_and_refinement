import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(filepath, header=0)

# Display the first few rows of the DataFrame
print(df.head())

# Check the data types of each column
print(df.dtypes)

# Check correlation between selected numerical features
print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Create regression plot for 'engine-size' vs 'price'
sns.regplot(x='engine-size', y='price', data=df)
plt.ylim(0,)
plt.show()
#Figure indicates positive direct correlation between these variables.
#since the regression line is almot perfect diagonal engine size seems like a good predictoe of price

# Compute and display correlation between 'engine-size' and 'price'
print(df[['engine-size', 'price']].corr())

# Create regression plot for 'highway-mpg' vs 'price'
sns.regplot(x='highway-mpg', y='price', data=df)
plt.show()
#indicates an inverse relationship

# Compute and display correlation between 'highway-mpg' and 'price'
print(df[['highway-mpg', 'price']].corr())

# Create regression plot for 'peak-rpm' vs 'price'
sns.regplot(x='peak-rpm', y='price', data=df)
plt.show()

# Compute and display correlation between 'peak-rpm' and 'price'
print(df[['peak-rpm', 'price']].corr())
#weak lineaar relationship, not a good predictor of price

# Create regression plot for 'stroke' vs 'price'
sns.regplot(x='stroke', y='price', data=df)
plt.show()


# Create boxplot for 'body-style' vs 'price'(boxplot is a good way to visualize categorical variables)
sns.boxplot(x='body-style', y='price', data=df)
plt.show()
#we can see a significant overlap bw diff body style categories, not a good predictor of price

# Create boxplot for 'engine-location' vs 'price'
sns.boxplot(x='engine-location', y='price', data=df)
plt.show()
#Distinct,  a potential good predictor of price

# Create boxplot for 'drive-wheels' vs 'price'
sns.boxplot(x='drive-wheels', y='price', data=df)
plt.show()
#potenially be a predictor of price

#Descriptive Statistical Analysys
# Display summary statistics for numerical columns
print(df.describe())

# Display summary statistics for object (categorical) columns
print(df.describe(include='object'))

#value counts
#convert the series to a dataframe
#save result and rename it
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

#rename index
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
#Engine location may not be a good predictor sice we only have 3 cars with rear engine.

# Group by 'drive-wheels' and 'body-style', compute mean price, and create pivot table
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

# Fill missing values with 0
grouped_pivot = grouped_pivot.fillna(0)

#grouping results
df_gptest2=df[['body-style','price']]
grouped_test_bodystyle=df_gptest2.groupby(['body-style'],as_index=False).mean()
grouped_test_bodystyle

# Create heatmap using to visualize the relationship bw body style vs price
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
#the default label conveys no useful informations

# Create heatmap with row and column labels
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()

# Compute and display correlation matrix for all numerical features
print(df.corr())#this will cause error since it include non numericals also
# Exclude non-numeric columns before calculating the correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute and display correlation matrix for numerical columns
print(numeric_df.corr())
from scipy import stats
#p value is the probability value that the correlation bw variable is statistically significant 
pearson_coef,p_value=stats.pearsonr(df['wheel-base'],df['price'])
print('The correlation coefficient is', pearson_coef, 'with a p-value of p =', p_value)
#since p value is <0.001 correlation is significant, although the linear relationship isn't extremely strong
pearson_coef, p_value=stats.pearsonr(df['horsepower'],df['price'])
print('The pearson correlation coefficient is', pearson_coef, 'with a p-value of p =', p_value)
#since p value is <0.001 correlation is significant, and the linear relationship is quote strong
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#since p value is <0.001 correlation is significant, although the linear relationship is moderately strong
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
#since p value is <0.001 correlation is significant, although the linear relationship is quite strong
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#since p value is <0.001 correlation is significant, although the linear relationship is quite strong
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#since p value is <0.001 correlation is significant, although the linear relationship is very strong
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 
#since p value is <0.001 correlation is significant, although the linear relationship is only moderate
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#since p value is <0.001 correlation is significant, although the linear relationship is negative and moderately strong
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value )
#since p value is <0.001 correlation is significant, although the linear relationship is negative and moderately strong 

#Conclusion : important Variables

# We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:
# Continuous numerical variables:

# Length
# Width
# Curb-weight 
# Engine-size
# Horsepower
# City-mpg
# Highway-mpg
# Wheel-base
# Bore

# Categorical variables:

# Drive-wheels
# As we now move into building machine learning models to automate our analysis, feeding the model with variables that meaningfully affect our target variable will improve our model's prediction performance.