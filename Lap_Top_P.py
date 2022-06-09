#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:20:21 2022

@author: venky
"""

################## Lap top price prediction #############

import pandas as pd ## Data manipulation
import numpy as np ## basic mathematical calculations

## Load the data

laptop = pd.read_csv("/home/venky/Desktop/Datascience_360/Laptop_Price_Project/Laptop_Price_Prediction-main/laptop_data.csv")

## Data understanding
laptop.shape ## (1303, 12)
laptop.info() ## inforamtion about the null, data type and meomory
laptop.describe() ## statistical information
laptop.columns
#['Unnamed: 0', 'Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']

laptop.drop(['Unnamed: 0'], axis = 1, inplace = True)
laptop.shape  ## (1303, 11)

## Data cleaning

laptop.duplicated().sum()  ## no duplicates
laptop.isna().sum()  ## no null values

## remove GB from RAM column, KG form Weight column and convert columns into numeric
laptop['Ram'] = laptop['Ram'].str.replace("GB", "")
laptop['Weight'] = laptop['Weight'].str.replace("kg", "")
laptop['Ram'] = laptop['Ram'].astype('int32')
#laptop['Weight'] = laptop['Weight'].astype('float32')

## EDA

import matplotlib.pyplot as plt
import seaborn as sns

## target column
sns.distplot(laptop['Price'])
## right skewed, low price laptops are sold more than branded laptops

## company vs brand

sns.barplot(x = laptop['Company'], y = laptop['Price'])
plt.xticks(rotation = "vertical")

## Typename vs price

sns.barplot(x = laptop['TypeName'], y = laptop['Price'])
plt.xticks(rotation = "vertical")

## Size vs Price

sns.barplot(x = laptop['Inches'], y = laptop['Price'])


## Feature engineering

## Screen resolution has too much data, so wen can exstract into different columns

## exstract touch screen information
laptop['Touchscreen'] = laptop['ScreenResolution'].apply(lambda x : 1 if 'Touchscreen' in x else 0)

## check howmany laptops are touchscreen
sns.countplot(laptop['Touchscreen'])

## touch screen vs Price
sns.boxplot(x = laptop['Touchscreen'], y = laptop['Price'])

## extract IPS 
laptop['Ips'] = laptop['ScreenResolution'].apply(lambda x : 1 if 'Ips' in x else 0)

sns.barplot(x = laptop['Ips'], y = laptop['Price'])

## exstract screen resolution(dimension (x and y axis))
def findXresolution(s):
    return s.split()[-1].split("x")[0]
def findYresolution(s):
    return s.split()[-1].split("x")[1]

## finding the x_res and y_res from screen resolution
laptop['X_res'] = laptop['ScreenResolution'].apply(lambda x : findXresolution(x))
laptop['Y_res'] = laptop['ScreenResolution'].apply(lambda y : findYresolution(y))

## convert to numeric
laptop['X_res'] = laptop['X_res'].astype('int')
laptop['Y_res'] = laptop['Y_res'].astype('int')

## Replacing X and Y resolution into PPI
laptop['ppi'] = (((laptop['X_res']**2) + (laptop['Y_res']))**0.5/laptop['Inches']).astype('float')

laptop.corr()['Price']

laptop.drop(['ScreenResolution', 'Inches', 'X_res', 'Y_res'], axis = 1, inplace = True)


## Exstract the information from the CPU column

def fetch_processor(x):
    cpu_name = " ".join(x.split()[0:3])
    if cpu_name == 'Intel Core i7' or cpu_name == 'Intel Core i5' or cpu_name == 'Intel Core i3':
        return cpu_name
    elif cpu_name.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'
laptop['Cpu_brand'] = laptop['Cpu'].apply(lambda x : fetch_processor(x))

## price vs processors

sns.barplot( x = laptop['Cpu_brand'], y = laptop['Price'])
plt.xticks(rotation = 'vertical')

## price vs Ram
sns.barplot(x = laptop['Ram'], y = laptop['Price'])

### Memory column

laptop['Memory'] = laptop['Memory'].astype(str).replace('\.0', '', regex=True)
laptop["Memory"] = laptop["Memory"].str.replace('GB', '')
laptop["Memory"] = laptop["Memory"].str.replace('TB', '000')
new2 = laptop["Memory"].str.split("+", n = 1, expand = True)
laptop["first"]= new2[0]
laptop["first"]=laptop["first"].str.strip()
laptop["second"]= new2[1]
laptop["Layer1HDD"] = laptop["first"].apply(lambda x: 1 if "HDD" in x else 0)
laptop["Layer1SSD"] = laptop["first"].apply(lambda x: 1 if "SSD" in x else 0)
laptop["Layer1Hybrid"] = laptop["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
laptop["Layer1Flash_Storage"] = laptop["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
laptop['first'] = laptop['first'].str.replace(r'\D', '')
laptop["second"].fillna("0", inplace = True)
laptop["Layer2HDD"] = laptop["second"].apply(lambda x: 1 if "HDD" in x else 0)
laptop["Layer2SSD"] = laptop["second"].apply(lambda x: 1 if "SSD" in x else 0)
laptop["Layer2Hybrid"] = laptop["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
laptop["Layer2Flash_Storage"] = laptop["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)
laptop['second'] = laptop['second'].str.replace(r'\D', '')
laptop["first"] = laptop["first"].astype(int)
laptop["second"] = laptop["second"].astype(int)
laptop["Total_Memory"]=(laptop["first"]*(laptop["Layer1HDD"]+laptop["Layer1SSD"]+laptop["Layer1Hybrid"]+laptop["Layer1Flash_Storage"])+laptop["second"]*(laptop["Layer2HDD"]+laptop["Layer2SSD"]+laptop["Layer2Hybrid"]+laptop["Layer2Flash_Storage"]))
laptop["Memory"]=laptop["Total_Memory"]
laptop["HDD"]=(laptop["first"]*laptop["Layer1HDD"]+laptop["second"]*laptop["Layer2HDD"])
laptop["SSD"]=(laptop["first"]*laptop["Layer1SSD"]+laptop["second"]*laptop["Layer2SSD"])
laptop["Hybrid"]=(laptop["first"]*laptop["Layer1Hybrid"]+laptop["second"]*laptop["Layer2Hybrid"])
laptop["Flash_Storage"]=(laptop["first"]*laptop["Layer1Flash_Storage"]+laptop["second"]*laptop["Layer2Flash_Storage"])


laptop['Gpu_brand'] = laptop['Gpu'].apply(lambda x:x.split()[0])
#there is only 1 row of ARM GPU so remove it
laptop = laptop[laptop['Gpu_brand'] != 'ARM']


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
laptop['os'] = laptop['OpSys'].apply(cat_os)


sns.barplot(x = laptop['os'], y = laptop['Price'])
plt.xticks(rotation='vertical')
plt.show()

laptop.drop(['Cpu', 'Gpu', 'OpSys'], axis = 1, inplace = True)

laptop=laptop.drop(['first','second','Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage','Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage','Total_Memory'],axis=1)


sns.distplot(np.log(laptop['Price']))
plt.show()


objList = laptop.select_dtypes(include = "object").columns
cols = ['Company', 'TypeName', 'Weight', 'Cpu_brand', 'Gpu_brand', 'os']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in cols:
    laptop[i] = le.fit_transform(laptop[i].astype(str))


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

laptop1 = norm_func(laptop)

laptop1.drop(['Ips'], axis = 1 , inplace = True)


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


## Split the train and test
X = laptop1.drop(columns=['Price'])
y = laptop1['Price']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=2)


## Randomforest Regression
## pipe line 

step1 = RandomForestRegressor(n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)

pipe = Pipeline([ ('step1',step1) ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))

print('MAE',mean_absolute_error(y_test,y_pred))


y_pred_train = pipe.predict(X_train)

print('R2 score',r2_score(y_train,y_pred_train))

print('MAE',mean_absolute_error(y_train,y_pred_train))


import pickle
laptop1.to_csv("df.csv", index=False)
pickle.dump(pipe,open('pipe.pkl','wb'))



## Multilinear Regression

from sklearn.linear_model import LinearRegression
model = LinearRegression() # intilize the model

model.fit(X_train, y_train) # Train model on x_train and y_train

y_pred_test = model.predict(X_test)

result_test = pd.DataFrame({'Actual':y_test, "Predicted": y_pred_test})
result_test.head(10)

## importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# predicting the accuracy score
score_test = r2_score(y_test, y_pred_test)

print('R2 score(test): ', score_test)
print('Mean squared error(test): ', mean_squared_error(y_test, y_pred_test))
print('Root Mean squared error(test): ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

y_pred_train = model.predict(X_train)

result_train = pd.DataFrame({'Actual':y_train, "Predicted": y_pred_train})
result_train.head(10)

# predicting the accuracy score
score_train = r2_score(y_train, y_pred_train)

print('R2 score(train): ', score_train)
print('Mean squared error(train): ', mean_squared_error(y_train, y_pred_train))
print('Root Mean squared error(train): ', np.sqrt(mean_squared_error(y_train, y_pred_train)))

plt.scatter(y_test, y_pred_test)



















