#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:23:59 2026

@author: atharvaaher
"""

import pandas as pd
from pathlib import Path

# This automatically finds your "home" directory (C:\Users\atharvaaher or /Users/atharvaaher)
base_path = Path.home()
full_path = base_path / "Desktop/data analytics projects/cost_of_living/Cost_of_Living_Index_by_Country_2024.csv"

data = pd.read_csv(full_path)


df=pd.DataFrame(data)
df.sort_values(by=["Country"],ascending=True,inplace=True)
df.reset_index(inplace=True)
df.dropna()
df.info()
df.drop(["Cost of Living Plus Rent Index","index"],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["Country"] = lb.fit_transform(df.Country)

import statsmodels.api as sm
fig = sm.qqplot(df["Cost of Living Index"], line='45')
# fig = sm.qqplot(df.Country, line='45')

import matplotlib.pyplot as plt
plt.hist(df["Cost of Living Index"], edgecolor='black', bins=10)

import numpy as np
data_log = np.sqrt(df["Cost of Living Index"])
plt.hist(data_log, edgecolor='black')

def min_max_normalization(column):
    return (column - column.min()) / (column.max() - column.min())
normalized_column = min_max_normalization(df["Cost of Living Index"])
df["Cost of Living Index new"] = normalized_column
df["Cost of Living Index"]

plt.scatter(df["Cost of Living Index"],df["Country"])
plt.xlabel("cost of living")
plt.ylabel("country code")
x=df.iloc[:,:-1]
y=df.iloc[:,2]

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
 
lr.fit(x_train,y_train)
lr.score(x_train,y_train)
lr.score(x_test,y_test)
lr.predict(x_test)
