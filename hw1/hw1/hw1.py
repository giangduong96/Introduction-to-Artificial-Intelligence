# Name: Giang Duong
# CS 156 SPRING 2021
# Professor Sanjoy Paul
# Homework 1
import numpy as np
import pandas as pd
import sklearn as sk
import django as dj
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("1. Read a file csv into read_file")
read_file = pd.read_csv('Data-hw1.csv')
# create a frame
frame = pd.DataFrame(read_file)
print("Table value: \n", frame)
# all rows except the last row
X = frame.iloc[:, :-1].values
# the last row of dataframe
Y = frame.iloc[:, -1].values
Y = Y.reshape(-1, 1)
print("X array: \n", X)
print("y1 array: \n", Y)
# print(frame)

print("2. Eliminate missing data")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit_transform(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')
Y = imputer2.fit_transform(Y)
print(Y)

print("3. Convert categories data\n")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

print("4. split training set and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print("X_test", X_test)
print("X_train", X_train)
print("y_test", y_test)
print("y_train", y_train)

# drop NaN from train sets
print("A")
imputer.fit_transform(X_train)
imputer.fit(y_train)
# print(X_train)

print("5. Do linear regression based on years of experience")
year_ex = X_train[:, -1]
year_ex = year_ex.reshape(-1, 1)
year_ex_test = X_test[:, -1]
year_ex_test = year_ex_test.reshape(-1, 1)

# print(year_ex,y_train)
reg = LinearRegression()
reg.fit(year_ex, y_train)

# score = reg.score(year_ex,y_train)
# print(score)
print("6. Predict the test set result")
year_predit = reg.predict(year_ex_test)
print(year_predit)

print("Predict value for new data points (YearsExperience = 3.1 and YearsExperience = 7.0)")
pre3 = reg.predict([[3.1]])
pre7 = reg.predict([[7]])
print("Predit YearEx: 3.1 ",pre3)
print("Predit YearEx: 7.0 ",pre7)

print(
    "Part (B): Visualization of Data – Training set, Test set, Linear Regression Line, Predicted value for new data points     Use matplotlib to visualize:  Training set data & Linear Regression fit")
print("Training set data and linear regression fit ")
train_viz = plt
train_viz.scatter(year_ex, y_train, color='red')
train_viz.plot(year_ex, reg.predict(year_ex), color='blue')
train_viz.title('Year experience vs Salary on Training set')
train_viz.xlabel('Year of experience')
train_viz.ylabel('Salary')
train_viz.show()

print("Testing set data and linear regression fit ")
test_viz = plt
test_viz.scatter(year_ex_test, y_test, color='red')
test_viz.plot(year_ex_test, reg.predict(year_ex_test), color='blue')
test_viz.title('Year experience vs Salary on Testing set')
test_viz.xlabel('Year of experience')
test_viz.ylabel('Salary')
test_viz.show()

print("Plot predit value for new data points")
predit_vix =plt
predit_vix.scatter(year_ex_test, year_predit, color='red')
test_viz.title('Year experience vs Salary on predit test set')
test_viz.xlabel('Year of experience')
test_viz.ylabel('Salary')
test_viz.show()