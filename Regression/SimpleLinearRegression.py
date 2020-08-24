# Simple Linear Regression

# Importing the libraries
import numpy
import matplotlib.pyplot as matplotlib
import pandas

# Importing the dataset
dataset = pandas.read_csv('Salary_Data.csv')
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into training dataset and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Simple Linear Regression Model to training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualing the traning set results
matplotlib.scatter(X_train, Y_train, color = 'red')
matplotlib.plot(X_train, regressor.predict(X_train), color = 'blue')
matplotlib.title('Salary vs Experiance (Training Set)')
matplotlib.xlabel('Years of Experiance')
matplotlib.ylabel('Salary')
matplotlib.show()

# Visualing the test set results
matplotlib.scatter(X_test, Y_test, color = 'red')
matplotlib.plot(X_train, regressor.predict(X_train), color = 'blue')
matplotlib.title('Salary vs Experiance (Test Set)')
matplotlib.xlabel('Years of Experiance')
matplotlib.ylabel('Salary')
matplotlib.show()
