# Importing Packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Preparing Data
def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out) #creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]) #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True) #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0) #cross validation

    response = [X_train,X_test , Y_train, Y_test , X_lately]
    return response

# Taking Input for Filename
fileName = input("Enter the file name to analyze (csv): ")

# Reading the file
df = pd.read_csv(fileName)

# Reading the symbol
stockName = input("Enter the stock to analyze: ")

# Filtering the Stock Prices
df = df[df.symbol == stockName]

# Choosing the open or close file
priceOption = input("Choose from the following:\n1. Open\n2. Close\n3. High\n4. Low\n\nYour Selection: ")

def processPriceOption(option):

    return 'close'

# Preparing Controller Variables
forecast_col = 'close'
forecast_out = 5
test_size = 0.2

# Initializing linear regression model
X_train, X_test, Y_train, Y_test , X_lately = prepare_data(df,forecast_col,forecast_out,test_size); #calling the method were the cross validation and data preperation is in
learner = LinearRegression() 

# Training the linear regression model
learner.fit(X_train,Y_train) 

# Testing the linear regression model
score=learner.score(X_test,Y_test)

# Set that will contain the forecasted data
forecast = learner.predict(X_lately) 

# Creating json object
response= {}
response['test_score']=score
response['forecast_set']=forecast

print(response)