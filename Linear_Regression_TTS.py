'''
Author:			Ezekiel Lutz
Contact Info:	xxZeke77xx@gmail.com
Time: 			14:00 UTC
Date:			05-22-2021
Data Source:	https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY2021%20Fuel%20Consumption%20Ratings.csv
IDE:		Sublime Text 3.2.2
'''

#imports all necessary libraries
import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl 
import numpy as np
from sklearn.metrics import r2_score
from sklearn import linear_model

#opens the .csv file with relevant vehicle information and places it in a dataframe (NOTE: .csv file must be saved in the same directory as this script)
with open ('MY2021 Fuel Consumption Ratings.csv','r') as csv_file:
	df = pd.read_csv(csv_file)
	df = df[['Engine Size','Cylinders','Fuel Consumption','CO2 Emissions']]

#formats the dataframe to only include cells with data
df = df.iloc[1:884]

#defines the datatype in each column of the dataframe
df = df.astype({'Engine Size': float, 'Cylinders': int, 'Fuel Consumption': float, 'CO2 Emissions': int})

#creates a histogram with all of the data that will be used to build the models
df.hist()
plt.suptitle('THE ENTIRE DATASET: VISUALIZED')
plt.show()

#creates a mask to select random rows of our data set for training/testing
#this model trains with 80% of the selected data and tests with the other 20%
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask] 


##### ENGINE SIZE VERSUS CO2 EMISSIONS #####


#produces a scatter plot of Engine Size against CO2 Emissions
plt.scatter(df[['Engine Size']], df[['CO2 Emissions']],  color='blue')
plt.title('DETERMINING THE LINEAR RELATIONSHIP FOR:\n Engine Size vs CO2 Emissions')
plt.xlabel('Engine Size (L)')
plt.ylabel('Emission (g/km)')
plt.show()

#selects the linear model we will use and defines our training variables
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Engine Size']])
train_y = np.asanyarray(train[['CO2 Emissions']])

#creates variables to hold the value of theta 0 and theta 1 when the model is run multiple times in the next section
eng_theta_0 = []
eng_theta_1 = []

#loops through the entire training process ten-thousand times and appends the values for theta 0 and theta 1 to a set
for i in range(10000):
	
	mask = np.random.rand(len(df)) < 0.8
	train = df[mask]
	train_x = np.asanyarray(train[['Engine Size']])
	train_y = np.asanyarray(train[['CO2 Emissions']])
	regression.fit (train_x, train_y) 
	eng_theta_0.append(float(regression.coef_))
	eng_theta_1.append(float(regression.intercept_))

#calculates the mean for theta 0
sum_eng_theta_0 = sum(eng_theta_0)
n_eng_theta_0 = len(eng_theta_0)
avg_eng_theta_0 = sum_eng_theta_0/n_eng_theta_0

#calculates the mean for theta 1
sum_eng_theta_1 = sum(eng_theta_1)
n_eng_theta_1 = len(eng_theta_1)
avg_eng_theta_1 = sum_eng_theta_1/n_eng_theta_1


#plots the fitted line to the model using the averaged values for theta 0 and theta 1
plt.scatter(df[['Engine Size']], df[['CO2 Emissions']],  color='blue')
plt.plot(train_x, avg_eng_theta_0*train_x + avg_eng_theta_1, '-r')
plt.title('THE BEST LINE FIT LINE FOR:\n Engine Size vs CO2 Emissions')
plt.xlabel('Engine Size (L)')
plt.ylabel('Emission (g/km)')
plt.show()

#creates variables to hold the values of x, y, and y_ when the model is run multiple times in the next section
eng_test_x = []
eng_test_y = []
eng_test_y_ = []

#loops through the entire training process ten-thousand times and append the values for x, y, and y_ to a set
for i in range(10000):

	x = np.asanyarray(test[['Engine Size']])
	y = np.asanyarray(test[['CO2 Emissions']])
	y_ = regression.predict(x)
	eng_test_x.append(x)
	eng_test_y.append(y)
	eng_test_y_.append(y_)

#calculates the mean for x
sum_eng_test_x = sum(eng_test_x)
n_eng_test_x = len(eng_test_x)
avg_eng_test_x = sum_eng_test_x/n_eng_test_x

#calculates the mean for y
sum_eng_test_y = sum(eng_test_y)
n_eng_test_y = len(eng_test_y)
avg_eng_test_y = sum_eng_test_y/n_eng_test_y

#calculates the mean for y_
sum_eng_test_y_ = sum(eng_test_y_)
n_eng_test_y_ = len(eng_test_y_)
avg_eng_test_y_ = sum_eng_test_y_/n_eng_test_y_

#calculates the Mean Absolute Error, Residual Sum of Squares, and R2 score for this linear relationship and stores them in variables
EvCO2_MAE = np.mean(np.absolute(avg_eng_test_y_ - avg_eng_test_y))
EvCO2_MSE = np.mean((avg_eng_test_y_ - avg_eng_test_y) ** 2)
EvCO2_R2 = r2_score(avg_eng_test_y , avg_eng_test_y_)

##### CYLINDERS VERSUS CO2 EMISSIONS #####


#produces a scatter plot of Cylinders against CO2 Emissions
plt.scatter(df[['Cylinders']], df[['CO2 Emissions']],  color='blue')
plt.title('DETERMINING THE LINEAR RELATIONSHIP FOR:\n Cylinders vs CO2 Emissions')
plt.xlabel('Cylinders (#)')
plt.ylabel('Emission (g/km)')
plt.show()

#selects the linear model we will use and defines our training variables
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Cylinders']])
train_y = np.asanyarray(train[['CO2 Emissions']])

#creates variables to hold the value of theta 0 and theta 1 when the model is run multiple times in the next section
cyc_theta_0 = []
cyc_theta_1 = []

#loops through the entire training process ten-thousand times and appends the values for theta 0 and theta 1 to a set
for i in range(10000):
	
	mask = np.random.rand(len(df)) < 0.8
	train = df[mask]
	train_x = np.asanyarray(train[['Cylinders']])
	train_y = np.asanyarray(train[['CO2 Emissions']])
	regression.fit (train_x, train_y) 
	cyc_theta_0.append(float(regression.coef_))
	cyc_theta_1.append(float(regression.intercept_))

#calculates the mean for theta 0
sum_cyc_theta_0 = sum(cyc_theta_0)
n_cyc_theta_0 = len(cyc_theta_0)
avg_cyc_theta_0 = sum_cyc_theta_0/n_cyc_theta_0

#calculates the mean for theta 1
sum_cyc_theta_1 = sum(cyc_theta_1)
n_cyc_theta_1 = len(cyc_theta_1)
avg_cyc_theta_1 = sum_cyc_theta_1/n_cyc_theta_1


#plots the fitted line to the model using the averaged values for theta 0 and theta 1
plt.scatter(df[['Cylinders']], df[['CO2 Emissions']],  color='blue')
plt.plot(train_x, avg_cyc_theta_0*train_x + avg_cyc_theta_1, '-r')
plt.title('THE BEST LINE FIT LINE FOR:\n Cylinders vs CO2 Emissions')
plt.xlabel('Cylinders (#)')
plt.ylabel('Emission (g/km)')
plt.show()

#creates variables to hold the values of x, y, and y_ when the model is run multiple times in the next section
cyc_test_x = []
cyc_test_y = []
cyc_test_y_ = []

#loops through the entire training process ten-thousand times and append the values for x, y, and y_ to a set
for i in range(10000):

	x = np.asanyarray(test[['Cylinders']])
	y = np.asanyarray(test[['CO2 Emissions']])
	y_ = regression.predict(x)
	cyc_test_x.append(x)
	cyc_test_y.append(y)
	cyc_test_y_.append(y_)

#calculates the mean for x
sum_cyc_test_x = sum(cyc_test_x)
n_cyc_test_x = len(cyc_test_x)
avg_cyc_test_x = sum_cyc_test_x/n_cyc_test_x

#calculates the mean for y
sum_cyc_test_y = sum(cyc_test_y)
n_cyc_test_y = len(cyc_test_y)
avg_cyc_test_y = sum_cyc_test_y/n_cyc_test_y

#calculates the mean for y_
sum_cyc_test_y_ = sum(cyc_test_y_)
n_cyc_test_y_ = len(cyc_test_y_)
avg_cyc_test_y_ = sum_cyc_test_y_/n_cyc_test_y_

#calculates the Mean Absolute Error, Residual Sum of Squares, and R2 score for this linear relationship and stores them in variables
CvCO2_MAE = np.mean(np.absolute(avg_cyc_test_y_ - avg_cyc_test_y))
CvCO2_MSE = np.mean((avg_cyc_test_y_ - avg_cyc_test_y) ** 2)
CvCO2_R2 = r2_score(avg_cyc_test_y , avg_cyc_test_y_)

##### FUEL CONSUMPTION VERSUS CO2 EMISSIONS #####


#produces a scatter plot of Fuel Consumption against CO2 Emissions
plt.scatter(df[['Fuel Consumption']], df[['CO2 Emissions']],  color='blue')
plt.title('DETERMINING THE LINEAR RELATIONSHIP FOR:\n Fuel Consumption vs CO2 Emissions')
plt.xlabel('Fuel Consumption (L/100km)')
plt.ylabel('Emission (g/km)')
plt.show()

#selects the linear model we will use and defines our training variables
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Fuel Consumption']])
train_y = np.asanyarray(train[['CO2 Emissions']])

#creates variables to hold the value of theta 0 and theta 1 when the model is run multiple times in the next section
fc_theta_0 = []
fc_theta_1 = []

#loops through the entire training process ten-thousand times and appends the values for theta 0 and theta 1 to a set
for i in range(10000):
	
	mask = np.random.rand(len(df)) < 0.8
	train = df[mask]
	train_x = np.asanyarray(train[['Fuel Consumption']])
	train_y = np.asanyarray(train[['CO2 Emissions']])
	regression.fit (train_x, train_y) 
	fc_theta_0.append(float(regression.coef_))
	fc_theta_1.append(float(regression.intercept_))

#calculates the mean for theta 0
sum_fc_theta_0 = sum(fc_theta_0)
n_fc_theta_0 = len(fc_theta_0)
avg_fc_theta_0 = sum_fc_theta_0/n_fc_theta_0

#calculates the mean for theta 1
sum_fc_theta_1 = sum(fc_theta_1)
n_fc_theta_1 = len(fc_theta_1)
avg_fc_theta_1 = sum_fc_theta_1/n_fc_theta_1


#plots the fitted line to the model using the averaged values for theta 0 and theta 1
plt.scatter(df[['Fuel Consumption']], df[['CO2 Emissions']],  color='blue')
plt.plot(train_x, avg_fc_theta_0*train_x + avg_fc_theta_1, '-r')
plt.title('THE BEST LINE FIT LINE FOR:\n Fuel Consumption vs CO2 Emissions')
plt.xlabel('Fuel Consumption (L/100km)')
plt.ylabel('Emission (g/km)')
plt.show()

#creates variables to hold the values of x, y, and y_ when the model is run multiple times in the next section
fc_test_x = []
fc_test_y = []
fc_test_y_ = []

#loops through the entire training process ten-thousand times and append the values for x, y, and y_ to a set
for i in range(10000):

	x = np.asanyarray(test[['Fuel Consumption']])
	y = np.asanyarray(test[['CO2 Emissions']])
	y_ = regression.predict(x)
	fc_test_x.append(x)
	fc_test_y.append(y)
	fc_test_y_.append(y_)

#calculates the mean for x
sum_fc_test_x = sum(fc_test_x)
n_fc_test_x = len(fc_test_x)
avg_fc_test_x = sum_fc_test_x/n_fc_test_x

#calculates the mean for y
sum_fc_test_y = sum(fc_test_y)
n_fc_test_y = len(fc_test_y)
avg_fc_test_y = sum_fc_test_y/n_fc_test_y

#calculates the mean for y_
sum_fc_test_y_ = sum(fc_test_y_)
n_fc_test_y_ = len(fc_test_y_)
avg_fc_test_y_ = sum_fc_test_y_/n_fc_test_y_

#calculates the Mean Absolute Error, Residual Sum of Squares, and R2 score for this linear relationship and stores them in variables
FvCO2_MAE = np.mean(np.absolute(avg_fc_test_y_ - avg_fc_test_y))
FvCO2_MSE = np.mean((avg_fc_test_y_ - avg_fc_test_y) ** 2)
FvCO2_R2 = r2_score(avg_fc_test_y , avg_fc_test_y_)

#determines the linear relationship with the highest R squared value and then instructs the user as to what that strongest relationship is.
if (EvCO2_R2 >= CvCO2_R2) and (EvCO2_R2 >= FvCO2_R2):
   largest = round(EvCO2_R2*100,2)
   print(f"""\
The strongest linear relationship exists between Engine Size and CO2 Emissions of vehicles produced in 2021.

The linear regression model developed here found an R squared value of {EvCO2_R2}.

This means that the model created here explains {largest}% of the variation in the response variable (CO2 Emissions) around its mean.

Therefore, using Engine Size to predict the CO2 Emissions for a newly manufactured car will produce the most accurate results. 
""")
elif (CvCO2_R2 >= EvCO2_R2) and (CvCO2_R2 >= FvCO2_R2):
   largest = round(CvCO2_R2*100,2)
   print(f"""\
The strongest linear relationship exists between Number of Cylinders and CO2 Emissions of vehicles produced in 2021.

The linear regression model developed here found an R squared value of {CvCO2_R2}.

This means that the model created here explains {largest}% of the variation in the response variable (CO2 Emissions) around its mean.

Therefore, using Number of Cylinders to predict the CO2 Emissions for a newly manufactured car will produce the most accurate results. 
""")

else:
   largest = round(FvCO2_R2*100,2)
   print(f"""\
The strongest linear relationship exists between Fuel Consumption and CO2 Emissions of vehicles produced in 2021.

The linear regression model developed here found an R squared value of {FvCO2_R2}.

This means that the model created here explains {largest}% of the variation in the response variable (CO2 Emissions) around its mean.

Therefore, using Fuel Consumption to predict the CO2 Emissions for a newly manufactured car will produce the most accurate results. 
""")

