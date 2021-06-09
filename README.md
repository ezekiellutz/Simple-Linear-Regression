# Simple Linear Regression Model: Train/Test Split Method
A linear regression model that accurately picks the best independent variable to predict vehicle CO2 emissions.

# Installation Instructions
Download both the .py and .csv files and place them in the same directory on the PC. Run the code from your IDE of choice.

It is recommended that this script be run in Sublime Text 3.2.2. 

# Description
This python script illustrates how to use linear regression models to determine linear relationships between two variables. To illustrate this, new vehicle data was used from the Natural Resources Canada website (https://www.nrcan.gc.ca/). This website provides .csv files of new vehicle information for a variety of model years. In this case, data collected for 2021 model year vehicles was used. However, the code can be easily modified to analyze data collected for other model years as well.

Once the .csv file has been downloaded and placed in the same directory as the .py file, the code can be ran to produce graphs. 

The python script will first produce a histogram of the entire data set for the user to see. This part of the code acts as a visual aid to help the user visualize the entire data set more easily.


   ![Figure_1](https://user-images.githubusercontent.com/83550613/119499749-a2a70e80-bd2c-11eb-89c4-2328489861e9.png)


After this histogram is produced, the script will begin building a model for each parameter of interest, plotted against CO2 emissions, to see which parameter has the strongest linear relationship with the CO2 emissions. In this case, the three parameters of interest are engine size, number of cylinders, and fuel consumption. 

For each parameter of interest the script will:

	1.) Create a scatter plot of the parameter of interest against CO2 emissions.
	2.) Create a linear regression of the model.
	3.) Train this model using randomly selected data points in the data set (80%). 
	4.) Loop through the model 10,000 times to generate 10,000 values for the slope (theta 0) and intercept (theta 1) of the best fit line.
	5.) Find the the average for theta 0 and theta 1 from the 10,000 models to find the parameters of the best fit line for the data.
	6.) Plot the best fit against the scatter plot of the parameter of interest against CO2 emissions. 
	7.) Test the model using randomly selected data points in the data set (20%).
	8.) Loop through the model 10,000 times to generate 10,000 values for x, y, and y_.
	9.) Find the average for x, y, and y_ from the 10,000 models. 
	10.) Use the average values for x, y, and y_ to find the average mean absolute error (MAE), residual sum of squares (MSE), and an R squared score.

Here are the scatter plots for each parameter of interest:


   ![Figure_2](https://user-images.githubusercontent.com/83550613/119589554-a5d7e400-bd98-11eb-8a8e-2d1a17aa3d98.png)
	 
	 
   ![Figure_3](https://user-images.githubusercontent.com/83550613/119589582-b12b0f80-bd98-11eb-8b89-dbcb545c959b.png)
	 
	 
   ![Figure_4](https://user-images.githubusercontent.com/83550613/119589587-b2f4d300-bd98-11eb-87a2-035b1b9c2d48.png)
	 
	 
   ![Figure_5](https://user-images.githubusercontent.com/83550613/119589592-b4be9680-bd98-11eb-8478-99d7e2344951.png)
	 
	 
   ![Figure_6](https://user-images.githubusercontent.com/83550613/119589593-b6885a00-bd98-11eb-8e31-7d783decf391.png)
	 
	 
   ![Figure_7](https://user-images.githubusercontent.com/83550613/119589597-b8eab400-bd98-11eb-8a21-4582ff054e6b.png)


Once the script has performed the above actions for each parameter of interest, it will then perform a comparison of the R squared score for each model. The model with the highest R squared score will be selected as having the strongest linear relationship with CO2 emissions. The script will then instruct the user as to which parameter of interest had the strongest linear relationship with CO2 emissions and print out the R squared score for that model. 

Here is an example of what the script will print to the console:

      The strongest linear relationship exists between Fuel Consumption and CO2 Emissions for vehicles produced in 2021.

      The linear regression model developed here found an R squared score of 0.958100128891766.

      This means that the model created here explains 95.81% of the variation in the response variable (CO2 Emissions) around its mean.

      Therefore, using Fuel Consumption to predict the CO2 Emissions for a newly manufactured car will produce the most accurate results. 

      [Finished in 311.3s]
      
# Why is the R² Score Important?

Good question!

The R squared score, often referred to as the coefficient of determination, is a statistical measure of how close the data fits the fitted regression line. While it is ordinarily used as a metric to quantify the percentage of response variable variation that is explained by a given linear model, it can also be used as a way to determine how linear the relationship between your dependent and independent variable is. 

For example:

![example](https://user-images.githubusercontent.com/83550613/121426750-8c5a9e80-c939-11eb-867b-84dea00d8c50.jpg)

In the figure above, the plot on the left has an R squared score of 0.38 or 38%. Conversely, the plot on the right has an R squared score of 0.87 or 87%. Because both of these models are created using the exact same algorithm, the R squared score here can be used to indicate how strongly linear the relationship between the dependent and independent variable is. This is possible because the model will always try to maximize the R squared score in order to make the model as accurate as possible. If the model is unable to obtain an acceptable R squared score, it is reasonable to assume that a strong linear relationship between the dependent and independent variable does not exist. 

It is extremely important in linear regression modeling to have a strong linear relationship between the independent variable(s) and the dependent variable. This is especially true with multiple linear regression, where it is extremely important to use the correct independent variables (as well as the correct amount) to create an accurate model that is not overfit. 

In summary, the script created here can assist in picking the best dependent variables to use when building a multiple linear regression model. To see this concept in action, check out : https://github.com/ezekiellutz/Multiple-Linear-Regression

# Out-of-Sample Accuracy

A common issue that can be encountered when building linear regression models is high training accuracy and low out-of-sample accuracy. When a model has an overly high training accuracy it may overfit the data. When overfitting of the data occurrs, the results of the linear regression model will be high for the data set used to build the model, but relatively low when a different (but still applicable) data set is used. 

The model built here uses the train/test split approach, which avoids this issue by preventing the testing set from being part of the training set, and vice-versa. To measure the out-of-sample accuracy of the model, the R squared score is used. The R squared score will be a value between 0 and 1, with a higher R squared score generally indicating a model that accurately fits the data. In most cases, the higher the R squared score, the better. 

This model was tested with three different data sets, each one available from the website mentioned above. (https://www.nrcan.gc.ca/)

For the 2021 Model Year, the linear regression model found the strongest linear relationship between fuel consumption and CO2 emissions. On average its R squared value was 0.95.

For the 2020 Model Year, the linear regression model found the strongest linear relationship between fuel consumption and CO2 emissions. On average its R squared value was 0.92.

For the 2019 Model Year, the linear regression model found the strongest linear relationship between fuel consumption and CO2 emissions. On average its R squared value was 0.88.
