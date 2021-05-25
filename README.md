# Simple Linear Regression Model: Train/Test Split Method
A linear model that accurately picks the best independent variable to predict vehicle CO2 emissions.

# Installation Instructions
Download both the .py and .csv files and place them in the same directory on the PC. Run the code from your IDE of choice.

It is recommended that this script be ran in Sublime Text 3.2.2. 

# Description
This python script illustrates how to use linear regression models to determine linear relationships between two variables. To illustrate this, new vehicle data was used from the Natural Resources Canada website (https://www.nrcan.gc.ca/). This website provides .csv files of new vehicle information for a variety of model years. In this case, data collected for 2021 model year vehicles was used. 

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

Once the script has performed the above actions for each parameter of interest, it will then perform a comparison of the R squared score for each model. The model with the highest R squared score will be selected as having the strongest linear relationship with CO2 emissions. The script will then instruct the user as to which parameter of interest had the strongest linear relationship with CO2 emissions and print out the R squared score for that model. 

      

