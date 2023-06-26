'''
TITLE:
Understanding the effect of "age" variable when obtaining mean observed price per unit.

AIM:
To compare and contrast two different models designed to predict levels of potential
sale price (fair value) of real estate assets in a specific area given predefined
asset and environmental characteristsics.

The price-changing variable in this specific experiment will be age.

CHOSEN TWO MODELS:
1. Linear Regression
   A Linear Regression model allows us to model the relationship between two
   continuous variables. As we are assuming that the price and age values have
   a linear relationship (as this is unknown for now) - a linear regression model
   will allow us to best predict the value of output variables (age) based on the
   value of our input variables (price).
2. RANSAC Regressor
   The RANSAC Regressor model allows for the exclusion of outliers which should
   result in higher accuracies when understanding the relationship between age and
   house prices. This model will also work best to compare with a Linear Regression
   model (given that the data actually has a linear relationship - however this
   is unknown as of now).

METHODOLOGY:
1. Import Python function-libraries for us to create effective and efficient code.
2. Import the given data from the csv file.
   This will be the data we used to test the models that we create.
3. Separate imported data into age and house price per unit values.
   These will be the input variables for our models.
4. Set up our RANSAC Regressor model and input our age and house price variables.
   We use Linear Regression model parameters as our estimator to help fit our model
   into the given data and values. We will use a minimum sample of 100, so that
   we capture at least a quarter of the given data in each sample; maximum trials
   of 200, capping the amount of iterations for a random sample selection; residual
   threshold of 10, capping the residual for each data sample (this will allow
   us to classify our inliers); and a random state of 0 to initialise the centres.
5. Set up trainers using the previously noted characteristics (apart from price).
   This training model is used to predict house prices from the data generated in
   our RANSAC Regressor. We will later test to see how successful our model is
   in predicting house prices based off age.
   Note: We allocate 30% of the original data points for testing and 70% for training.
         This is so that we can achieve more accurate estimates which are
         - valid - do not overestimate the accuracy, and
         - are more accuracte among the valid estimates.
   Then, predict the values using the training set.
6. Plot the data the RANSAC Regressor model has generated onto a scatterplot
   (noting inliers/outliers) to show the relationship between age and house
   prices - calculating the RANSAC Slope and RANSAC Intercept.
   AND
   Plot the training data (using the Linear Regression model) against the test
   subset on a scatterplot.
7. Measure the success of our model through Mean Square Error and Coefficient of Determination.
   Note: We do this by comparing our data (trained and tested against predicted).
   
NOTE: I have omitted the usage of other variables (apart from age) to decrease
any discrepancies within the models and instead predicted values onto a second
model.
'''

# IMPORT FUNCTION LIBRARIES AND INDIVIDUAL FUNCTIONS
# This will allow us to utilise prebuilt python functions for more effective and
# efficient analysis/code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def main():
    # START DATA EXTRACTION
    
    # Function to import data from given csv file AND
    # Separate the data into columns
    # Note: This is placed into a function as you will need to change the
    #       parameter in which you are reading the csv file from as well as what
    #       each column is referring to
    def read_data():
        df = pd.read_csv("REDACTED.csv")
        df.columns = ["No", "trade_date", "age", "distance_to_MTR", "number_of_stores", "latitude", "longitude", "house_price_per_unit"]
        return df
    # Call function to read in data objects from the given csv file
    df = read_data()

    # END DATA EXTRACTION
    # START RANSAC REGRESSOR MODEL

    # Define test variables - choosing one X variable age compared to the house price
    x_age = df[["age"]].values
    y_price = df["house_price_per_unit"].values

    # Set up our RANSAC Regressor - utilising the function from the sklearn.linear_model function library,
    # Then fitting model to training target values defined above
    ransac_regressor = RANSACRegressor(LinearRegression(), min_samples=100, max_trials=200, residual_threshold=10, random_state=0)
    ransac_regressor.fit(x_age, y_price)

    # Define Masks - noting the boolean mask of inliers (classified as TRUE)
    mask_inliers = ransac_regressor.inlier_mask_
    mask_outliers = np.logical_not(mask_inliers)

    # Set up predicted values using the linear model
    # These values will be used to create the line on the scatter plot
    line_x_age = np.arange(3, 40, 1)
    line_y_price = ransac_regressor.predict(line_x_age[:, np.newaxis])

    # END RANSAC REGRESSOR MODEL
    # START LINEAR REGRESSION MODEL
    
    # Set up data to train on
    lin_x = df.iloc[:,:-1].values
    #y_price = df["house_price_per_unit"].values - predefined variable above
    # Split data up - 30% for testing and 70% on training
    x_training, x_testing, y_training, y_testing = train_test_split(lin_x, y_price, test_size=0.3, random_state=0)

    # Create model for linear regression - fitting a linear model with coefficients w = (w1,...wp)
    lin_reg = LinearRegression()
    # Predict values for our linear regression model
    lin_reg.fit(x_training, y_training)
    predict_y_training = lin_reg.predict(x_training)
    predict_y_testing = lin_reg.predict(x_testing)
    
    # END LINEAR REGRESSION MODEL
    # START SCATTER PLOT FUNCTIONS
    
    # Set up plotting function
    def plotter(x_1,y_1,x_2,y_2):
        plt.scatter(x_1, y_1, c='grey', edgecolors='black', marker='o', label='Inliers')
        plt.scatter(x_2, y_2, c='lightblue', edgecolors='blue', marker='s', label='Outliers')
        plt.legend(loc='upper left')
    def showplot():
        plt.tight_layout()
        plt.show()

    # Plot the RANSAC Regressor data from our model onto a scatterplot
    plotter(x_age[mask_inliers], y_price[mask_inliers],x_age[mask_outliers], y_price[mask_outliers])
    plt.plot(line_x_age, line_y_price, c='red')
    plt.xlabel('Property Age [age]')
    plt.ylabel('Property Price [house_price_per_unit]')
    showplot()
    
    # Plot predicted data from Linear Regression model onto a scatterplot
    plotter(predict_y_training, predict_y_training - y_training,predict_y_testing, predict_y_testing - y_testing)
    plt.xlabel('Predicted value(s)')
    plt.ylabel('Residual(s)')
    plt.hlines(y=0, xmin=0, xmax=60, lw=2, color='red')
    plt.xlim([0, 60])
    showplot()
    
    # END SCATTER PLOT FUNCTIONS
    # PRINT RESULTS
    
    # Measure success of our model using predictions from test data and print the result
    # We will explain this result at the end
    print('RANSAC Slope: %.3f' % ransac_regressor.estimator_.coef_[0])
    print('RANSAC Intercept: %.3f' % ransac_regressor.estimator_.intercept_)
    print('MSE training: %.3f, testing: %.3f' % (mean_squared_error(y_training, predict_y_training), mean_squared_error(y_testing, predict_y_testing)))
    print('Average Net Difference training: %.3f, testing: %.3f' % (math.sqrt(mean_squared_error(y_training, predict_y_training)), math.sqrt(mean_squared_error(y_testing, predict_y_testing))))
    print('Difference between Average Net Differences: %.3f' % (abs(math.sqrt(mean_squared_error(y_training, predict_y_training)) - math.sqrt(mean_squared_error(y_testing, predict_y_testing)))))
    print('Coefficient of Determination (R^2) training: %.3f, testing: %.3f' % (r2_score(y_training, predict_y_training), r2_score(y_testing, predict_y_testing)))
    print('Difference between R^2''s: %.3f' % (abs(r2_score(y_training, predict_y_training)-r2_score(y_testing, predict_y_testing))))

if __name__ == "__main__":
    main()

'''
(PRINTED) RESULTS:
RANSAC Slope: -0.331
RANSAC Intercept: 46.932
MSE training: 79.211, testing: 73.107
Average Net Difference training: 8.900, testing: 8.550
Difference between Average Net Differences: 0.350
Coefficient of Determination (R^2) training: 0.584, testing: 0.571
Difference between R^2s: 0.013

EXPLANATION:
RANSAC Slope: -0.331
    A RANSAC Slope of -0.331 tells us that there is a slight negative correlation
    between house prices and age. (i.e. House prices decrease as the age increases)
    The model has predicted that the house price will decrease by 0.331 units
    for each 1 year that the age increases.

RANSAC Intercept: 46.932
    The RANSAC Intercept tells us where the predicted slope intercepts the y-axis.
    The model has predicted that a house of age 0 will have a price of 46.932 units.

MSE training: 79.211, testing: 73.107
Average Net Difference training: 8.900, testing: 8.550
Difference between Average Net Differences: 0.350
    The Mean Square Error measures the success of our model when predicting
    outcomes - how close our regression line is to the data points).
    The MSE is calculated by squaring the difference between our predicted values
    and the actual values.
    An MSE of 79.211 (for training) and 73.107 (for testing) tells us that our
    model is not very successful in predicting outcomes.
    An MSE of 79.211 (for training) tells us that we have an average net
    difference of 8.900 units (for training).
    An MSE of 73.107 (for testing) tells us that we have an average net
    difference of 8.550 units (for testing).
    The Difference between Average Net Differences of 0.350 tells us that our
    testing and training values are fairly similar.

Coefficient of Determination (R^2) training: 0.584, testing: 0.571
Difference between R^2s: 0.013
    The coefficient of determination (R^2) tells us how well our model predicts
    an outcome from the given event (house ages vs prices).
    A R^2 value of 0.584 (for training) tells us that our model partially predicts
    the outcome as 0<0.584<1.
    A R^2 value of 0.571 (for training) tells us that our model partially predicts
    the outcome as 0<0.571<1.
    Note that as the two R^2 are >0.5, they can be considered as moderately accurate.
    The Difference of R^2's tells us that our testing and training values are fairly similar.

CONCLUSION:
The use of a RANSAC Regressor is most likely attributes to the inaccuracies in our
model. While our RANSAC Regressor has removed the known outliers, it may still cause
inaccuracies in our model as RANSAC Regressors are mostly used when dealing with
linear data and values.

This can be noted through the given data - which is heavily varied.
This will cause inaccuracies in our model as it will not be able to properly find
a clear linear relationship between house prices and their age. The omittance of
other factors of the house (e.g. asset and environmental characteristics) also create
inaccuracies in our model as they are also important in determining the house prices.

In conclusion, our results tell us there is a negative correlation between a
house's price and age and that our model was moderately accruate in predicting results.
'''