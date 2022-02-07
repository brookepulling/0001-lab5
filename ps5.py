# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: brooke pulling
# Collaborators: NONE
# Time: 6 hours

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re
from sklearn.cluster import KMeans

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

# KMeans class not required until Problem 7
class KMeansClustering(KMeans):

    def __init__(self, data, k):
        super().__init__(n_clusters=k, random_state=0)
        self.fit(data)
        self.labels = self.predict(data)

    def get_centroids(self):
        'return np array of shape (n_clusters, n_features) representing the cluster centers'
        return self.cluster_centers_

    def get_labels(self):
        'Predict the closest cluster each sample in data belongs to. returns an np array of shape (samples,)'
        return self.labels

    def total_inertia(self):
        'returns the total inertia of all clusters, rounded to 4 decimal points'
        return round(self.inertia_, 4)



class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """

        # NOTE: TO BE IMPLEMENTED IN PART 4B OF THE PSET
        lst = []
        for year in years:
            numerator = 0
            denominator = 0
            for city in cities:
                numerator += sum(self.get_daily_temps(city, year))
                #would this just be 365?
                denominator += len(self.get_daily_temps(city, year))
            lst.append(numerator/denominator)
        return lst

def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    #find the average of x and y
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    
    #find the denominator and numerator of the m function 
    num = 0 
    denom = 0
    for i in range(len(x)):
        num += (x[i]-x_avg)*(y[i]-y_avg)
        denom += (x[i]-x_avg)**2
    
    #solve for m and b
    m = num/denom
    b = y_avg - (m*x_avg)
    
    return (m,b)

def squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    se = 0
    #using the regression line calculate the y' points and standard error
    for i in range(len(y)):
        y_p = m*x[i] + b
        se += (y[i]-y_p)**2
    return se
        
def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    coeffs = []
    # creates a polynomial regression model for each degree with the given x and y arrays
    for degree in degrees: 
        coeffs.append(np.polyfit(x,y,degree))
    return coeffs


def evaluate_models(x, y, models, display_graphs=True):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    r_lst = []
    for model in models:
        #get the y prime values 
        y_p = np.polyval(model,x)
        #calculate r squared
        r_sqr = round(r2_score(y,y_p), 4)
        r_lst.append(r_sqr)
        if display_graphs:
            deg = len(model)-1
            m = round(standard_error_over_slope(x,y, y_p, model), 4)
            plt.plot(x,y, 'bo')
            plt.plot(x, y_p)
            plt.xlabel('year')
            plt.ylabel('temperature in celsius)')
            # creates different titles based on whether it is a linear model or not
            if deg == 1:
                plt.title('Degree: '+ str(deg) + ', R-squared: ' + str(r_sqr) + ', SE/slope: ' + str(m))
            else:
                plt.title('Degree: '+ str(deg) + ', R-squared: ' + str(r_sqr))
            plt.show()
    return r_lst

def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    
    max_j = length-1
    max_i = 0 
    m = 0
    for i in range(len(x)-(length-1)):
        j =  i+length 
        x_lst = x[i:j]
        y_lst = y[i:j]
        lin_mod = generate_polynomial_models(x_lst, y_lst, [1])
        # finds the first value in the linear model, which is the slope
        current_m = lin_mod[0][0]
        # finds best slope within the range
        if not positive_slope:
            if current_m < m - 1e-8:
                max_j = j
                max_i = i
                m = current_m
        else: 
            if current_m > m + 1e-8:
                max_j = j
                max_i = i
                m = current_m
    # returns none if no maximum slope is found           
    if m == 0:
        return None
    else:
        return (max_i, max_j, m)


def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    slopes = []
    for i in range(2, len(x)+1):
        p_max = get_max_trend(x, y, i, True)
        n_max = get_max_trend(x, y, i, False)
        if n_max == None and p_max == None:
            slopes.append((0, i, None))
        elif p_max == None and n_max != None: 
            slopes.append(n_max)
        elif n_max == None and p_max != None: 
            slopes.append(p_max)
        elif abs(p_max[2] - abs(n_max[2])) <= 1e-8:
            if p_max[0]<n_max[0]:
                slopes.append(p_max)
            else:
                slopes.append(n_max)
        elif p_max[2] > abs(n_max[2]):
            slopes.append(p_max)
        else:
            slopes.append(n_max)
    return slopes

def calculate_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    sum = 0
    for i in range(len(y)):
        val = y[i]- estimated[i]
        sum += (val)**2
    # calculates and returns rmse
    rmse = (sum/len(y))**0.5   
    return rmse

def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    values = []
    for model in models: 
        y_p = np.polyval(model, x)
        val = round(calculate_rmse(y,y_p), 4)
        values.append(val)
        if display_graphs:
            # finds degree as length of model array - 1 to account for constants
            deg = -1+ len(model)
            plt.plot(x,y, 'bo')
            plt.plot(x, y_p, c= 'r')
            plt.xlabel('year')
            plt.ylabel('temp in egrees celsius)')
            plt.title('Degree: ' + str(deg) + ', RMSE: ' + str(val))
    return values

def cluster_cities(cities, years, data, n_clusters):
    '''
    Clusters cities into n_clusters clusters using their average daily temperatures
    across all years in years. Generates a line plot with the average daily temperatures
    for each city. Each cluster of cities should have a different color for its
    respective plots.

    Args:
        cities: a list of the names of cities to include in the average
                daily temperature calculations
        years: a list of years to include in the average daily
                temperature calculations
        data: a Dataset instance
        n_clusters: an int representing the number of clusters to use for k-means

    Note that this part has no test cases, but you will be expected to show and explain
    your plots during your checkoff
    '''
    raise NotImplementedError


if __name__ == '__main__':
    pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    data = Dataset('data.csv')
    x = np.arange(1961,2016)
    y = []
    for year in x:
        y.append(data.get_temp_on_date('BOSTON', 12, 1, year))
    y_vals = np.array(y)
    models = generate_polynomial_models(x, y_vals, [1])
    evaluate_models(x, y_vals, models,  True)

    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    # avg_temps = data.calculate_annual_temp_averages(['BOSTON'], x)
    # models = generate_polynomial_models(x, avg_temps, [1])
    # evaluate_models(x, avg_temps, models,  True)

    ##################################################################################
    # Problem 5B: INCREASING TRENDS
    # y_vals = data.get_annual_averages(['SEATTLE'], x)
    # max_trend = get_max_trend(x,y_seattle, 30, True)
    # x = x[max_trend[0]:max_trend[1]]
    # y = y_vals[max_trend[0]:max_trend[1]]
    # increasing = generate_polynomial_models(x, y, [1])
    # evaluate_models(x, y, increasing, True)
    
    ##################################################################################
    # Problem 5C: DECREASING TRENDS
    # decr_trend = get_max_trend(x,y_seattle, 15, False)
    # x = x[max_trend[0]:max_trend[1]]
    # y = y_vals[max_trend[0]:max_trend[1]]
    # decreasing = generate_polynomial_models(x, y, [1])
    # evaluate_models(x, y, decreasing, True)
    
    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.

    ##################################################################################
    # Problem 6B: PREDICTING
    x = np.array(TRAIN_INTERVAL)
    averages = data.calculate_annual_temp_averages(CITIES, x)
    models = generate_polynomial_models(x, averages, [2,10])
    evaluate_models(x, averages, [models[0]],True)
    evaluate_models(x, averages, [models[1]],True)
    
    x = np.array(TEST_INTERVAL)
    averages = data.calculate_annual_temp_averages(CITIES, x)
    models = generate_polynomial_models(x, averages, [2,10])
    evaluate_models(x, averages, [models[0]],True)
    evaluate_models(x, averages, [models[1]],True)

    ##################################################################################
    # Problem 7: KMEANS CLUSTERING (Checkoff Question Only)


    ####################################################################################