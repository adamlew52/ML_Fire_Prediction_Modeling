import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time


#rewrite from "GPT_better_data_rep.py"

grid_size = 20  #this will be dictated by the data collection process from GIS, but for now I am going to use 20x20 for simulations sake

#Practice Data Generation
"""
Continuous Uniform Distribution is used here, but Discrete Uniform Distribution can be used.

continuous distribution comes from a range (i.e. 20, 40), vs Discrete Distribution using discrete values
ex: array = np.random.randint(1, 11, (3, 4))  # Note: upper bound is exclusive

"""
weather_temp = np.random.uniform(20,40,(grid_size, grid_size)) #random number selection, uniform(low,high,shape of the array)
weather_humidity = np.random.uniform(0, 100, (grid_size, grid_size))  # Humidity in percentage
elevation = np.random.uniform(0, 1000, (grid_size, grid_size))  # Elevation in meters
fuel_loading = np.random.uniform(0, 1, (grid_size, grid_size))  # Fuel density (normalized 0 to 1)

#find the slope
slope = np.gradient(elevation, axis=(0, 1))
#gradient(The input array, Optional array of coordinates corresponding to the dimensions of f, rows (axis 0) and columns (axis 1) of the array)
slope_magnitude = np.sqrt(slope[0]**2 + slope[1]**2)
"""
slope_magnitude= ((gradient_x)^2)+((gradient_y)^2) the gradient comes from the slope calculation.
slope_magnitude: Represents the overall steepness of the terrain at each grid point, combining changes in both the row and column directions.
higher the magnitude = Indicate steeper slopes, where the terrain changes more rapidly
how it works: the Slope Magnitude is measured by finding the overall steepness of the terrain by combining the gradient values in two perpendicular directions.
"""

scaler = MinMaxScaler()
"""
MinMaxScaling i.e. Normalization

Formula: 
x_scaled= x - min(x) / max(x)-min(x)
 
Purpose: Transforms features to a range between 0 and 1 (or any other range). It’s useful when features have different units or scales.
Example: Scaling temperatures from a range of 30-100°F to a range of 0-1.

"""

# Normalize the data
norm_temp = scaler.fit_transform(weather_temp.flatten().reshape(-1.1)).reshape(grid_size, grid_size)
"""
weather_temp = data set
.flatten() = flatten the 2D array into a 1D array for processing
.reshape(-1.1)) = reshape the flattened array into a 
    --> -1 means "infer this dimension based on the length of the array and the other given dimension", 
    --> 1 specifies that there should be one column.
.reshape(grid_size, grid_size) = reshape scaler.fit_transform(weather_temp.flatten().reshape(-1.1)) into the grid size defined by the data set (in the example its 20)

--> all of the other normalizations done here follow the exact same format

"""
norm_humid = scaler.fit_transform(weather_humidity.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
norm_slope = scaler.fit_transform(slope_magnitude.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
norm_fuel_load = scaler.fit_transform(fuel_loading.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)

#Create a single score

fuel_loading_weighting = 0.3
slope_normalized_weighting = 0.2
weather_temp_weighting = 0.5
weather_humid_weighting = 0.2

# Combine the features into a single score
fire_spread_risk = (norm_temp * weather_temp_weighting + 
                    (1 - norm_humid) * weather_humid_weighting + 
                    norm_slope * slope_normalized_weighting + 
                    norm_fuel_load * fuel_loading_weighting)

"""
the weighting will need to be modified based on the data 
multiple techniques for this:
    1. look at nearby squares, if it is above 

"""

