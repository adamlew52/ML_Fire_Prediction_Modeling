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

def compute_slope(elevation):
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
    return NormalizeData(slope_magnitude)

def NormalizeData(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.flatten().reshape(-1.1)).reshape(grid_size, grid_size)
    """
    MinMaxScaling i.e. Normalization

    Formula: 
    x_scaled= x - min(x) / max(x)-min(x)
    
    Purpose: Transforms features to a range between 0 and 1 (or any other range). It’s useful when features have different units or scales.
    Example: Scaling temperatures from a range of 30-100°F to a range of 0-1.

    """

    # Normalize the data
    #norm_temp = scaler.fit_transform(weather_temp.flatten().reshape(-1.1)).reshape(grid_size, grid_size)
    """
    weather_temp = data set
    .flatten() = flatten the 2D array into a 1D array for processing
    .reshape(-1.1)) = reshape the flattened array into a 
        --> -1 means "infer this dimension based on the length of the array and the other given dimension", 
        --> 1 specifies that there should be one column.
    .reshape(grid_size, grid_size) = reshape scaler.fit_transform(weather_temp.flatten().reshape(-1.1)) into the grid size defined by the data set (in the example its 20)

    --> all of the other normalizations done here follow the exact same format


    essentially what is happening below: 
    #norm_humid = scaler.fit_transform(weather_humidity.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
    #norm_temp = scaler.fit_transform(weather_temp.flatten().reshape(-1.1)).reshape(grid_size, grid_size)
    #norm_slope = scaler.fit_transform(slope_magnitude.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
    #norm_fuel_load = scaler.fit_transform(fuel_loading.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)

    """

norm_temp = NormalizeData(weather_temp)
norm_humid = NormalizeData(weather_humidity)
norm_slope = NormalizeData(compute_slope(elevation))
norm_fuel_load = NormalizeData(fuel_loading)

#Create a single score

fuel_loading_weighting = 0.4
slope_normalized_weighting = 0.5
weather_temp_weighting = 0.2
weather_humid_weighting = 0.2

# Combine the features into a single score
fire_spread_risk = (norm_temp * weather_temp_weighting + 
                    (1 - norm_humid) * weather_humid_weighting + 
                    norm_slope * slope_normalized_weighting + 
                    norm_fuel_load * fuel_loading_weighting)

"""
the weighting will need to be modified based on the data 
multiple techniques for this:
    1. look at nearby squares: take the average of the two, and multiply the current squares value by the previous added [ex: 0.55 + (0.55*0.63) = adjusted 0.8965]
    2. 

"""

def spread_fire(fire_spread_risk, start_location):
    fire_map = np.zeros_like(fire_spread_risk)
    """
    np.zeros_like() is a NumPy function that creates a new array filled with zeros.
        - The array it creates has the same shape and data type as the input array, which in this case is fire_spread_risk.

    fire_map[start_location] = 1 is showing where the fire has started, and thus will be the first square populated with data
    """

    """
    example calcualtion for spread_risk_weighting :
    #fire_spread_risk[i,j]=(temperature[i,j]/max_temperature)*(/wind_speed[i,j]/max_wind_speed)*(1/humidity[i,j])*(topography_factor[i,j])*(fuel_density[i,j])
    """
    spread_risk_weight = 0.5 # create algorithm for not just saying "if it has above a 0.5 risk, set to 'burning'"

    for fire_row in range(grid_size): #outer loop
        for fire_col in range(grid_size): #inner loop
            if fire_map[fire_row, fire_col] == 1: #condition for change
                for di in [-1, 0, 1]:
                    for dj in [-1, 0 ,1]:
                        ni, nj = fire_row + di, fire_col + dj
                        if 0<= ni < grid_size and 0 <= nj < grid_size:
                            if fire_spread_risk[ni, nj] > spread_risk_weight:
                                fire_map[ni,nj] = 1 

    """
    for i or fire_row in range(grid_size): #outer loop
        - iterates over each space in the grid, defined by grid_size
        - grid_size is the size of the grid, so range(grid_size) generates a sequence of indices from 0 to grid_size - 1.
    for fire_col or j in range(grid_size): #inner loop
        - For each row i, this loop iterates over each column j, effectively going through every cell in the grid.
    if fire_map[i,j] == 1: #condition for change
        - checks if the current tile is on fire
        - if not burning
        - 
    for di in [-1, 0, 1]:
        - di (Delta i): Vertical shift (up or down).
        - dj (Delta j): Horizontal shift (left or right).
        - used to check the neighboring cells of the current burning cell (i, j).
    for dj in [-1, 0 ,1]:
        - di = -1 and dj = 0: Move up one row.
        - di = 1 and dj = 0: Move down one row.
        - di = 0 and dj = -1: Move left one column.
        - di = 0 and dj = 1: Move right one column.
    ni, nj =fire_row+ di, j + dj
        - ni and nj represent the actual row and column indices of the neighboring cell that the code is currently checking.
        - They are calculated by adding di and dj to the current cell's indices (i, j):
            - ni = i + di: New row index, after applying the vertical shift di.
            - nj = j + dj: New column index, after applying the horizontal shift dj.

            (di, dj) pairs:        Neighboring cells (ni, nj):
        (-1, -1) (-1, 0) (-1, +1)     (1, 1) (1, 2) (1, 3)
        ( 0, -1) ( i, j) ( 0, +1)  -> (2, 1) (2, 2) (2, 3)
        (+1, -1) (+1, 0) (+1, +1)     (3, 1) (3, 2) (3, 3)

    if 0 <= ni < grid_size and 0 <= nj < grid_size:
        - boundary check
        - This ensures that the neighboring cell (ni, nj) is within the grid boundaries.
            - If ni or nj is out of bounds, the code skips this neighbor.

    if fire_spread_risk[ni, nj] > spread_risk_weight:
        - Risk Check
        - This checks if the fire risk for the neighboring cell (ni, nj) is above a certain threshold (in this case, 0.5).
            - If the risk is high enough, the fire spreads to this cell.
    
    fire_map[ni,nj] = 1
        - spreading the fire
        - If the neighboring cell passes the risk check, it's marked as burning by setting fire_map[ni, nj] to 1.
    """

def new_spread_fire():
    """
    Simulates the spread of fire across the grid based on the fire spread risk.
    
    Parameters:
    - fire_spread_risk: 2D array of risk values for fire spread
    - fire_map: 2D array indicating which cells are burning
    - threshold: Risk threshold above which the fire will spread
    """
    


