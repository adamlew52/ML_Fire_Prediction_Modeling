import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

# Set grid size
grid_size = 10

# Generate random data
weather_temp = np.random.uniform(20, 40, (grid_size, grid_size))  # Temperature in degrees Celsius
weather_humidity = np.random.uniform(0, 100, (grid_size, grid_size))  # Humidity in percentage
elevation = np.random.uniform(0, 1000, (grid_size, grid_size))  # Elevation in meters
fuel_loading = np.random.uniform(0, 1, (grid_size, grid_size))  # Fuel density (normalized 0 to 1)

# Compute slope from elevation
slope = np.gradient(elevation, axis=(0, 1))
slope_magnitude = np.sqrt(slope[0]**2 + slope[1]**2)

#------------------------------------------------------------------------------------------------------

# Create a scaler
scaler = MinMaxScaler()

# Normalize the data
weather_temp_normalized = scaler.fit_transform(weather_temp.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
weather_humidity_normalized = scaler.fit_transform(weather_humidity.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
slope_normalized = scaler.fit_transform(slope_magnitude.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
fuel_loading_normalized = scaler.fit_transform(fuel_loading.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)

#------------------------------------------------------------------------------------------------------

fuel_loading_weighting = 0.3
slope_normalized_weighting = 0.2
weather_temp_weighting = 0.5
weather_humid_weighting = 0.2

# Combine the features into a single score
fire_spread_risk = (weather_temp_normalized * weather_temp_weighting + 
                    (1 - weather_humidity_normalized) * weather_humid_weighting + 
                    slope_normalized * slope_normalized_weighting + 
                    fuel_loading_normalized * fuel_loading_weighting)

#------------------------------------------------------------------------------------------------------

def spread_fire(fire_spread_risk, start_location):
    fire_map = np.zeros_like(fire_spread_risk)
    fire_map[start_location] = 1  # Fire starts here
    
    for i in range(grid_size):
        for j in range(grid_size):
            if fire_map[i, j] == 1:
                # Spread to neighboring cells based on risk
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            if fire_spread_risk[ni, nj] > 0.5:  # Arbitrary threshold
                                fire_map[ni, nj] = 1
    return fire_map



def new_spread_fire(fire_spread_risk, fire_map, threshold=0.5):
    """
    Simulates the spread of fire across the grid based on the fire spread risk.
    
    Parameters:
    - fire_spread_risk: 2D array of risk values for fire spread
    - fire_map: 2D array indicating which cells are burning
    - threshold: Risk threshold above which the fire will spread
    """
    grid_size = fire_spread_risk.shape[0]
    new_fire_map = fire_map.copy()  # Create a copy to update fire spread without modifying the original immediately
    
    for i in range(grid_size):
        for j in range(grid_size):
            if fire_map[i, j] == 1:  # If this cell is burning
                # Check all 8 possible directions (including diagonals)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # Skip the current cell itself
                        
                        ni, nj = i + di, j + dj  # Neighbor indices
                        
                        # Check if the neighbor is within grid boundaries
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            if fire_spread_risk[ni, nj] > threshold:  # If risk is above the threshold
                                new_fire_map[ni, nj] = 1  # Set the neighbor cell as burning
    
    return new_fire_map

start_location = (5, 5)  # Fire starts in the middle
fire_map = spread_fire(fire_spread_risk, start_location)

#------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Fire Spread Risk")
plt.imshow(fire_spread_risk, cmap='hot', interpolation='nearest')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Simulated Fire Spread")
plt.imshow(fire_map, cmap='hot', interpolation='nearest')
plt.colorbar()

plt.show()
