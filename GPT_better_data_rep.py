import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler
import time

# Define the grid size
grid_size = 10

class Simulate_Data:
    @staticmethod
    def elevation_data():
        # Simulate elevation data (elevation in feet)
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        x, y = np.meshgrid(x, y)
        elevation = 50 + 20 * np.sin(x / 10) * np.cos(y / 10)  # Simulated elevation data
        return x, y, elevation

    @staticmethod
    def weather_data():
        # Simulate weather data
        weather_temp = np.random.uniform(20, 40, (grid_size, grid_size))  # Temperature in degrees Celsius
        weather_humidity = np.random.uniform(0, 100, (grid_size, grid_size))  # Humidity in percentage
        return weather_temp, weather_humidity

    @staticmethod
    def fuel_loading():
        # Simulate fuel loading
        fuel_loading = np.random.uniform(0, 1, (grid_size, grid_size))  # Fuel density (normalized 0 to 1)
        return fuel_loading

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)

def compute_slope(elevation):
    slope = np.gradient(elevation, axis=(0, 1))
    slope_magnitude = np.sqrt(slope[0]**2 + slope[1]**2)
    return normalize_data(slope_magnitude)

def combine_features(weather_temp, weather_humidity, slope, fuel_loading):
    weather_temp_normalized = normalize_data(weather_temp)
    weather_humidity_normalized = normalize_data(weather_humidity)
    slope_normalized = slope
    fuel_loading_normalized = normalize_data(fuel_loading)

    # Weighting factors
    fuel_loading_weighting = 0.3
    slope_weighting = 0.2
    weather_temp_weighting = 0.5
    weather_humidity_weighting = 0.2

    # Combine the features into a single risk score
    fire_spread_risk = (
        weather_temp_normalized * weather_temp_weighting + 
        (1 - weather_humidity_normalized) * weather_humidity_weighting + 
        slope_normalized * slope_weighting + 
        fuel_loading_normalized * fuel_loading_weighting
    )
    return fire_spread_risk

def spread_fire(fire_spread_risk, start_location):
    fire_map = np.zeros_like(fire_spread_risk)
    fire_map[start_location] = 1  # Fire starts here
    
    for _ in range(grid_size * grid_size):
        new_fire_map = fire_map.copy()
        for i in range(grid_size):
            for j in range(grid_size):
                if fire_map[i, j] == 1:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                if fire_spread_risk[ni, nj] > 0.5:
                                    new_fire_map[ni, nj] = 1
        fire_map = new_fire_map
        yield fire_map  # Yield the updated fire map

def plot_3d(x, y, elevation, title, zlabel, cmap):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, elevation, cmap=cmap, edgecolor='k')
    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

# Generate data
x, y, elevation = Simulate_Data.elevation_data()
weather_temp, weather_humidity = Simulate_Data.weather_data()
fuel_loading = Simulate_Data.fuel_loading()

# Normalize and compute features
slope = compute_slope(elevation)
fire_spread_risk = combine_features(weather_temp, weather_humidity, slope, fuel_loading)

# Plot 3D visualizations
plot_3d(x, y, normalize_data(elevation), '3D Topographic Map (Normalized Elevation)', 'Normalized Elevation', cmap='hot')

# Simulate fire spread and plot the animation
start_location = (5, 5)  # Fire starts in the middle
for fire_map in spread_fire(fire_spread_risk, start_location):
    plt.imshow(fire_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Simulated Fire Spread')
    plt.pause(1)  # Pause to create the animation effect
    plt.clf()  # Clear the figure to prepare for the next frame

plt.show()
