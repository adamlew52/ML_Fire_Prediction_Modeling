import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Define the grid size (10x10 cells, each 10ft x 10ft)
grid_size = (10, 10)

class Simulate_Data:
    def alt_elev():
        elevation = np.random.uniform(0, 1000, (grid_size, grid_size))  # Elevation in meters
        return elevation

    def elevation_data():
        # Simulate elevation data (e.g., elevation in feet)
        x = np.linspace(0, 100, grid_size[0])
        y = np.linspace(0, 100, grid_size[1])
        x, y = np.meshgrid(x, y)
        elevation = 50 + 20 * np.sin(x / 10) * np.cos(y / 10)  # Simulated elevation data
        return x,y, elevation
    
    def weather_data():
        weather_temp = np.random.uniform(20, 40, (grid_size, grid_size))  # Temperature in degrees Celsius
        weather_humidity = np.random.uniform(0, 100, (grid_size, grid_size))  # Humidity in percentage
        return weather_temp, weather_humidity

    def fuel_loading():
        fuel_loading = np.random.uniform(0, 1, (grid_size, grid_size))  # Fuel density (normalized 0 to 1)
        return fuel_loading

# Min-Max Normalization
elevation = Simulate_Data.elevation_data()[2]
elevation_min = np.min(elevation)
elevation_max = np.max(elevation)
elevation_normalized = (elevation - elevation_min) / (elevation_max - elevation_min)

'''normalized_value = value-min(dataset)/max(dataset) - min(dataset)'''

class vz:
    def normalizedData():
    # Visualize the normalized data
        plt.imshow(elevation_normalized, cmap='terrain', origin='lower')
        plt.colorbar(label='Normalized Elevation')
        plt.title('Normalized Topographic Map (0-1 scale)')
        #plt.show()

    def heatmap():
        # Visualize the topographic data as a heatmap
        plt.imshow(elevation, cmap='terrain', origin='lower')
        plt.colorbar(label='Elevation (ft)')
        plt.title('Simulated Topographic Map (10ft x 10ft resolution)')
        #plt.show()
    
    def var_def_heatmap():
        # Visualize the topographic data as a heatmap
        plt.imshow(elevation, cmap='terrain', origin='lower')
        plt.colorbar(label='Top = more likely')
        plt.title('whatever the data set is dude idk this is a test')
        #plt.show()

    def three_dimensional():
        # 3D Visualization of Original Elevation Data
        x = Simulate_Data.elevation_data()[0]
        y = Simulate_Data.elevation_data()[1]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, elevation, cmap='terrain', edgecolor='k')

        ax.set_xlabel('X (ft)')
        ax.set_ylabel('Y (ft)')
        ax.set_zlabel('Elevation (ft)')
        ax.set_title('3D Topographic Map (Original Elevation)')
        plt.show()
    
    def three_dimension_normalized():
        # 3D Visualization of Normalized Elevation Data
        x = Simulate_Data.elevation_data()[0]
        y = Simulate_Data.elevation_data()[1]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, elevation_normalized, cmap='terrain', edgecolor='k')

        ax.set_xlabel('X (ft)')
        ax.set_ylabel('Y (ft)')
        ax.set_zlabel('Normalized Elevation')
        ax.set_title('3D Topographic Map (Normalized Elevation)')
        plt.show()


elevation_data = Simulate_Data.elevation_data()
vz.three_dimension_normalized()
#vz.var_def_heatmap()
