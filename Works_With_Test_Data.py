import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
#import noise
import rasterio  # for reading GeoTIFFs
import os
from scipy.ndimage import zoom

grid_size = 10  #this will be dictated by the data collection process from GIS, but for now I am going to use 20x20 for simulations sake

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


class Data_Sim:
    @staticmethod
    def realistic_data_gen():
        # Create a grid of x and y coordinates
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        x, y = np.meshgrid(x, y)

        # Define desired wavelength range and parameters
        min_wavelength = 20  # Minimum wavelength for larger spacing
        max_wavelength = 50  # Maximum wavelength for larger spacing
        max_elevation = 60   # Maximum elevation threshold for your data

        # Convert wavelength range to frequency range
        min_frequency = 1 / max_wavelength
        max_frequency = 1 / min_wavelength

        # Create a frequency map with random frequencies for each point
        frequency_map_x = np.random.uniform(min_frequency, max_frequency, size=x.shape)
        frequency_map_y = np.random.uniform(min_frequency, max_frequency, size=y.shape)

        # Reduce the amplitude of the sine and cosine functions
        amplitude = 10  # Decreased amplitude for less aggressive peaks

        # Compute the elevation data with adjusted amplitude
        elevation = 50 + amplitude * np.sin(x * frequency_map_x * 2 * np.pi) * np.cos(y * frequency_map_y * 2 * np.pi)

        # Optional: Add Gaussian noise for more randomness
        elevation += 5 * np.random.normal(0, 1, size=x.shape)

        # Add vertical offset to ensure most data is below the max elevation
        elevation -= elevation.max() - max_elevation

        # Optional: Clip values to ensure they do not exceed the max elevation
        elevation = np.clip(elevation, None, max_elevation)

        #elevation += 5 * np.random.normal(0, 1, size=x.shape)
        return x,y,elevation

    @staticmethod
    def several_elevation_data_testing():
        #simulating elecation data (elevation in feet)
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        x, y = np.meshgrid(x, y)
        elevation = 50 + 20 * np.sin(x/10) * np.cos(y/10) + 5 * np.random.normal(0, 1, size=x.shape)
 # Simulated elevation data
        return x,y,elevation

    @staticmethod
    def elevation_data():
        #simulating elecation data (elevation in feet)
        x = np.linspace(0, 100, grid_size)
        y=np.linspace(0, 100, grid_size)
        x, y = np.meshgrid(x, y)
        elevation = 50 + 20 * np.sin(x/10) * np.cos(y/10) # Simulated elevation data
        return x,y,elevation
    
    @staticmethod
    def weather_data():
        #simulating weather data
        weather_temp = np.random.uniform(20, 40, (grid_size, grid_size)) # Temperature in degrees Celsius
        weather_humidity = np.random.uniform(0, 100, (grid_size, grid_size)) # Humidity in percentage
        return weather_temp, weather_humidity
    
    @staticmethod
    def fuel_loading():
        #simulate fuel loading
        fuel_loading = np.random.uniform(0, 1, (grid_size,grid_size))# Fuel density (normalized 0 to 1)
        return fuel_loading

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
class Obtain_Actual_Data:
    def load_and_normalize_gis_data(file_path, grid_size_pulled, min_value, max_value):
        """
        Load GIS data from a GeoTIFF file and normalize it.
        """
        with rasterio.open(file_path) as src:
            # Read the data from the first band
            data = src.read(1)
            
            # Resample to the desired grid size if necessary
            # This example assumes the data needs to be resized to grid_size x grid_size
            data_resampled = np.interp(np.linspace(0, data.shape[0], grid_size_pulled),
                                    np.arange(data.shape[0]),
                                    data)
            data_resampled = np.interp(np.linspace(0, data_resampled.shape[1], grid_size_pulled),
                                    np.arange(data_resampled.shape[1]),
                                    data_resampled.T).T
            
            # Normalize the data
            normalized_data = normalize_GIS_data(data_resampled, min_value, max_value)
         
        return normalized_data
    
    def NormalizeRaster(raster_path, output_path=None):
        """
        Reads a raster file, normalizes its data to a range between 0 and 1, and optionally saves the normalized raster.
        
        Parameters:
        - raster_path (str): Path to the input raster file.
        - output_path (str, optional): Path to save the normalized raster file. If None, the file is not saved.
        
        Returns:
        - normalized_array (numpy.ndarray): The normalized raster data.
        """

        with rasterio.open(raster_path) as src:
            # Read the data from the raster file
            data = src.read(1)  # Read the first band
            
            # Flatten the data to a 1D array
            data_flat = data.flatten()
            
            # Normalize the data
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data_flat.reshape(-1, 1)).reshape(data.shape)
            
            if output_path:
                # Save the normalized data to a new raster file
                with rasterio.open(output_path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
                                count=1, dtype=data_normalized.dtype, crs=src.crs, transform=src.transform) as dst:
                    dst.write(data_normalized, 1)
            
            return data_normalized

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Normalize: 
    def normalize_GIS_data(data, min_value, max_value):
        """
        Normalize the GIS data to a specified range [min_value, max_value].
        """
        scaler = MinMaxScaler(feature_range=(min_value, max_value))
        return scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(data.shape)

    def NormalizeTiff(input_tiff_path, output_tiff_path=None):
        """
        Reads a TIFF raster file, normalizes its data to a range between 0 and 1, and optionally saves the normalized raster.

        Parameters:
        - input_tiff_path (str): Path to the input TIFF file.
        - output_tiff_path (str, optional): Path to save the normalized TIFF file. If None, the file is not saved.

        Returns:
        - normalized_array (numpy.ndarray): The normalized raster data.
        """
        with rasterio.open(input_tiff_path) as src:
            # Read the data from the raster file
            data = src.read(1)  # Read the first band
            
            # Flatten the data to a 1D array
            data_flat = data.flatten()
            
            # Normalize the data
            scaler = MinMaxScaler()
            data_normalized_flat = scaler.fit_transform(data_flat.reshape(-1, 1)).flatten()
            data_normalized = data_normalized_flat.reshape(data.shape)
            
            if output_tiff_path:
                # Save the normalized data to a new TIFF file
                with rasterio.open(output_tiff_path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
                                count=1, dtype=data_normalized.dtype, crs=src.crs, transform=src.transform) as dst:
                    dst.write(data_normalized, 1)
                    
            return data_normalized

    def NormalizeData(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)
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
        return Normalize.NormalizeData(slope_magnitude)

def combine_features(weather_temp, weather_humidity, slope, fuel_loading):
    norm_temp = Normalize.NormalizeData(weather_temp)
    norm_humid = Normalize.NormalizeData(weather_humidity)
    norm_slope = Normalize.NormalizeData(Normalize.compute_slope(elevation))
    norm_fuel_load = Normalize.NormalizeData(fuel_loading)

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

    return fire_spread_risk

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
        new_fire_map = fire_map.copy() 
        for fire_col in range(grid_size): #inner loop
            if fire_map[fire_row, fire_col] == 1: #condition for change
                for di in [-1, 0, 1]:
                    for dj in [-1, 0 ,1]:
                        ni, nj = fire_row + di, fire_col + dj
                        if 0<= ni < grid_size and 0 <= nj < grid_size:
                            if fire_spread_risk[ni, nj] > spread_risk_weight:
                                fire_map[ni,nj] = 1 

    fire_map = new_fire_map
    #yield fire_map  # Yield the updated fire map

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

#Everything below here needs explained by gpt/documenation------------------------------------------------------------------------------------------------------

def plot_3d(x,y,elevation, title, zlabel, cmap):
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, elevation, cmap=cmap, edgecolor='k')
    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

def testing_all_generation_data():
    # generate data
    x,y,elevation = Data_Sim.realistic_data_gen()
    weather_temp, weather_humidity = Data_Sim.weather_data()
    fuel_loading = Data_Sim.fuel_loading()


    #normalize and combine
    slope = Normalize.compute_slope(elevation)
    fire_spread_risk = combine_features(weather_temp, weather_humidity, slope, fuel_loading)

    #plot 3d visualizations
    plot_3d(x, y, Normalize.NormalizeData(elevation), '3D Topographic Map (Normalized Elevation)', 'Normalized Elevation', cmap='hot')

    # Simulate fire spread and plot the animation
    start_location = (5, 5)  # Fire starts in the middle
    #for fire_map in spread_fire(fire_spread_risk, start_location):
    for fire_map in spread_fire(fire_spread_risk, start_location):
        plt.imshow(fire_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Simulated Fire Spread')
        plt.pause(1)  # Pause to create the animation effect
        plt.clf()  # Clear the figure to prepare for the next frame

    plt.show()

testing_all_generation_data()


""" data resources
https://earthexplorer.usgs.gov/



"""