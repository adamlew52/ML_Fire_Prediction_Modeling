import numpy as np
from sklearn.preprocessing import MinMaxScaler
import rasterio
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os

grid_size = 10  # Example grid size

class DataSim:
    @staticmethod
    def realistic_data_gen():
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        x, y = np.meshgrid(x, y)

        min_wavelength, max_wavelength = 20, 50
        max_elevation = 60

        min_frequency = 1 / max_wavelength
        max_frequency = 1 / min_wavelength

        frequency_map_x = np.random.uniform(min_frequency, max_frequency, size=x.shape)
        frequency_map_y = np.random.uniform(min_frequency, max_frequency, size=y.shape)

        amplitude = 10
        elevation = 50 + amplitude * np.sin(x * frequency_map_x * 2 * np.pi) * np.cos(y * frequency_map_y * 2 * np.pi)
        elevation += 5 * np.random.normal(0, 1, size=x.shape)

        elevation -= elevation.max() - max_elevation
        elevation = np.clip(elevation, None, max_elevation)
        return x, y, elevation

    @staticmethod
    def generate_weather_data():
        weather_temp = np.random.uniform(20, 40, (grid_size, grid_size))
        weather_humidity = np.random.uniform(0, 100, (grid_size, grid_size))
        return weather_temp, weather_humidity

    @staticmethod
    def generate_fuel_loading():
        return np.random.uniform(0, 1, (grid_size, grid_size))

class ObtainActualData:
    @staticmethod
    def load_and_normalize_gis_data(file_path, grid_size_pulled, min_value, max_value):
        with rasterio.open(file_path) as src:
            data = src.read(1)
            data_resampled = zoom(data, (grid_size_pulled / data.shape[0], grid_size_pulled / data.shape[1]), order=1)
            normalized_data = ObtainActualData.normalize_gis_data(data_resampled, min_value, max_value)
        return normalized_data

    @staticmethod
    def normalize_gis_data(data, min_value, max_value):
        scaler = MinMaxScaler(feature_range=(min_value, max_value))
        return scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(data.shape)

    @staticmethod
    def normalize_tiff(input_tiff_path, output_tiff_path=None):
        with rasterio.open(input_tiff_path) as src:
            data = src.read(1)
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(data.shape)

            if output_tiff_path:
                with rasterio.open(output_tiff_path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
                                   count=1, dtype=data_normalized.dtype, crs=src.crs, transform=src.transform) as dst:
                    dst.write(data_normalized, 1)

        return data_normalized

    @staticmethod
    def extract_raster_info(file_path):
        # Open the raster file
        with rasterio.open(file_path) as src:
            # Extract grid size (number of rows and columns)
            grid_size = (src.height, src.width)
            
            # Read the raster data as a 2D numpy array
            data = src.read(1)  # Read the first band
            
            # Calculate min and max values
            min_value = np.min(data)
            max_value = np.max(data)
        
        return file_path, grid_size, min_value, max_value

class FireSpreadModel:
    @staticmethod
    def compute_slope(elevation):
        slope = np.gradient(elevation, axis=(0, 1))
        slope_magnitude = np.sqrt(slope[0] ** 2 + slope[1] ** 2)
        return FireSpreadModel.normalize_data(slope_magnitude)

    @staticmethod
    def combine_features(weather_temp, weather_humidity, slope, fuel_loading):
        norm_temp = FireSpreadModel.normalize_data(weather_temp)
        norm_humid = FireSpreadModel.normalize_data(weather_humidity)
        norm_slope = FireSpreadModel.normalize_data(slope)
        norm_fuel_load = FireSpreadModel.normalize_data(fuel_loading)

        fire_spread_risk = (norm_temp * 0.2 + 
                            (1 - norm_humid) * 0.2 + 
                            norm_slope * 0.5 + 
                            norm_fuel_load * 0.4)

        return fire_spread_risk

    @staticmethod
    def spread_fire(fire_spread_risk, start_location):
        fire_map = np.zeros_like(fire_spread_risk)
        fire_map[start_location] = 1
        spread_risk_weight = 0.5

        for fire_row in range(grid_size):
            new_fire_map = fire_map.copy()
            for fire_col in range(grid_size):
                if fire_map[fire_row, fire_col] == 1:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = fire_row + di, fire_col + dj
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                if fire_spread_risk[ni, nj] > spread_risk_weight:
                                    new_fire_map[ni, nj] = 1
            fire_map = new_fire_map

        return fire_map

    @staticmethod
    def normalize_data(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(grid_size, grid_size)

def plot_3d(x, y, elevation, title, zlabel, cmap):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)  # Ensure x and y are properly meshed
    surf = ax.plot_surface(x, y, elevation, cmap=cmap, edgecolor='k')
    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

# Example usage
weather_temp, weather_humidity = DataSim.generate_weather_data()
elevation = DataSim.realistic_data_gen()[2]
fuel_loading = DataSim.generate_fuel_loading()

slope = FireSpreadModel.compute_slope(elevation)
fire_spread_risk = FireSpreadModel.combine_features(weather_temp, weather_humidity, slope, fuel_loading)
fire_map = FireSpreadModel.spread_fire(fire_spread_risk, (5, 5))

#plt.imshow(fire_map, cmap='hot')

file_path = '/Users/adam/Documents/GitHub/ML_Fire_Prediction_Modeling/GIS_Data/raw_tiff/Conifer.tiff'
file_path, grid_size, min_value, max_value = ObtainActualData.extract_raster_info(file_path)
normalized_data = ObtainActualData.load_and_normalize_gis_data(file_path, grid_size[0], min_value, max_value)

x = np.linspace(0, 100, normalized_data.shape[1])  # Adjust based on data shape
y = np.linspace(0, 100, normalized_data.shape[0])  # Adjust based on data shape

title = "test grid"
zlabel = "zlabel"
cmap = 'hot'

plot_3d(x, y, normalized_data, title, zlabel, cmap)
