import numpy as np
from sklearn.preprocessing import MinMaxScaler
import rasterio
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

grid_size = 10  # Example grid size

class DataSim:
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
    def downsample_data(data, factor):
        if factor < 1:
            raise ValueError("Downsampling factor must be greater than or equal to 1")
        
        new_shape = (data.shape[0] // factor, data.shape[1] // factor)
        downsampled_data = zoom(data, (new_shape[0] / data.shape[0], new_shape[1] / data.shape[1]), order=1)
        
        return downsampled_data
    
    @staticmethod
    def remove_outliers(data, threshold=3):
        mean = np.mean(data)
        std = np.std(data)
        outlier_mask = np.abs(data - mean) > threshold * std
        data[outlier_mask] = mean
        return data

    @staticmethod
    def normalize_gis_data(data, min_value, max_value):
        scaler = MinMaxScaler(feature_range=(min_value, max_value))
        return scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(data.shape)

    @staticmethod
    def extract_raster_info(file_path):
        with rasterio.open(file_path) as src:
            grid_size = (src.height, src.width)
            data = src.read(1)
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
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, elevation, cmap=cmap, edgecolor='k')
    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

def plot_2d(data, title, cmap):
    plt.figure(figsize=(10, 7))
    plt.imshow(data, cmap=cmap, origin='lower')
    plt.colorbar(label='Elevation (ft)')
    plt.title(title)
    plt.xlabel('X (grid units)')
    plt.ylabel('Y (grid units)')
    plt.show()

# User-defined parameters for tuning
outlier_threshold = 3  # Change this value to adjust outlier detection
downsampling_factor = 100  # Change this value to adjust downsampling

# Example usage
weather_temp, weather_humidity = DataSim.generate_weather_data()
fuel_loading = DataSim.generate_fuel_loading()

file_path = '/Users/adam/Documents/GitHub/ML_Fire_Prediction_Modeling/GIS_Data/raw_tiff/FSTopo Berthoud Pass 394510545.tiff'
file_path, grid_size, min_value, max_value = ObtainActualData.extract_raster_info(file_path)
normalized_data = ObtainActualData.load_and_normalize_gis_data(file_path, grid_size[0], min_value, max_value)

# Apply outlier removal and downsampling
normalized_data = ObtainActualData.remove_outliers(normalized_data, threshold=outlier_threshold)
downsampled_data = ObtainActualData.downsample_data(normalized_data, downsampling_factor)

# Now plot the downsampled data in 2D and 3D
x = np.linspace(0, 100, downsampled_data.shape[1])
y = np.linspace(0, 100, downsampled_data.shape[0])

title = "Downsampled Grid"
zlabel = "Elevation (ft)"
cmap = 'hot'

# 3D plot
plot_3d(x, y, downsampled_data, title, zlabel, cmap)

# 2D plot
plot_2d(downsampled_data, title, cmap)
