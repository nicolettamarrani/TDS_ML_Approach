import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
from scipy.interpolate import interp1d
import os


class ExpDataProcessing:
    """Process experimental TDS data from file"""

    def __init__(self, file_name, material, hyperparameters):
        self.file_name = file_name
        self.material = material
        self.hp = hyperparameters
        
        # Initialize data attributes
        self.temperature_raw = None
        self.desorption_rate_raw = None
        self.Temperature = None
        self.Flux = None
        self.TDS_Curve = None
        
        # Process the data
        self._load_data()
        self._process_data()

    def _load_data(self):
        """Load experimental data from Excel file"""
        data = pd.read_excel(self.file_name, header=0)
        self.temperature_raw = data.iloc[:, 0].to_numpy()
        self.desorption_rate_raw = data.iloc[:, 1].to_numpy()

    def _process_data(self):
        """Apply corrections, smooth, and downsample the data"""
        # Apply material-specific corrections
        corrected_rate = self._apply_material_corrections()
        
        # Smooth the data
        smoothed_rate = self._smooth_data(corrected_rate)
        
        # Downsample to model grid
        self._downsample_data(smoothed_rate)
        
        # Create TensorFlow tensor
        self.TDS_Curve = [tf.convert_to_tensor(self.Flux)]

    def _apply_material_corrections(self):
        """Apply density and thickness corrections to desorption rate"""
        coef = self.material.mass_density / 1.01
        return self.desorption_rate_raw * coef * self.material.Thickness / 2

    def _smooth_data(self, desorption_rate):
        """Apply Savitzky-Golay smoothing filter."""
        window_length = max(3, 2 * round(len(desorption_rate) / self.material.ntp))
        
        # Ensure window_length is odd and not larger than data
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(desorption_rate))
        
        return signal.savgol_filter(desorption_rate, window_length, 2)

    def _downsample_data(self, desorption_rate_smoothed):
        """Downsample data to model temperature grid using interpolation"""
        # Create model temperature grid (ntp points from TMin to TMax, excluding TMax)
        temperature_model = np.linspace(
            self.material.TMin, 
            self.material.TMax, 
            self.material.ntp + 1
        )

        # Create interpolation function
        interp_function = interp1d(
            self.temperature_raw, 
            desorption_rate_smoothed, 
            kind='cubic', 
            bounds_error=False, 
            fill_value=(desorption_rate_smoothed[0], self.hp.flux_threshold)
        )

        # Interpolate to model grid
        desorption_rate_downsampled = interp_function(temperature_model)
        
        # Set extrapolated values to threshold
        beyond_range = temperature_model > np.max(self.temperature_raw)
        desorption_rate_downsampled[beyond_range] = self.hp.flux_threshold

        # Remove last point (TMax) and store results
        self.Temperature = temperature_model[:-1]
        self.Flux = desorption_rate_downsampled[:-1]

    def get_raw_data(self):
        """Get original unprocessed data"""
        return self.temperature_raw, self.desorption_rate_raw

    def get_processed_data(self):
        """Get processed data ready for modeling"""
        return self.Temperature, self.Flux

    def plot_comparison(self):
        """Plot raw vs processed data for visualization"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Raw data
            ax1.plot(self.temperature_raw, self.desorption_rate_raw, 'b-', alpha=0.7)
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Raw Desorption Rate (wppm/s)')
            ax1.set_title('Raw Data')
            ax1.grid(True, alpha=0.3)
            
            # Processed data
            ax2.plot(self.Temperature, self.Flux, 'r-', linewidth=2)
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Processed Flux (mol/m^3/s)')
            ax2.set_title('Processed Data')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")