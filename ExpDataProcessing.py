import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import TDS_Material
import Model_Parameters


class ExpDataProcessing:

    M:TDS_Material.TDS_Material = None

    hp:Model_Parameters.Model_Parameters = None
    
    FileName = None
    
    def __init__(self, FileName, Material, HyperParameters):
        
        self.hp = HyperParameters

        self.filename = FileName

        self.M = Material
        
        self.load_and_process_data()
        self.smooth_and_downsample_data()
        

    def load_and_process_data(self):
        """Load experimental data from a file."""
        
        folder_path = os.path.dirname(self.filename)
        file_name = os.path.basename(self.filename)
        
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_excel(file_path, header = 0)

        self.temperature_raw = data.iloc[:, 0].to_numpy()
        self.desorption_rate_raw = data.iloc[:, 1].to_numpy()
        
    def smooth_and_downsample_data(self):
        """Smooth and downsample data based on desired number of data points (ntp)."""

        coef = self.M.mass_density/1.01
        desorption_rate = self.desorption_rate_raw*coef*self.M.Thickness/2

        window_length = max(3, 2 * round(len(desorption_rate) / self.M.ntp))
        desorption_rate_smoothed = signal.savgol_filter(desorption_rate, window_length, 2)

        #Model temperature range (ntp points evenly spaced from Tmin K to Tmax K, not including Tmax) 
        temperature_model = np.linspace(self.M.TMin, self.M.TMax, self.M.ntp+1)

        # Experimental data
        temperature_raw = np.array(self.temperature_raw)  # Experimental temperatures (not evenly spaced)
        desorption_rate_smoothed = np.array(desorption_rate_smoothed)     # Corresponding fluxes at T_exp

        # Create an interpolation function based on experimental data
        interp_function = interp1d(temperature_raw, desorption_rate_smoothed, kind='cubic', bounds_error=False, fill_value=(desorption_rate_smoothed[0], self.hp.flux_threshold))

        # Interpolate flux values at the model's temperature points 
        desorption_rate_downsampled = interp_function(temperature_model)
        desorption_rate_downsampled[(temperature_model > np.max(temperature_raw))] = self.hp.flux_threshold

        # Remove datapoint corresponding to Tmax
        temperature_model = temperature_model[:-1]
        desorption_rate_downsampled = desorption_rate_downsampled[:-1]

        desorption_rate = desorption_rate_downsampled[:]
        temperature = temperature_model[:]

        # Model inputs
        self.Temperature = temperature
        self.Flux = desorption_rate
        TDS_Curve = desorption_rate
        self.TDS_Curve = [tf.convert_to_tensor(TDS_Curve)]