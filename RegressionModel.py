import os.path
import joblib
import pickle
import numpy as np
import math as math
import h5py
from datetime import datetime
import matplotlib.pyplot as plt

# Libraries for machine learning
import tensorflow as tf
from keras import models, layers, saving, initializers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Union

import TDS_Sim
import TDS_Material
import Model_Parameters

class RegressionModel:
    """Neural-Network Model that determines the concentrations and energies"""

    NumTraps:int = 0
    """Number of traps this model is trained for"""

    Concentrations:Union[float, str] = "Random"
    """Whether a pre-determined concentration of trapping sites is used (provide the concentration as mol/m^3), or if this concentration is unknown"""

    NumTraining:int = 10000
    """Number of training datapoints used for training"""

    n_cpu_cores:int = 16
    """Number of cpu cores to use for training/dataset generation"""

    DataName:str = None
    """Name of training dataset for saving/loading"""

    TrainedModelName:str = None
    """Name of trained model for saving/loading"""

    TrainedModel = None
    """Trained regression model"""

    OutputScalerName:str = None
    """Name of input/output scaler for saving/loading"""

    OutputScaler = None
    """Scaler for pre/post-processing"""

    n_variables:int = 0
    """Number of variables predicted by this model"""

    Material:TDS_Material.TDS_Material = None
    """Material parameters being used"""

    hp:Model_Parameters.Model_Parameters = None
    """Model hyperparameters and training parameters used, see ModelTrainingParameters.py"""

    def __init__(self, Material, MaxTraps, NumTraps, Concentrations, HyperParameters, NumTraining, Regenerate_Data, Regenerate_Training, n_cpu_cores):
        """Initializes the regression model. If data-sets are not present, generates these. If the model is not trained, trains it.

        Args:
            Material (TDS_Material.TDS_Material): Material parameters and TDS setup
            NumTraps (int): Number of trapping sites to use
            Concentrations (Union[float, str]): Either "Random" or a set concentration
            NumTraining (int): Number of data points to use for training
            Regenerate_Data (bool): Should the training data be re-generated
            Regenerate_Training (bool): Should the training be re-done
            n_cpu_cores (int): Number of cpu cores to use for training/dataset generation
            num_layers (int): Number of hidden layers in Neural-Network
            num_nodes (list): Number of nodes in each hidden layer of the NN
            Hyperparameter (dict): Dictiornary of hyperparameters (learning_rate, weight_decay, dropout_rate and epoch_coeff) defined by user
        """

        # Material and trapping parameters
        self.Material = Material
        self.MaxTraps = MaxTraps
        self.NumTraps = NumTraps
        self.Concentrations = Concentrations

        # Hyperparameters and model training parameters
        self.hp = HyperParameters
        self.NumTraining = NumTraining

        self.n_cpu_cores = n_cpu_cores

        # hYperparameter scaling factor
        self.n_variables = self.NumTraps
        if (isinstance(self.Concentrations, str)==True):
            self.n_variables *= 2

        dir_data = f"DataFiles/{Material.ExpName}/"
        dir_model = f"TrainedModels/{Material.ExpName}/"

        self.SettingsName =  f"{self.Material.ExpName}_{int(self.Material.dEMin)}EdMin_{self.Material.ntp}ntp_{self.NumTraps}_{self.Concentrations}_{self.Material.NRange[0]}Nmin_{self.Material.NRange[1]}Nmax_{self.NumTraining}"
        self.DataName = dir_data + self.SettingsName + ".hdf5"
        self.TrainedModelName = dir_model + self.SettingsName+".keras"
        self.OutputScalerEnergyName = dir_model + self.SettingsName+"_E"+".OutScale"
        self.OutputScalerConcentrationName = dir_model +self.SettingsName+"_N"+".OutScale"

        if ((os.path.isfile(self.DataName) == False) or (Regenerate_Data)):
            #Data needs to be generated
            self.GenerateData()

        if ((os.path.isfile(self.TrainedModelName) == False) or (Regenerate_Training)):
            #Train model
            self.Train()

        #Save model
        self.TrainedModel = saving.load_model(self.TrainedModelName)
        self.OutputScalerEnergy = pickle.load( open( self.OutputScalerEnergyName, "rb" ) )
        self.OutputScalerConcentration = pickle.load( open( self.OutputScalerConcentrationName, "rb" ) )

    def GenerateData(self):
        """Generates a dataset required for training"""
    
        # Define lists to store generated data
        Sample_NumTraps = []
        Sample_C_Traps = []
        Sample_E_Traps = []
        Sample_TDS_Curve_T = []
        Sample_TDS_Curve_j = []
       
        Res = joblib.Parallel(n_jobs=self.n_cpu_cores)(joblib.delayed(TDS_Sim.GenerateDataPoint)(i, self.Material, self.MaxTraps, self.NumTraps, self.Concentrations) for i in range(0, self.NumTraining))

        # Extracts data from results
        for i in range(0,self.NumTraining):
            Sample_NumTraps.append(Res[i][0])
            Sample_C_Traps.append(Res[i][1])
            Sample_E_Traps.append(Res[i][2])
            Sample_TDS_Curve_T.append(Res[i][3])
            Sample_TDS_Curve_j.append(Res[i][4])
            
        Sample_NumTraps = np.array(Sample_NumTraps)
        Sample_C_Traps = np.array(Sample_C_Traps)
        Sample_E_Traps = np.array(Sample_E_Traps)
        Sample_TDS_Curve_T = np.array(Sample_TDS_Curve_T)
        Sample_TDS_Curve_j = np.array(Sample_TDS_Curve_j)
        

        # Saves data into a h5py file
        with h5py.File(self.DataName, 'w') as hf:
            hf.create_dataset('Date', data=datetime.today().strftime('%Y-%m-%d') )

            hf.create_dataset('tds_curves', data=Sample_TDS_Curve_j)
            hf.create_dataset('temperature_data', data=Sample_TDS_Curve_T)
            hf.create_dataset('number_trapping_sites', data=Sample_NumTraps)
            hf.create_dataset('concentration_trapping_sites', data=Sample_C_Traps)
            hf.create_dataset('energy_trapping_sites', data=Sample_E_Traps)

    def Train(self):
        """Trains regression model"""

        with h5py.File(self.DataName, 'r') as hf:
            tds_curves = hf['tds_curves'][:]
            concentration_trapping_sites = hf['concentration_trapping_sites'][:]
            energy_trapping_sites = hf['energy_trapping_sites'][:]
        
        # Data pre-processing
        tds_curves = np.where(tds_curves < self.hp.flux_thershold, self.hp.flux_thershold, tds_curves)
        if self.hp.log_transformation:
            tds_curves = np.log10(tds_curves)

        # Zero-mean white Gaussian noise
        tds_curves_noise = tds_curves.copy()
        selected_features = range(tds_curves.shape[1])
        for feature_idx in selected_features:
            noise = np.random.normal(0, self.hp.noise_std_dev, tds_curves.shape[0])
            tds_curves_noise[:,feature_idx] += noise

        # Solution vector
        SolutionVector = []
        for i in range(0,self.NumTraining):
            sol = []
            for j in range(0,self.NumTraps):
                sol.append(energy_trapping_sites[i][j][1])

            if (isinstance(self.Concentrations, str)):
                for j in range(0,self.NumTraps):
                    sol.append(concentration_trapping_sites[i][j])
            SolutionVector.append(sol)
        SolutionVector = np.array(SolutionVector)
        
        # Output scalers
        SolScalerEnergy = MinMaxScaler(feature_range=(0,1))
        SolScalerConcentration = MinMaxScaler(feature_range=(0,1))

        energies = SolutionVector[:,:self.NumTraps]
        concentrations = SolutionVector[:,self.NumTraps:]

        energies_scaled = SolScalerEnergy.fit_transform(energies)
        concentrations_scaled = SolScalerConcentration.fit_transform(concentrations)

        SolutionVector_scaled = np.hstack((energies_scaled, concentrations_scaled))
        
        # Split train and test datasets
        TestsetSize = min(200, math.floor(0.2*self.NumTraining))
        X_train, X_test, y_train, y_test = train_test_split(tds_curves_noise, SolutionVector_scaled, test_size=TestsetSize)

        # Initialiser
        initialiser = initializers.HeNormal()

        # Build model
        NormLayer = layers.Normalization()
        NormLayer.adapt(X_train)
        model = models.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        model.add(NormLayer)

        for i in range(0,self.hp.num_layers_reg):
            model.add(layers.Dense(self.hp.nodes_reg[i]*self.n_variables, kernel_initializer=initialiser))
            model.add(layers.ReLU())
    
        model.add(layers.Dense(self.n_variables))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=self.hp.lr_reg, weight_decay=self.hp.wd_reg),
                        loss='mean_squared_error')
        
        # Train model
        model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=self.hp.batch_size,
                            epochs=self.hp.epoch_coeff_reg*self.n_variables,
                            verbose=1)
        
        # Save trained model
        model.save(self.TrainedModelName)

        # Save both output scalers
        pickle.dump(SolScalerEnergy, open(self.OutputScalerEnergyName, 'wb'))
        pickle.dump(SolScalerConcentration, open(self.OutputScalerConcentrationName, 'wb'))

        # Make predictions
        predictions = model.predict(X_test)

        predictions_energies = predictions[:,:self.NumTraps]
        predictions_concentrations = predictions[:,self.NumTraps:]

        predictions_energies = SolScalerEnergy.inverse_transform(predictions_energies)
        predictions_concentrations = SolScalerConcentration.inverse_transform(predictions_concentrations)

        predictions = np.hstack((predictions_energies, predictions_concentrations))

        y_test_energies = y_test[:,:self.NumTraps]
        y_test_concentrations = y_test[:,self.NumTraps:]

        y_test_energies = SolScalerEnergy.inverse_transform(y_test_energies)
        y_test_concentrations = SolScalerConcentration.inverse_transform(y_test_concentrations)

        y_test = np.hstack((y_test_energies, y_test_concentrations))

    def predict_energies_concentrations(self, TDS_Curve):
        """Uses the model to predict trapping energies and concentrations

        Args:
            TDS_Curve (np.ndarray): Input TDS curve

        Returns:
            outputs (tuple): containing
            - e (np.ndarray): Trapping energies
            - c (np.ndarray): Trapping site concentrations
        """
        # Data pre-processing
        TDS_Curve = np.where(TDS_Curve < self.hp.flux_thershold, self.hp.flux_thershold, TDS_Curve)
        if self.hp.log_transformation:
            TDS_Curve = np.log10(TDS_Curve)

        # Make predictions
        predictions = self.TrainedModel.predict(TDS_Curve)

        # Separate the predictions into energy and concentrations output
        predictions_energies = predictions[:,:self.NumTraps]
        predictions_concentrations = predictions[:,self.NumTraps:]

        # Inverse transform each output separately
        predictions_energies = self.OutputScalerEnergy.inverse_transform(predictions_energies)
        predictions_concentrations = self.OutputScalerConcentration.inverse_transform(predictions_concentrations)

        energies = predictions_energies[0]
        concentrations = predictions_concentrations[0]
        
        return energies,concentrations

    def Plot_Comparison(self, y_test, predictions):
        """Plots comparison between correct and predicted data

        Args:
            y_test (np.ndarray): Correct values
            predictions (np.ndarray): predicted values
        """

        if (isinstance(self.Concentrations, str)==False and self.NumTraps==1): # Single trap, fixed concentration
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[:]*1e-3, predictions[0][:],'r*')
            plt.xlabel('E_sample')
            plt.xlabel('E_predict')
        elif (isinstance(self.Concentrations, str)==True and self.NumTraps==1): # Single trap, Variable concentration
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[0][0]*1e-3, y_test[0][1],'ro', label='Correct')
            plt.plot(predictions[0][0]*1e-3,predictions[0][1],'b*', label='Predicted')
            for i in range(0,y_test.shape[0]):
                plt.plot(y_test[i][0]*1e-3, y_test[i][1],'ro')
                plt.plot(predictions[i][0]*1e-3,predictions[i][1],'b*')
                plt.plot([y_test[i][0]*1e-3, predictions[i][0]*1e-3],[y_test[i][1], predictions[i][1]],'k')
            plt.xlabel(r'$E_\text{trap}\;[\text{kJ}/\text{mol}]$')
            plt.ylabel(r'$C_\text{trap}\;[\text{mol}/\text{m}^3]$')
            plt.legend()
        elif (isinstance(self.Concentrations, str)==False and self.NumTraps==2): # Two traps, fixed concentration
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[0][0]*1e-3, y_test[0][1]*1e-3,'ro', label='Correct')
            plt.plot(predictions[0][0]*1e-3,predictions[0][1]*1e-3,'b*', label='Predicted')
            for i in range(0,y_test.shape[0]):
                plt.plot(y_test[i][0]*1e-3, y_test[i][1]*1e-3,'ro')
                plt.plot(predictions[i][0]*1e-3,predictions[i][1]*1e-3,'b*')
                plt.plot([y_test[i][0]*1e-3, predictions[i][0]*1e-3],[y_test[i][1]*1e-3, predictions[i][1]*1e-3],'k')
            plt.xlabel(r'$E_\text{1}\;[\text{kJ}/\text{mol}]$')
            plt.ylabel(r'$E_\text{2}\;[\text{kJ}/\text{mol}]$')
            plt.legend()
        elif (isinstance(self.Concentrations, str)==True and self.NumTraps==2): # Two traps, Variable concentration
            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.plot(y_test[0][0]*1e-3, y_test[0][1]*1e-3,'ro', label='Correct')
            ax1.plot(predictions[0][0]*1e-3,predictions[0][1]*1e-3,'b*', label='Predicted')
            for i in range(0,y_test.shape[0]):
                ax1.plot(y_test[i][0]*1e-3, y_test[i][1]*1e-3,'ro')
                ax1.plot(predictions[i][0]*1e-3,predictions[i][1]*1e-3,'b*')
                ax1.plot([y_test[i][0]*1e-3, predictions[i][0]*1e-3],[y_test[i][1]*1e-3, predictions[i][1]*1e-3],'k')
            ax1.set(xlabel=r'$E_\text{1}\;[\text{kJ}/\text{mol}]$')
            ax1.set(ylabel=r'$E_\text{2}\;[\text{kJ}/\text{mol}]$')
            ax1.legend()

            ax2.plot(y_test[0][2], y_test[0][3],'ro', label='Correct')
            ax2.plot(predictions[0][2],predictions[0][3],'b*', label='Predicted')
            for i in range(0,y_test.shape[0]):
                ax2.plot(y_test[i][2], y_test[i][3],'ro')
                ax2.plot(predictions[i][2],predictions[i][3],'b*')
                ax2.plot([y_test[i][2], predictions[i][2]],[y_test[i][3], predictions[i][3]],'k')
            ax2.set(xlabel=r'$C_\text{1}\;[\text{mol}/\text{m}^3]$')
            ax2.set(ylabel=r'$C_\text{2}\;[\text{mol}/\text{m}^3]$')
            ax2.legend()
        elif (isinstance(self.Concentrations, str)==False and self.NumTraps==3): # Three traps, fixed concentration
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(y_test[0][0]*1e-3, y_test[0][1]*1e-3, y_test[0][2]*1e-3,'ro', label='Correct')
            ax.plot(predictions[0][0]*1e-3,predictions[0][1]*1e-3,predictions[0][2]*1e-3,'b*', label='Predicted')
            for i in range(0,y_test.shape[0]):
                ax.plot(y_test[i][0]*1e-3, y_test[i][1]*1e-3, y_test[i][2]*1e-3,'ro')
                ax.plot(predictions[i][0]*1e-3,predictions[i][1]*1e-3,predictions[i][2]*1e-3,'b*')
                ax.plot([y_test[i][0]*1e-3, predictions[i][0]*1e-3],[y_test[i][1]*1e-3, predictions[i][1]*1e-3],[y_test[i][2]*1e-3, predictions[i][2]*1e-3],'k')

            ax.set_xlabel(r'$E_\text{1}\;[\text{kJ}/\text{mol}]$')
            ax.set_ylabel(r'$E_\text{2}\;[\text{kJ}/\text{mol}]$')
            ax.set_zlabel(r'$E_\text{3}\;[\text{kJ}/\text{mol}]$')
            ax.legend()
        elif (isinstance(self.Concentrations, str)==True and self.NumTraps==3): # Three traps, Variable concentration
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(121, projection='3d')
            ax.plot(y_test[0][0]*1e-3, y_test[0][1]*1e-3, y_test[0][2]*1e-3,'ro', label='Correct')
            ax.plot(predictions[0][0]*1e-3,predictions[0][1]*1e-3,predictions[0][2]*1e-3,'b*', label='Predicted')
            for i in range(0,y_test.shape[0]):
                ax.plot(y_test[i][0]*1e-3, y_test[i][1]*1e-3, y_test[i][2]*1e-3,'ro')
                ax.plot(predictions[i][0]*1e-3,predictions[i][1]*1e-3,predictions[i][2]*1e-3,'b*')
                ax.plot([y_test[i][0]*1e-3, predictions[i][0]*1e-3],[y_test[i][1]*1e-3, predictions[i][1]*1e-3],[y_test[i][2]*1e-3, predictions[i][2]*1e-3],'k')

            ax.set_xlabel(r'$E_\text{1}\;[\text{kJ}/\text{mol}]$')
            ax.set_ylabel(r'$E_\text{2}\;[\text{kJ}/\text{mol}]$')
            ax.set_zlabel(r'$E_\text{3}\;[\text{kJ}/\text{mol}]$')
            ax.legend()

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot(y_test[0][3], y_test[0][4], y_test[0][5],'ro', label='Correct')
            ax2.plot(predictions[0][3],predictions[0][4],predictions[0][5],'b*', label='Predicted')
            for i in range(0,y_test.shape[0]):
                ax2.plot(y_test[i][3], y_test[i][4], y_test[i][5],'ro')
                ax2.plot(predictions[i][3],predictions[i][4],predictions[i][5],'b*')
                ax2.plot([y_test[i][3], predictions[i][3]],[y_test[i][4], predictions[i][4]],[y_test[i][5], predictions[i][5]],'k')

            ax2.set_xlabel(r'$C_\text{1}\;[\text{mol}/\text{m}^3]$')
            ax2.set_ylabel(r'$C_\text{2}\;[\text{mol}/\text{m}^3]$')
            ax2.set_zlabel(r'$C_\text{3}\;[\text{mol}/\text{m}^3]$')
            ax2.legend()

        plt.draw()
        plt.pause(1.0e-10)