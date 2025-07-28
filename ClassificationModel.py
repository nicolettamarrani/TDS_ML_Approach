import os.path
import numpy as np
import h5py
import math as math
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers, models, saving, initializers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from typing import Union
import seaborn as sns

import TDS_Material
import Model_Parameters

class ClassificationModel:
    """Model which predicts the number of traps contained within a TDS curve"""

    MaxTraps:int = 1
    """Range of traps which can be predicted by this model"""

    Concentrations:Union[float, str] = "Random"
    """Whether a pre-determined concentration of trapping sites is used (provide the concentration as mol/m^3), or if this concentration is unknown"""

    NumTraining:int = 100
    """Number of training datapoints used per trapping site number"""

    SettingsName:str = None
    """Name used to save/load files"""

    TrainedModelName = None
    """Name used to save/load files"""

    TrainedModel = None
    """Trained model (as saved/loaded from file)"""

    n_cpu_cores = 8
    """Number of cpu cores to use for training models"""
    
    Material:TDS_Material.TDS_Material = None
    """Material parameters used, see TDS_Material.py"""

    hp:Model_Parameters.Model_Parameters = None
    """Model hyperparameters and training parameters used, see ModelTrainingParameters.py"""

    def __init__(self, Material, MaxTraps, Concentrations, HyperParameters, NumTraining, RegModels, Regenerate_Training, n_cpu_cores):
        """Creates the classification model. If the model is not yet trained, also performs the training

        Args:
            Material (TDS_Material.TDS_Material): Material/TDS setup
            MaxTraps (int): Max number of traps to consider
            Concentrations (Union[float, str]): Either "Random" or a prescribed concentration
            NumTraining (int): Number of data points used for training
            RegModels (list[Models]) (or (list[DataNames]): Regression models (or list of Regression Model names) that predict concentrations for traps, used to re-cycle training data
            Regenerate_Training (bool): Should the previously trained model be used (false) or this model trained anew (true)
            n_cpu_cores (int): Number of cpu cores to use for training
        """

        # Link input parameters to class parameters
        self.Material = Material
        self.MaxTraps = MaxTraps
        self.Concentrations = Concentrations

        self.hp = HyperParameters

        self.NumTraining = NumTraining
        self.n_cpu_cores = n_cpu_cores

        dir_model = f"TrainedModels/{self.Material.ExpName}"

        # Check if model is pre-trained and load-able, otherwise train it
        self.SettingsName = f"{self.Material.ExpName}_{int(self.Material.dEMin)}EdMin_{self.Material.ntp}ntp_{self.MaxTraps}_{self.Concentrations}_{self.Material.NRange[0]}Nmin_{self.Material.NRange[1]}Nmax_{self.NumTraining}"
        self.TrainedModelName = f"{dir_model}/Classification_"+self.SettingsName+".keras"

        if ((os.path.isfile(self.TrainedModelName) == False) or (Regenerate_Training)):
            self.Train(RegModels)

        self.TrainedModel = saving.load_model(self.TrainedModelName)


    def GetData(self, RegModels):
        """Collects data used in regression models

        Args:
            RegModels (Models): Regression models for varying trap numbers

        Returns:
            outputs (tuple): containing
            - All_TDS (np.ndarray): Matrix containing all TDS curves
            - All_Traps (np.ndarray): Vector containing the number of traps per TDS curve
        """
        All_TDS = []
        All_Traps = []

        for i in range(0,self.MaxTraps):
            if isinstance(RegModels[i], str):
                DName = RegModels[i]
            else:
                DName = RegModels[i].DataName

            with h5py.File(DName, 'r') as hf:
                tds_curves = hf['tds_curves'][:]
                Number_trapping_sites = hf['number_trapping_sites'][:]

            for j in range(0,self.NumTraining):
                All_TDS.append(tds_curves[j])
                All_Traps.append(Number_trapping_sites[j])

        All_TDS = np.array(All_TDS)
        All_Traps = np.array(All_Traps)

        return All_TDS, All_Traps

    def Train(self, RegModels):
        """Trains classification model without cross-validation"""

        """Args:
            RegModels (Models): Regression models used to get training data
        """

        All_TDS, All_Traps = self.GetData(RegModels)
        Labels = label_binarize(All_Traps, classes=range(1,self.MaxTraps+1))

        # Data pre-processing
        All_TDS = np.where(All_TDS < self.hp.flux_thershold, self.hp.flux_thershold, All_TDS)
        if self.hp.log_transformation:
            All_TDS = np.log10(All_TDS)

        # Zero-mean white Gaussian noise
        All_TDS_noise = All_TDS.copy()
        selected_features = range(All_TDS.shape[1])
        for feature_idx in selected_features:
            noise = np.random.normal(0, self.hp.noise_std_dev, All_TDS.shape[0])
            All_TDS_noise[:,feature_idx] += noise

        # Re-build and train model with complete dataset - to save
        TestsetSize = min(200, math.floor(0.2*self.NumTraining))*self.MaxTraps
        X_train, X_test, y_train, y_test = train_test_split(All_TDS_noise, Labels, test_size=TestsetSize)

        # Initialiser
        initialiser = initializers.HeNormal()

        # Build model
        NormLayer = layers.Normalization()
        NormLayer.adapt(X_train)
        model = models.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        model.add(NormLayer)

        # Add Dense layers based on num_layers
        for i in range(0,self.hp.num_layers_class):
            model.add(layers.Dense(self.hp.nodes_class[i]*self.MaxTraps, kernel_initializer=initialiser))
            model.add(layers.ReLU())

        # Add Output layer
        model.add(layers.Dense(self.MaxTraps, activation="softmax"))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=self.hp.lr_class, weight_decay=self.hp.wd_class),
                    loss='categorical_crossentropy')

        model.fit(x=X_train, y=y_train,
                                validation_data=(X_test, y_test),
                                batch_size=self.hp.batch_size,
                                epochs=self.hp.epoch_coeff_class*self.MaxTraps,
                                verbose=1)

        # Make predictions and plot comparison
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)+1
        y_test_classes = np.argmax(y_test, axis=1)+1
        self.PlotConfMatrix(y_test_classes, y_pred_classes) 

        # Save trained model
        model.save(self.TrainedModelName)

    def predict_traps(self, TDS_Curve):
        """Predicts the number of traps for a given TDS curve

        Args:
            TDS_Curve (np.ndarray): Input TDS curve

        Returns:
            n_traps (int): Number of trapping site types
        """

        # Data pre-processing
        TDS_Curve = np.where(TDS_Curve < self.hp.flux_thershold, self.hp.flux_thershold, TDS_Curve)
        if self.hp.log_transformation:
            TDS_Curve = np.log10(TDS_Curve)

        # Model prediction
        prediction = self.TrainedModel.predict(TDS_Curve, batch_size=1, verbose=0)
        n_traps = np.argmax(prediction, axis=1)+1

        return n_traps[0]

    def PlotConfMatrix(self, y_test, y_pred):
        """Plots the confusion matrix, indicating how accurate the results are

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            outputs (tuple): containing
            - f (plt.figure): Matplotlib figure handle
            - ax (plt.axis): matplotlib axis handle
        """

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=range(1,self.MaxTraps+1), normalize='true')
        print("Confusion Matrix:\n", cm)

        # Plot confusion matrix
        f = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=range(1,self.MaxTraps+1), yticklabels=range(1,self.MaxTraps+1))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix for Number of Traps Prediction')
        plt.draw()
        return f, ax
    