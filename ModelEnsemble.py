import RegressionModel
import ClassificationModel
import TDS_Material
import Model_Parameters
import TDS_Sim

import matplotlib.pyplot as plt
import numpy as np
import os

class ModelEnsemble:
    """Combined classification and regression models, used to determine the number of trapping sites, and for this number, return concentrations and energies"""
    Material:TDS_Material.TDS_Material = None
    """Contains material, numerical parameters and TDS set-up"""

    HyperParameters:Model_Parameters.Model_Parameters = None
    """Model hyperparameters and training parameters used, see Model_Parameters.py"""

    RegModels:list[RegressionModel.RegressionModel] = None
    """list of regression models, each trained for a single amount of traps"""

    ClassModel = None
    """Classification model, trained to determine number of different traps"""

    MaxTraps = 0
    """Maximum amount of different trapping sites"""

    Concentrations = "Random"
    """Either "Random" or a fixed concentration of trapping sites"""

    NumTraining = 100
    """Number of data points used for training"""

    n_cpu_cores = 8
    """Number of cpu cores to use for training"""

    def __init__(self, Material, Traps, MaxTraps, Concentrations, HyperParameters, NumTraining, Regenerate_Data, Regenerate_Training, n_cpu_cores):
        """Initializes the set of models (and trains them if they are not yet trained)

        Args:
            Material (TDS_Material.TDS_Material): Material parameters and TDS setup
            NumTraps (str): Either a string (Random) or consider this model solely for a single amount of trapping sites
            MaxTraps (int): Maximum number of trapping sites to consider
            Concentrations (str): Either a string (random) or a prescribed concentration of trapping sites
            NumTraining (int): Number of fata points to use for training
            Regenerate_Data (bool): Should training data be generated anew (true), or re-use saved data (false)
            Regenerate_Training (bool): _Should the model be re-trained
            n_cpu_cores (int): Number of cpu cores to use for training
            HyperParameters (Model_Parameters.Model_Parameters): model training parameters
        """

        # Check directories and create missing ones
        directories_to_check = ['Figures', 'DataFiles', 'TrainedModels']
        material_exp_name = Material.ExpName  # Assume this is defined somewhere

        for dir_name in directories_to_check:
            directory_path = os.path.join(dir_name, material_exp_name)
            os.makedirs(directory_path, exist_ok=True)

        self.Material = Material
        self.MaxTraps = MaxTraps
        self.Concentrations = Concentrations
        self.NumTraining = NumTraining
        self.n_cpu_cores = n_cpu_cores
        self.Traps = Traps

        self.dir = f"Figures/{self.Material.ExpName}"

        if (isinstance(self.Traps, str)):
            # Classification + Regression Model
            self.RegModels = []
            for n_trap in range(1,MaxTraps+1):
                self.RegModels.append(RegressionModel.RegressionModel(Material, MaxTraps, n_trap, Concentrations, HyperParameters, NumTraining, Regenerate_Data, Regenerate_Training, n_cpu_cores))

            self.ClassModel = ClassificationModel.ClassificationModel(Material, MaxTraps, Concentrations, HyperParameters, NumTraining, self.RegModels, Regenerate_Training, n_cpu_cores)

        else:
            # Regression Only
            self.RegModel = RegressionModel.RegressionModel(Material, MaxTraps, Traps, Concentrations, HyperParameters, NumTraining, Regenerate_Data, Regenerate_Training, n_cpu_cores)

    def predict(self, TDS_Curves):
        """Predicts the number of trapping sites, energy and concentrations for a series of TDS curves

        Args:
            TDS_Curves (list[np.ndarray]): TDS curves to consider

        Returns:
            outputs (tuple): containing
            - Predicted_Traps (list[int]): Number of trapping sites predicted
            - Predicted_Concentrations (list[np.ndarray]): Predicted concentrations associated with traps
            - Predicted_Energies (list[np.ndarray]): Predicted trapping energies associated with traps
        """

        Predicted_Traps = []
        Predicted_Concentrations = []
        Predicted_Energies = []

        for TDS_Curve in TDS_Curves:

            TDS_Curve = np.array([TDS_Curve])

            if (isinstance(self.Traps, str)):
                n_traps = self.ClassModel.predict_traps(TDS_Curve)
                e, c = self.RegModels[n_traps-1].predict_energies_concentrations(TDS_Curve)
            else:
                n_traps = self.Traps
                e, c = self.RegModel.predict_energies_concentrations(TDS_Curve)

            Predicted_Traps.append(n_traps)
            Predicted_Energies.append(e)
            Predicted_Concentrations.append(c)

        return Predicted_Traps, Predicted_Concentrations, Predicted_Energies
    
    def PlotComparisonExpData(self, Temperature, TDS_Curve, Predicted_Concentrations, Predicted_Energies):
        
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel(r"$\text{Temperature} \;[\text{K}]$", fontsize=10)
        ax1.set_ylabel(r"$\text{Hydrogen Flux} \;[\text{mol}/\text{m}^2/\text{s}]$", fontsize=10)

        ax1.scatter(Temperature, TDS_Curve, label="Experimental Data", color="black", s=10)
        ax1.legend(loc='upper right', fontsize=8)

        fig2, ax2 = plt.subplots()
        ax2.set_xlabel(r"$\text{Temperature} \;[\text{K}]$", fontsize=10)
        ax2.set_ylabel(r"$\text{Hydrogen Flux} \;[\text{mol}/\text{m}^2/\text{s}]$", fontsize=10)

        N_traps = [lst.tolist() for array in Predicted_Concentrations for lst in array]
        E_traps = [lst.tolist() for array in Predicted_Energies for lst in array]



        for i in range(0,len(N_traps)):
            if N_traps[i] > 0 and E_traps[i] > 0:
                trap_num = i + 1
                N = [N_traps[i]]
                if (self.Material.TrapModel == TDS_Material.TRAPMODELS.MCNABB):
                    E = [[self.Material.E_Diff, E_traps[i]]]
                elif (self.Material.TrapModel == TDS_Material.TRAPMODELS.ORIANI):
                    E = [[self.Material.E_Diff, E_traps[i]+self.Material.E_Diff]]
                Sample = TDS_Sim.TDS_Sample(self.Material, N, E, False)
                Sample.Charge()
                Sample.Rest()
                T, J = Sample.TDS()

                ax2.plot(T, J, label=f"Trap {trap_num}")
            else:
                pass

        N = []
        E = []
        for i in range(len(E_traps)):
            if N_traps[i] > 0 and E_traps[i] > 0:
                N.append(N_traps[i])
                if (self.Material.TrapModel == TDS_Material.TRAPMODELS.MCNABB):
                    E.append([self.Material.E_Diff, E_traps[i]])
                elif (self.Material.TrapModel == TDS_Material.TRAPMODELS.ORIANI):
                    E.append([self.Material.E_Diff, E_traps[i]+self.Material.E_Diff])

        Sample = TDS_Sim.TDS_Sample(self.Material, N, E, False)
        Sample.Charge()    
        Sample.Rest()                             
        [T,J] = Sample.TDS() 

        ax1.plot(T, J,'--',color="blue",label=f"TDS Simulation Prediction")
        ax1.legend(loc='upper right', fontsize=8)

        ax2.plot(T, J, '--', label=f"TDS Simulation Prediction")
        ax2.legend(loc='upper right', fontsize=8)

        plt.draw()
        fig1.savefig(f'{self.dir}/ComparisonExpSim_'+self.ClassModel.SettingsName+r'.png', dpi=600)
        fig2.savefig(f'{self.dir}/TrapContribution_'+self.ClassModel.SettingsName+r'.png', dpi=600)

    def PlotComparisonTraps(self, predicted, actual, TDS_Curves, TDS_temp):
        """Plots comparisons between predicted and actual number of trapping sites

        Args:
            predicted (list[int]): Predicted number of trapping sites
            actual (list[int]): Actual number of trapping sites
            TDS_Curves (list[np.ndarray]): TDS curves used for plotting which are identified incorrectly
            TDS_temp (np.ndarray): Same as above
        """

        f, ax = self.ClassModel.PlotConfMatrix(actual, predicted)
        f.savefig(f'{self.dir}/confMat_'+self.ClassModel.SettingsName+r'.png', dpi=600)

        f = plt.figure(figsize=[18,6])
        ax = f.add_subplot(111)

        for i in range(0,len(actual)):
            if (actual[i]!=predicted[i]):
                labText = "Correct: "+str(actual[i])+", predicted: "+str(predicted[i])
                ax.plot(TDS_temp, TDS_Curves[i], label=labText)

        ax.set_xlabel(r'$T\;[\degree\text{C}]$')
        ax.set_ylabel(r'$j\;[\text{mol}/\text{m}^2\text{s}]$')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.draw()
        f.savefig(f'{self.dir}/IncTraps_'+self.ClassModel.SettingsName+r'.png', dpi=600)

    def PlotComparisonEnergies(self, predicted, actual):
        """Plots the comparison between predicted and correct trapping energies

        Args:
            predicted (list[np.adarray]):
            actual (list[np.adarray]):
        """
        self.PlotComparison(predicted, actual, 1e-3, r'E', r'[\text{kJ}]')

    def PlotComparisonConcentrations(self, predicted, actual):
        """Plots comparison between actual and predicted trapping site concentrations

        Args:
            predicted (list[np.adarray]):
            actual (list[np.adarray]):
        """
        self.PlotComparison(predicted, actual, 1.0, r'C', r'[\text{mol}/\text{m}^3]')

    def PlotComparison(self, predicted, actual, scale, axis_prefix, axis_suffix):
        fg = plt.figure(figsize=[18,10])

        ax = [None] * 5
        ax[0] = fg.add_subplot(3,3,1)
        ax[0].set_xlabel(r'$'+axis_prefix+r'_{real}'+axis_suffix+r'$')
        ax[0].set_ylabel(r'$'+axis_prefix+r'_{pred}'+axis_suffix+r'$')
        ax[0].set_title(r'Single Trap')

        ax[1] = fg.add_subplot(3,3,2)
        ax[1].set_xlabel(r'$'+axis_prefix+r'_{1}'+axis_suffix+r'$')
        ax[1].set_ylabel(r'$'+axis_prefix+r'_{2}'+axis_suffix+r'$')
        ax[1].set_title(r'Two Traps')

        ax[2] = fg.add_subplot(3,3,3, projection='3d')
        ax[2].set_xlabel(r'$'+axis_prefix+r'_{1}'+axis_suffix+r'$')
        ax[2].set_ylabel(r'$'+axis_prefix+r'_{2}'+axis_suffix+r'$')
        ax[2].set_ylabel(r'$'+axis_prefix+r'_{3}'+axis_suffix+r'$')
        ax[2].set_title(r'Three Traps')

        ax[3] = fg.add_subplot(3,3,4)
        ax[3].set_xlabel(r'$'+axis_prefix+r'_{1}'+axis_suffix+r'$')
        ax[3].set_ylabel(r'$'+axis_prefix+r'_{2}'+axis_suffix+r'$')
        ax[3].set_title(r'Four Traps (1/2)')

        ax[4] = fg.add_subplot(3,3,5)
        ax[4].set_xlabel(r'$'+axis_prefix+r'_{3}'+axis_suffix+r'$')
        ax[4].set_ylabel(r'$'+axis_prefix+r'_{4}'+axis_suffix+r'$')
        ax[4].set_title(r'Four Traps (2/2)')

        for i in range(0,len(actual)):
            ntraps_pred   = len(predicted[i])
            ntraps_actual = len(actual[i])

            if (ntraps_pred==ntraps_actual):
                match ntraps_pred:
                    case 1:
                        ax[0].plot(actual[i][0], predicted[i][0],'ko')
                    case 2:
                        ax[1].plot(actual[i][0]*scale, actual[i][1]*scale,'ro')
                        ax[1].plot(predicted[i][0]*scale, predicted[i][1]*scale,'b*')
                        ax[1].plot([actual[i][0]*scale, predicted[i][0]*scale],[actual[i][1]*scale, predicted[i][1]*scale],'k')
                    case 3:
                        ax[2].plot(actual[i][0]*scale, actual[i][1]*scale, actual[i][2]*scale,'ro')
                        ax[2].plot(predicted[i][0]*scale, predicted[i][1]*scale, predicted[i][2]*scale,'b*')
                        ax[2].plot([actual[i][0]*scale, predicted[i][0]*scale],[actual[i][1]*scale, predicted[i][1]*scale],[actual[i][2]*scale, predicted[i][2]*scale],'k')
                    case 4:
                        ax[3].plot(actual[i][0]*scale, actual[i][1]*scale,'ro')
                        ax[3].plot(predicted[i][0]*scale, predicted[i][1]*scale,'b*')
                        ax[3].plot([actual[i][0]*scale, predicted[i][0]*scale],[actual[i][1]*scale, predicted[i][1]*scale],'k')

                        ax[4].plot(actual[i][2]*scale, actual[i][3]*scale,'ro')
                        ax[4].plot(predicted[i][2]*scale, predicted[i][3]*scale,'b*')
                        ax[4].plot([actual[i][2]*scale, predicted[i][2]*scale],[actual[i][3]*scale, predicted[i][3]*scale],'k')

        plt.tight_layout()
        if (isinstance(self.Traps, str)):
            fg.savefig(f'{self.dir}/'+axis_prefix+"_"+self.ClassModel.SettingsName+r'.png', dpi=600)
        else:
            fg.savefig(f'{self.dir}/'+axis_prefix+"_"+self.RegModel.SettingsName+r'.png', dpi=600)


