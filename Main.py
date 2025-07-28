import ModelEnsemble
import TDS_Material
import TDS_Sim
import Model_Parameters
import ExpDataProcessing

import matplotlib.pyplot as plt

# Training parameters
NumTraining = 10
NumVerification = 50
Regenerate_Data = False
Regenerate_Training = False
n_cpu_cores = 16

Traps = "Random"
Concentrations = "Random"
MaxTraps = 4

# Material, test and numerical parameters
ExpName = "Novak_Test"
trap_model = "McNabb"
Material = TDS_Material.TDS_Material(ExpName, trap_model)

# Model hyperparameters
HyperParameters = Model_Parameters.Model_Parameters(ParameterSet="optimised")

# Model creation and training
Model = ModelEnsemble.ModelEnsemble(Material, Traps, MaxTraps, Concentrations, HyperParameters, NumTraining, Regenerate_Data, Regenerate_Training, n_cpu_cores)

# Verification
TDS_Curves, Actual_Traps, Actual_Concentrations, Actual_Energies, TDS_Temp = TDS_Sim.SimDataSet(Material, NumVerification, MaxTraps, Traps, Concentrations, n_cpu_cores)
Predicted_Traps, Predicted_Concentrations, Predicted_Energies = Model.predict(TDS_Curves)

Model.PlotComparisonTraps(Predicted_Traps, Actual_Traps, TDS_Curves, TDS_Temp)
Model.PlotComparisonConcentrations(Predicted_Concentrations, Actual_Concentrations)
Model.PlotComparisonEnergies(Predicted_Energies, Actual_Energies)

plt.close('all')

# Experimental data fit
FileName = 'filename.xlsx'
Exp_Processed_Data = ExpDataProcessing.ExpDataProcessing(FileName, Material)
Exp_Temp = Exp_Processed_Data.Temperature
Exp_Flux = Exp_Processed_Data.Flux
Exp_TDS_Curve = Exp_Processed_Data.TDS_Curve

Exp_Predicted_Traps, Exp_Predicted_Concentrations, Exp_Predicted_Energies = Model.predict(Exp_TDS_Curve)
Model.PlotComparisonExpData(Exp_Temp, Exp_Flux, Exp_Predicted_Concentrations, Exp_Predicted_Energies)