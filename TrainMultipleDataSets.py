import ModelEnsemble
import TDS_Material
import TDS_Sim
import Model_Parameters

import matplotlib.pyplot as plt

# Training parameters
NumVerification = 500
Regenerate_Data = False
Regenerate_Training = False
n_cpu_cores = 16

Traps = "Random"
Concentrations = "Random"
MaxTraps = 4

# Material, test and numerical parameters
ExpName = "Novak_200"
trap_model = "McNabb"
Material = TDS_Material.TDS_Material(ExpName, trap_model)

# Model hyperparameters
HyperParameters = Model_Parameters.Model_Parameters(ParameterSet="optimised")


for NumTraining in [100, 1000, 10000, 25000, 100000]:
    # Model creation and training
    Model = ModelEnsemble.ModelEnsemble(Material, Traps, MaxTraps, Concentrations, HyperParameters, NumTraining, Regenerate_Data, Regenerate_Training, n_cpu_cores)

    # Verification
    TDS_Curves, Actual_Traps, Actual_Concentrations, Actual_Energies, TDS_Temp = TDS_Sim.SimDataSet(Material, NumVerification, MaxTraps, Traps, Concentrations, n_cpu_cores)
    Predicted_Traps, Predicted_Concentrations, Predicted_Energies = Model.predict(TDS_Curves)

    Model.PlotComparisonTraps(Predicted_Traps, Actual_Traps, TDS_Curves, TDS_Temp)
    Model.PlotComparisonConcentrations(Predicted_Concentrations, Actual_Concentrations)
    Model.PlotComparisonEnergies(Predicted_Energies, Actual_Energies)

    plt.close('all')