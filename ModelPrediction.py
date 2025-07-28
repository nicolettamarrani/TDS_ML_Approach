import ModelEnsemble
import TDS_Material
import ExpDataProcessing

# Training parameters
NumTraining = 100000
NumVerification = 500
Regenerate_Data = False
Regenerate_Training = False
n_cpu_cores = 16

Traps = "Random"
Concentrations = "Random"
MaxTraps = 4

# Material, test and numerical parameters
ExpName = "Novak"
trap_model = "McNabb"
Material = TDS_Material.TDS_Material(ExpName, trap_model)

# Model creation and training
Model = ModelEnsemble.ModelEnsemble(Material, Traps, MaxTraps, Concentrations, NumTraining, Regenerate_Data, Regenerate_Training, n_cpu_cores)

# Experimental data fit
FileName = 'filename.xlsx'
Exp_Processed_Data = ExpDataProcessing.ExpDataProcessing(FileName, Material)
Exp_Temp = Exp_Processed_Data.Temperature
Exp_Flux = Exp_Processed_Data.Flux
Exp_TDS_Curve = Exp_Processed_Data.TDS_Curve

Exp_Predicted_Traps, Exp_Predicted_Concentrations, Exp_Predicted_Energies = Model.predict(Exp_TDS_Curve)
Model.PlotComparisonExpData(Exp_Temp, Exp_Flux, Exp_Predicted_Concentrations, Exp_Predicted_Energies)