# TDS_ML_Approach
This repository provides a machine learning-based framework for parameter identification from thermal desorption spectroscopy (TDS) spectra. A multi-Neural Network (NN) model is developed and trained exclusively on synthetic
data to predict trapping parameters directly from experimental data. The model comprises two multi-layer, fully connected, feed-forward NNs trained with backpropagation.

Code to perform analysis of TDS measurements through machine learning. Files to run:
- **Main.py**: Generates data and trains (or loads) models for a selected test case, performs model evaluation, loads and pre-processes experimental data,
  fits experimental data (i.e., predicts the number of trap sites and corresponding binding energies/densities), plots the fitted model prediction (including trap contributions) against experimental data
- **TrainMultipleDataSets.py**: Parameter sweep to compare multiple trained models

Other relevant files:
- **TDS_Material.py**: Contains the material-specific parameters and TDS setup configurations
- **Model_Parameters.py**: Defines model hyperparameters and training parameters
- **TDS_Sim.py**: Numerical TDS simulation
- **Classificationmodel.py**: Neural network to determine the number of trapping sites in the material
- **RegressionModel.py**: Neural network that, for a given number of trapping sites, determines the binding energy and density of those sites
- **ModelEnsemble.py**: Combines both of the above neural networks into a single model, first predicting the number of traps and then their density/energies
- **ExpDataParameters.py**: Stores material parameters, TDS test conditions and numerical simulation settings for experimental datasets
- **ExpDataProcessing.py**: Handles experimental data processing (i.e., loading, smoothing and downsampling)

Current setup:
All inputs and files required to analyse the experimental data from Novak et al. for a high-strength AISI 4340 tempered martensitic steel (test case 1 of the manuscript) are included, allowing the results from Fig. 11 to be reproduced.
Model hyperparameters are defined as specified in the manuscript.
