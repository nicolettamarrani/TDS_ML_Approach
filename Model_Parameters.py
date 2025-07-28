class Model_Parameters:
    """Defines model hyper-parameters"""

    # Pre-processing parameters
    flux_thershold:float = 1e-12
    """Clips all flux values to 1e-12. Added for stability"""

    log_transformation:bool = True
    """Flag to apply log transformation"""

    # Zero-mean white Gaussian noise related parameters
    noise_std_dev:float = 0.05

    # Training parameters
    epoch_coeff_reg:int = 20

    epoch_coeff_class:int = 10

    batch_size:int = 32

    def __init__(self, ParameterSet:str, UserDefinedParameters:dict = None):
        """Initializes the model hyper-parameters (allowing for easy switching between optimised and userdefined parameter sets)

        Args:
            ParameterSet (str): reference for choice of parameter set ("optimised" or "user_defined")
            UserDefinedParameters (dict): contains user defined parameters, which will be adopted for model training if ParameterSet is set to "user_defined"
        """

        match ParameterSet:

            case 'optimised':

                # Regression model
                self.num_layers_reg = 5
                self.nodes_reg = [64, 64, 32, 16, 8]

                self.lr_reg = 1e-3
                self.wd_reg = 1e-3

                # Classification model
                self.num_layers_class = 5
                self.nodes_class = [64, 64, 32, 16, 8]

                self.lr_class = 1e-3
                self.wd_class = 1e-3

            case 'user_defined':
                pass
