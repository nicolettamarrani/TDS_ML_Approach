class ExpDataParameters:
    """defines material, test and numerical parameters of the experimental data sets"""
    
    # Class variable to store all registered experiments
    _experiments = {}
    
    def __init__(self, ExpName):

        self.ExpName = ExpName

        match ExpName:

            case 'Novak_200':
                material = {
                        'NL' : 8.47e5,
                        'E_Diff' : 5690,
                        'D0' : 7.23e-8,
                        'C0' : 0.06,
                        'TrapRate': 1e13,
                        'MolMass': 55.847,
                        'MassDensity': 7.8474
                        }

                test = {      
                        'tRest' : 2700,
                        'HeatingRate': 0.055, # 200 C/h        
                        'Thickness' : 0.0063,
                        'TMax' : 873.15,
                        'TMin' : 293.15
                    }
                
                numerical = {
                        'dEMin': 10e3,
                        'ntp': 64,
                        'SampleFreq': 10,
                        'NRange': [1e-1, 1e1],
                        'ERange': [50e3, 150e3],
                        }
                
                HD_Trap = {
                    'HDT': False,
                    'HDT_NRange': None,
                    'HDT_ERange': None,
                }
                
            case 'Novak_100':
                material = {
                        'NL' : 8.47e5,
                        'E_Diff' : 5690,
                        'D0' : 7.23e-8,
                        'C0' : 0.06,
                        'TrapRate': 1e13,
                        'MolMass': 55.847,
                        'MassDensity': 7.8474
                        }

                test = {      
                        'tRest' : 2700,
                        'HeatingRate': 0.0275, # 100 C/h        
                        'Thickness' : 0.0063,
                        'TMax' : 873.15,
                        'TMin' : 293.15
                    }
                
                numerical = {
                        'dEMin': 10e3,
                        'ntp': 64,
                        'SampleFreq': 10,
                        'NRange': [1e-1, 1e1],
                        'ERange': [50e3, 150e3],
                        }
                
                HD_Trap = {
                    'HDT': False,
                    'HDT_NRange': None,
                    'HDT_ERange': None,
                }
                
            case 'Novak_50':
                material = {
                        'NL' : 8.47e5,
                        'E_Diff' : 5690,
                        'D0' : 7.23e-8,
                        'C0' : 0.06,
                        'TrapRate': 1e13,
                        'MolMass': 55.847,
                        'MassDensity': 7.8474
                        }

                test = {      
                        'tRest' : 2700,
                        'HeatingRate': 0.0139, # 50 C/h      
                        'Thickness' : 0.0063,
                        'TMax' : 873.15,
                        'TMin' : 293.15
                    }
                
                numerical = {
                        'dEMin': 10e3,
                        'ntp': 64,
                        'SampleFreq': 10,
                        'NRange': [1e-1, 1e1],
                        'ERange': [50e3, 150e3],
                        }
                
                HD_Trap = {
                    'HDT': False,
                    'HDT_NRange': None,
                    'HDT_ERange': None,
                }
                
            case 'Wei_Tsuzaki':
                material = {
                        'NL' : 8.47e5,
                        'E_Diff' : 5690,
                        'D0' : 7.23e-8,
                        'C0' : 0.6,
                        'TrapRate': 1e13,
                        'MolMass': 55.847,
                        'MassDensity': 7.8474
                        }

                test = {      
                        'tRest' : 120,
                        'HeatingRate': 0.0278,      
                        'Thickness' : 0.005,
                        'TMax' : 1500,
                        'TMin' : 293.15
                    }
                
                numerical = {
                        'dEMin': 10e3,
                        'ntp': 64,
                        'SampleFreq': 10,
                        'NRange': [1, 20],
                        'ERange': [40e3, 140e3],
                        }
                
                HD_Trap = {
                    'HDT': True,
                    'HDT_NRange': [1e4, 1e5],
                    'HDT_ERange': [10e3, 20e3],
                }

            
            case 'Depover':
                material = {
                        'NL' : 8.47e5,
                        'E_Diff' : 5690,
                        'D0' : 7.23e-8,
                        'C0' : 0.06,
                        'TrapRate': 1e13,
                        'MolMass': 55.847,
                        'MassDensity': 7.8474
                        }

                test = {      
                        'tRest' : 3600,
                        'HeatingRate': 0.167,    
                        'Thickness' : 0.001,
                        'TMax' : 873.15,
                        'TMin' : 293.15
                    }
                
                numerical = {
                        'dEMin': 10e3,
                        'ntp': 64,
                        'SampleFreq': 10,
                        'NRange': [1e-1, 1e1],
                        'ERange': [50e3, 110e3],
                        }
                
                HD_Trap = {
                    'HDT': True,
                    'HDT_NRange': [40, 100],
                    'HDT_ERange': [30e3, 50e3],
                }

        self.material = material
        self.test = test
        self.numerical = numerical
        self.HD_Trap = HD_Trap
        
        # Store this experiment in the class registry
        ExpDataParameters._experiments[ExpName] = {
            'material': material.copy(),
            'test': test.copy(), 
            'numerical': numerical.copy(),
            'HD_Trap': HD_Trap.copy()
        }
    
    @classmethod
    def register_experiment(cls, ExpName, material_param, test_param, numerical_param, HD_Trap_param):
        """Register a new experiment with the given parameters
        
        Args:
            ExpName (str): Name of the experiment
            material_param (dict): Material parameters
            test_param (dict): Test parameters  
            numerical_param (dict): Numerical parameters
        """
        cls._experiments[ExpName] = {
            'material': material_param.copy(),
            'test': test_param.copy(),
            'numerical': numerical_param.copy(),
            'HD_Trap': HD_Trap_param.copy()
        }
    
    @classmethod
    def get_experiment(cls, ExpName):
        """Get parameters for a registered experiment
        
        Args:
            ExpName (str): Name of the experiment
            
        Returns:
            tuple: (material_param, test_param, numerical_param) or (None, None, None) if not found
        """
        if ExpName in cls._experiments:
            exp_data = cls._experiments[ExpName]
            return exp_data['material'], exp_data['test'], exp_data['numerical'], exp_data['HD_Trap']
        return None, None, None
    
    @classmethod
    def list_experiments(cls):
        """List all registered experiments
        
        Returns:
            list: List of experiment names
        """
        return list(cls._experiments.keys())
    
    @classmethod
    def experiment_exists(cls, ExpName):
        """Check if an experiment is registered
        
        Args:
            ExpName (str): Name of the experiment
            
        Returns:
            bool: True if experiment exists, False otherwise
        """
        return ExpName in cls._experiments