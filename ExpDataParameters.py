class ExpDataParameters:
    """defines material, test and numerical parameters of the experimental data sets"""
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

        self.material = material
        self.test = test
        self.numerical = numerical