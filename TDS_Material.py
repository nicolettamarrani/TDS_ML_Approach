import math as math
from enum import Enum
from ExpDataParameters import ExpDataParameters

TRAPMODELS = Enum('TRAPMODELS','MCNABB ORIANI MCNABB2')
"""Indicates which trapping model is to be used, either MCNABB or ORIANI"""

class TDS_Material:
    """Contains the material parameters and TDS set-up"""    

    # Experimental Data Reference
    ExpName:str    = None
    """Label for experimental data (user defined)"""

    # General constants
    R:float         = 8.3144
    """Gas constant"""
    
    NA:float        =6.022e23  
    """Avogadro Constant"""
    
    RoomTemp:float  = 293.15   
    """Room Temperature [K]"""
    
    f_H2_room:float = 0.0      
    """Fugacity of hydrogen within room/TDS [Pa]"""
    
    f0:float        = 1.0e6    
    """Reference pressure [Pa]"""
    
    # Material Parameters
    TrapRate:float  = None      
    """rate constant for trapping/detrapping (Note: should be a Debye frequency-like constant (10^13))"""

    NL:float = None
    """Concentration of lattice sites [mol/m3]"""
    
    D0:float = None         
    """Diffusion rate constant (before being modified by temperature) [m^2/s]"""
    
    E_Diff:float = None     
    """diffusion energy (used for temperature scaling) [J]"""
    
    k_T:list[float]  = [1.0e4, 1.0e3] 
    """boundary flux absorbtion-desorbtion rates       [mol/s]"""

    C0:float = None
    """Initial hydrogen lattice concentration [mol/m3]"""

    mass_density:float = None
    """Mass density [g/cm3]"""

    molar_mass:float = None
    """Molar mass [g/mol]"""

    # Transport Theory
    TrapModel = TRAPMODELS.MCNABB
    """Model used to determine trapping rate, either TRAPMODELS.MCNABB or TRAPMODELS.ORIANI"""

    # TDS setup
    time_Charge:float     = 12*3600   
    """time left charging [s]"""
    
    Temp_Charge:float     = 293.15    
    """temperature at which samples are charged [K]"""
    
    pCharge:float         = 1e6     
    """hydrogen fugacity during charging [Pa]"""

    tRest:float       = None     
    """time sample is being transfered from charging to TDS [s]"""
    
    HeatingRate:float = None     
    """heating rate of TDS [K/s]"""

    TMin:float            = None
    """minimum temperature of TDS experiment [K]"""

    TMax:float            = None
    """maximum temperature of TDS experiment [K]"""
    
    tTDS:float        = None     
    """total time over which TDS is performed [s]"""

    Thickness:float = None     
    """sample thickness [m]"""
    
    nElems:int   = 25           
    """for obtaining the solution, number of linear finite elements used"""
    
    deltaTime:float = None          
    """time increment used [s]"""
    
    SampleFreq:int = 10         
    """time increments between saves"""

    dEMin:float = 10e3
    """minimum difference in energy of traps [J]"""

    High_Density_Trap = False
    """If True modifies data generation and model training procedures to account for a high-density low-energy trap"""

    HDT_NRange = None
    """Density range of high-density low-energy trap"""

    HDT_ERange = None
    """De-trapping energy range of high-density low-energy trap"""

    def __init__(self, ExpName:str, material_param:dict=None, test_param:dict=None, numerical_param:dict=None, HD_Trap_param:dict=None, trap_model:str='McNabb'):
        """Initializes the parameters based on the provided reference experiment name (allowing for easy switching between investigating different experimental datasets)

        Args:
            ExpName (str): reference for experimental data
            material_param (dict, optional): Material parameters. If None, will try to load from ExpDataParameters
            test_param (dict, optional): Test parameters. If None, will try to load from ExpDataParameters  
            numerical_param (dict, optional): Numerical parameters. If None, will try to load from ExpDataParameters
            HDT (tuple): Indicates presence of high-density low-energy trap. If True, trap exists and density/energy range needs to be specified
            trap_model (str): indicates which trapping model is used, either MCNABB or ORIANI
        """
        self.ExpName = ExpName

        # Try to get parameters from ExpDataParameters if not provided
        if material_param is None or test_param is None or numerical_param is None:
            # First check if experiment is already registered
            exp_material, exp_test, exp_numerical, exp_HD_Trap = ExpDataParameters.get_experiment(ExpName)
            
            if exp_material is None:
                # If not registered, try to create ExpDataParameters object (this will work for predefined experiments)
                try:
                    exp_params = ExpDataParameters(ExpName)
                    exp_material = exp_params.material
                    exp_test = exp_params.test
                    exp_numerical = exp_params.numerical
                    exp_HD_Trap = exp_params.HD_Trap
                except:
                    # If that fails and no parameters provided, raise error
                    if material_param is None or test_param is None or numerical_param is None or exp_HD_Trap is None:
                        raise ValueError(f"Experiment '{ExpName}' not found in predefined experiments and insufficient parameters provided")
            
            # Use loaded parameters as defaults, override if parameters provided by user
            material_param = material_param if material_param is not None else exp_material
            test_param = test_param if test_param is not None else exp_test
            numerical_param = numerical_param if numerical_param is not None else exp_numerical
            HD_Trap_param = HD_Trap_param if HD_Trap_param is not None else exp_HD_Trap

        # High-density, low-energy trap parameters
        if HD_Trap_param:
            self.High_Density_Trap = HD_Trap_param['HDT']
            self.HDT_NRange = HD_Trap_param['HDT_NRange']
            self.HDT_ERange = HD_Trap_param['HDT_ERange']
        else:
            self.High_Density_Trap = False
            self.HDT_NRange = None
            self.HDT_ERange = None
            HD_Trap_param = {
                'HDT': False,
                'HDT_NRange': None,
                'HDT_ERange': None,
            }

        # Register the experiment with ExpDataParameters
        ExpDataParameters.register_experiment(ExpName, material_param, test_param, numerical_param, HD_Trap_param)

        # Material properties
        self.NL = material_param['NL']
        self.E_Diff = material_param['E_Diff']
        self.D0 = material_param['D0']
        if self.D0 is None:
            D_Roomtemp = 1e-9
            self.D0 = D_Roomtemp*math.exp(self.E_Diff/self.R/self.RoomTemp)  
        self.C0 = material_param['C0']
        self.TrapRate = material_param['TrapRate']
        self.mass_density = material_param['MassDensity']
        self.molar_mass = material_param['MolMass']

        # Test parameters
        self.tRest = test_param['tRest']
        self.HeatingRate = test_param['HeatingRate']
        self.Thickness = test_param['Thickness']
        self.TMax = test_param['TMax']
        self.TMin = test_param['TMin']

        # TDS time calculation
        self.tTDS = (self.TMax-self.RoomTemp)/self.HeatingRate 
        
        # Numerical parameters 
        self.dEMin = numerical_param['dEMin']
        self.ntp = numerical_param['ntp'] 
        self.SampleFreq = numerical_param['SampleFreq']
        self.NRange = numerical_param['NRange']
        self.ERange = numerical_param['ERange']
        self.deltaTime = self.tTDS/self.ntp/self.SampleFreq # for high trapping rate can increase sample frequency to use smaller time increments while still keeping the same amount of ntp

        # Transport model
        if trap_model == 'McNabb':
            self.TrapModel = TRAPMODELS.MCNABB
        elif trap_model == 'Oriani':
            self.TrapModel = TRAPMODELS.ORIANI
    
    def PrintProperies(self):
        """Prints the material parameters and TDS setup"""
        
        print("Material Parameters:")
        print("\t Concentration of lattice sites: "+str(self.NL)+" mol/m3 ("+str(self.NL*self.NA)+" sites/m3)")
        print("\t Base diffusivity: "+str(self.D0)+" m2/s")
        print("\t Lattice diffusivity energy: "+str(self.E_Diff*1e-3)+" kJ/mol")
        print("\t Boundary absorption/desorption rate constant: "+str(self.k_T)+" mol/s")
        print("\t Trapping model: "+str(self.TrapModel))
        print("\t Trap rate: "+str(self.TrapRate))
        print("\n")
        print("TDS parameters:")
        print("\t Initial lattice hydrogen concentration: "+str(self.C0)+"mol/m3")
        print("\t Resting for "+str(self.tRest))
        print("\t TDS until T="+str(self.RoomTemp+self.tTDS*self.HeatingRate)+"K with heating rate "+str(self.HeatingRate)+" K/s ("+str(self.tTDS)+"s total time)")
        print("\t Sample Thickness: "+str(self.Thickness*1e3)+" mm")
        print("\n")