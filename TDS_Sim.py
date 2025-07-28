import numpy as np
import scipy as sp
import math as math
import matplotlib.pyplot as plt
import random
import joblib
from typing import Union
import time
import pandas as pd

import TDS_Material

class TDS_Sample:

    #Material and simulation settings
    
    M:TDS_Material.TDS_Material = None
    """Reference to material parameters being used (see TDS_Material)"""
    
    Visualize:bool = None
    """Whether to use live-plotting of concentrations, only use for debugging, very slow"""
    
    TStep:float = None
    """Time increment used for each time step [s]"""
    
    conv_crit:float = 1.0e-9  
    """concergence criteria, below which the simulations consider themselves converged"""

    N_traps:list[float] = None
    """Trapping concentrations, formatted as [C_1, C_2]"""
    
    E_traps:list[list:[float]] = None
    """Trapping energies, formatted as [[E_trap1, E_detrap1],[E_trap2, E_detrap2]]"""
    
    E_T:float = None
    """Surface absorption energy [J/mol]"""
    
    E_Tr:float = None
    """Surface desorption energy [J/mol]"""
    
    #History parameters (containing trapping concentrations)
    HistNew:np.ndarray = None
    HistOld:np.ndarray = None

    #shape function values and integration weights
    N_ip = 2
    N = None
    G = None
    w = None
    NDofs = None
    XPlot = None
    StateVec = None
    OldStateVec = None

    def __init__(self, 
                 Material:  TDS_Material.TDS_Material, #Material parameters
                 N_traps:   list[float], 
                 E_traps:   list[float],
                 viz:       bool):        #plotting while simulating (very slow)
        """Initializes TDS simulation settings

        Args:
            Material (TDS_Material.TDS_Material): Material parameters to use (See TDS_Material)
            N_traps (list[float]): Concentrations of trapping sites, formatted as [C_trap1, C_trap2]
            E_traps (list[float]): Trapping energies, formatted as [[E_trap1, E_detrap1],[E_trap2, E_detrap2]]
            viz (bool): Whether to do live plotting, very slow
        """
        
        self.M = Material
        self.Visualize = viz                            
        self.TStep = self.M.deltaTime
        self.C0 = self.M.C0

        # Charging Time
        self.time_Charge = self.M.time_Charge

        # State vector, saving lattice concentrations
        self.NDofs = self.M.nElems+1
        self.StateVec = np.zeros([self.NDofs,1])
        self.OldStateVec = np.zeros([self.NDofs,1])
    
        # pre-calculate shape functions
        self.N_ip = 2   #Number of integration points per element
        self.N = np.zeros((self.M.nElems, self.N_ip, 2)) # Element shape functions
        self.G = np.zeros((self.M.nElems, self.N_ip, 2)) # Element shape function gradients
        self.w = np.zeros((self.M.nElems, self.N_ip))    # Integration point weights

        for el in range(0, self.M.nElems):
            dx = self.M.Thickness/2/self.M.nElems    #element size, using symmetry to simulate half the thickness
            x_elem = (np.array([0.0, 1.0])+el)*dx
            
            # integration locations and weights for an element of length [0,1]
            x_ip, w_ip = sp.special.roots_legendre(self.N_ip)
            x_ip = (x_ip+1)/2
            w_ip = w_ip/2

            for ip in range(0,self.N_ip):
                self.N[el][ip][0] = x_ip[ip]
                self.N[el][ip][1] = 1-x_ip[ip]
                self.G[el][ip][0] = -1/dx
                self.G[el][ip][1] = 1/dx
                self.w[el][ip] = w_ip[ip]*dx
        
        self.Set_Params(N_traps, E_traps)

        # required for plotting while solving
        self.XPlot = np.arange(0, self.M.Thickness/2+1.0e-10, dx)
        if (self.Visualize):
            self.PlotState(np.zeros(1), np.zeros(1))

    def PlotState(self,
                  tvec: np.ndarray,     #time vector
                  jvec: np.ndarray):    #TDS flux vector
        """Plots the current hydrogen contents, and time response

        Args:
            tvec (np.ndarray): At which times are datapoints saved
            jvec (np.ndarray): Hydrogen fluxes that are saved
        """
        
        fig, self.ax = plt.subplots(2)
        if (self.M.TrapModel==TDS_Material.TRAPMODELS.MCNABB):
            self.p1, = self.ax[0].plot(self.XPlot, self.StateVec[0:self.M.nElems+1])
        else:
            self.p1, = self.ax[0].plot(self.XPlot, self.StateVec)
        self.p2, = self.ax[1].plot(tvec/3600, jvec)
        plt.draw()
        plt.pause(1.0e-10)

    # updates any plots
    def UpdatePlot(self, 
                   tvec: np.ndarray,     #time vector
                   jvec: np.ndarray):    #TDS flux vector
        """Updates previously generated plots

        Args:
            tvec (np.ndarray): At which times are datapoints saved
            jvec (np.ndarray): Hydrogen fluxes that are saved
        """
        if (self.M.TrapModel==TDS_Material.TRAPMODELS.MCNABB):
            self.p1.set_ydata(self.StateVec[0:self.M.nElems+1])
        else:
            self.p1.set_ydata(self.StateVec)
        self.ax[0].set_ylim(min(0.0,np.min(self.StateVec)), np.max(self.StateVec)*1.1)
        self.p2.set_data(tvec/3600, jvec)
        self.ax[1].set_xlim(0, np.max(tvec)/3600)
        self.ax[1].set_ylim(np.min(jvec), np.max(jvec))
        plt.draw()
        plt.pause(1.0e-10)
    
    def Set_Params(self, 
                   N_traps: list[float], #Concentration of trapping sites [mol/m^3]
                   E_traps: list[float]): #Energy levels of trapping sites
        """ Sets the trapping site material parameters, and initializes history data structures

        Args:
            N_traps (list[float]): Concentrations of trapping sites, formatted as [C_trap1, C_trap2]
            E_traps (list[float]): Trapping energies, formatted as [[E_trap1, E_detrap1],[E_trap2, E_detrap2]]
        """
        
        self.N_traps = N_traps
        self.E_traps = E_traps

        #calculate temperature dependent rate parameters based on room temp inputs
        self.E_T = -self.M.R*self.M.RoomTemp*math.log(self.M.k_T[0])
        self.E_Tr = -self.M.R*self.M.RoomTemp*math.log(self.M.k_T[1])

        #initialize data structure for storing trapping site occupancy
        if (self.M.TrapModel==TDS_Material.TRAPMODELS.MCNABB): #if using McNabb, save trapping site concentrations as well
            self.NDofs = (self.M.nElems+1)*(len(self.N_traps)+1)
            self.StateVec = np.zeros([self.NDofs,1])
            self.OldStateVec = np.zeros([self.NDofs,1])

    #Copies New state to Old state, to proceed to next time increment
    def Commit(self):
        """ Overwrites the old history state (trapping concentrations) to now match the new state
        """
        self.OldStateVec = self.StateVec.copy()
            
    def AverageConcentrations(self):
        """Returns the average lattice and trapping site concentrations

        Returns:
            outputs (tuple): containing
            - CL (float): Average lattice concentration
            - CT (float): Average trapping site concentration
        """
        
        CL = np.average(self.StateVec[0:self.M.nElems+1])
        CT = np.zeros(len(self.N_traps))
        for i in range(0,len(self.N_traps)):
            if (self.M.TrapModel==TDS_Material.TRAPMODELS.ORIANI):
                E = self.E_traps[i] 
                NT = self.N_traps[i]
                eTerm = math.exp(E/self.M.R/self.T)
                CT[i] = NT * eTerm * CL/self.M.NL/(1.0+(eTerm-1.0)*CL/self.M.NL)
            elif (self.M.TrapModel==TDS_Material.TRAPMODELS.MCNABB):
                CT[i] = np.average(self.StateVec[(1+i)*(self.M.nElems+1):(i+2)*(self.M.nElems+1)+1])
        return CL, CT
        
    def Charge(self):
        """ Performs the charging process of a sample, using the charging conditions set in M
        """

        self.T = self.M.Temp_Charge
        self.TOLD = self.M.Temp_Charge
        self.p = self.M.pCharge

        if self.M.C0 is not None:
            self.StateVec.fill(self.M.C0)
            self.OldStateVec.fill(self.M.C0)

            if (self.M.TrapModel == TDS_Material.TRAPMODELS.MCNABB):
                for i,E,Nt in zip(range(0,len(self.E_traps)), self.E_traps, self.N_traps):
                    dE = E[1] - E[0] 
                    eTerm = math.exp(dE/self.M.R/self.T)
                    CT = Nt * eTerm * self.M.C0/self.M.NL/(1.0+(eTerm-1.0)*self.M.C0/self.M.NL)
                    for n in range(0,self.M.nElems+1):
                        self.StateVec[(1+i)*(self.M.nElems+1)+n,0] = CT
            self.Commit()

        else:
            t = 0.0 #current time

            # Resize vector to store time data
            jVec = np.zeros(math.ceil(self.time_Charge/self.TStep))
            tVec = np.zeros(math.ceil(self.time_Charge/self.TStep))

            #time stepping loop
            step = 0
            while t<self.time_Charge:

                tVec[step] = t     
                jVec[step] = self.DoStep()

                t += self.TStep
                self.Commit()
                step += 1

                if ((self.Visualize) & (step%100==0)):
                    self.UpdatePlot(tVec[0:step], jVec[0:step])

            if (self.Visualize):
                self.UpdatePlot(tVec, jVec)

    def Rest(self):
        """leave the sample at room temperature/pressure (e.g. during transferring from charging to TDS)
        """
        t = 0.0

        self.T = self.M.RoomTemp
        self.TOLD = self.M.RoomTemp
        self.p = self.M.f_H2_room

        jVec = np.zeros(math.ceil(self.M.tRest/self.TStep))
        tVec = np.zeros(math.ceil(self.M.tRest/self.TStep))

        step = 0
        while t<self.M.tRest:

            tVec[step] = t     
            jVec[step] = self.DoStep()

            t += self.TStep
            self.Commit()
            step += 1

            if (((self.Visualize) & (step%100==0))):
                self.UpdatePlot(tVec[0:step], jVec[0:step])

        if (self.Visualize):
            self.UpdatePlot(tVec[0:step], jVec[0:step])
        
    def TDS(self):
        """Performs TDS, keeping track of hydrogen fluxes and temperatures

        Returns:
            outputs (tuple): containing
            - TVec (list[float]): Vector containing temperature data points [K]
            - jVec (list[float]): Vector containing hydrogen exit fluxes for an unit surface [mol/m^2/s]
        """
        
        t = 0.0

        self.T = self.M.RoomTemp
        self.p = 0.0

        # Initialize ouytput vectors
        jVec = np.zeros(self.M.ntp) #hydrogen flux [mol/m^2/s]
        tVec = np.zeros(self.M.ntp) #time vector [s]
        TVec = np.zeros(self.M.ntp) #Temperature Vector [K]

        #perform actual simulation
        step = 0
        nSample = 0
        while step < self.M.ntp*self.M.SampleFreq:
            #print("TDS: "+str(t), end='\r')
            self.TOLD = self.T
            self.T = self.M.RoomTemp + t*self.M.HeatingRate
            
            j = self.DoStep()
            if (step%self.M.SampleFreq==0):
                tVec[nSample] = t     
                jVec[nSample] = j
                TVec[nSample] = self.T
                nSample += 1

            t += self.TStep
            self.Commit()
            step += 1

            if (((self.Visualize) & (step%100==0))):
                self.UpdatePlot(tVec[0:step], jVec[0:step])

        if (self.Visualize):
            self.UpdatePlot(tVec[0:step], jVec[0:step])   
            
        return TVec, jVec   

    def DoStep(self):
        """Calculates a single time increment, return exit flux for this increment

        Returns:
            j (float): hydrogen exit fluxes for an unit surface [mol/m^2/s]
        """
        
        conv = False
        K, f, j = self.GetKf()

        it = 0
        while (conv==False and it<20):
            dState = -np.linalg.solve(K, f)
            
            #line-search
            if True:
                e0 = np.tensordot(f, dState)
                self.StateVec += dState
                
                K, f, j = self.GetKf()
                e1 = np.tensordot(f, dState)
                
                if (e0==e1):
                    lineSearch = 1.0
                else:
                    lineSearch = -e0/(e1-e0)
                lineSearch = min(1.0, max(0.1, lineSearch))
                self.StateVec += dState*-(1-lineSearch)
            else:
                self.StateVec += dState
            
            K, f, j = self.GetKf()
            
            err = 0
            for i in range(0,len(f)):
                err += abs(f[i,0]*dState[i,0])/self.NDofs

            if (it==0):
                err0 = err

            it += 1
            if (err<self.conv_crit or err/err0<1e-3):
                conv = True
        return j

    def GetKf(self):
        """Gets the tangent matrix, required within DoStep to perform non-linear NR iterations

        Returns:
            outputs (tuple): containing
            - K (np.ndarray): Tangent Matrix
            - f (np.ndarray): Out-of-balance force vector
            - j (float): hydrogen exit flux [mol/m^2/s]
        """

        HeatRate = (self.T-self.TOLD)/self.TStep

        # Initialize tangent matrix and force vector
        K = np.zeros([self.NDofs, self.NDofs])
        f = np.zeros([self.NDofs, 1])

        #integrate all elements
        for el in range(0,self.M.nElems):

            C = self.StateVec[el:el+2,0]    #lattice concentrations relevant to current element
            COld = self.OldStateVec[el:el+2,0]  # lattice concentrations at previous time increment

            WLumped = self.N[el][0]*0.0     #Lumped integration weights

            #Sum over integration points
            for ip in range(0,self.N_ip):

                #Calculate local concentrations, and time derivative
                cloc = np.matmul(self.N[el][ip], C)
                dcloc= np.matmul(self.N[el][ip], C-COld)/self.TStep

                #lumped integration weights
                WLumped += self.w[el][ip]*self.N[el][ip]

                #diffusion
                GtG = np.outer(self.G[el][ip].T,self.G[el][ip])
                D_eff = self.M.D0*math.exp(-self.M.E_Diff/self.M.R/self.T)
                f[el:el+2,0]        += self.w[el][ip]*D_eff*np.matmul(GtG,C)
                K[el:el+2,el:el+2]  += self.w[el][ip]*D_eff*GtG    

            for n in range(0,2):
                CL_dof = el+n
                
                #capacity
                f[CL_dof,0]       += WLumped[n]*(C[n]-COld[n])/self.TStep
                K[CL_dof,CL_dof]  += WLumped[n]/self.TStep

                #traps
                if (self.M.TrapModel == TDS_Material.TRAPMODELS.ORIANI):
                    for E,Nt in zip(self.E_traps, self.N_traps):
                        eTerm = math.exp(E/self.M.R/self.T)
                        deTerm_dt = -E/self.M.R/self.T**2*eTerm*HeatRate
                        theta_L = C[n]/self.M.NL
                        theta_L = min(1.0,max(0.0,theta_L))
                        
                        #trapping due to changes in concentration, dC_T/dC_L
                        cap =       Nt/self.M.NL * eTerm                      /(1.0+theta_L*(eTerm-1.0))**2
                        dcap = -2.0*Nt/self.M.NL * eTerm*(eTerm-1.0)/self.M.NL/(1.0+theta_L*(eTerm-1.0))**3

                        #trapping due to changes in temperature, dC_T/deTerm
                        TrapThermal  = Nt*(theta_L-theta_L**2)/((1.0+theta_L*(eTerm-1.0))**2)
                        dTrapThermal = Nt/self.M.NL/((1.0+theta_L*(eTerm-1.0))**2)* ( 1.0-2.0*theta_L+2*(eTerm-1.0)*(theta_L-theta_L**2)/(1.0+theta_L*(eTerm-1.0)) )

                        f[CL_dof,0]       += WLumped[n] * cap * (C[n]-COld[n])/self.TStep        + WLumped[n] * TrapThermal * deTerm_dt
                        K[CL_dof,CL_dof]  += WLumped[n] * (cap + dcap*(C[n]-COld[n]))/self.TStep + WLumped[n] * dTrapThermal * deTerm_dt
    
                if (self.M.TrapModel == TDS_Material.TRAPMODELS.MCNABB):
                    for i,E,Nt in zip(range(0,len(self.E_traps)), self.E_traps, self.N_traps):
                        CT_Dof = (1+i)*(self.M.nElems+1)+el+n 
                        CT = self.StateVec[CT_Dof,0]
                        CT_Old = self.OldStateVec[CT_Dof,0]
                        
                        theta_L = C[n]/self.M.NL
                        dthetaL_dCL = 1.0/self.M.NL
                        theta_L = min(1.0,max(0.0,theta_L))
                        
                        theta_T = CT/Nt
                        dthetaT_dCT = 1.0/Nt
                        theta_T = min(1.0,max(0.0,theta_T))
                        
                        c_fac = self.M.TrapRate*Nt
                        
                        r1_div_r2 = math.exp((E[1]-E[0])/self.M.R/self.T)
                        r2 = math.exp(-E[1]/self.M.R/self.T)
                        
                        #  v_abs = k_abs \theta_L * (1-\theta_T)
                        v_abs     = theta_L*(1.0-theta_T)
                        dv_abs_dL =         (1.0-theta_T)
                        dv_abs_dT = theta_L*    -1.0     

                        #  v_des = k_des \theta_T * (1-\theta_L) 
                        v_des     = theta_T*(1.0-theta_L)
                        dv_des_dL = theta_T*    -1.0     
                        dv_des_dT =         (1.0-theta_L)

                        #net trapping rate (positive=increase in trapping)
                        v     = c_fac*r2*            (v_abs*r1_div_r2    -v_des)
                        dv_dT = c_fac*r2*dthetaT_dCT*(dv_abs_dT*r1_div_r2-dv_des_dT) 
                        dv_dL = c_fac*r2*dthetaL_dCL*(dv_abs_dL*r1_div_r2-dv_des_dL)
                        
                        f[CL_dof,0]      += WLumped[n] * v
                        K[CL_dof,CL_dof] += WLumped[n] * dv_dL
                        K[CL_dof,CT_Dof] += WLumped[n] * dv_dT
                        
                        f[CT_Dof,0]      -= WLumped[n] * v
                        K[CT_Dof,CL_dof] -= WLumped[n] * dv_dL
                        K[CT_Dof,CT_Dof] -= WLumped[n] * dv_dT
                        
                        f[CT_Dof,0]      += WLumped[n] * (CT-CT_Old)/self.TStep
                        K[CT_Dof,CT_Dof] += WLumped[n] * 1.0/self.TStep

        if (np.isnan(K).any() or np.isnan(f).any()):
            print("D_eff=")
            print(D_eff)
            print("C=")
            print(self.StateVec)
            input("NanHewr Enter to continue...")

        #boundary
        Cb = self.StateVec[0,0]
        k_in = math.exp(-self.E_T/self.M.R/self.T)
        k_out =  math.exp(-self.E_Tr/self.M.R/self.T)
        j = - k_in * math.sqrt(self.p/self.M.f0) + k_out * max(0.0,Cb)
        f[0]   += j
        if (Cb>=0):
            K[0,0] += k_out

        return K, f, j

    def TrappingDynamic(self, CT_Old, CT_New, CL_New, E, Nt, TStep):
        """Calculates the trapping site concentrations, and trapping rates
        
        Args:
            CT_Old (np.ndarray): Trapping site concentrations at previous time increment
            CT_New (np.ndarray): Trapping site concentrations at current time increment
            CL_New (float): Lattice concentration at current time increment
            E (list[float]): Trapping energies
            Nt (list[float]): Trapping site concentrations
            TStep (float): Time increment
            
        Returns:
            outputs (tuple): containing
            - CT_New (np.ndarray): Trapping site concentrations at current time increment
            - v_trap (float): Trapping rate
            - dv_dCL (float): Derivative of trapping rate with respect to lattice concentration
        """
        
        ## return mapping scheme, defining local state vector as [CT_New, v_trapping]
        conv = False
        it = 0

        Loc_State = np.zeros([2*len(self.N_traps)+2,1])
        Loc_State[0,0] = CL_New
        for i in range(0,len(Nt)):
            Loc_State[2*(i+1),0] = CT_New[i]
            
        while (conv == False):
            # apply sensible limits to CL, CT to retain stability
            
            CL = max(0.0,Loc_State[0,0])
            CT = []
            for i in range(0,len(Nt)):
                CT[i] = min(max(0.0, Loc_State[2*(i+1),0]), Nt[i])

            fvec = np.zeros([2*len(Nt)+2,1])
            KMat = np.zeros([2*len(Nt)+2,2*len(Nt)+2])
            
            # Change in concentrations
            fvec[0] = (Loc_State[0,0]-CL_New)/TStep - Loc_State[1,0]
            KMat[0,0] = 1.0/TStep
            KMat[0,1] = -1.0
            for i in range(0,len(Nt)):
                fvec[2*(i+1),0] = (Loc_State[2*(i+1),0]-CT_Old[i])/TStep - Loc_State[2*(i+1)+1,0]
                KMat[2*(i+1),2*(i+1)] = 1.0/TStep
                KMat[2*(i+1),2*(i+1)+1] = -1.0
            
            # Trapping rates
            for i in range(0,len(Nt)):
                c_fac = self.M.TrapRate*Nt[i]
                        
                #calculate absorption and desorption rates as:
                #  v_abs = k_abs \theta_L * (1-\theta_T) 
                v_abs = CL/self.M.NL*(1-CT[i]/Nt[i]) *math.exp(-E[i][0]/self.M.R/self.T)
                dv_abs_dL = (1-CT[i]/Nt[i])/self.M.NL*math.exp(-E[i][0]/self.M.R/self.T)
                dv_abs_dT = -CL/self.M.NL/Nt[i]      *math.exp(-E[i][0]/self.M.R/self.T)

                #  v_des = k_des \theta_T * (1-\theta_L) 
                v_des = CT[i]/Nt[i]*(1-CL/self.M.NL)  *math.exp(-E[i][1]/self.M.R/self.T)
                dv_des_dL = -1.0/self.M.NL*CT[i]/Nt[i]*math.exp(-E[i][1]/self.M.R/self.T)
                dv_des_dT = (1-CL/self.M.NL)/Nt[i]    *math.exp(-E[i][1]/self.M.R/self.T)

                #net trapping rate
                v     = c_fac*(v_abs-v_des)
                dv_dT = c_fac*(dv_abs_dT-dv_des_dT)
                dv_dL = c_fac*(dv_abs_dL-dv_des_dL)

                fvec[2*(i+1)+1,0] = Loc_State[2*(i+1)+1,0] - v
                KMat[2*(i+1)+1,2*(i+1)+1] = 1.0
                KMat[2*(i+1)+1,0] = -dv_dL
                KMat[2*(i+1)+1,2*(i+1)] = -dv_dT
            
            #de-trapping rate (for C_L)
            fvec[1,0] = Loc_State[1,0]
            KMat[1,1] = 1.0
            for i in range(0,len(Nt)):
                fvec[1,0] += Loc_State[2*(i+1)+1,0]
                KMat[1,2*(i+1)+1] = 1.0

            # update increment
            dLocState = -np.linalg.solve(KMat,fvec)
            Loc_State +=  dLocState

            err = 0.0
            for i in range(0,len(dLocState)):
                err += abs(dLocState[i]*fvec[i])

            # check for convergence
            it += 1
            if (it>40 or err<1.0e-18):
                conv = True

        #extract a consistent tangent matrix, and provide outputs
        CMat = np.zeros([2*len(Nt)+2,1])
        CMat[0,0] = -1.0/TStep

        dMat = np.matmul(np.linalg.inv(KMat), CMat)

        v_trap = Loc_State[1,0]
        dv_dCL = dMat[1,0]
        
        for i in range(0,len(Nt)):
            CT_New[i] = Loc_State[2*(i+1),0]
        
        return CT_New, v_trap, dv_dCL


def takeSecond(elem):
    return elem[1]

# Generates a single TDS curve based on (mostly) random binding energies and trapping site concentration
def GenerateDataPoint(i             :int,
                      Material      :TDS_Material.TDS_Material,
                      MaxTraps      :int,
                      NumTraps      :Union[int, str],
                      Concentration :Union[float, str]):
    """Generates a single data point with randomly set energies and trapping site concentrations

    Args:
        i (int): Only used for printing data point number
        Material (TDS_Material.TDS_Material): Material properties and TDS set-up
        NumTraps (Union[int, str]): either an int to indicate the number of traps to use, or "Random" to generate a random distribution
        Concentration (Union[float, str]): either a float to indicate the concentration of traps to use for all sites, or "Random" to generate a random distribution

    Returns:
        outputs (tuple): containing
        - NumTraps (int): Number of trapping sites used
        - N_traps (list[float]): Concentrations of trapping sites
        - E_traps (list[float]): Trapping energies
        - T (np.ndarray): Output TDS curve, temperature axis
        - J (np.ndarray): Output TDS curve, hydrogen flux axis
    """
    
    PlotWhileSolving = False #whether to generate plots when simulating (much slower)

    if (isinstance(NumTraps, int)): #number of trapping sites, either an integer or "random"
        NumTraps = NumTraps       
    else :
        NumTraps = random.randint(1,MaxTraps)
    
    traps = []

    for t in range(0,NumTraps):
        validPoint = False
        while validPoint == False:
            E_abs = Material.E_Diff #set equal to lattice activation energy
            E_des = random.uniform(Material.ERange[0], Material.ERange[1])

            if (isinstance(Concentration, str)):
                N = random.uniform(Material.NRange[0], Material.NRange[1])
            else:
                N = Concentration
    
            goodDist = True
            for E in traps:
                if abs(E[1]-E_des)<Material.dEMin:
                    goodDist = False

            validPoint = goodDist

        traps.append([E_abs, E_des, N])
        traps.sort(key=takeSecond)
        
    N_traps = []
    E_traps = []
    for t in traps:
        N_traps.append(t[2])
        if (Material.TrapModel == TDS_Material.TRAPMODELS.MCNABB):
            E_traps.append([t[0], t[1]])
        elif (Material.TrapModel == TDS_Material.TRAPMODELS.ORIANI):
            E_traps.append(t[1]-t[0])

    
    print()
    print(str(i)+":"+"\n\t N="+str(N_traps)+"\n\t E="+str(E_traps))

    #perform TDS experiment within simulation
    Sample = TDS_Sample(Material, N_traps, E_traps, PlotWhileSolving) #initializes material
    Sample.Charge()    #performs charging
    Sample.Rest()                              #leave at atmospheric pressure
    [T,J] = Sample.TDS()       #perform TDS

    return NumTraps, N_traps, E_traps, T, J

def SimDataSet(Material         :TDS_Material.TDS_Material,
               NumVerification  :int,
               MaxTraps         :int,              
               NumTraps         :Union[int, str],   
               Concentrations   :Union[float, str],
               n_cpu_cores      :int              
               ):
    """Generates a full data set

    Args:
        Material (TDS_Material.TDS_Material): Material parameters
        NumVerification (int): Number of data points to generate
        NumTraps (Union[int, str]): Number of trapping sites, either a defined number or "Random"
        Concentrations (Union[float, str]): #Concentration of trapping sites, either a defined number or "Random"
        n_cpu_cores (int): Number of cpu cores to run simulations in parallel

    Returns:
        outputs (tuple): containing
        - Sample_TDS_Curve_j (list[np.ndarray]): All output TDS curves their hydrogen flux axis
        - Sample_NumTraps (list[int]): Number of trapping sites used in each sample
        - Sample_C_Traps (list[list[float]]): Trapping sites concentrations
        - Sample_E_Traps (list[list[float]]): Trapping site binding energies
        - TDS_temp (np.ndarray): TDS curve temperature axis, assumed the same for all data points and thus only outputted once
    """

    Res = joblib.Parallel(n_jobs=n_cpu_cores)(joblib.delayed(GenerateDataPoint)(i, Material, MaxTraps, NumTraps, Concentrations) for i in range(0, NumVerification))

    # Extracts data from results
    Sample_NumTraps = []
    Sample_C_Traps = []
    Sample_E_Traps = []
    Sample_TDS_Curve_T = []
    Sample_TDS_Curve_j = []

    for i in range(0,NumVerification):
        Sample_NumTraps.append(Res[i][0])
        Sample_C_Traps.append(Res[i][1])
        Sample_TDS_Curve_T.append(Res[i][3])
        Sample_TDS_Curve_j.append(Res[i][4])

        sol = []
        for j in range(0,Res[i][0]): #only save de-trapping energy
            sol.append(Res[i][2][j][1])
        Sample_E_Traps.append(sol)

    TDS_temp = Sample_TDS_Curve_T[0]

    return Sample_TDS_Curve_j, Sample_NumTraps, Sample_C_Traps, Sample_E_Traps, TDS_temp