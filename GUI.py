import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import queue
import os

from ModelEnsemble import ModelEnsemble
import TDS_Material
import TDS_Sim
from Model_Parameters import Model_Parameters
from ExpDataProcessing import ExpDataProcessing

def run_thermal_desorption_analysis(params, result_queue):
    """
    This function runs the analysis in a background thread and puts results in a queue.
    Now includes more frequent stop flag checking.
    """
    try:
        # Extract stop flag
        stop_flag = params.get('stop_flag')
        
        material_params = params['material']
        test_params = params['test']
        numerical_params = params['numerical']
        ExpName = params['ExpName']
        trap_model = params['trap_model']
        exp_file = params['exp_file']
        training_parameters = params['training_parameters']
        HD_Trap = params['HD_Trap']

        # Check for stop before each major step
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return

        # Check if an experimental file was provided
        if not exp_file:
            result_queue.put({'status': 'error', 'message': 'No experimental data file provided.', 'progress': 100})
            return

        # Step 1: Initialize objects
        result_queue.put({'status': 'progress', 'progress': 10, 'message': "Initializing material and model parameters..."})
        
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return
            
        Material = TDS_Material.TDS_Material(ExpName, 
                               material_param=material_params, 
                               test_param=test_params, 
                               numerical_param=numerical_params,
                               HD_Trap_param=HD_Trap, 
                               trap_model=trap_model)
        HyperParameters = Model_Parameters(ParameterSet=training_parameters['ParameterSet'])
        
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return
        
        # Step 2: Set additional parameters
        result_queue.put({'status': 'progress', 'progress': 15, 'message': "Setting up training parameters..."})
        Traps = training_parameters['Traps']
        Concentrations = training_parameters['Concentrations']
        MaxTraps = training_parameters['MaxTraps']
        Regenerate_Training = training_parameters['Regenerate_Training'] == "True"
        Regenerate_Data = training_parameters['Regenerate_Data'] == "True"
        
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return
        
        # Step 3: Create and train the Model Ensemble (this is the longest step)
        result_queue.put({'status': 'progress', 'progress': 20, 'message': "Generating data and training models... (this may take several hours)"})
        
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return

        # Note: The ModelEnsemble creation is a long-running operation that doesn't respect stop flags
        # This is a limitation of the underlying library - it will continue until completion
        try:
            Model = ModelEnsemble(Material, Traps, MaxTraps, Concentrations, HyperParameters,
                                  numerical_params['NumTraining'], Regenerate_Data,
                                  Regenerate_Training, numerical_params['n_cpu_cores'])
        except Exception as model_error:
            if stop_flag and stop_flag.is_set():
                result_queue.put({'status': 'stopped'})
                return
            else:
                raise model_error

        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return

        # Step 4: Run verification
        result_queue.put({'status': 'progress', 'progress': 60, 'message': "Running model verification..."})
        
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return
            
        TDS_Curves_Ver, Actual_Traps, Actual_Concentrations, Actual_Energies, TDS_Temp_Ver = TDS_Sim.SimDataSet(
            Material, numerical_params['NumVerification'], MaxTraps, Traps, Concentrations, numerical_params['n_cpu_cores'])
        
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return
        
        Predicted_Traps_Ver, Predicted_Concentrations_Ver, Predicted_Energies_Ver = Model.predict(TDS_Curves_Ver)

        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return

        # Step 5: Process experimental data
        result_queue.put({'status': 'progress', 'progress': 80, 'message': f"Processing experimental data..."})
        
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return
            
        Exp_Processed_Data = ExpDataProcessing(exp_file, Material, HyperParameters)
        Exp_Temp = Exp_Processed_Data.Temperature
        Exp_Flux = Exp_Processed_Data.Flux
        Exp_TDS_Curve = Exp_Processed_Data.TDS_Curve

        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return

        result_queue.put({'status': 'progress', 'progress': 90, 'message': f"Making predictions and generating plots..."})
        Exp_Predicted_Traps, Exp_Predicted_Concentrations, Exp_Predicted_Energies = Model.predict(Exp_TDS_Curve)

        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
            return

        # Return all data needed for plotting in the main thread
        result_data = {
            'status': 'success',
            'progress': 100,
            'message': 'Analysis completed successfully!',
            'Material': Material,
            'Model': Model,
            'Predicted_Traps_Ver': Predicted_Traps_Ver,
            'Predicted_Concentrations_Ver': Predicted_Concentrations_Ver,
            'Predicted_Energies_Ver': Predicted_Energies_Ver,
            'TDS_Curves_Ver': TDS_Curves_Ver,
            'TDS_Temp_Ver': TDS_Temp_Ver,
            'Actual_Traps': Actual_Traps,
            'Actual_Concentrations': Actual_Concentrations,
            'Actual_Energies': Actual_Energies,
            'Exp_Temp': Exp_Temp,
            'Exp_Flux': Exp_Flux,
            'Predicted_Traps': Exp_Predicted_Traps,
            'Predicted_Concentrations': Exp_Predicted_Concentrations,
            'Predicted_Energies': Exp_Predicted_Energies
        }
        
        result_queue.put(result_data)

    except Exception as e:
        # Check if this was due to stopping
        if stop_flag and stop_flag.is_set():
            result_queue.put({'status': 'stopped'})
        else:
            result_queue.put({'status': 'error', 'message': f"An error occurred during analysis: {str(e)}", 'progress': 100})

class ScrolledFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.canvas = tk.Canvas(self, borderwidth=0, bg='white', highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='white')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TDS Analysis - ML Approach")
        self.root.geometry("1200x800")
        self.root.config(bg='white')
        
        # Queue for thread communication
        self.result_queue = queue.Queue()
        self.analysis_thread = None
        
        # Add stop flag for analysis control
        self.stop_analysis_flag = threading.Event()
        
        # Store experimental data for immediate plotting
        self.exp_data = None
        
        # Try to configure styles, but continue if it fails
        self.configure_styles()
        
        # Create main layout
        self.create_layout()
        
        # Start checking for thread results - THIS SHOULD BE THE ONLY PLACE
        self.check_thread_results()
        
    def configure_styles(self):
        """Configure ttk styles for consistent appearance - with error handling"""
        try:
            style = ttk.Style()
            
            # Try to use a simple theme that should work on most systems
            available_themes = style.theme_names()
            if 'clam' in available_themes:
                style.theme_use('clam')
            elif 'alt' in available_themes:
                style.theme_use('alt')
            elif 'default' in available_themes:
                style.theme_use('default')
            
            # Configure basic styles only
            style.configure("TFrame", background="white")
            style.configure("TLabel", background="white", foreground="black")
            style.configure("TEntry", fieldbackground="white")
            style.configure("TButton", background="white")
            
            style.configure("TProgressbar", 
                       background="green",      # Progress bar color (green)
                       troughcolor="white",     # Background color (white)
                       bordercolor="gray",      # Border color
                       lightcolor="white",      # Light edge color
                       darkcolor="gray")        # Dark edge color
        
            
        except Exception as e:
            print(f"Warning: Could not configure styles: {e}")
            # Continue without custom styling
        
    def create_layout(self):
        """Create the main GUI layout"""
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left container for inputs
        self.left_container = tk.Frame(main_frame, width=450, bg='white')
        self.left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_container.pack_propagate(False)

        # Scrollable frame for inputs
        self.scrollable_left_frame = ScrolledFrame(self.left_container, relief=tk.GROOVE)
        self.scrollable_left_frame.pack(fill=tk.BOTH, expand=True)

        # Right frame for plots and output
        self.right_frame = tk.Frame(main_frame, bg='white')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_input_panels()
        self.create_output_panels()
        
    def create_input_panels(self):
        """Create all input panels on the left side"""
        parent_frame = self.scrollable_left_frame.scrollable_frame

        # Experimental Data Input Panel
        self.create_exp_data_panel(parent_frame)

        # Material Properties Panel
        self.create_material_panel(parent_frame)
        
        # Trap Model Panel
        self.create_trap_model_panel(parent_frame)
        
        # Test Parameters Panel
        self.create_test_panel(parent_frame)
        
        # Numerical Parameters Panel
        self.create_numerical_panel(parent_frame)
        
        # ML Parameter Panel
        self.create_ml_panel(parent_frame)
        
        # Button frame
        self.create_button_panel()

    def create_exp_data_panel(self, parent):
        """Create experimental data input panel using basic tk widgets"""
        exp_data_frame = tk.LabelFrame(parent, text="Experimental Data Input", 
                                    padx=10, pady=10, bg='white')
        exp_data_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(exp_data_frame, text="Excel File:", bg='white').grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.exp_file_entry = tk.Entry(exp_data_frame, width=30, bg='white')
        self.exp_file_entry.grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        
        self.browse_button = tk.Button(exp_data_frame, text="Browse", command=self.browse_file, bg='white')
        self.browse_button.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        self.load_plot_button = tk.Button(exp_data_frame, text="Plot", command=self.load_and_plot_data, bg='white')
        self.load_plot_button.grid(row=1, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        fields = [
            ("Test case:", "exp_name", "Novak_200"),
        ]
        
        for i, (label, attr, default) in enumerate(fields):
            tk.Label(exp_data_frame, text=label, bg='white').grid(row=i+2, column=0, sticky=tk.W, pady=2)
            entry = tk.Entry(exp_data_frame, width=15, bg='white')
            entry.grid(row=i+3, column=0, sticky=tk.W, pady=2)  # Changed from row=i+2 to row=i+3
            entry.insert(0, default)
            setattr(self, attr, entry)

    def create_material_panel(self, parent):
        """Create material properties panel"""
        material_frame = tk.LabelFrame(parent, text="Material Properties", 
                                     padx=10, pady=10, bg='white')
        material_frame.pack(fill=tk.X, pady=(0, 10))

        fields = [
            ("Density of lattice sites, NL (mol/m³):", "nl", "8.47e5"),
            ("Activation energy for lattice diffusion, EL (J/mol):", "e_diff", "5690"),
            ("Pre-exponential diffusion factor, D₀ (m²/s):", "d0", "7.23e-8"),
            ("Initial H concentration in the lattice, C₀ (mol/m³):", "c0", "0.06"),
            ("Molar mass (g/mol):", "mol_mass", "55.847"),
            ("Mass density (g/cm³):", "mass_density", "7.8474")
        ]
        
        for i, (label, attr, default) in enumerate(fields):
            tk.Label(material_frame, text=label, bg='white').grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = tk.Entry(material_frame, width=15, bg='white')
            entry.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            entry.insert(0, default)
            setattr(self, attr, entry)

    def create_trap_model_panel(self, parent):
        """Create trap model panel"""
        trap_frame = tk.LabelFrame(parent, text="Trap Model", padx=10, pady=10, bg='white')
        trap_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(trap_frame, text="Model Type:", bg='white').grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Use ttk.Combobox but with error handling
        try:
            self.trap_model = ttk.Combobox(trap_frame, values=["McNabb-Foster", "Oriani"], width=12, state="readonly")
        except:
            # Fallback to a simple approach if ttk fails
            self.trap_model = tk.StringVar()
            self.trap_model.set("McNabb-Foster")
            combo_frame = tk.Frame(trap_frame, bg='white')
            combo_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            tk.Label(combo_frame, text="McNabb-Foster", bg='white').pack()
        else:
            self.trap_model.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            self.trap_model.set("McNabb-Foster")
            self.trap_model.bind("<<ComboboxSelected>>", self.on_trap_model_change)

        tk.Label(trap_frame, text="Trapping rate (1/s):", bg='white').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.trap_rate_trap_model = tk.Entry(trap_frame, width=15, bg='white')
        self.trap_rate_trap_model.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        self.trap_rate_trap_model.insert(0, "1e13")

    def create_test_panel(self, parent):
        """Create test parameters panel"""
        test_frame = tk.LabelFrame(parent, text="Test Parameters", padx=10, pady=10, bg='white')
        test_frame.pack(fill=tk.X, pady=(0, 10))
        
        fields = [
            ("Resting time (s):", "t_rest", "2700"),
            ("Heating rate (K/s):", "heating_rate", "0.055"),
            ("Sample thickness (m):", "thickness", "0.0063"),
            ("Maximum temperature (K):", "t_max", "873.15"),
            ("Minimum temperature (K):", "t_min", "293.15")
        ]
        
        for i, (label, attr, default) in enumerate(fields):
            tk.Label(test_frame, text=label, bg='white').grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = tk.Entry(test_frame, width=15, bg='white')
            entry.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            entry.insert(0, default)
            setattr(self, attr, entry)

    def create_numerical_panel(self, parent):
        """Create numerical parameters panel"""
        numerical_frame = tk.LabelFrame(parent, text="Numerical Parameters", padx=10, pady=10, bg='white')
        numerical_frame.pack(fill=tk.X, pady=(0, 10))
        
        fields = [
            ("Minimum difference in trap binding energies (J/mol):", "de_min", "10e3"),
            ("Number temperature evaluations:", "ntp", "64"),
            ("Sample frequency:", "sample_freq", "10"),
            ("Density lower bound (mol/m³):", "n_range_min", "1e-1"),
            ("Density upper bound (mol/m³):", "n_range_max", "1e1"),
            ("Binding energy lower bound (J/mol):", "e_range_min", "50e3"),
            ("Binding energy upper bound (J/mol):", "e_range_max", "150e3"),
            ("High-density low-energy trap:", "HDT", "False"),
            ("HDT density lower bound (mol/m³):", "HDT_n_range_min", "0"),
            ("HDT density upper bound (mol/m³):", "HDT_n_range_max", "0"),
            ("HDT binding energy lower bound (J/mol):", "HDT_e_range_min", "0"),
            ("HDT binding energy upper bound (J/mol):", "HDT_e_range_max", "0")
        ]
        
        for i, (label, attr, default) in enumerate(fields):
            tk.Label(numerical_frame, text=label, bg='white').grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = tk.Entry(numerical_frame, width=15, bg='white')
            entry.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            entry.insert(0, default)
            setattr(self, attr, entry)

    def create_ml_panel(self, parent):
        """Create ML parameters panel"""
        ml_frame = tk.LabelFrame(parent, text="ML Model Training Parameters", padx=10, pady=10, bg='white')
        ml_frame.pack(fill=tk.X, pady=(0, 10))

        fields = [
            ("CPU cores:", "n_cpu_cores", "16"),
            ("Number training datapoints:", "num_training", "50000"),
            ("Number verification datapoints:", "num_verification", "500"),
            ("Hyperparameter set:", "hp_set", "optimised"),
            ("Maximum traps:", "max_traps", "4"),
            ("Traps:", "traps", "Random"),
            ("Concentrations:", "concentrations", "Random"),
            ("Regenerate data:", "regenerate_data", "False"),
            ("Regenerate training:", "regenerate_training", "False")
        ]
        
        for i, (label, attr, default) in enumerate(fields):
            tk.Label(ml_frame, text=label, bg='white').grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = tk.Entry(ml_frame, width=15, bg='white')
            entry.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            entry.insert(0, default)
            setattr(self, attr, entry)

    def create_button_panel(self):
        """Create button panel outside scrollable area"""
        button_frame = tk.Frame(self.left_container, bg='white')
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.run_button = tk.Button(button_frame, text="Run Analysis", command=self.run_analysis, bg='white')
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Add stop button
        self.stop_button = tk.Button(button_frame, text="Stop Analysis", command=self.stop_analysis,  bg='white', state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_inputs, bg='white')
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_button = tk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults, bg='white')
        self.reset_button.pack(side=tk.LEFT)

    def create_output_panels(self):
        """Create the output panels on the right side"""
        # Graph panel
        graph_frame = tk.LabelFrame(self.right_frame, text="Plots", padx=5, pady=5, bg='white')
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_widget.configure(bg='white')
        self.figure.patch.set_facecolor('white')
        
        self.ax.set_title("Plots")
        self.ax.set_xlabel(r"Temperature [K]")
        self.ax.set_ylabel(r"Hydrogen Desorption Flux, J [mol/m²/s]")
        self.ax.grid(True)
        
        # Progress panel
        progress_frame = tk.LabelFrame(self.right_frame, text="Analysis Progress", padx=5, pady=5, bg='white')
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        
        # Try ttk.Progressbar first, fallback to basic implementation
        try:
            self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                                maximum=100, length=400)
        except:
            # Simple fallback progress indicator
            progress_canvas = tk.Canvas(progress_frame, height=20, width=400, bg='white')
            progress_canvas.pack(fill=tk.X, padx=5, pady=(0, 5))
            self.progress_bar = progress_canvas
        else:
            self.progress_bar.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.progress_status = tk.Label(progress_frame, text="Ready to start analysis", 
                                      fg="blue", bg='white')
        self.progress_status.pack(pady=(0, 5))
        
        # Output panel
        output_frame = tk.LabelFrame(self.right_frame, text="Output", padx=5, pady=5, bg='white')
        output_frame.pack(fill=tk.X, pady=0)
        output_frame.configure(height=150)

        self.output_text = tk.Text(output_frame, height=5, wrap="word", state=tk.DISABLED, bg='white')
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        output_scrollbar = tk.Scrollbar(output_frame, command=self.output_text.yview)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text['yscrollcommand'] = output_scrollbar.set

    def browse_file(self):
        """Open file dialog to select Excel file"""
        file_path = filedialog.askopenfilename(
            title="Select Experimental Data File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if file_path:
            self.exp_file_entry.delete(0, tk.END)
            self.exp_file_entry.insert(0, file_path)

    def load_and_plot_data(self):
        """Load and plot experimental data immediately"""
        file_path = self.exp_file_entry.get().strip()
        if not file_path:
            messagebox.showwarning("No File Selected", "Please select an experimental data file first.")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("File Not Found", f"The file '{file_path}' does not exist.")
            return
        
        self.load_and_plot_experimental_data(file_path)

    def load_and_plot_experimental_data(self, file_path):
        """Load experimental data and plot it immediately"""
        try:
            temp_params = self.collect_parameters()
            temp_material = TDS_Material.TDS_Material(
                temp_params['ExpName'], 
                material_param=temp_params['material'], 
                test_param=temp_params['test'], 
                numerical_param=temp_params['numerical'],
                HD_Trap_param = temp_params['HD_Trap'],
                trap_model=temp_params['trap_model']
            )
            temp_hyperparams = Model_Parameters(ParameterSet=temp_params['training_parameters']['ParameterSet'])
            
            self.exp_data = ExpDataProcessing(file_path, temp_material, temp_hyperparams)
            
            self.plot_experimental_data_only(self.exp_data.Temperature, self.exp_data.Flux)
            
            self.update_output_text(f"Experimental data loaded and plotted from: {file_path}\n")
            self.update_output_text(f"Temperature range: {min(self.exp_data.Temperature):.1f} - {max(self.exp_data.Temperature):.1f} K\n")
            self.update_output_text(f"Max flux: {max(self.exp_data.Flux):.2e} mol/m²/s\n")
            
        except Exception as e:
            messagebox.showerror("Error Loading Data", f"Could not load experimental data:\n{str(e)}")
            self.update_output_text(f"Error loading experimental data: {str(e)}\n")

    def on_trap_model_change(self, event):
        """Handle trap model change"""
        try:
            selected_model = self.trap_model.get()
            if selected_model == "McNabb-Foster":
                self.trap_rate_trap_model.config(state=tk.NORMAL)
            else:
                self.trap_rate_trap_model.config(state=tk.DISABLED)
        except:
            pass

    def stop_analysis(self):
        """Stop the running analysis"""
        self.stop_analysis_flag.set()
        self.progress_status.config(text="Stop requested - waiting for safe stopping point...", fg="orange")
        self.update_output_text("Stop requested. Waiting for current operation to complete safely...\n")
        
        # Disable stop button to prevent multiple clicks
        self.stop_button.config(state=tk.DISABLED)

    def run_analysis(self):
        """Start analysis in background thread"""
        self.update_output_text("--- Running Analysis ---\n", clear=True)
        self.reset_progress()
        
        # Reset stop flag and manage button states
        self.stop_analysis_flag.clear()
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        try:
            params = self.collect_parameters()
            # Add stop flag to parameters
            params['stop_flag'] = self.stop_analysis_flag
            self.display_parameters(params)

            # Start analysis in background thread
            self.analysis_thread = threading.Thread(
                target=run_thermal_desorption_analysis, 
                args=(params, self.result_queue),
                daemon=True
            )
            self.analysis_thread.start()

        except ValueError as e:
            self.handle_error(f"Invalid input value: {str(e)}")
        except Exception as e:
            self.handle_error(f"An unexpected error occurred: {str(e)}")


    def check_thread_results(self):
        """Check for results from background thread"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                
                if result['status'] == 'progress':
                    self.update_progress(result['progress'], result['message'])
                elif result['status'] == 'success':
                    self.handle_success(result)
                elif result['status'] == 'error':
                    self.handle_error(result['message'])
                elif result['status'] == 'stopped':
                    self.handle_stopped()
                        
        except queue.Empty:
            pass
    
        # THIS LINE IS CRITICAL - it keeps the checking loop running:
        self.root.after(100, self.check_thread_results)

    def handle_success(self, result):
        """Handle successful analysis completion"""
        self.progress_var.set(100)
        self.progress_status.config(text="Analysis completed!", fg="green")
        self.update_output_text(f"\n{result['message']}\n")
        
        # Reset button states
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Generate plots in main thread (safe for matplotlib)
        try:
            self.generate_saved_plots(result)
            self.update_plot_with_contributions(
                result['Exp_Temp'], 
                result['Exp_Flux'], 
                result['Predicted_Concentrations'], 
                result['Predicted_Energies'],
                result['Material']
            )
            
            # Display results
            self.display_trap_results(result)
            
        except Exception as plot_error:
            print(f"Plotting error: {plot_error}")
            self.plot_experimental_data_only(result['Exp_Temp'], result['Exp_Flux'])
            self.update_output_text(f"Warning: Could not plot trap contributions: {plot_error}\n")
        
        self.progress_status.config(text="All tasks completed!", fg="green")
        self.run_button.config(state=tk.NORMAL)

    def handle_error(self, message):
        """Handle analysis errors"""
        self.progress_var.set(100)
        self.progress_status.config(text=f"Error: {message}", fg="red")
        self.update_output_text(f"Error: {message}\n")
        
        # Reset button states
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def handle_stopped(self):
        """Handle analysis stop"""
        self.progress_var.set(100)
        self.progress_status.config(text="Analysis stopped by user", fg="orange")
        self.update_output_text("\n=== ANALYSIS STOPPED ===\n")
        self.update_output_text("Note: Some operations (like model training) may continue briefly in the background\n")
        self.update_output_text("before fully stopping. This is normal behavior.\n")
        
        # Reset button states
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def generate_saved_plots(self, result):
        """Generate and save plots to files (called from main thread)"""
        try:
            model = result['Model']
            
            # Create directory if it doesn't exist
            figures_dir = f"Figures/{result['Material'].ExpName}"
            os.makedirs(figures_dir, exist_ok=True)
            
            # Generate comparison plots
            model.PlotComparisonTraps(
                result['Predicted_Traps_Ver'], 
                result['Actual_Traps'], 
                result['TDS_Curves_Ver'], 
                result['TDS_Temp_Ver']
            )
            
            model.PlotComparisonConcentrations(
                result['Predicted_Concentrations_Ver'], 
                result['Actual_Concentrations']
            )
            
            model.PlotComparisonEnergies(
                result['Predicted_Energies_Ver'], 
                result['Actual_Energies']
            )
            
            model.PlotComparisonExpData(
                result['Exp_Temp'], 
                result['Exp_Flux'], 
                result['Predicted_Concentrations'], 
                result['Predicted_Energies']
            )
            
            # Close all plots to free memory
            plt.close('all')
            
            self.update_output_text(f"Plots saved to '{figures_dir}/' folder\n")
            
        except Exception as e:
            self.update_output_text(f"Warning: Could not generate saved plots: {e}\n")

    def display_trap_results(self, result):
        """Display trap analysis results in text"""
        try:
            self.update_output_text(f"\nPredicted Traps: {result['Predicted_Traps'][0]}\n")
            
            # Handle trap model selection
            try:
                energy_label = "Binding energy" if self.trap_model.get() == "Oriani" else "De-trapping energy"
            except:
                energy_label = "De-trapping energy"  # Default
            
            predicted_energies = result['Predicted_Energies'][0]
            predicted_concentrations = result['Predicted_Concentrations'][0]

            for i in range(result['Predicted_Traps'][0]):
                energy = predicted_energies[i]
                concentration = predicted_concentrations[i]
                self.update_output_text(f"Trap {i+1}: {energy_label} = {energy:.2f} J/mol, Trap density = {concentration:.2f} mol/m³\n")

        except Exception as e:
            self.update_output_text(f"Warning: Could not display trap parameters: {e}\n")

    def collect_parameters(self):
        """Collect all parameters from GUI inputs"""
        # Handle trap model selection safely
        try:
            if hasattr(self.trap_model, 'get'):
                trap_model_value = self.trap_model.get()
            else:
                trap_model_value = "McNabb-Foster"  # Default fallback
            
            # Convert GUI names to backend names
            if trap_model_value == "McNabb-Foster":
                trap_model_backend = "McNabb"
                trap_rate_value = float(self.trap_rate_trap_model.get())
            elif trap_model_value == "Oriani":
                trap_model_backend = "Oriani"
                trap_rate_value = float(self.trap_rate_trap_model.get())
            else:
                trap_model_backend = "McNabb"
                trap_rate_value = 1e13
        except:
            trap_model_backend = "McNabb"
            trap_rate_value = 1e13  # Safe default

        material = {
            'NL': float(self.nl.get()) if self.nl.get() else 8.47e5,
            'E_Diff': float(self.e_diff.get()) if self.e_diff.get() else 5690,
            'D0': float(self.d0.get()) if self.d0.get() else 7.23e-8,
            'C0': float(self.c0.get()) if self.c0.get() else 0.06,
            'TrapRate': trap_rate_value,
            'MolMass': float(self.mol_mass.get()) if self.mol_mass.get() else 55.847,
            'MassDensity': float(self.mass_density.get()) if self.mass_density.get() else 7.8474
        }
        
        test = {
            'tRest': float(self.t_rest.get()) if self.t_rest.get() else 2700,
            'HeatingRate': float(self.heating_rate.get()) if self.heating_rate.get() else 0.055,
            'Thickness': float(self.thickness.get()) if self.thickness.get() else 0.0063,
            'TMax': float(self.t_max.get()) if self.t_max.get() else 873.15,
            'TMin': float(self.t_min.get()) if self.t_min.get() else 293.15
        }
        
        numerical = {
            'dEMin': float(self.de_min.get()) if self.de_min.get() else 10e3,
            'ntp': int(self.ntp.get()) if self.ntp.get() else 64,
            'SampleFreq': int(self.sample_freq.get()) if self.sample_freq.get() else 10,
            'NRange': [float(self.n_range_min.get()) if self.n_range_min.get() else 1e-1, 
                    float(self.n_range_max.get()) if self.n_range_max.get() else 1e1],
            'ERange': [float(self.e_range_min.get()) if self.e_range_min.get() else 50e3, 
                    float(self.e_range_max.get()) if self.e_range_max.get() else 150e3],
            'NumTraining': int(self.num_training.get()) if self.num_training.get() else 50000,
            'NumVerification': int(self.num_verification.get()) if self.num_verification.get() else 500,
            'n_cpu_cores': int(self.n_cpu_cores.get()) if self.n_cpu_cores.get() else 16
        }

        training_parameters = {
            'Traps': self.traps.get() if self.traps.get() else "Random",
            'Concentrations': self.concentrations.get() if self.concentrations.get() else "Random",
            'MaxTraps': int(self.max_traps.get()) if self.max_traps.get() else 4,
            'ParameterSet': self.hp_set.get() if self.hp_set.get() else "optimised",
            'Regenerate_Data': self.regenerate_data.get() if self.regenerate_data.get() else "False",
            'Regenerate_Training': self.regenerate_training.get() if self.regenerate_training.get() else "False"
        }

        if self.HDT.get() == 'False' or self.HDT.get() is None:
            HDT_Flag = False
        elif self.HDT.get() == 'True':
            True

        HD_Trap = {
            "HDT": HDT_Flag,
            'HDT_NRange': [float(self.HDT_n_range_min.get()) if self.HDT_n_range_min.get() else None, 
                    float(self.HDT_n_range_max.get()) if self.HDT_n_range_max.get() else None],
            'HDT_ERange': [float(self.HDT_e_range_min.get()) if self.HDT_e_range_min.get() else None, 
                    float(self.HDT_e_range_max.get()) if self.HDT_e_range_max.get() else None]
        }

        return {
            'material': material, 
            'test': test, 
            'numerical': numerical, 
            'trap_model': trap_model_backend,  # Use converted name
            'exp_file': self.exp_file_entry.get(),
            'ExpName': self.exp_name.get(),
            'training_parameters': training_parameters,
            'HD_Trap': HD_Trap
        }

    def display_parameters(self, params):
        """Display collected parameters in output text"""
        self.update_output_text(f"Test case: {params['ExpName']}\n\n")
        self.update_output_text("--- Inputs for Analysis ---\n")
        self.update_output_text(f"Experimental Data File: {params['exp_file']}\n\n")
        
        self.update_output_text("Material parameters:\n")
        for key, value in params['material'].items():
            if isinstance(value, float) and value >= 1e4:
                self.update_output_text(f"  {key}: {value:.2e}\n")
            else:
                self.update_output_text(f"  {key}: {value}\n")

        self.update_output_text("\nTest parameters:\n")
        for key, value in params['test'].items():
            if isinstance(value, float) and value >= 1e4:
                self.update_output_text(f"  {key}: {value:.2e}\n")
            else:
                self.update_output_text(f"  {key}: {value}\n")

        self.update_output_text("\nNumerical parameters:\n")
        for key, value in params['numerical'].items():
            if isinstance(value, float) and value >= 1e4:
                self.update_output_text(f"  {key}: {value:.2e}\n")
            else:
                self.update_output_text(f"  {key}: {value}\n")

        self.update_output_text(f"\nTrap model: {params['trap_model']}\n")
        self.update_output_text("Parameters collected!\nStarting analysis...\n")

    def update_progress(self, progress_percent, status_text):
        """Update progress bar and status with better text handling and color management"""
        self.progress_var.set(progress_percent)
        
        # Determine text color based on content
        if "completed" in status_text.lower() or "success" in status_text.lower():
            text_color = "green"
        elif "stop" in status_text.lower() or "error" in status_text.lower():
            text_color = "red"
        else:
            text_color = "black"
        
        # Clear and update the status text properly
        self.progress_status.config(text="", fg=text_color)  # Clear first with color
        self.root.update_idletasks()  # Force update
        self.progress_status.config(text=status_text, fg=text_color)  # Then set new text with color
        self.root.update_idletasks()
            
    def reset_progress(self):
        """Reset progress bar to initial state"""
        self.progress_var.set(0)
        self.progress_status.config(text="Ready to start analysis")

    def update_output_text(self, text, clear=False):
        """Update output text widget"""
        self.output_text.configure(state=tk.NORMAL)
        if clear:
            self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state=tk.DISABLED)
        self.output_text.see(tk.END)

    def plot_experimental_data_only(self, Temperature, TDS_Curve):
        """Plot only experimental data"""
        self.ax.clear()
        self.ax.scatter(Temperature, TDS_Curve, label="Experimental Data", color="black", s=15, alpha=0.7)
        self.ax.set_title("TDS Experimental Data")
        self.ax.set_xlabel(r"Temperature, T [K]")
        self.ax.set_ylabel(r"Hydrogen Desorption Flux, J [mol/m²/s]")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot_with_contributions(self, Temperature, TDS_Curve, Predicted_Concentrations, Predicted_Energies, Material):
        """Update plot with trap contributions"""
        self.ax.clear()
        
        # Plot experimental data
        try:
            self.ax.scatter(Temperature, TDS_Curve, label="Experimental Data", color="black", s=15, alpha=0.7)
        except Exception as e:
            print(f"Error plotting experimental data: {e}")
            return
        
        try:
            N_traps = Predicted_Concentrations[0]
            E_traps = Predicted_Energies[0]

            # Ensure we have proper arrays
            if hasattr(N_traps, 'tolist'):
                N_traps = N_traps.tolist()
            if hasattr(E_traps, 'tolist'):
                E_traps = E_traps.tolist()
                
            # Convert to lists if they're still numpy arrays
            if not isinstance(N_traps, list):
                N_traps = list(N_traps)
            if not isinstance(E_traps, list):
                E_traps = list(E_traps)
            
            # Plot individual trap contributions
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            valid_traps = []
            
            for i in range(len(N_traps)):
                if E_traps[i] > 0 and N_traps[i] > 0:  # don't plot negative density or energy traps (not valid)
                    try:
                        trap_num = i + 1
                        N = [N_traps[i]]
                        if (Material.TrapModel == TDS_Material.TRAPMODELS.MCNABB):
                            E = [[Material.E_Diff, E_traps[i]]]
                        elif (Material.TrapModel == TDS_Material.TRAPMODELS.ORIANI):
                            E = [[Material.E_Diff, E_traps[i]+Material.E_Diff]]
                        
                        # Create individual trap simulation
                        Sample = TDS_Sim.TDS_Sample(Material, N, E, False)
                        Sample.Charge()
                        Sample.Rest()
                        T, J = Sample.TDS()
                        
                        color = colors[i % len(colors)]
                        self.ax.plot(T, J, '--', color=color, alpha=0.7, linewidth=1.5, 
                                label=f"Trap {trap_num}")
                        valid_traps.append(i)
                        
                    except Exception as e:
                        print(f"Error plotting individual trap {i+1}: {e}")
                        continue
            
            # Plot total predicted curve
            if valid_traps:
                try:
                    E_full = []
                    N_full = []
                    for i in valid_traps:
                        if (Material.TrapModel == TDS_Material.TRAPMODELS.MCNABB):
                            E_full.append([Material.E_Diff, E_traps[i]])
                        elif (Material.TrapModel == TDS_Material.TRAPMODELS.ORIANI):
                            E_full.append([Material.E_Diff, E_traps[i]+Material.E_Diff])
                        N_full.append(N_traps[i])
                    
                    Sample_total = TDS_Sim.TDS_Sample(Material, N_full, E_full, False)
                    Sample_total.Charge()
                    Sample_total.Rest()
                    T_total, J_total = Sample_total.TDS()
                    
                    self.ax.plot(T_total, J_total, '-', color="blue", linewidth=2, 
                            label="Total TDS Prediction", alpha=0.8)
                            
                except Exception as e:
                    print(f"Error plotting total prediction: {e}")
                    self.update_output_text(f"Warning: Could not plot total prediction: {e}\n")
            
        except Exception as e:
            print(f"Error in trap plotting: {e}")
            self.update_output_text(f"Warning: Could not plot trap contributions: {e}\n")
        
        # Format the plot
        self.ax.set_title("Re-constructed TDS spectrum and individual trap contributions")
        self.ax.set_xlabel(r"Temperature, T [K]")
        self.ax.set_ylabel(r"Hydrogen Desorption Flux, J [mol/m²/s]")
        self.ax.grid(True, alpha=0.3)
        
        # Handle legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc='upper right', fontsize=8)
        
        try:
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating canvas: {e}")

    def clear_inputs(self):
        """Clear all input fields"""
        # Clear all entry widgets
        entry_widgets = [
            self.exp_name, self.exp_file_entry, self.nl, self.e_diff, self.d0, self.c0,
            self.mol_mass, self.mass_density, self.t_rest, self.heating_rate, self.thickness,
            self.t_max, self.t_min, self.de_min, self.ntp, self.sample_freq, self.n_range_min,
            self.n_range_max, self.e_range_min, self.e_range_max, self.num_training,
            self.num_verification, self.n_cpu_cores, self.traps, self.concentrations,
            self.max_traps, self.regenerate_data, self.regenerate_training, self.hp_set,
            self.trap_rate_trap_model, self.HDT, self.HDT_n_range_min, self.HDT_n_range_max, self.HDT_e_range_min, self.HDT_e_range_max
        ]
        
        for widget in entry_widgets:
            try:
                widget.delete(0, tk.END)
            except:
                pass  # Skip if widget doesn't support this operation

        # Reset button states properly
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Clear the stop flag
        self.stop_analysis_flag.clear()

        # Clear output and reset plot
        self.update_output_text("Inputs cleared. Ready for analysis...\n", clear=True)
        self.reset_progress()
        self.reset_plot()

    def reset_defaults(self):
        """Reset all inputs to default values"""
        self.clear_inputs()
        
        # Set default values
        defaults = {
            'exp_name': "Novak_200",
            'nl': "8.47e5",
            'e_diff': "5690",
            'd0': "7.23e-8",
            'c0': "0.06",
            'mol_mass': "55.847",
            'mass_density': "7.8474",
            't_rest': "2700",
            'heating_rate': "0.055",
            'thickness': "0.0063",
            't_max': "873.15",
            't_min': "293.15",
            'de_min': "10e3",
            'ntp': "64",
            'sample_freq': "10",
            'n_range_min': "1e-1",
            'n_range_max': "1e1",
            'e_range_min': "50e3",
            'e_range_max': "150e3",
            'HDT': "False",
            'HDT_n_range_min': "0",
            'HDT_n_range_max': "0",
            'HDT_e_range_min': "0",
            'HDT_e_range_max': "0",
            'num_training': "50000",
            'num_verification': "500",
            'n_cpu_cores': "16",
            'traps': "Random",
            'concentrations': "Random",
            'max_traps': "4",
            'regenerate_data': "False",
            'regenerate_training': "False",
            'hp_set': "optimised",
            'trap_rate_trap_model': "1e13"
        }
        
        for attr, value in defaults.items():
            try:
                widget = getattr(self, attr)
                widget.insert(0, value)
            except:
                pass  # Skip if attribute doesn't exist or operation fails
        
        # Set trap model safely
        try:
            if hasattr(self.trap_model, 'set'):
                self.trap_model.set("McNabb-Foster")
        except:
            pass
        
        # Reset button states properly
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Clear the stop flag
        self.stop_analysis_flag.clear()
        
        self.update_output_text("Reset to Novak_200 default values.\n", clear=True)
        self.reset_progress()
        self.reset_plot()

    def reset_plot(self):
        """Reset plot to initial state"""
        self.ax.clear()
        self.ax.set_title("Plots")
        self.ax.set_xlabel(r"Temperature, T [K]")
        self.ax.set_ylabel(r"Hydrogen Desorption Flux, J [mol/m²/s]")
        self.ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()