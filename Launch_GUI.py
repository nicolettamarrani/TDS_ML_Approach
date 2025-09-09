#!/usr/bin/env python3
"""
TDS ML-based Analysis GUI Launcher
Automatically starts the TDS analysis GUI
"""

import sys
import os
import subprocess

def main():
    # Get the directory where this launcher script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main GUI file
    gui_file = os.path.join(script_dir, "GUI.py")
    
    # Check if GUI file exists
    if not os.path.exists(gui_file):
        print(f"Error: GUI.py not found in {script_dir}")
        input("Press Enter to exit...")
        return
    
    try:
        # Run the GUI
        subprocess.run([sys.executable, gui_file], check=True)
    except Exception as e:
        print(f"Error launching GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()