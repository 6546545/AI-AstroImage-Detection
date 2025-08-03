#!/usr/bin/env python3
"""
Demo script for Space Anomaly Detection System GUI
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

def show_gui_demo():
    """Show a demo of the GUI features."""
    
    # Create a simple demo window
    demo_window = tk.Tk()
    demo_window.title("Space Anomaly Detection System - GUI Demo")
    demo_window.geometry("600x400")
    demo_window.configure(bg='#f0f0f0')
    
    # Title
    title_label = tk.Label(demo_window, text="🚀 Space Anomaly Detection System", 
                          font=('Arial', 18, 'bold'), bg='#f0f0f0')
    title_label.pack(pady=20)
    
    subtitle_label = tk.Label(demo_window, text="GUI Demo & Features", 
                             font=('Arial', 12), bg='#f0f0f0')
    subtitle_label.pack(pady=10)
    
    # Features frame
    features_frame = tk.Frame(demo_window, bg='#f0f0f0')
    features_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Feature list
    features = [
        "🏠 Main Tab - System status and quick actions",
        "🎯 Training Tab - Train models with visual progress",
        "🔍 Detection Tab - Detect anomalies with adjustable parameters",
        "📊 Results Tab - View and browse detected anomalies",
        "⚙️ Settings Tab - Configure system settings and view logs"
    ]
    
    for i, feature in enumerate(features):
        feature_label = tk.Label(features_frame, text=feature, 
                               font=('Arial', 10), bg='#f0f0f0', anchor='w')
        feature_label.pack(fill=tk.X, pady=5)
    
    # Buttons frame
    button_frame = tk.Frame(demo_window, bg='#f0f0f0')
    button_frame.pack(fill=tk.X, padx=20, pady=20)
    
    def launch_gui():
        demo_window.destroy()
        try:
            from gui_app import main as gui_main
            gui_main()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch GUI: {e}")
    
    def show_help():
        help_text = """
Space Anomaly Detection System GUI

Quick Start:
1. Launch the GUI using 'python launch_gui.py'
2. Check system status on the Main tab
3. Train a model using the Training tab
4. Detect anomalies using the Detection tab
5. View results using the Results tab

Key Features:
- User-friendly interface with tabs
- Real-time progress monitoring
- Background processing (GUI stays responsive)
- Image viewer for detected anomalies
- Settings management
- Comprehensive logging

For detailed instructions, see README_GUI.md
        """
        messagebox.showinfo("Help", help_text)
    
    def check_system():
        status = []
        
        # Check dependencies
        try:
            import tkinter
            status.append("✅ tkinter - Available")
        except ImportError:
            status.append("❌ tkinter - Missing")
        
        try:
            import matplotlib
            status.append("✅ matplotlib - Available")
        except ImportError:
            status.append("❌ matplotlib - Missing")
        
        try:
            import torch
            status.append("✅ PyTorch - Available")
        except ImportError:
            status.append("❌ PyTorch - Missing")
        
        # Check files
        if os.path.exists('space_anomaly_detector.py'):
            status.append("✅ Core system - Available")
        else:
            status.append("❌ Core system - Missing")
        
        if os.path.exists('gui_app.py'):
            status.append("✅ GUI application - Available")
        else:
            status.append("❌ GUI application - Missing")
        
        if os.path.exists('models/anomaly_detector.pth'):
            status.append("✅ Trained model - Available")
        else:
            status.append("⚠️ Trained model - Not found (will need to train)")
        
        if os.path.exists('sdss_images'):
            status.append("✅ Data directory - Available")
        else:
            status.append("⚠️ Data directory - Not found (will need to specify)")
        
        status_text = "\n".join(status)
        messagebox.showinfo("System Check", status_text)
    
    # Buttons
    tk.Button(button_frame, text="🚀 Launch GUI", command=launch_gui, 
              bg='#4CAF50', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=10)
    
    tk.Button(button_frame, text="❓ Help", command=show_help, 
              bg='#2196F3', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
    
    tk.Button(button_frame, text="🔍 Check System", command=check_system, 
              bg='#FF9800', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
    
    tk.Button(button_frame, text="❌ Close", command=demo_window.destroy, 
              bg='#f44336', fg='white', font=('Arial', 12)).pack(side=tk.RIGHT, padx=10)
    
    # Run the demo
    demo_window.mainloop()

def main():
    """Main function."""
    print("🎯 Space Anomaly Detection System - GUI Demo")
    print("=" * 50)
    print("This demo will show you the GUI features and help you get started.")
    print("The GUI provides a user-friendly interface for:")
    print("- Training anomaly detection models")
    print("- Detecting anomalies in space imagery")
    print("- Viewing and browsing results")
    print("- Managing system settings")
    print("=" * 50)
    
    show_gui_demo()

if __name__ == "__main__":
    main() 