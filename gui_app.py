#!/usr/bin/env python3
"""
GUI Application for Space Anomaly Detection System
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from space_anomaly_detector import SpaceAnomalyDetectionSystem, DataPreprocessor, AnomalyDetector
from config import *

# Configure matplotlib for GUI
plt.style.use('default')
matplotlib.use('TkAgg')

class LogHandler(logging.Handler):
    """Custom log handler for GUI text widget."""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.text_widget.after(100, self.check_queue)
    
    def emit(self, record):
        self.queue.put(record)
    
    def check_queue(self):
        while True:
            try:
                record = self.queue.get_nowait()
                msg = self.format(record)
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()
            except queue.Empty:
                break
        self.text_widget.after(100, self.check_queue)

class SpaceAnomalyGUI:
    """Main GUI application for Space Anomaly Detection System."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Space Anomaly Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize variables
        self.data_directory = tk.StringVar(value="sdss_images")
        self.model_path = tk.StringVar(value="models/anomaly_detector.pth")
        self.confidence_threshold = tk.DoubleVar(value=0.8)
        self.train_epochs = tk.IntVar(value=10)
        self.batch_size = tk.IntVar(value=32)
        self.learning_rate = tk.DoubleVar(value=1e-3)
        
        # Results storage
        self.current_results = None
        self.anomaly_images = []
        self.current_image_index = 0
        
        # Create GUI components
        self.setup_gui()
        self.setup_logging()
        
        # Check if model exists
        self.check_model_status()
    
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.setup_main_tab()
        self.setup_training_tab()
        self.setup_detection_tab()
        self.setup_results_tab()
        self.setup_settings_tab()
    
    def setup_main_tab(self):
        """Setup the main tab with overview and quick actions."""
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Main")
        
        # Title
        title_label = ttk.Label(main_frame, text="Space Anomaly Detection System", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Checking system status...")
        self.status_label.pack()
        
        # Quick actions frame
        actions_frame = ttk.LabelFrame(main_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Buttons
        ttk.Button(actions_frame, text="Run Complete Pipeline", 
                  command=self.run_complete_pipeline).pack(fill=tk.X, pady=5)
        ttk.Button(actions_frame, text="Load Existing Model", 
                  command=self.load_existing_model).pack(fill=tk.X, pady=5)
        ttk.Button(actions_frame, text="Detect Anomalies", 
                  command=self.detect_anomalies).pack(fill=tk.X, pady=5)
        ttk.Button(actions_frame, text="View Results", 
                  command=self.view_results).pack(fill=tk.X, pady=5)
    
    def setup_training_tab(self):
        """Setup the training tab."""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Training")
        
        # Training parameters
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Data directory
        ttk.Label(params_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(params_frame, textvariable=self.data_directory, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(params_frame, text="Browse", command=self.browse_data_directory).grid(row=0, column=2)
        
        # Training parameters
        ttk.Label(params_frame, text="Training Epochs:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.train_epochs, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=1, to=128, textvariable=self.batch_size, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(params_frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(params_frame, textvariable=self.learning_rate, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Training buttons
        button_frame = ttk.Frame(training_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(button_frame, text="Start Training", 
                  command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Training", 
                  command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(training_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        # Training log
        log_frame = ttk.LabelFrame(training_frame, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=10)
        self.training_log.pack(fill=tk.BOTH, expand=True)
    
    def setup_detection_tab(self):
        """Setup the detection tab."""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="Detection")
        
        # Detection parameters
        params_frame = ttk.LabelFrame(detection_frame, text="Detection Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(params_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.confidence_threshold).grid(row=0, column=2)
        
        ttk.Label(params_frame, text="Model Path:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(params_frame, textvariable=self.model_path, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(params_frame, text="Browse", command=self.browse_model_path).grid(row=1, column=2)
        
        # Detection buttons
        button_frame = ttk.Frame(detection_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(button_frame, text="Detect Anomalies", 
                  command=self.detect_anomalies).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        
        # Detection log
        log_frame = ttk.LabelFrame(detection_frame, text="Detection Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.detection_log = scrolledtext.ScrolledText(log_frame, height=10)
        self.detection_log.pack(fill=tk.BOTH, expand=True)
    
    def setup_results_tab(self):
        """Setup the results tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results summary
        summary_frame = ttk.LabelFrame(results_frame, text="Detection Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=8, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X)
        
        # Image viewer
        viewer_frame = ttk.LabelFrame(results_frame, text="Anomaly Images", padding=10)
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(viewer_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        self.image_label = ttk.Label(nav_frame, text="No images loaded")
        self.image_label.pack(side=tk.LEFT, padx=20)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_display = ttk.Label(viewer_frame, text="No image to display")
        self.image_display.pack(expand=True, fill=tk.BOTH)
    
    def setup_settings_tab(self):
        """Setup the settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # System settings
        system_frame = ttk.LabelFrame(settings_frame, text="System Settings", padding=10)
        system_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Device selection
        ttk.Label(system_frame, text="Device:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(system_frame, textvariable=self.device_var, 
                                   values=["auto", "cpu", "cuda", "mps"], state="readonly")
        device_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Output directory
        ttk.Label(system_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value="anomalies_export")
        ttk.Entry(system_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(system_frame, text="Browse", command=self.browse_output_directory).grid(row=1, column=2)
        
        # Buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(button_frame, text="Save Settings", 
                  command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_settings).pack(side=tk.LEFT, padx=5)
        
        # Log viewer
        log_frame = ttk.LabelFrame(settings_frame, text="System Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.system_log = scrolledtext.ScrolledText(log_frame, height=15)
        self.system_log.pack(fill=tk.BOTH, expand=True)
    
    def setup_logging(self):
        """Setup logging for the GUI."""
        # Create custom handler for GUI
        handler = LogHandler(self.system_log)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Setup logger
        self.logger = logging.getLogger("gui")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
    
    def check_model_status(self):
        """Check if trained model exists and update status."""
        if os.path.exists(self.model_path.get()):
            self.status_label.config(text="✅ Model found and ready for detection")
        else:
            self.status_label.config(text="⚠️ No trained model found. Please train a model first.")
    
    def browse_data_directory(self):
        """Browse for data directory."""
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.data_directory.set(directory)
    
    def browse_model_path(self):
        """Browse for model file."""
        filename = filedialog.askopenfilename(title="Select Model File", 
                                           filetypes=[("PyTorch models", "*.pth")])
        if filename:
            self.model_path.set(filename)
    
    def browse_output_directory(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def run_complete_pipeline(self):
        """Run the complete pipeline in a separate thread."""
        def run_pipeline():
            try:
                self.logger.info("Starting complete pipeline...")
                system = SpaceAnomalyDetectionSystem(self.data_directory.get())
                results = system.run_complete_pipeline(
                    confidence_threshold=self.confidence_threshold.get(),
                    train_epochs=self.train_epochs.get()
                )
                self.current_results = results
                self.logger.info("Pipeline completed successfully!")
                self.root.after(0, lambda: messagebox.showinfo("Success", "Pipeline completed successfully!"))
            except Exception as e:
                self.logger.error(f"Pipeline failed: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Pipeline failed: {e}"))
        
        threading.Thread(target=run_pipeline, daemon=True).start()
    
    def load_existing_model(self):
        """Load an existing trained model."""
        try:
            detector = AnomalyDetector(model_path=self.model_path.get())
            self.logger.info(f"Model loaded successfully from {self.model_path.get()}")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def detect_anomalies(self):
        """Detect anomalies using the trained model."""
        def run_detection():
            try:
                self.logger.info("Starting anomaly detection...")
                
                # Load preprocessed data
                X_test = np.load(os.path.join("preprocessed_data", "X_test.npy"))
                
                # Create detector
                detector = AnomalyDetector(model_path=self.model_path.get())
                
                # Detect anomalies
                results = detector.detect_anomalies(
                    X_test, 
                    confidence_threshold=self.confidence_threshold.get()
                )
                
                # Export results
                exported_files = detector.export_anomalies(X_test, results)
                
                self.current_results = results
                self.anomaly_images = exported_files
                self.current_image_index = 0
                
                self.logger.info(f"Detection completed. Found {len(results['high_confidence_anomalies'])} anomalies.")
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                    f"Detection completed!\nFound {len(results['high_confidence_anomalies'])} anomalies."))
                
            except Exception as e:
                self.logger.error(f"Detection failed: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {e}"))
        
        threading.Thread(target=run_detection, daemon=True).start()
    
    def view_results(self):
        """View detection results."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to view. Please run detection first.")
            return
        
        # Switch to results tab
        self.notebook.select(3)  # Results tab
        
        # Update summary
        summary = f"""
Detection Results:
================
Total images processed: {self.current_results.get('total_images_processed', 'N/A')}
High-confidence anomalies: {len(self.current_results['high_confidence_anomalies'])}
Total anomalies: {len(self.current_results['all_anomalies'])}
Confidence threshold: {self.current_results.get('confidence_threshold', 'N/A')}
Error threshold: {self.current_results.get('error_threshold', 'N/A'):.6f}
        """
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
        
        # Load first image
        if self.anomaly_images:
            self.load_image(0)
    
    def load_image(self, index):
        """Load and display an image."""
        if 0 <= index < len(self.anomaly_images):
            try:
                # Load image
                img = Image.open(self.anomaly_images[index])
                
                # Resize for display
                display_size = (400, 400)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Update display
                self.image_display.configure(image=photo, text="")
                self.image_display.image = photo  # Keep a reference
                
                # Update label
                self.image_label.config(text=f"Image {index + 1} of {len(self.anomaly_images)}")
                self.current_image_index = index
                
            except Exception as e:
                self.image_display.configure(image="", text=f"Error loading image: {e}")
    
    def next_image(self):
        """Show next image."""
        if self.anomaly_images:
            self.load_image((self.current_image_index + 1) % len(self.anomaly_images))
    
    def previous_image(self):
        """Show previous image."""
        if self.anomaly_images:
            self.load_image((self.current_image_index - 1) % len(self.anomaly_images))
    
    def export_results(self):
        """Export results to a file."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export. Please run detection first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_results, f, indent=2)
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def start_training(self):
        """Start model training in a separate thread."""
        def run_training():
            try:
                self.logger.info("Starting model training...")
                
                # Load data
                X_train = np.load(os.path.join("preprocessed_data", "X_train.npy"))
                
                # Create detector
                detector = AnomalyDetector()
                
                # Train model
                losses = detector.train(
                    X_train,
                    epochs=self.train_epochs.get(),
                    batch_size=self.batch_size.get(),
                    learning_rate=self.learning_rate.get()
                )
                
                self.logger.info("Training completed successfully!")
                self.root.after(0, lambda: messagebox.showinfo("Success", "Training completed successfully!"))
                
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
        
        threading.Thread(target=run_training, daemon=True).start()
    
    def stop_training(self):
        """Stop training (placeholder for future implementation)."""
        messagebox.showinfo("Info", "Stop training functionality will be implemented in future versions.")
    
    def save_settings(self):
        """Save current settings."""
        settings = {
            'data_directory': self.data_directory.get(),
            'model_path': self.model_path.get(),
            'confidence_threshold': self.confidence_threshold.get(),
            'train_epochs': self.train_epochs.get(),
            'batch_size': self.batch_size.get(),
            'learning_rate': self.learning_rate.get(),
            'device': self.device_var.get(),
            'output_directory': self.output_dir_var.get()
        }
        
        filename = filedialog.asksaveasfilename(
            title="Save Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=2)
                messagebox.showinfo("Success", f"Settings saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def reset_settings(self):
        """Reset settings to defaults."""
        self.data_directory.set("sdss_images")
        self.model_path.set("models/anomaly_detector.pth")
        self.confidence_threshold.set(0.8)
        self.train_epochs.set(10)
        self.batch_size.set(32)
        self.learning_rate.set(1e-3)
        self.device_var.set("auto")
        self.output_dir_var.set("anomalies_export")
        
        messagebox.showinfo("Info", "Settings reset to defaults.")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = SpaceAnomalyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 