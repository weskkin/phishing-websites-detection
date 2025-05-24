import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import joblib
import json
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PhishingDetectorApp:
    def __init__(self, root):
        self.root = root # Main window
        self.root.title("Phishing Website Detector")
        self.models = {}
        self.scaler = joblib.load("../models/scaler.pkl") # Load pretrained scaler

        # Load feature names used during training
        with open('../data/processed/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)

        # GUI components
        self.create_widgets()

    def create_widgets(self):
        # Load data button
        self.btn_load = ttk.Button(self.root, text="Load Dataset", command=self.load_data)
        self.btn_load.pack(pady=10)

        # Model Selection Dropdown
        self.model_var = tk.StringVar() # keeps track of the current selection in the combo box
        self.model_dropdown = ttk.Combobox(self.root, textvariable=self.model_var, 
                                         values=["LogisticRegression", "DecisionTree", "RandomForest"]) # Any change in the combo box updates self.model_var automatically
        self.model_dropdown.pack(pady=10)

        # Run Button
        self.btn_run = ttk.Button(self.root, text="Run Model", command=self.run_model)
        self.btn_run.pack(pady=10)

        # Results Display
        self.results_text = tk.Text(self.root, height=15, width=50)
        self.results_text.pack(pady=10)

        # ROC Curve Plot
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.results_text.insert(tk.END, "Data loaded successfully!\n")

    def run_model(self):
        model_name = self.model_var.get()
        if not hasattr(self, 'data') or model_name == "": # Checks whether load_data has run and set self.data and whether the user has picked a model
            self.results_text.insert(tk.END, "Error: Load data and select a model first!\n")
            return
        
        try:
            # Preprocess data (drop columns and scale)
            columns_to_drop = ['FILENAME', 'URL', 'Domain', 'Title', 'TLD']
            if 'label' in self.data.columns:
                columns_to_drop.append('label')
            X = self.data.drop(columns=columns_to_drop)
            
            # Align features with training data
            X = X[self.feature_names]  # Use the saved feature names to select columns

            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)

            # Load model
            model = joblib.load(f'../models/{model_name}.pkl')

            # Predict
            y_pred = model.predict(X_scaled)
            self.results_text.insert(tk.END, f"Predictions using {model_name}: {y_pred[:10]}...\n")

            # Display saved metrics (from metrics.json)
            with open('../reports/metrics.json', 'r') as f:
                metrics = json.load(f)[model_name]
                for metric, value in metrics.items():
                    self.results_text.insert(tk.END, f"{metric}: {value:.4f}\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhishingDetectorApp(root)
    root.mainloop()