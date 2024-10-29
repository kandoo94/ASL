# main.py
import subprocess
import sys
import os

# Path to Python executable in the virtual environment
python_executable = os.path.join(os.path.dirname(sys.executable), "python")

# Step 1: Preprocess the dataset
print("Starting data preprocessing...")
subprocess.run([python_executable, "preprocess_data.py"])

# Step 2: Train the model
print("Starting model training...")
subprocess.run([python_executable, "train_model.py"])

# Step 3: Start real-time prediction with GUI
print("Starting real-time ASL translation...")
subprocess.run([python_executable, "real_time_predict.py"])
