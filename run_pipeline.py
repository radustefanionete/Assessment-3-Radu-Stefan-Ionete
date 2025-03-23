import os
import yaml
import subprocess

# Loading configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Step 1: Preprocessing Data
print("Running Text Preprocessing...")
subprocess.run(["python", "src/preprocess/preprocess_text.py"])

print("Running Image Preprocessing...")
subprocess.run(["python", "src/preprocess/preprocess_image.py"])

# Step 2: Training Models
print("Training Text Model...")
subprocess.run(["python", "src/models/train_nlp.py"])

print("Training Image Model...")
subprocess.run(["python", "src/models/train_cv.py"])

print("Training Fusion Model...")
subprocess.run(["python", "src/models/final_model.py"])

# Step 3: Evaluating Model
print("Evaluating Model Performance...")
subprocess.run(["python", "src/evaluation/evaluate.py"])

print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Running the pipeline
    main()
