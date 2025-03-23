import os
import yaml
import subprocess

# Loading configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Step 1: Preprocessing Data
print("Running Text Preprocessing...")
subprocess.run(["python", "src/preprocess_text.py"])

print("Running Image Preprocessing...")
subprocess.run(["python", "src/preprocess_images.py"])

# Step 2: Training Models
print("Training Text Model...")
subprocess.run(["python", "train/train_text_model.py"])

print("Training Image Model...")
subprocess.run(["python", "train/train_image_model.py"])

print("Training Fusion Model...")
subprocess.run(["python", "train/train_fusion.py"])

# Step 3: Evaluating Model
    print("Evaluating Model Performance...")
    run_script(EVALUATE_MODEL)

print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
