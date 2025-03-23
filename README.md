# Assessment-3-Radu-Stefan-Ionete

## Overview
This project implements a **multi-modal sentiment analysis system** that combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** techniques to classify sentiment from both text and images. The model predicts sentiment as **positive**, **negative**, or **neutral** based on the input data.

## Features
- **Text Processing**: Tokenization, stopword removal, word embeddings (Word2Vec, GloVe, BERT)
- **Image Processing**: CNN-based feature extraction (ResNet, VGG)
- **Fusion Model**: Merging text and image features for final classification
- **End-to-End Pipeline**: From data preprocessing to model evaluation

## Project Structure
```
├── config.yaml # Configuration file
├── requirements.txt # Dependencies
├── run_pipeline.py # Main script to run the pipeline
├── src/
│ ├── preprocessing/
│ │ ├── preprocess_text.py # Text preprocessing
│ │ ├── preprocess_image.py # Image preprocessing
│ ├── models/
│ │ ├── train_text_model.py # NLP model training
│ │ ├── train_image_model.py # CV model training
│ │ ├── train_fusion.py # Fusion model
│ ├── evaluation/
│ │ ├── evaluate.py # Model evaluation
│ ├── utils/
│ │ ├── helpers.py # Utility functions
│ ├── data/
│ │ ├── dataset/ # Local dataset storage
│ ├── download_data.py # Local dataset handling script
```

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/radustefanionete/Assessment-3-Radu-Stefan-Ionete.git
cd Assessment-3-Radu-Stefan-Ionete
```

### 2. Install Dependencies
Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Ensure that the dataset is stored locally inside the data/dataset/ folder.

### 4. Run the Pipeline
To run the full pipeline (data preprocessing, training, and evaluation):

```bash
python run_pipeline.py
```
