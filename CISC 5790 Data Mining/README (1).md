# Stroke Prediction Data Mining Project

## Overview
This project is a data mining analysis for predicting stroke occurrences based on healthcare data. The dataset contains various health indicators, and the goal is to train machine learning models to predict the likelihood of a stroke. This project was conducted as a final project for a graduate course at Fordham University.

## Dataset
- The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
- Users must **download the dataset** and place it in their local system.
- Update the dataset path in the notebook:
  ```python
  pd.read_csv('C:/Users/seths/Desktop/healthcare-dataset-stroke-data.csv')
  ```
  to the correct file path where the dataset is saved on their computer.

## Requirements
- The only requirement is **Jupyter Notebook**.
- No additional dependencies need to be installed beyond standard libraries available in Python.

## Running the Notebook
1. Open Jupyter Notebook.
2. Load the provided notebook.
3. Run the notebook **one cell at a time** to preprocess data, train models, and evaluate results.

## Key Features
- **Preprocessing:** Handles categorical variables, standardizes numerical features, and fills missing values (BMI).
- **Machine Learning Models:**
  - Decision Trees
  - Ensemble Methods
  - Linear Models
  - Naïve Bayes
  - Neural Networks
- **Model Evaluation:**
  - ROC-AUC Score
  - Precision-Recall Curve
  - ROC Curve
- **Performance Metrics:**
  - The best-trained model achieved an **80% true positive rate** with a **25% false positive rate**.
  - Performance was **0.55 lower than random guessing**.

## Results
- The project evaluates model effectiveness and compares predictions with actual stroke occurrences.
- Charts and comparison results between **training and test data** are included.

## Acknowledgments
This project was completed as part of a **graduate-level course final project** for a **Master's program at Fordham University**.
