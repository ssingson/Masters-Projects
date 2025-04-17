# Predicting Breast Cancer from Tumor Cell Nuclei

## Overview
This project investigates a classification approach to determine whether a breast tumor is benign or malignant based on features extracted from tumor cell nuclei. The study leverages several key techniques including data normalization, Principal Component Analysis (PCA) for feature selection, and a Support Vector Machine (SVM) model optimized through gradient descent and hyperparameter tuning. Out of 240 model variations tested, the final model—using a polynomial PCA kernel with 4 components, an RBF SVM kernel, and a linear slack variable—achieved an accuracy rate of 91% on the test dataset. This project was conducted as a final project for a graduate Machine Learning course at Fordham University.

## Dataset
- The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset).
- Dataset summary: 
  - Includes 569 patient records, each with 30 features extracted from cell nuclei
  - Features consist of 10 cell attributes (e.g., size, texture), each with 3 statistical metrics: mean, standard error, and worst case
  - Target column (`diagnosis`) was converted to 1 for malignant and -1 for benign

## Requirements
- The only requirement is **Jupyter Notebook**.
- No additional dependencies need to be installed beyond standard Python libraries.
- Packages used:
  - numpy
  - pandas
  - sklearn
  - imblearn
  - seaborn
  - time
  - datetime

## Running the Notebook
1. Open Jupyter Notebook.
2. Load the provided notebook.
3. Update the dataset path in the notebook to the correct file path where the dataset is saved on your computer:
   ```python
   bc_data = pd.read_csv('/Users/ssingson-robbins/Desktop/breast-cancer.csv').to_numpy()
   ```
4. Run the notebook **one cell at a time** to preprocess the data, perform PCA, train SVM models, and evaluate results.

## Key Features
- **Preprocessing:** Converts categorical labels, normalizes features, and balances class distributions using oversampling.
- **Feature Selection:** Principal Component Analysis (PCA) used to reduce dimensionality and minimize overfitting.
  - Multiple PCA kernels tested: Linear, Polynomial, RBF, Sigmoid
  - Up to 10 components used based on explained variance
- **Machine Learning Model:**
  - Support Vector Machine (SVM) trained using custom implementation with:
    - Dual Kernel method
    - Gradient descent optimization
    - Slack variable regularization
    - Kernel options: Linear, RBF
- **Model Evaluation:**
  - Accuracy
  - Model runtime and convergence tracking
  - Support vector distribution analysis

## Results
- 240 models were trained across combinations of PCA kernel, PCA components, SVM kernel, and slack variable type.
- The best-performing model achieved **91% accuracy** on the test dataset using:
  - **Poly PCA Kernel**
  - **4 PCA Components**
  - **RBF SVM Kernel**
  - **Linear Slack Variable**
- Analysis showed that models with more support vectors tended to perform better, and polynomial PCA kernels outperformed others.

## Acknowledgments
This project was developed by **Seth Singson-Robbins** at **Fordham University**.  
For further details, contact [seth.singson@gmail.com](mailto:seth.singson@gmail.com).