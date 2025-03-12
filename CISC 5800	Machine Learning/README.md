# Predicting Breast Cancer from Tumor Cell Nuclei

## Overview
This project investigates a classification approach to determine whether a breast tumor is benign or malignant based on features extracted from tumor cell nuclei. The study leverages several key techniques including data normalization, Principal Component Analysis (PCA) for feature selection, and a Support Vector Machine (SVM) model optimized through gradient descent and hyperparameter tuning. Out of 240 model variations tested, the final model—using a polynomial PCA kernel with 4 components, an RBF SVM kernel, and a linear slack variable—achieved an accuracy rate of 91% on the test dataset.

## Dataset
The data is sourced from Kaggle’s [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset).

Download the dataset from Kaggle and update the file path in your project accordingly:

```python
pd.read_csv('path/to/your/breast-cancer-dataset.csv')
```

## Requirements
- Python (preferably with Anaconda)
- Jupyter Notebook
- Required libraries:
  - NumPy
  - scikit-learn
  - imbalanced-learn (for oversampling)
  - Pandas

## Running the Project
1. **Setup Environment:** Install the required Python libraries.
2. **Download the Dataset:** Get the dataset from Kaggle and save it locally.
3. **Update File Path:** Modify the data-loading script in the notebook:

   ```python
   pd.read_csv('path/to/your/breast-cancer-dataset.csv')
   ```

4. **Run Notebook:** Open the Jupyter Notebook and execute the cells sequentially to preprocess data, train multiple SVM models, and evaluate their performance.

## Key Features
### Data Preprocessing:
- Conversion of class labels (malignant set to 1, benign set to -1).
- Normalization of features to zero mean and unit variance.

### Feature Selection:
- Utilizes PCA (with various kernels such as linear, poly, RBF, and sigmoid) to reduce dimensionality and mitigate overfitting.

### Modeling:
- Implements an SVM with dual kernel optimization and gradient descent for support vector learning.
- Hyperparameter tuning over 240 different models varying PCA components, PCA kernel, SVM kernel, and slack variable type.

### Performance Evaluation:
- The best model is chosen based on validation accuracy and simplicity (Occam’s razor), achieving a 91% accuracy on unseen test data.

## Results
### Best Model Configuration:
- **PCA:** Polynomial kernel with 4 components.
- **SVM:** RBF kernel with a linear slack variable.
- **Accuracy:** The final test accuracy achieved was **91%**.
- Additional insights from the experiments include the impact of support vector percentage on model performance and the influence of slack variables on overfitting.

## Acknowledgments
This project was developed by **Seth Singson-Robbins** at **Fordham University**.  
For further details, contact [seth.singson@gmail.com](mailto:seth.singson@gmail.com).
