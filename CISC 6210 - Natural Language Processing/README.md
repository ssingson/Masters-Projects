# Jeopardy Valuation Prediction Project

## Overview
This project investigates a classification approach to determine the appropriate round and monetary value for Jeopardy! questions using Natural Language Processing (NLP) techniques. The study leverages various embeddings and machine learning models to predict one of eleven possible Jeopardy! categories. The best-performing model—an RNN using GloVe embeddings—achieved an accuracy rate of 15.2% on the test dataset, significantly outperforming the baseline accuracy of 10.3% from random guessing. This project was conducted as a final project for a graduate Natural Language Processing course at Fordham University.

## Dataset
- The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions).
- Dataset summary:
  - Contains 216,930 Jeopardy! questions from 1984 to 2012
  - Includes metadata such as category, air date, round, and value
  - Converted to JSON and preprocessed to normalize values, filter out daily doubles, and remove stopwords (not included due to github constraints)

## Files 
- Data Modeling
   - Includes X training and test data, and various embedding models in pickle form to push into the code.
- NLP Models
  - NLP Models after both embedding and machine learning models are implemented.
    
## Requirements
- The only requirement is **Jupyter Notebook**.
- No additional dependencies need to be installed beyond standard libraries available in Python.
- Packages used:
  - pandas
  - re
  - nltk
  - sklearn
  - tensorflow
  - gensim

## Running the Notebook
1. Open Jupyter Notebook.
2. Load the provided notebook.
3. Update the dataset path in the notebook to the correct file path where the dataset is saved on your computer:
   ```python
   pd.read_json('/Users/ssingson-robbins/Desktop/jeopardy_questions.json')
   ```
4. Run the notebook **one cell at a time** to preprocess the data, embed text, train models, and evaluate results.

## Key Features
- **Preprocessing:** Data cleaning, removal of daily doubles, stopword filtering, text normalization
- **Text Embeddings:**
  - Term Frequency-Inverse Document Frequency (TfIdf)
  - GloVe
  - Word2Vec
  - BERT
- **Machine Learning Models:**
  - Neural Network (NN)
  - Recurrent Neural Network (RNN)
  - Convolutional Neural Network (CNN)
  - Naïve Bayes
  - Logistic Regression
- **Model Evaluation:**
  - Accuracy
  - Precision & Recall (per class)

## Results
- Out of all models tested, the **RNN with GloVe embeddings** performed best with a **15.2% accuracy** on the test set.
- This is a **47% improvement** over random guessing (10.3%).
- Recall was strongest for **Jeopardy! $100** and **Double Jeopardy! $400**, though precision across categories remained relatively low.
- Additional analysis explored accuracy trends over time and performance impact of including or excluding solution text.

## Acknowledgments
This project was developed by **Seth Singson-Robbins** at **Fordham University**.  
For further details, contact [seth.singson@gmail.com](mailto:seth.singson@gmail.com).
