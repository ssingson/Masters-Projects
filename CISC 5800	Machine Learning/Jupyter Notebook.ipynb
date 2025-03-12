#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U imbalanced-learn


# In[2]:


import pandas as pd 
import numpy as np
from sklearn import preprocessing
import seaborn as sb
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import KernelPCA, PCA
from imblearn.over_sampling import RandomOverSampler
import time
from datetime import timedelta


# # Cleaning and Normalizing the Data

# In[3]:


#Pull csv file in, change to numpy type
bc_data = pd.read_csv('/Users/ssingson-robbins/Desktop/breast-cancer.csv').to_numpy()

#remove ID columns since not a dimension
bc_data = np.delete(bc_data, 0, 1)

#replace 'benign' and 'malignant' to -1 and 1, respectively, to more align with SVM model 
bc_data = np.select([bc_data == 'B', bc_data == 'M'], [-1,1], bc_data)

#separate classifier from the data
y_bc_data = bc_data[:,0]
x_bc_data = bc_data[:, 1:len(bc_data)]

# Normalize the data 
# #x_bc_data = (x_bc_data - x_bc_data.mean(axis=0)) / x_bc_data.std(axis=0)
for i in range(len(x_bc_data[0,:])): 
    x_bc_data[:, i] = (x_bc_data[:, i] - x_bc_data[:, i].mean(axis=0)) / x_bc_data[:, i].std()


# # Split Training and Test Data

# In[4]:


#split up test vs training data
from sklearn.model_selection import train_test_split

#10% for validating the data, 10% for testing the final model 
X_train, X_test, y_train, y_test = train_test_split(x_bc_data, y_bc_data, test_size=0.2, stratify=y_bc_data, random_state=0)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=0)

#ros = RandomOverSampler(sampling_strategy = 'minority', random_state=0)
#X_train, y_train = ros.fit_resample(X_train, y_train.astype('int'))

y_test.shape


# # PCA Intro Chart

# In[5]:


#PCA Analysis, equivalent of KernelPCA with the 'linear' Kernel

pca = PCA(n_components = 10)
new_data = pca.fit_transform(X_train)

print(pca.explained_variance_ratio_.cumsum())
plt.plot(pca.explained_variance_ratio_.cumsum())


# In the paper, explain how it's a different feature space so didn't include charts for other kernels but will used
# for different model and performance


    


# # Model Functions

# In[6]:


#find sum of positive and negative classes, make summation equal by shrinking the lower values proportionally
def regularize_alphas(y_train, a): 
    #pushes down examples with negative a's to 0 
    a = np.maximum(a, 0)
    
    if np.abs(np.minimum(a * y_train,0)).sum() > 0: 
        sum_ratio = np.maximum(a * y_train,0).sum() / np.abs(np.minimum(a * y_train,0)).sum()

        # ratios the negative class items so that they're equal with the positive class when summed up separately 
        a = np.multiply(a, np.where(y_train == -1, sum_ratio,1))
    else: 
        a = 0 * a
    return a

test = np.ones(len(X_train))
a = regularize_alphas(y_train, test)


# In[7]:


# SVM Model from scratch 
def SVM_scratch(X_train, y_train, SVM_Kernel = 'linear', slack_variable = 'no'):
    #base Lagrange: max[a] of sum(a[j]) - ((1/2) sum[i,j](y[i]*y[j])*a[i]*a[j]*K(x[i], x[j]))
    #base gradient descent: learning rate[0.001] * (1 - y[i] * y[j] * K(x[i], x[j]) * a[i?j?])
    
    #sets which kernel to user 
    if SVM_Kernel == 'linear': 
        K = np.dot(X_train, X_train.transpose())
    elif SVM_Kernel == 'rbf': 
        K = np.zeros([len(X_train), len(X_train)])
        for i in range(len(X_train)): 
            for j in range(len(X_train)): 
                K[i,j] = math.e ** (-0.5 * ((X_train[i] - X_train[j]) ** 2).sum())
    else:
        print('Only options allowed are linear and rbf.')
        return 'Error'
    
    #creates a yi * yb matrix for every combo of examples
    Y = np.dot(y_train[:, np.newaxis], y_train[np.newaxis])
    
    a = np.ones(len(X_train))
    a = regularize_alphas(y_train, a)

    #gradient descent of the alpha values
    for i in range(100):
        #adds a slack variable 
        if slack_variable == 'no': 
            misaligned = 0
            slack_descent = 0
        else: 
            misaligned = (np.sign(np.dot(X_train, np.dot((a * y_train), X_train))) != y_train)
            
            if slack_variable == 'linear':
                slack_descent = - y_train * (np.dot((y_train), X_train) * X_train).sum(axis=1) * misaligned
                
            elif slack_variable == 'squared':
                slack_descent = 2 * (1 - (y_train * (np.dot((a * y_train), X_train) * X_train).sum(axis=1))) * -(y_train * (np.dot(y_train, X_train) * X_train).sum(axis=1)) * misaligned
            else: 
                print('Only options allowed are no, linear, and squared.')
                return 'Error'

        a = a + 0.001 * (1 - np.dot(Y*K,a)) + 0.002 * slack_descent
        a = regularize_alphas(y_train, a) 

    w = np.dot((a * y_train), X_train) 
    return a, w #returns the alpha values, weights, and loss per round 


# # Prediction Model Object Creator

# In[8]:


class predictionModel():
    def __init__(self, weights, alphas, PCA_model, PCA_Kernel, PCA_components, SVM_Kernel, slack_variable):
        self.name = 'PCA + SVM Model: ' + str(PCA_Kernel) + ' PCA Kernel with ' + str(PCA_components) + ' components, ' + str(SVM_Kernel) + ' SVM Kernel with ' + str(slack_variable) + ' slack variable' 
        self.PCA_model = PCA_model
        self.PCA_Kernel = PCA_Kernel
        self.PCA_components = PCA_components
        self.SVM_Kernel = SVM_Kernel
        self.slack_variable = slack_variable
        self.weights = weights
        self.support_vectors = (alphas > 0).sum()
    
    def accuracy(self, X_test, y_test): 
        kpca_test_data = kpca.transform(X_test)
        

        
        data = np.dot(kpca_test_data, self.weights)

        if self.slack_variable == 'linear': 
            data += 0.2 * np.minimum(1, np.maximum(0, 1 - y_test * data))
        if self.slack_variable == 'squared':
            data += 0.2 * (np.minimum(1, np.maximum(0, 1 - y_test * data)) ** 2)
            
        accuracy = np.zeros(len(data))
        for i in range(len(data)): 
            if np.abs(data[i]) < 1: 
                accuracy[i] = -1
            elif np.sign(data[i]) == y_test[i]: 
                accuracy[i] = 1
            else:
                accuracy[i] = 0
        return (accuracy == 1).sum() / len(accuracy)
    
    def performance_results(self, X_test,y_test): 
        kpca_test_data = self.PCA_model.transform(X_test)
        
        data = np.dot(kpca_test_data, self.weights)        
        
        if self.slack_variable == 'linear': 
            data += 0.2 * np.minimum(1, np.maximum(0, 1 - y_test * data))
        if self.slack_variable == 'squared':
            data += 0.2 * (np.minimum(1, np.maximum(0, 1 - y_test * data)) ** 2)
                
        accuracy = np.zeros(len(data))
        for i in range(len(data)): 
            if np.abs(data[i]) < 1: 
                accuracy[i] = -1
            elif np.sign(data[i]) == y_test[i]: 
                accuracy[i] = 1
            else:
                accuracy[i] = 0
        exs_unid = (accuracy == -1).sum()
        exs_incorrect = (accuracy == 0).sum()
        exs_correct = (accuracy == 1).sum()
        return exs_unid, exs_incorrect, exs_correct
    
    def predict(self, x_test): 
        if len(x_test.shape) == 1: 
            x_test = x_test[np.newaxis,:]
            
        kpca_test_data = self.PCA_model.transform(x_test)

        model_value = np.dot(new_data, w)

        predicted_values = []
        for i in range(len(model_value)): 
            if model_value[i] >= 1: 
                predicted_values.append('Malignant')
            if model_value <= -1: 
                predicted_values.append('Benign')
            else: 
                predicted_values.append('Unidentifiable')

    #Chart loss
        
    


# # Run of All the Model Iterations

# In[14]:


#Run through of all 240 models; checks time to run the models 
start_time = time.monotonic()

model_runs = []    

for PCA_components in range(1, 11, 1): 
    for PCA_Kernel in ['linear', 'poly', 'rbf', 'sigmoid']: 
   
        kpca = KernelPCA(n_components = PCA_components, kernel = PCA_Kernel)
        kpca_data = kpca.fit_transform(X_train) 
        
        for SVM_Kernel in ['linear', 'rbf']: 
            for slack_variable in ['no', 'linear', 'squared']: 
              

                a,w = SVM_scratch(kpca_data, y_train, SVM_Kernel = SVM_Kernel, slack_variable = slack_variable)
                model_runs.append(predictionModel(w,a, kpca, PCA_Kernel, PCA_components, SVM_Kernel, slack_variable))
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
print(timedelta(seconds=end_time - start_time) / 240)        


# In[10]:


# Save the results in a pandas dataframe and push it into a csv file
df = pd.DataFrame(columns=['PCA Kernal','PCA_Components','SVM Kernel', 'Slack Variable', 'Support Vectors', 'Unidentifiable Examples', 'Incorrect Examples', 'Correct Examples', 'Accuracy', 'Weights'])
for i in model_runs: 
    exs_unid, exs_incorrect, exs_correct = i.performance_results(X_validation,y_validation)
    df2 = {'PCA Kernal': i.PCA_Kernel,'PCA_Components': i.PCA_components,'SVM Kernel': i.SVM_Kernel, 'Slack Variable': i.slack_variable, 'Support Vectors': i.support_vectors, 'Unidentifiable Examples': exs_unid, 'Incorrect Examples': exs_incorrect, 'Correct Examples': exs_correct, 'Accuracy': exs_correct / (exs_unid + exs_incorrect + exs_correct), 'Weights': i.weights.sum()}
    df = df.append(df2, ignore_index = True)
df.to_csv('/Users/ssingson-robbins/Desktop/breast-cancer-classification-results.csv')


# In[11]:


#check time to run the models 
start_time = time.monotonic()
model_runs = []    

for PCA_components in range(1, 11, 1): 
    for PCA_Kernel in ['linear', 'poly', 'rbf', 'sigmoid']: 
   
        kpca = KernelPCA(n_components = PCA_components, kernel = PCA_Kernel)
        kpca_data = kpca.fit_transform(X_train) 
        
        for SVM_Kernel in ['linear', 'rbf']: 
            for slack_variable in ['no', 'linear', 'squared']: 
                
                a,w = SVM_scratch(kpca_data, y_train, SVM_Kernel = SVM_Kernel, slack_variable = slack_variable)
                model_runs.append(predictionModel(w,a, kpca, PCA_Kernel, PCA_components, SVM_Kernel, slack_variable))
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
print(timedelta(seconds=end_time - start_time) / 240)


# In[13]:


# Results of the final model
PCA_components = 4
PCA_Kernel = 'poly'
SVM_Kernel = 'rbf' 
slack_variable = 'linear'  

kpca = KernelPCA(n_components = PCA_components, kernel = PCA_Kernel)
kpca_data = kpca.fit_transform(X_train) 

a,w = SVM_scratch(kpca_data, y_train, SVM_Kernel = SVM_Kernel, slack_variable = slack_variable)
final_model = predictionModel(w,a, kpca, PCA_Kernel, PCA_components, SVM_Kernel, slack_variable)
exs_unid, exs_incorrect, exs_correct = final_model.performance_results(X_test, y_test)
print('Accuracy:', final_model.accuracy(X_test, y_test))


# In[ ]:




