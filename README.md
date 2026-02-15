# Machine Learning Assignment 2

## Problem Statement
The objective of this assignment is to implement multiple classification models on a chosen dataset, evaluate their performance using standard metrics, and deploy an interactive Streamlit web application. The workflow demonstrates end-to-end ML deployment: modeling, evaluation, UI design, and deployment.

## Dataset Description
The dataset chosen is the **Wine Quality Dataset (UCI Repository)**.  
- **Instances:** ~1600  
- **Features:** 12 (physicochemical properties of wine)  
- **Target:** Wine quality (converted into binary classification: good vs bad wine, where quality ≥ 6 is considered good)

This dataset meets the minimum requirements (≥ 12 features, ≥ 500 instances) and provides a real-world classification problem.

## Models Used
The following machine learning models were implemented on the dataset:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using the following metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Model Performance Metrics

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression  | 0.7406   | 0.8190| 0.7857    | 0.7374 | 0.7608| 0.4793|
| Decision Tree        | 0.7281   | 0.7254| 0.7614    | 0.7486 | 0.7549| 0.4498|
| kNN                  | 0.7063   | 0.7737| 0.7202    | 0.7765 | 0.7473| 0.3994|
| Naive Bayes          | 0.7344   | 0.7927| 0.7582    | 0.7710 | 0.7645| 0.4600|
| Random Forest        | 0.8000   | 0.8954| 0.8177    | 0.8268 | 0.8222| 0.5937|
| XGBoost              | 0.8125   | 0.8787| 0.8362    | 0.8268 | 0.8315| 0.6203|

### Observations

| ML Model Name        | Observation about model performance |
|----------------------|-------------------------------------|
| Logistic Regression  | Performs reasonably well with balanced precision and recall, but limited in capturing complex non-linear relationships. |
| Decision Tree        | Captures non-linear patterns but shows signs of overfitting, leading to slightly lower generalization. |
| kNN                  | Performance depends heavily on scaling and choice of k; recall is relatively strong but overall accuracy is lower. |
| Naive Bayes          | Fast and effective with probabilistic assumptions; performs decently but struggles with correlated features. |
| Random Forest        | Strong ensemble method; achieves high accuracy and balanced metrics, reducing overfitting compared to a single tree. |
| XGBoost              | Delivers the best overall performance with high accuracy, precision, and MCC; efficient boosting makes it robust and reliable. |

## Streamlit App Features
- Dataset upload option (CSV)  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  

## Checkout the deployed App:
-  https://mlassignment2-cujrwsmh66wh4awmsaujzq.streamlit.app/ 


