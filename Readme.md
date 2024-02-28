# Indian Liver Patient Analysis

## Overview
This project utilizes machine learning techniques to analyze liver patient data and predict the presence of liver disease. Through a series of steps, including data preprocessing, model training, and evaluation, the project aims to provide insights into effective diagnosis and treatment strategies.

## Summary of Steps

### 1. Importing Necessary Libraries
   - Libraries such as NumPy, Pandas, Seaborn, Matplotlib, and scikit-learn were imported.

### 2. Loading Dataset
   - The dataset was loaded using Pandas from the provided CSV file.
   - Initial exploration of the dataset including data description, shape, and information was performed.

### 3. Handling Null Values
   - Checked for null values in the dataset and filled missing values using mean imputation.

### 4. Checking Dataset Balance
   - Visualized the class distribution to identify data imbalance.

### 5. Over Sampling
   - Applied RandomOverSampler to balance the dataset for better model performance.

### 6. Label Encoding
   - Encoded categorical features using LabelEncoder.

### 7. Correlation Analysis
   - Created a correlation matrix to understand the relationships between features.

### 8. KNN Model
   - Prepared data for modeling by splitting into train and test sets.
   - Standardized the features and trained a K-Nearest Neighbors (KNN) classifier.
   - Conducted hyperparameter tuning using GridSearchCV to find the best parameters.

### 9. Model Evaluation
   - Evaluated the model's performance using confusion matrix, accuracy score, and classification report.

### 10. Saving and Loading Model
   - Saved the trained model using pickle for future use.
   - Loaded the saved model for prediction on new data.

## Conclusion
The application of machine learning algorithms to liver patient data provides valuable insights for healthcare professionals. By leveraging predictive analytics, this project contributes to the development of effective diagnosis and treatment strategies for liver diseases.

