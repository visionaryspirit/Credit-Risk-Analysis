# CREDIT RISK ANALYSIS REPORT
![logo](12-5-challenge-image.png)

>Applying Logistic Regression models with imbalanced, and balanced data sets. Understanding the metrics to assess each model, and predictions.

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

### Explain the purpose of the analysis 

* The purpose of the analysis is to build 2 models (one model training on imbalanced data, and the second model training on balanced data) utilizing Logistic Regression, and resampling method. Then, each model is assessed by their performance metrics such as Confusion Matrix, Classification Report, Accuracy, Precision, and Recall scores. Upon comparing both models, a recomendation is made.
* The data set provided (lending_data csv file ) was utilized to build both models.
  
### Explain what financial information the data was on, and what you needed to predict
 The financial information had the following feature columns (X):
* loan_size,
* interest_rate,
*  borrower_income,
*  debt_to_income,
*  number_of_accounts,
*  derogatory_marks,
*  total_debt
  The objective is to predict the loan_status(y) which is our target variable compiled of class 0 (healthy loan), and class 1 (high risk loan).

### Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

* The value_counts function provided me with the number of data points that I was working on in my data set. In the first model where I worked with an imbalanced data set the value counts for my target column (y) was 75036 data points for class 0 (healthy loan), and 2500 for class 1 (high risk loan).
* In contrast, for the second model I worked on with the same data set but this time utilizing the RandomOverSampler module allowed me to balance my data points in my target column (y), this time the value counts function displayed 56277 data points for class 0 (healthy loan), and the same value 56277 data points for class 1 (high risk loan). 

### Describe the stages of the machine learning process you went through as part of this analysis.

### As part of this analysis I took the following steps:

* Preprocessing the data: I first took a good observation at my csv file to grasp the content of my features, and target column that I was going to predict, I needed to put into perspective the complexity of the data set that I will be working with. The feature columns were all numerical, so there was no need to encode any categorical values. I did notice that in the instructions (Starter Code), it asked us to split the data into X, and y but there wasn't an instruction to use the StandardScaler to normalize the data into the same scale since the X feature columns had a variety of numerical dimensions such as interest_rate, borrower_income, loan_size, etc. I did used the StandardScaler, trained the model, predicted the model, and observed my metrics and ended up with an accuracy score of **0.98**. However, I commented this step out in my jupyter notebook. My purpose for scaling the data first was to observe my metrics, and how much difference it makes when preprocessing data.
* Train the model and predict: The data split into training data (75 %), and testing data (25%)
* Analyse predictions with metrics to assess model: The metrics to assess both models were to use the Confusion Matrix, Classification report, Accuracy score, precision, recall, and such.

### Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

* The LogisticRegression classification method was utilized in both models (Imbalanced model, Balanced model).
* The resampling method was utilized in the second model to balance minority class data for class 1 (high risk loan). Once balanced, the LogisticRegression method was applied.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

### Machine Learning Model 1:
>Description of Model 1 Accuracy, Precision, and Recall scores.
    
* Balanced accuracy score: The score is **0.94.4** (As previously mentioned before, this score is the result of using the original data without scaling the X features, as this step wasn't part of the Starter Code file. However, I did try the first model by scaling the numerical X columns after spliting the data into X and y, and the balanced accuracy score I got was 98%. I wanted to experiment how much difference scaling the X features would have changed my results. I commented out that step of mine since it wasn't a requirement in the jupyter notebook). The 94% accuracy score represents the actual imbalance in the classes (0 and 1) as the model was built with more input data for the class 0 (healthy loan) than class 1 (high risk loan).
  
* Precision score for class 0 (healthy loan): **1.00**
* Precision score for class 1 (high risk loan): **0.87**  
* Recall score for class 0 (healthy loan): **1.00**
* Recall score for class 1 (high risk loan): **0.89** 


### Machine Learning Model 2:
>Description of Model 2 Accuracy, Precision, and Recall scores.

* Balanced accuracy score: The score for this model is **0.99.5**
* Precision score for class 0 (healthy loan): **1.00**
* Precision score for class 1 (high risk loan): **0.87**  
* Recall score for class 0 (healthy loan): **1.00**
* Recall score for class 1 (high risk loan): **1.00** 

## Summary
>Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
Upon assessment of both models' performance metrics. The model that seems to perform best is model 2. The accuracy score is **0.99**, whereas the accuracy score for model 1 is **0.94**. Model 1 is built using imbalanced data, which is therefore a bias model to begin with. The classification of our target label is Class 0: Healhty loan, or Class 1: High risk loans. Beyond accuracy scores, high precision scores relates to a low false positive rate, whereas high recall scores relates to a low false negative rate.
Therefore, is it more costly/detrimental/painful to have low false positive rates or low false negative rates?
Low false positive rates means low rates of false alarms, or low rates of accounts flagged as high risk loan incorrectly.
Low false negative rates means low rates of accounts flagged as healthy loan incorrectly. Therefore, it seems that recall would be the metric to highlight in the performance of the models. The recall score of class 0, and class 1 is **1.00** in model 2. The tradeoff would be to have some accounts being flagged as high risk incorrectly, but we wouldn't miss an account flagged as healthy incorrectly, which appears to be the most painful, costly, and detrimental outcome. 
I would be recommending model 2 for those reasons.
