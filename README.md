# **Customer Churn Prediction using Deep Learning** #


## Project Overview ##


  
  Customer churn is one of the most critical challenges faced by telecom companies. Retaining existing customers is significantly more cost-effective than acquiring new ones.
This project focuses on predicting customer churn using deep learning techniques to help businesses proactively identify customers who are likely to leave and take preventive actions.


The project implements and compares multiple deep learning models and provides a Streamlit-based deployment demo for practical understanding.


## Problem Statement ##


To predict whether a telecom customer is likely to churn or stay based on their demographic, service usage, and billing information using deep learning models.



## Dataset Description ##


#### Dataset Used ####


**Telecom Customer Churn Dataset**


**Files Used**


   ---> telecom_customer_churn.csv – main dataset

   ---> telecom_data_dictionary.csv – column descriptions

   ---> telecom_zipcode_population.csv – demographic enrichment




**Dataset Contains**



 ---> Customer demographics (Age, Gender, Dependents)


 ---> Service details (Internet type, phone service, streaming)


 ---> Billing information (Monthly charges, total charges)


 ---> Contract and payment details


 ---> Customer status (Stayed / Churned)

 
 
 ## Step-by-Step Methodology ##

 
## **1.Data Loading & Understanding** ##


   ---> Loaded dataset using pandas


  ---> Explored feature distributions and target variable

  
 ---> Identified categorical and numerical columns



## **2️.Data Preprocessing** ##



 ---> To ensure clean and reliable modeling, the following preprocessing steps were applied
 

 ---> Removed irrelevant identifiers


 ---> Handled missing values


 ---> Encoded categorical variables using one-hot encoding


 ---> Scaled numerical features using StandardScaler


 ---> Removed data leakage columns:


 ---> Customer Status


 ---> Churn Category


 ---> Churn Reason



Why is it important?

Data leakage can artificially inflate model performance and must be avoided in real-world systems.


## **3️ Feature Engineering** ##



High-cardinality columns (City, Zip Code) were removed to avoid feature explosion



Final dataset prepared for deep learning models



# *Models Implemented* #



#### The project implements three different modeling approaches: ####

  ##### 1. Artificial Neural Network (ANN) #####
  

   ---> Used as a baseline deep learning model
   

   ---> Captures non-linear relationships in customer behavior
   

   ---> Well-suited for structured tabular data
   

##### 2. Long Short-Term Memory (LSTM) #####


   ---> Sequence-based deep learning model


   ---> Applied by reshaping tabular data into sequence format


   ---> Used to compare ANN vs sequence learning performance

  


 ##### 3. Autoencoder (Anomaly Detection) #####

 

   ---> Treats churn as an anomalous behavior


  ---> Trained only on non-churn customers


   ---> High reconstruction error indicates potential churn


   ---> This approach adds advanced analytical depth to the project.


 
 
 ## **Model Evaluation Metrics** ##

 

Each model was evaluated using:



  ---> Accuracy

  ---> Precision

  ---> Recall

  ---> F1-Score

  ---> ROC Curve


## **Model Performance Comparison (Reference)** ##



| Model        | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| ANN         | 0.9368   | 0.9495    | 0.8048 | **0.8712** |
| LSTM        | 0.9070   | 0.8809    | 0.7513 | 0.8110 |
| Autoencoder| 0.7097   | 0.2535    | 0.0481 | 0.0809 |



##  **Best Performing Model** ##


| Metric     | Value |
|-----------|-------|
| Model     | ANN |
| Accuracy  | 0.9368 |
| Precision | 0.9495 |
| Recall    | 0.8048 |
| F1-Score  | **0.8712** |




Based on F1-Score, which balances precision and recall:


## **Artificial Neural Network (ANN)** ##


Accuracy: 93.68%


Precision: 94.95%


Recall: 80.48%


F1-Score: 87.12%



 --->  The ANN model achieved the best balance between correctly identifying churners and minimizing false positives.




## **Model Persistence** ##


   ---> Trained models saved as .h5 files
   

   ---> Scaler saved using joblib
   

   ---> Enables reuse for deployment and inference


   

# **Deployment (Streamlit)** #



## **Deployment Approach** ##



 ---> Built a Streamlit web application
 

 ---> Allows users to input customer details



 ---> Displays churn probability and prediction result

 


## **Tools, Technologies & Libraries Used** ##



   ---> Python


   ---> pandas ( data handling)


   ---> numpy ( numerical computation)

  
   ---> scikit-learn ( preprocessing & metrics)


   ---> TensorFlow / Keras ( deep learning models)

   
   ---> joblib  ( model persistence)


   ---> Streamlit ( web application )


   ---> GitHub  ( version control)




## **Final Output** ##



   ---> Trained deep learning churn prediction models
   

   ---> Model comparison analysis
   

   ---> Best model identification


   ---> Streamlit-based deployment demo


   ---> End-to-end reproducible pipeline




##  **Key Learning Outcomes** ##



   ---> Real-world data preprocessing


   ---> Handling data leakage
   

   ---> Deep learning model comparison
   

   ---> Anomaly detection using autoencoders


   ---> Deployment constraints and solutions


   ---> Professional project structuring



## **Conclusion** ##



   This project demonstrates a complete end-to-end customer churn prediction system, from raw data preprocessing to model comparison and deployment. By implementing multiple deep learning approaches and selecting the best model based on robust metrics, the system provides meaningful business insights that can help telecom companies reduce customer churn and improve retention strategies.


