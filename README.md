# Sales Forecasting and Demand Prediction Research

## Overview

This project aims to enhance sales forecasting accuracy by leveraging advanced machine learning techniques. Accurate demand prediction is crucial for businesses to optimize inventory management, reduce costs, and improve customer satisfaction. The research explores various models and methodologies to identify the most effective approaches for sales forecasting
---

## 📌 Problem Statement

The core challenge addressed in this research is the development of reliable and accurate sales forecasting models. Traditional forecasting methods often fall short in capturing complex patterns and seasonality in sales data. This project investigates the application of machine learning models to overcome these limitations and improve forecast precision.
---

## ⚙️ Technologies Used

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, LightGBM, TensorFlow/Keras, statsmodels
- **Environment**: Jupyter Notebook
- **Version Control**: Git & GitHub

---

## 📊 Exploratory Data Analysis (EDA)

The EDA process, as documented in the Updated_EDA_and_Feature_Selection_for_Sales_Optimization.ipynb, involved several key steps to prepare the data for modeling:

Data Cleaning: Addressed missing values and outliers to ensure data quality.

Feature Engineering: Created new features to capture temporal patterns, such as lag features and rolling statistics.

Correlation Analysis: Assessed relationships between variables to identify significant predictors.

Visualization: Utilized plots to understand sales trends, seasonality, and the impact of promotions.

These steps provided a solid foundation for the subsequent modeling efforts.

---

## 🧠 Implemented Models

### 📌Model 1: Logistic Regression
Notebook: Updated_Mariam_Mohamed_Model_One(Logistic_Regression).ipynb

Purpose: Predicts the probability of a sales event occurring, making it suitable for classification tasks such as determining whether a product will be sold on a given day.

Strengths: Simple and interpretable model,Effective for binary classification problems.

Limitations: Assumes a linear relationship between independent variables and the log odds of the dependent variable.
Not suitable for capturing complex nonlinear patterns.

### 📌 Model 2 : Hybrid Model (ANN + ARIMA)
Notebook: Mariam_Mohamed_Model_Two(Combination_Between_ANN_&_ARIMA).ipynb

Purpose: Combines the strengths of ARIMA and Artificial Neural Networks (ANN) to model both linear and nonlinear components of the sales data.
ScienceDirect

Strengths: ARIMA effectively captures linear trends and seasonality. ANN models complex nonlinear relationships and residual patterns.

Limitations: Increased model complexity and computational requirements. Requires careful tuning of both ARIMA and ANN components

### 📌 Model 3: Decision Trees
Notebook: Mariam_Mohamed_Goda_Model_Three(Decision_Trees).ipynb

Purpose: Utilizes a tree-like model of decisions to predict sales outcomes based on input features.

Strengths: Captures nonlinear relationships and interactions between variables. Easy to interpret and visualize.

Limitations: Prone to overfitting, especially with deep trees. Sensitive to small variations in the data.

### 📌 Model 4: XGBoost
Purpose: Handles nonlinear relationships and interactions between variables.

Strengths: High predictive accuracy and efficiency.

Limitations: Requires careful tuning to prevent overfitting.

### 📌 Model 5: Random Forest
Notebook: Updated_Reem_Ehab_Model_Two(RandomForest).ipynb

Purpose: Ensemble of decision trees that improves robustness and accuracy.

Strengths: Reduces variance through averaging. Handles overfitting better than individual trees.

Limitations: Less interpretable than a single tree. May be slower with many trees.

### 📌Model 6: LSTM (Long Short-Term Memory)
Notebook: Reem_Ehab_Model_Three(LSTM).ipynb

Purpose: Deep learning model designed for time-series forecasting.

Strengths: Captures long-term dependencies and sequential data behavior.

Limitations: Computationally expensive. Needs large datasets and careful training.

### 📌Model 7: K-Nearest Neighbors (KNN)
Notebook: Reem_Ehab_Model_Four(KNN).ipynb

Purpose: Non-parametric method that predicts based on proximity to training examples.

Strengths: Simple and easy to understand.

Limitations:Sensitive to noisy data and feature scaling. Not suitable for large datasets due to computation.

### 📌 Model 8: Support Vector Machines (SVM)
Notebook: Updated Tasneem Ashraf Model One(SVM_Model).ipynb

Purpose: Classifies or regresses data by finding optimal separating hyperplanes.

Strengths: Effective in high-dimensional spaces.

Limitations: Poor performance on large datasets. Needs careful kernel selection.

### 📌 Model 9: ELMs (Extreme Learning Machines)
Notebook: Updated Tasneem Ashraf Model Two(ELMs).ipynb

Purpose: Fast training algorithm for single-layer feedforward networks.

Strengths: Very fast learning speed.

Limitations: Can be less accurate than deeper architectures.

### 📌 Model 10: Gradient Boosting Machines (GBM)
Notebook: TasneemAshraf Model Three(GBM_Model).ipynb

Purpose: Sequentially builds trees to minimize errors.

Strengths: High accuracy and effective with structured data.

Limitations: Training can be slow. Sensitive to noise.

### 📌 Model 11: LightGBM
Notebook: TasneemAshraf Model Four(LightGBM_Model_).ipynb

Purpose: Gradient boosting framework that uses tree-based learning algorithms.

Strengths: Extremely fast and efficient. Scales well to large datasets.

Limitations: Can overfit on small datasets. Sensitive to hyperparameters.

---

## 🏆 Results Summary

- Hybrid models (e.g., ANN + ARIMA and Tree-based Ensembles) generally outperformed single-method approaches.
- LightGBM and XGBoost delivered excellent balance between speed and accuracy.
- LSTM showed potential on longer sequences but required significant training time.

---

## 📦 Used Data

### 🔗 Dataset Source

We used the publicly available dataset from Kaggle:  
**[Sales for Furniture Store - by Zahraa Alaa Tageldein](https://www.kaggle.com/datasets/zahraaalaatageldein/sales-for-furniture-store)**

This dataset contains transactional sales data with columns such as:
- `Order Date`, `Ship Date`
- `Sales`, `Profit`, `Quantity`
- `Segment`, `Region`, `Category`, `Sub-Category`
- `Customer ID`, `Product ID`, and more

---

### 🧹 After EDA and Preprocessing

File: `Super_Store_Data_After_EDA.csv`

Main operations:
- **Datetime Conversion**: Extracted year, month, day, and day-of-week from date fields.
- **Feature Engineering**: Lag features, rolling stats, time-based segments.
- **Outlier Treatment**: Applied winsorization to mitigate extreme values in `Sales` and `Profit`.
- **Encoding**: One-hot encoded categorical variables like `Region`, `Segment`, and `Category`.
- **Scaling**: Applied Min-Max Scaling for all numerical columns.

---

### 🔁 After Oversampling Techniques


File: `Balanced_super_store_data.csv`

- **SMOTE (Synthetic Minority Over-sampling Technique)**:
  - Applied to increase the representation of minority sales groups.
  - Ensures better generalization and fairness in model predictions.
  - Balanced dataset used in training classification models like Logistic Regression, SVM, and Decision Trees.

---
