# Scania-Predictive-Maintenance-XAI
Deep Learning and XGBoost predictive maintenance models applied to the Scania Component X dataset, utilizing SHAP for Explainable AI.


# Explainable AI for Predictive Maintenance in Industrial Systems

## Overview
This repository contains the code and academic report for predicting Air Processing System (APS) failures in heavy-duty Scania commercial vehicles. The project addresses extreme class imbalances in industrial sensor data by benchmarking a Deep Learning approach (LSTM) against a Tree-based ensemble (XGBoost), optimized for a specific financial cost-matrix. 

## Key Features
* **Time-Series Preprocessing:** Execution of relational merging, temporal imputation (forward/backward fill), and Z-score standardization on 1.1 million rows of operational data.
* **Feature Engineering:** Extraction of rolling statistical time-domain features and Mutual Information dimensionality reduction (selecting the top 20 critical sensors).
* **Model Benchmarking:** Comparison of a 3D sequential LSTM network against a 2D flattened XGBoost ensemble with dynamic threshold tuning (Recall: 0.77).
* **Explainable AI (XAI):** Integration of SHAP (`GradientExplainer` and `TreeExplainer`) to map model predictions to physical sensor contributions, highlighting multicollinearity between highly correlated proxy variables (e.g., sensors 167_6 and 158_8).

## Files Included
* `Scania_Predictive_Maintenance.ipynb`: The fully documented Python codebase containing data preprocessing, model training, and SHAP visualizations.
##Dtaset Link
*Dataset Link: https://researchdata.se/en/catalogue/dataset/2024-34/2

## Technologies Used
* Python, Pandas, NumPy
* TensorFlow / Keras (LSTM)
* XGBoost
* Scikit-Learn (Mutual Information, Class Weights)
* SHAP (Explainable AI)
