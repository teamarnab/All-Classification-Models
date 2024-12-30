# Air Quality Prediction Using KNN Classifier

## Overview
This repository contains a project where a **Logistic Regression** was implemented to predict air quality using a dataset. The performance of the model was assessed using metrics such as **Accuracy Score** and the **Confusion Matrix**.

## Dataset
The air quality dataset used in this project contains features related to various environmental parameters. The dataset has been preprocessed to ensure its suitability for machine learning tasks.

### Features
- Temperature, Humidity, PM2.5, PM10, NO2, SO2, CO, Proximity_to_Industrial_Areas, Population_Density 

### Target Variable
- Air Quality

## Methodology
1. **Data Preprocessing**:
    - Handled missing values (if any).
    - Normalized the feature values for better performance of the KNN algorithm.
    - Split the data into training and testing sets.

2. **Model Training**:
    - Implemented a Logistic Regression algorithm.
     
3. **Evaluation**:
    - Assessed the model using the **Accuracy Score**.
    - Visualized the **Confusion Matrix** to understand the performance.


## Files in the Repository
- `logistic_regression.py`: Python script for implementing and evaluating the KNN Classifier.
- `README.md`: Project documentation.

## Prerequisites
To run the project, make sure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (optional, for visualization)

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/teamarnab/All-Classification-Models/tree/main/Logistic-Regression
   ```
2. Navigate to the project directory:
   ```bash
   cd air-quality-classifier
   ```
3. Run the Python script:
   ```bash
   python logistic_regression.py
   ```

## Future Work
- Experiment with other classifiers like **K Neighbors Classifier**, **Random Forest**, and **Support Vector Classifier (SVC)**.
- Improve feature engineering and dataset preprocessing.
- Perform hyperparameter tuning to further optimize the KNN model.

## Acknowledgments
- dataset source (kaggle): https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment?resource=download