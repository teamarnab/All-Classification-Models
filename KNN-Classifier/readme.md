# Air Quality Prediction Using KNN Classifier

## Overview
This repository contains a project where a **K-Nearest Neighbors (KNN) Classifier** was implemented to predict air quality using a dataset. The performance of the model was assessed using metrics such as **Accuracy Score** and the **Confusion Matrix**.

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
    - Implemented a K-Nearest Neighbors Classifier.
     
3. **Evaluation**:
    - Assessed the model using the **Accuracy Score**.
    - Visualized the **Confusion Matrix** to understand the performance.


## Files in the Repository
- `knn_classifier.py`: Python script for implementing and evaluating the KNN Classifier.
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
   git clone https://github.com/teamarnab/All-Classification-Models/tree/main/KNN-Classifier
   ```
2. Navigate to the project directory:
   ```bash
   cd air-quality-knn-classifier
   ```
3. Run the Python script:
   ```bash
   python knn_classifier.py
   ```

## Future Work
- Experiment with other classifiers like **Logistic Regression**, **Random Forest**, and **Support Vector Classifier (SVC)**.
- Improve feature engineering and dataset preprocessing.
- Perform hyperparameter tuning to further optimize the KNN model.

## Acknowledgments
- dataset source (kaggle): https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment?resource=download