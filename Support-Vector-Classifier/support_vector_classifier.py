# -*- coding: utf-8 -*-
"""support_vector_classifier

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19_W7LOKYf_ht1ZY44naNPnNKj6A1ov9F

About Dataset
This dataset focuses on air quality assessment across various regions. The dataset contains 5000 samples and captures critical environmental and demographic factors that influence pollution levels.

Key Features:

Temperature (°C): Average temperature of the region.      
Humidity (%): Relative humidity recorded in the region.       
PM2.5 Concentration (µg/m³): Fine particulate matter levels.       
PM10 Concentration (µg/m³): Coarse particulate matter levels.      
NO2 Concentration (ppb): Nitrogen dioxide levels.       
SO2 Concentration (ppb): Sulfur dioxide levels.      
CO Concentration (ppm): Carbon monoxide levels.      
Proximity to Industrial Areas (km): Distance to the nearest industrial zone.    
Population Density (people/km²): Number of people per square kilometer in the region.      
Target Variable: Air Quality Levels       
        
Good: Clean air with low pollution levels.     
Moderate: Acceptable air quality but with some pollutants present.     
Poor: Noticeable pollution that may cause health issues for sensitive groups.     
Hazardous: Highly polluted air posing serious health risks to the population.
"""

# Importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
warnings.filterwarnings('ignore')

# Loading dataset
dataset = pd.read_csv("/content/updated_pollution_dataset.csv")
dataset

"""### Checking the dataset"""

# Head of the dataset
dataset.head()

# Shape of the dataset
print(f"Number of rows: {dataset.shape[0]}")
print(f"Number of columns: {dataset.shape[1]}")

# Statistical insights on the dataset
dataset.describe()

# General insights on the dataset
dataset.info()

# Unique categories of Air Quality column
dataset['Air Quality'].unique()

# Distribution of each categories
dataset['Air Quality'].value_counts()

# plotting the distribution for visuals
air_quality_data = dataset['Air Quality'].value_counts()
air_quality_data.plot(kind="bar")

"""Dataset biased towards "good" and "moderate" and might not predict "poorp" and hazardous" very accurately.

### Data preprocessing
"""

# Splitting into input and output labels
X = dataset.drop(columns='Air Quality', axis=1)
y = dataset['Air Quality']

# Scaling input labels

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

# encoding output label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

"""### Traing logistic regression"""

# Instance of Logistic regression
sv_model = SVC()

# Training logistic regression with training data
sv_model.fit(X_train, y_train)

# Using trained model to predict test input labels
y_pred = sv_model.predict(X_test)

# Checking accuracy score
model_accuracy_score = accuracy_score(y_test, y_pred)
print(f"Model accuracy score: {model_accuracy_score}")

# Checking confusion matrix
confusioMatrix = confusion_matrix(y_test, y_pred)
confusioMatrix

