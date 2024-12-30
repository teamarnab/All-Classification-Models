# Air Quality Classification Models

This repository contains machine learning models used for classifying air quality based on a dataset. The models implemented include Logistic Regression, K-Nearest Neighbors (KNN) Classifier, Decision Tree Classifier, Random Forest Classifier, and Support Vector Classifier (SVC). The models are assessed using metrics such as Accuracy Score and Confusion Matrix.

## Dataset
The dataset used in this project contains information about air quality measurements. The features and target variable are preprocessed and prepared for classification tasks.

## Models Implemented
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN) Classifier**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
5. **Support Vector Classifier (SVC)**

## Performance Metrics
The models are evaluated using the following metrics:
- **Accuracy Score**: Measures the proportion of correctly predicted instances.
- **Confusion Matrix**: Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

## Project Structure
```
├── data
│   └── updated_pollution_dataset.csv  # The dataset used for training and evaluation
├── Logistic Regression
│   └── logistic_regression.py
│   └── README.md
├── *K-Nearest Neighbors (KNN) Classifier
│   └── knn_classifier.py
│   └── README.md
├── Decision Tree Classifier
│   └── decision_tree_classifier.py
│   └── README.md
├── Random Forest Classifier
│   └── random_forest_classifier.py
│   └── README.md
├── Support Vector Classifier (SVC)
│   └── support_vector_classifier.py
│   └── README.md
├── README.md  # Project documentation
```

## Getting Started
### Prerequisites
- Python 3.8 or above
- Jupyter Notebook

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/air-quality-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd air-quality-classification
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/air_quality_classification.ipynb
   ```
2. Run the cells to train models and evaluate their performance.

## Results
- The results of the accuracy scores and confusion matrices are presented in the notebook.
- Confusion matrices are saved in the `results/confusion_matrices` directory for each model.

## Contributing
Feel free to fork this repository and submit pull requests for improvements or additional features.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- Dataset source (kaggle): https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment?resource=download
- Python libraries used, including scikit-learn, pandas, and matplotlib.

## Contact
For any inquiries or feedback, please reach out at teamarnab2014@outlook.com.
